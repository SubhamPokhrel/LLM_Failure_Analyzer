"""
Labeler
-------
Sends each cluster's representative transcripts to an Ollama LLM
and asks it to assign a failure category and human-readable label.

Strategy (in order):
  1. Call Ollama with a strict JSON system prompt
  2. If the model wraps output in markdown fences or prose, extract JSON anyway
  3. If JSON is malformed, try a second simpler prompt ("just the label")
  4. If Ollama is unreachable or the model times out, fall back to
     keyword-based heuristic labeling from the transcript text itself
     — so the report always has meaningful labels, never "Cluster 0"

Returns enriched ClusterResult objects.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter

import httpx
from rich.console import Console

from src.models import ClusterResult, EmbeddedTranscript, FailureCategory

console = Console()

# ──────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a research analyst specializing in LLM agent failure modes.
You will be given failed agent transcripts that share a common failure pattern.
Your job: identify that pattern and respond ONLY with a JSON object.

IMPORTANT: Output raw JSON only. No markdown, no code fences, no explanation.

Schema (fill in the angle-bracket placeholders):
{"label": "<2-5 word failure category name>", "category": "<one of: ambiguity_errors|code_convention_violations|incomplete_solutions|hallucinated_apis|communication_failure|design_decision_errors|context_loss|reasoning_errors|unknown>", "description": "<one sentence describing the shared failure pattern>"}"""

LABEL_PROMPT = """\
These {n} agent transcripts all failed in a similar way. Identify the failure mode.

{examples}

Reply with JSON only — no markdown, no preamble."""

# Simpler retry prompt when the first attempt returns garbled JSON
RETRY_PROMPT = """\
Given these failed agent transcripts, reply with ONLY this JSON (no other text):
{{"label": "X", "category": "Y", "description": "Z"}}

Where:
- label = a short 2-5 word name for the failure type
- category = one of: ambiguity_errors, code_convention_violations, incomplete_solutions, hallucinated_apis, communication_failure, design_decision_errors, context_loss, reasoning_errors, unknown
- description = one sentence

Transcripts:
{examples}"""

# ──────────────────────────────────────────────
# Keyword-based fallback (no LLM needed)
# ──────────────────────────────────────────────

KEYWORD_RULES: list[tuple[list[str], FailureCategory, str]] = [
    (
        ["hallucin", "doesn't exist", "does not exist", "no such", "invented",
         "made up", "fabricat", "nonexistent", "not a real", "fake library",
         "fake api", "fake function", "wrong import"],
        FailureCategory.HALLUCINATION,
        "Hallucinated APIs / Libraries",
    ),
    (
        ["ambiguous", "unclear", "which one", "wrong module", "wrong file",
         "assumed", "assumption", "misunderstood", "meant the other",
         "not what i meant", "misinterpreted"],
        FailureCategory.AMBIGUITY,
        "Ambiguous Instruction Handling",
    ),
    (
        ["convention", "wrong framework", "wrong pattern", "should use",
         "we use", "our style", "existing pattern", "codebase", "project style",
         "async", "decorator", "structlog", "print statement"],
        FailureCategory.CODE_CONVENTION,
        "Code Convention Violations",
    ),
    (
        ["incomplete", "missing", "forgot", "only part", "partial",
         "didn't finish", "left out", "not all", "you missed", "half"],
        FailureCategory.INCOMPLETE,
        "Incomplete Solutions",
    ),
    (
        ["context", "earlier", "you said", "we discussed", "forgot",
         "lost track", "previous", "beginning of", "start of session"],
        FailureCategory.CONTEXT_LOSS,
        "Context Loss in Long Conversations",
    ),
    (
        ["design", "architecture", "scale", "million", "performance",
         "wrong approach", "frontend filter", "server side", "pattern"],
        FailureCategory.DESIGN,
        "Design Decision Errors",
    ),
    (
        ["miscommun", "misunderstand", "did not clarify", "failed to ask",
         "did not confirm", "wrong requirement"],
        FailureCategory.COMMUNICATION,
        "Communication Failures",
    ),
    (
        ["reasoning", "logic", "wrong conclusion", "incorrect analysis",
         "bad assumption", "root cause", "band-aid", "symptom"],
        FailureCategory.REASONING,
        "Reasoning / Diagnostic Errors",
    ),
]


def keyword_label(members: list[EmbeddedTranscript]) -> tuple[str, FailureCategory, str]:
    """
    Score each failure category by keyword hits across all cluster members.
    Returns (label, category, description).
    """
    combined = " ".join(
        et.transcript.to_embed_text().lower() for et in members
    )
    scores: Counter[int] = Counter()
    for i, (keywords, _cat, _label) in enumerate(KEYWORD_RULES):
        for kw in keywords:
            if kw in combined:
                scores[i] += 1

    if scores:
        best_idx = scores.most_common(1)[0][0]
        _, category, label = KEYWORD_RULES[best_idx]
        desc = f"Agent transcripts suggest a pattern of {label.lower()}."
        return label, category, desc

    # Absolute fallback: use the most common words in the notes/tasks
    words = re.findall(r"\b[a-z]{4,}\b", combined)
    stop = {"that", "this", "with", "from", "have", "been", "will", "were",
            "they", "when", "what", "just", "more", "some", "task", "user",
            "assistant", "notes", "outcome", "failure"}
    top = [w for w, _ in Counter(words).most_common(10) if w not in stop]
    label = " ".join(top[:3]).title() if top else "Unclassified Failures"
    return label, FailureCategory.UNKNOWN, f"Common theme: {', '.join(top[:5])}."


# ──────────────────────────────────────────────
# Labeler
# ──────────────────────────────────────────────

class ClusterLabeler:
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        max_examples: int = 3,
        max_chars_per_example: int = 800,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_examples = max_examples
        self.max_chars_per_example = max_chars_per_example
        self._client = httpx.Client(timeout=timeout)
        self._ollama_available: bool | None = None  # cached after first check

    # ──────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────

    def label_clusters(
        self,
        cluster_groups: dict[int, list[EmbeddedTranscript]],
        total_clustered: int,
    ) -> list[ClusterResult]:
        """Label every cluster (skips noise cluster -1)."""
        # Check Ollama once up front so we don't spam connection errors
        ollama_ok = self._check_ollama()
        if not ollama_ok:
            console.print(
                "[yellow]⚠  Ollama unreachable — using keyword-based labeling for all clusters.\n"
                "   Start Ollama with [bold]ollama serve[/bold] for AI-powered labels.[/yellow]"
            )

        results: list[ClusterResult] = []
        for cluster_id, members in sorted(cluster_groups.items()):
            if cluster_id == -1:
                continue
            console.print(
                f"\n[bold]Labeling cluster {cluster_id}[/bold] "
                f"({len(members)} transcript(s))…"
            )
            result = self._label_one(cluster_id, members, total_clustered, ollama_ok)
            results.append(result)
            console.print(
                f"  → [cyan]{result.label}[/cyan]  "
                f"[dim]({result.category.value})[/dim]  {result.percentage:.1f}%"
            )

        return results

    # ──────────────────────────────────────────
    # Internal — labeling
    # ──────────────────────────────────────────

    def _label_one(
        self,
        cluster_id: int,
        members: list[EmbeddedTranscript],
        total_clustered: int,
        ollama_ok: bool,
    ) -> ClusterResult:
        examples = members[: self.max_examples]
        examples_text = self._format_examples(examples)
        label, category, description = None, None, None

        if ollama_ok:
            label, category, description = self._llm_label(examples_text, len(members))

        # Always fall back to keyword labeling if LLM failed or was skipped
        if not label:
            console.print(
                "  [yellow]→ keyword fallback[/yellow] "
                "(LLM unavailable or returned unparseable output)"
            )
            label, category, description = keyword_label(members)

        return ClusterResult(
            cluster_id=cluster_id,
            label=label,
            description=description,
            category=category,
            transcript_ids=[et.transcript.id for et in members],
            size=len(members),
            percentage=round(100 * len(members) / max(total_clustered, 1), 1),
            representative_examples=[
                et.transcript.to_embed_text()[:300] for et in examples
            ],
        )

    def _llm_label(
        self, examples_text: str, n: int
    ) -> tuple[str | None, FailureCategory | None, str | None]:
        """
        Attempt LLM labeling with one retry on JSON parse failure.
        Returns (label, category, description) or (None, None, None) on failure.
        """
        # Attempt 1: strict JSON prompt
        try:
            raw = self._call_ollama(
                LABEL_PROMPT.format(n=n, examples=examples_text),
                system=SYSTEM_PROMPT,
            )
            console.print(f"  [dim]LLM raw response: {raw[:120].strip()!r}[/dim]")
            parsed = self._extract_json(raw)
            if parsed:
                return self._unpack(parsed)
        except Exception as e:
            console.print(f"  [yellow]Attempt 1 failed: {e}[/yellow]")

        # Attempt 2: simpler prompt, no system message
        try:
            time.sleep(0.5)
            raw2 = self._call_ollama(
                RETRY_PROMPT.format(examples=examples_text),
                system=None,
            )
            console.print(f"  [dim]Retry raw response: {raw2[:120].strip()!r}[/dim]")
            parsed2 = self._extract_json(raw2)
            if parsed2:
                return self._unpack(parsed2)
        except Exception as e:
            console.print(f"  [yellow]Attempt 2 failed: {e}[/yellow]")

        return None, None, None

    @staticmethod
    def _unpack(parsed: dict) -> tuple[str, FailureCategory, str]:
        label = str(parsed.get("label") or "").strip() or None
        description = str(parsed.get("description") or "").strip()
        cat_str = str(parsed.get("category") or "unknown").strip().lower()
        # Normalise: strip surrounding quotes, spaces
        cat_str = cat_str.strip("'\"")
        try:
            category = FailureCategory(cat_str)
        except ValueError:
            # Try partial match
            category = FailureCategory.UNKNOWN
            for member in FailureCategory:
                if member.value in cat_str or cat_str in member.value:
                    category = member
                    break
        return label, category, description

    def _format_examples(self, examples: list[EmbeddedTranscript]) -> str:
        parts = []
        for i, et in enumerate(examples, 1):
            text = et.transcript.to_embed_text()[: self.max_chars_per_example]
            parts.append(f"[Example {i}]\n{text}")
        return "\n\n".join(parts)

    # ──────────────────────────────────────────
    # Internal — Ollama I/O
    # ──────────────────────────────────────────

    def _call_ollama(self, prompt: str, system: str | None) -> str:
        """
        Call Ollama using whichever API endpoint is available.
        Newer Ollama (≥0.1.14): /api/chat with messages list.
        Older Ollama:           /api/generate with a single prompt string.
        Auto-detected once and cached in self._use_generate.
        """
        endpoint, payload = self._build_request(prompt, system)
        resp = self._client.post(endpoint, json=payload)

        # If /api/chat 404s, fall back to /api/generate permanently
        if resp.status_code == 404 and "chat" in endpoint:
            console.print(
                "[dim]/api/chat not found — switching to /api/generate "
                "(older Ollama version)[/dim]"
            )
            self._use_generate = True
            endpoint, payload = self._build_request(prompt, system)
            resp = self._client.post(endpoint, json=payload)

        resp.raise_for_status()
        return self._parse_ollama_response(resp)

    def _build_request(self, prompt: str, system: str | None) -> tuple[str, dict]:
        """Return (url, payload) for whichever endpoint we're using."""
        if getattr(self, "_use_generate", False):
            # /api/generate — older Ollama
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            return (
                f"{self.base_url}/api/generate",
                {"model": self.model, "prompt": full_prompt, "stream": False},
            )
        else:
            # /api/chat — newer Ollama (default)
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            return (
                f"{self.base_url}/api/chat",
                {"model": self.model, "stream": False, "messages": messages},
            )

    @staticmethod
    def _parse_ollama_response(resp: httpx.Response) -> str:
        """Extract the text content from either /api/chat or /api/generate response."""
        data = resp.json()
        # /api/chat response shape
        if "message" in data:
            content = data["message"].get("content", "")
        # /api/generate response shape
        elif "response" in data:
            content = data["response"]
        else:
            raise ValueError(f"Unrecognised Ollama response shape: {list(data.keys())}")
        if not content or not content.strip():
            raise ValueError("Empty response from Ollama")
        return content

    @staticmethod
    def _extract_json(raw: str) -> dict | None:
        """
        Robustly extract a JSON object from raw LLM output.
        Handles: plain JSON, markdown fences, JSON embedded in prose.
        """
        if not raw:
            return None

        # Strip markdown fences
        clean = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE)
        clean = re.sub(r"```", "", clean).strip()

        # Try parsing the whole cleaned string first
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            pass

        # Extract the first {...} block (handles prose preamble)
        match = re.search(r"\{[^{}]*\}", clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Greedy: find the outermost { ... } span
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(clean[start: end + 1])
            except json.JSONDecodeError:
                pass

        return None

    def _check_ollama(self) -> bool:
        """
        Returns True if Ollama is reachable AND the requested model is available.
        Uses /api/tags to list models; also probes /api/chat vs /api/generate.
        """
        if self._ollama_available is not None:
            return self._ollama_available
        try:
            resp = self._client.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            model_base = self.model.split(":")[0]
            if not any(model_base in m for m in models):
                console.print(
                    f"[yellow]⚠  Model '{self.model}' not pulled yet.\n"
                    f"   Run: [bold]ollama pull {self.model}[/bold]\n"
                    f"   Available models: {', '.join(models) or 'none'}[/yellow]"
                )
                # Model missing → can't label; fall back to keywords
                self._ollama_available = False
                return False

            # Probe which chat endpoint this Ollama version supports
            probe = self._client.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "stream": False,
                      "messages": [{"role": "user", "content": "hi"}]},
                timeout=10,
            )
            if probe.status_code == 404:
                console.print(
                    "[dim]Ollama /api/chat not available — will use /api/generate.[/dim]"
                )
                self._use_generate = True
            else:
                self._use_generate = False

            self._ollama_available = True
        except Exception as e:
            console.print(f"[yellow]⚠  Ollama unreachable ({e}).[/yellow]")
            self._ollama_available = False
        return self._ollama_available
