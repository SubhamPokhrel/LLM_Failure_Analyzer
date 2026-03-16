"""
Transcript Collector
--------------------
Ingests agent transcripts from multiple file formats and normalizes
them into the shared Transcript schema.

Supported formats:
  - .json  — single transcript or list of transcripts
  - .jsonl — one transcript per line
  - .txt   — labeled plain-text format
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Iterator

from rich.console import Console

from src.models import Transcript, Turn

console = Console()

# ──────────────────────────────────────────────
# Parser protocol
# ──────────────────────────────────────────────

class TranscriptParser:
    """Base class for transcript parsers."""

    def can_parse(self, path: Path) -> bool:
        raise NotImplementedError

    def parse(self, path: Path) -> list[Transcript]:
        raise NotImplementedError


# ──────────────────────────────────────────────
# JSON parser
# ──────────────────────────────────────────────

class JSONParser(TranscriptParser):
    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() == ".json"

    def parse(self, path: Path) -> list[Transcript]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return [self._parse_one(item, path) for item in data]
        return [self._parse_one(data, path)]

    def _parse_one(self, data: dict, path: Path) -> Transcript:
        turns = [
            Turn(role=t.get("role", "unknown"), content=t.get("content", ""))
            for t in data.get("turns", [])
        ]
        return Transcript(
            id=str(data.get("id", uuid.uuid4())),
            task=data.get("task", data.get("description", "")),
            turns=turns,
            outcome=data.get("outcome", "failure"),
            notes=data.get("notes", data.get("error", "")),
            source_file=str(path),
        )


# ──────────────────────────────────────────────
# JSONL parser
# ──────────────────────────────────────────────

class JSONLParser(TranscriptParser):
    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() == ".jsonl"

    def parse(self, path: Path) -> list[Transcript]:
        jp = JSONParser()
        results = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    results.append(jp._parse_one(data, path))
                except json.JSONDecodeError as e:
                    console.print(f"[yellow]  Skipping malformed line {i+1} in {path.name}: {e}[/yellow]")
        return results


# ──────────────────────────────────────────────
# Plain-text parser
# ──────────────────────────────────────────────

class TextParser(TranscriptParser):
    """
    Parses a simple labeled text format:

        TASK: <task description>
        USER: <message>
        ASSISTANT: <response>
        OUTCOME: failure
        NOTES: <optional notes>
    """

    LABEL_RE = re.compile(
        r"^(TASK|USER|ASSISTANT|SYSTEM|OUTCOME|NOTES)\s*:\s*(.*)$",
        re.IGNORECASE,
    )

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in (".txt", ".md")

    def parse(self, path: Path) -> list[Transcript]:
        text = path.read_text(encoding="utf-8")

        # Support multiple transcripts separated by "---" in one file
        raw_blocks = re.split(r"\n---+\n", text)
        results = []

        for block_idx, block in enumerate(raw_blocks):
            block = block.strip()
            if not block:
                continue

            task = ""
            outcome = "failure"
            notes = ""
            turns: list[Turn] = []
            current_role: str | None = None
            current_lines: list[str] = []

            def flush():
                nonlocal current_role, current_lines
                if current_role and current_lines:
                    turns.append(Turn(role=current_role, content=" ".join(current_lines).strip()))
                current_role = None
                current_lines = []

            for line in block.splitlines():
                m = self.LABEL_RE.match(line)
                if m:
                    label, value = m.group(1).upper(), m.group(2).strip()
                    if label == "TASK":
                        flush()
                        task = value
                    elif label in ("USER", "ASSISTANT", "SYSTEM"):
                        flush()
                        current_role = label.lower()
                        current_lines = [value]
                    elif label == "OUTCOME":
                        flush()
                        outcome = value.lower()
                    elif label == "NOTES":
                        flush()
                        notes = value
                elif current_role is not None:
                    current_lines.append(line)

            flush()

            file_id = f"{path.stem}-{block_idx}" if len(raw_blocks) > 1 else path.stem
            results.append(
                Transcript(
                    id=file_id,
                    task=task,
                    turns=turns,
                    outcome=outcome,
                    notes=notes,
                    source_file=str(path),
                    raw_text=block,
                )
            )

        return results


# ──────────────────────────────────────────────
# Collector
# ──────────────────────────────────────────────

PARSERS: list[TranscriptParser] = [JSONParser(), JSONLParser(), TextParser()]


def collect(transcripts_dir: str | Path) -> list[Transcript]:
    """
    Walk ``transcripts_dir`` and parse every supported file.
    Returns a deduplicated list of Transcript objects.
    """
    root = Path(transcripts_dir)
    if not root.exists():
        raise FileNotFoundError(f"Transcripts directory not found: {root}")

    all_transcripts: list[Transcript] = []
    seen_ids: set[str] = set()
    files_parsed = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        parser = next((p for p in PARSERS if p.can_parse(path)), None)
        if parser is None:
            continue

        try:
            parsed = parser.parse(path)
            for t in parsed:
                if t.id in seen_ids:
                    t.id = f"{t.id}-{uuid.uuid4().hex[:6]}"
                seen_ids.add(t.id)
                all_transcripts.append(t)
            files_parsed += 1
            console.print(f"  [green]✓[/green] {path.name} ({len(parsed)} transcript(s))")
        except Exception as e:
            console.print(f"  [red]✗[/red] {path.name}: {e}")

    console.print(
        f"\n[bold]Collected {len(all_transcripts)} transcript(s) from {files_parsed} file(s)[/bold]"
    )
    return all_transcripts


def iter_transcripts(transcripts_dir: str | Path) -> Iterator[Transcript]:
    """Generator variant of collect()."""
    yield from collect(transcripts_dir)
