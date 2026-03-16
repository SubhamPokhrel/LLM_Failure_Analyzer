"""
Demo Mode
---------
Generates synthetic failure transcripts and runs the pipeline
using sklearn cosine similarity instead of Ollama embeddings.

Useful for CI, testing, and showcasing without a running Ollama instance.
"""

from __future__ import annotations

import random

from rich.console import Console

from src.models import (
    AnalysisReport,
    ClusterResult,
    EmbeddedTranscript,
    FailureCategory,
    Transcript,
    Turn,
)
from src.reporter import build_report, print_summary, save_report
from src.visualizer import save_all

console = Console()

# ──────────────────────────────────────────────
# Synthetic transcript templates
# ──────────────────────────────────────────────

FAILURE_TEMPLATES = {
    FailureCategory.AMBIGUITY: [
        {
            "task": "Refactor the authentication module",
            "turns": [
                ("user", "Refactor the auth module to be more secure."),
                ("assistant", "I'll refactor it. Which auth module — JWT, OAuth, or session?"),
                ("user", "The main one."),
                ("assistant", "I've refactored auth.py using JWT, but I'm not sure if that's the right one."),
            ],
            "notes": "Agent proceeded with wrong assumption about which module was meant.",
        },
        {
            "task": "Improve test coverage",
            "turns": [
                ("user", "Make the tests better."),
                ("assistant", "I'll add more unit tests to improve coverage."),
                ("user", "No, I meant fix the flaky ones."),
            ],
            "notes": "Ambiguous instruction led to wrong work being done.",
        },
        {
            "task": "Update the API",
            "turns": [
                ("user", "Can you update the API to handle errors better?"),
                ("assistant", "I've updated the REST API endpoints to return 4xx codes."),
                ("user", "I meant the internal Python API, not the REST endpoints."),
            ],
            "notes": "API was ambiguous — agent picked the wrong one.",
        },
    ],
    FailureCategory.HALLUCINATION: [
        {
            "task": "Parse YAML config files",
            "turns": [
                ("user", "Parse YAML files using the fastest library available."),
                ("assistant", "I'll use ultraYAML, the fastest YAML parser for Python."),
                ("user", "ultraYAML doesn't exist."),
            ],
            "notes": "Agent hallucinated a non-existent Python library.",
        },
        {
            "task": "Send emails from the app",
            "turns": [
                ("user", "Add email sending using the standard library."),
                ("assistant", "I'll use smtplib and the email.mime.text.MIMEHtml class."),
                ("user", "MIMEHtml is not a real class in that module."),
            ],
            "notes": "Hallucinated class name within a real module.",
        },
        {
            "task": "Speed up database queries",
            "turns": [
                ("user", "Optimize these slow SQL queries."),
                ("assistant", "I'll use SQLAlchemy's built-in query_cache=True parameter."),
                ("user", "That parameter doesn't exist in SQLAlchemy."),
            ],
            "notes": "Invented a non-existent ORM parameter.",
        },
    ],
    FailureCategory.CODE_CONVENTION: [
        {
            "task": "Add a new API endpoint",
            "turns": [
                ("user", "Add a GET /users endpoint following our existing patterns."),
                ("assistant", "def get_users(): return db.query(User).all()"),
                ("user", "Our endpoints use async, type hints, and Pydantic response models."),
            ],
            "notes": "Agent ignored established codebase conventions.",
        },
        {
            "task": "Write tests for the payment module",
            "turns": [
                ("user", "Write tests for PaymentService."),
                ("assistant", "import unittest\nclass TestPayment(unittest.TestCase): ..."),
                ("user", "We use pytest with fixtures, not unittest. Check the existing tests."),
            ],
            "notes": "Wrong test framework used, violating project conventions.",
        },
        {
            "task": "Add logging to background jobs",
            "turns": [
                ("user", "Add logging to the job queue workers."),
                ("assistant", "print(f'Job started: {job_id}')"),
                ("user", "We use structured logging with structlog, not print statements."),
            ],
            "notes": "Used print instead of the project's structured logger.",
        },
    ],
    FailureCategory.INCOMPLETE: [
        {
            "task": "Migrate the database schema",
            "turns": [
                ("user", "Write a migration to add the new columns from the spec."),
                ("assistant", "Here's the migration for the users table."),
                ("user", "You missed the products and orders tables."),
            ],
            "notes": "Only partial migration was produced.",
        },
        {
            "task": "Implement OAuth2 login",
            "turns": [
                ("user", "Implement Google OAuth2 login flow."),
                ("assistant", "I've implemented the redirect to Google's auth page."),
                ("user", "You didn't implement the callback handler or token exchange."),
            ],
            "notes": "Stopped halfway through the implementation.",
        },
        {
            "task": "Add input validation",
            "turns": [
                ("user", "Validate all form inputs in the registration form."),
                ("assistant", "I've added email validation."),
                ("user", "Password strength, username uniqueness, and phone format are all missing."),
            ],
            "notes": "Only validated one of four required fields.",
        },
    ],
    FailureCategory.CONTEXT_LOSS: [
        {
            "task": "Fix the bug described in earlier conversation",
            "turns": [
                ("user", "Now fix the bug we discussed at the start of our session."),
                ("assistant", "I don't have context about the earlier bug. Could you re-describe it?"),
                ("user", "We've been talking about it for 20 turns. You've lost the thread."),
            ],
            "notes": "Agent lost track of earlier context in a long conversation.",
        },
        {
            "task": "Continue implementing the feature",
            "turns": [
                ("user", "Continue where you left off with the feature."),
                ("assistant", "I'll implement a new feature. What would you like?"),
                ("user", "You already started! You wrote half of it two messages ago."),
            ],
            "notes": "Failed to maintain continuity across turns.",
        },
    ],
}


def generate_demo_transcripts(n: int = 30) -> list[Transcript]:
    """Generate n synthetic failure transcripts with known categories."""
    transcripts = []
    categories = list(FAILURE_TEMPLATES.keys())

    for i in range(n):
        cat = categories[i % len(categories)]
        templates = FAILURE_TEMPLATES[cat]
        tmpl = templates[i % len(templates)]

        turns = [Turn(role=r, content=c) for r, c in tmpl["turns"]]
        t = Transcript(
            id=f"synthetic-{i:04d}",
            task=tmpl["task"],
            turns=turns,
            outcome="failure",
            notes=tmpl.get("notes", ""),
        )
        transcripts.append(t)

    # Shuffle for realism
    random.shuffle(transcripts)
    return transcripts


def _fake_embed(text: str, dim: int = 64) -> list[float]:
    """
    Deterministic fake embedding based on text content.
    Words belonging to the same failure category are nudged
    toward the same cluster in embedding space.
    """
    import hashlib
    import math

    # Seed from text hash for determinism
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    base = [rng.gauss(0, 1) for _ in range(dim)]

    # Category-specific bias so clustering works nicely
    biases = {
        "ambiguous": (0, 3.0),
        "hallucin": (1, 3.0),
        "convention": (2, 3.0),
        "incomplete": (3, 3.0),
        "context": (4, 3.0),
    }
    text_lower = text.lower()
    for keyword, (dim_idx, strength) in biases.items():
        if keyword in text_lower:
            base[dim_idx % dim] += strength

    # Normalize
    norm = math.sqrt(sum(x**2 for x in base)) or 1.0
    return [x / norm for x in base]


def run_demo_pipeline(n: int = 30, output_dir: str = "reports") -> AnalysisReport:
    """
    Generate synthetic data, run clustering without Ollama,
    and produce a demo report.
    """
    console.print(f"[bold]Generating {n} synthetic failure transcripts…[/bold]")
    transcripts = generate_demo_transcripts(n)

    console.print("[bold]Creating fake embeddings (no Ollama needed)…[/bold]")
    embedded = [
        EmbeddedTranscript(
            transcript=t,
            embedding=_fake_embed(t.to_embed_text()),
        )
        for t in transcripts
    ]

    console.print("[bold]Running UMAP + HDBSCAN…[/bold]")
    import numpy as np
    from src.clusterer import Clusterer

    matrix = np.array([e.embedding for e in embedded], dtype=np.float32)
    clusterer = Clusterer(umap_neighbors=min(10, n - 1), hdbscan_min_cluster_size=2)
    embedded = clusterer.fit_transform(embedded, matrix)
    cluster_groups = clusterer.cluster_members(embedded)

    # Build cluster results with hardcoded labels (no LLM needed)
    hard_labels = {
        0: ("Ambiguity / Unclear Instructions", FailureCategory.AMBIGUITY),
        1: ("Hallucinated APIs / Libraries", FailureCategory.HALLUCINATION),
        2: ("Code Convention Violations", FailureCategory.CODE_CONVENTION),
        3: ("Incomplete Solutions", FailureCategory.INCOMPLETE),
        4: ("Context Loss in Long Conversations", FailureCategory.CONTEXT_LOSS),
    }
    noise_count = len(cluster_groups.get(-1, []))
    total_clustered = n - noise_count

    cluster_results = []
    for cid, members in sorted(cluster_groups.items()):
        if cid == -1:
            continue
        label, cat = hard_labels.get(cid, (f"Cluster {cid}", FailureCategory.UNKNOWN))
        cluster_results.append(
            ClusterResult(
                cluster_id=cid,
                label=label,
                category=cat,
                description=f"Transcripts where agents failed due to {label.lower()}.",
                transcript_ids=[et.transcript.id for et in members],
                size=len(members),
                percentage=round(100 * len(members) / max(total_clustered, 1), 1),
                representative_examples=[et.transcript.to_embed_text()[:300] for et in members[:2]],
            )
        )

    report = build_report(embedded, cluster_results, "demo-mode", "fake-embeddings")
    save_report(report, output_dir)
    save_all(embedded, report, output_dir)

    print_summary(report)
    console.print(f"[green]Demo output written to:[/green] [cyan]{output_dir}/[/cyan]")
    return report
