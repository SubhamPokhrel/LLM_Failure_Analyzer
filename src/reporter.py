"""
Reporter
--------
Generates a structured Markdown report + machine-readable JSON
from the analysis results.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.models import AnalysisReport, ClusterResult, EmbeddedTranscript

console = Console()

# ──────────────────────────────────────────────
# Report builder
# ──────────────────────────────────────────────

def build_report(
    embedded: list[EmbeddedTranscript],
    cluster_results: list[ClusterResult],
    model: str,
    embed_model: str,
) -> AnalysisReport:
    """Assemble the AnalysisReport from all pipeline outputs."""
    total = len(embedded)
    noise_count = sum(1 for et in embedded if et.cluster_id == -1)
    clustered = total - noise_count

    # Re-calculate percentages relative to clustered (not total)
    for cr in cluster_results:
        cr.percentage = round(100 * cr.size / max(clustered, 1), 1)

    clusters_sorted = sorted(cluster_results, key=lambda c: c.percentage, reverse=True)

    return AnalysisReport(
        total_transcripts=total,
        clustered_transcripts=clustered,
        noise_count=noise_count,
        n_clusters=len(cluster_results),
        clusters=clusters_sorted,
        model_used=model,
        embed_model_used=embed_model,
        generated_at=datetime.now(timezone.utc).isoformat(),
        summary=_make_summary(clusters_sorted, total, noise_count),
    )


def _make_summary(
    clusters: list[ClusterResult], total: int, noise: int
) -> str:
    lines = [
        f"Analyzed {total} transcript(s). "
        f"Found {len(clusters)} failure cluster(s) with {noise} unclustered (noise) point(s).",
        "",
        "Top failure modes:",
    ]
    for c in clusters[:5]:
        lines.append(f"  • {c.label} ({c.percentage:.1f}%): {c.description}")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# File writers
# ──────────────────────────────────────────────

def save_report(report: AnalysisReport, output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    md_path = out / "report.md"
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    saved["markdown"] = md_path

    json_path = out / "report.json"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    saved["json"] = json_path

    return saved


def _render_markdown(report: AnalysisReport) -> str:
    lines: list[str] = []

    lines += [
        "# LLM Agent Failure Mode Analysis",
        "",
        f"> Generated: {report.generated_at}  ",
        f"> Embed model: `{report.embed_model_used}`  ",
        f"> Label model: `{report.model_used}`",
        "",
        "---",
        "",
        "## Summary",
        "",
        report.summary,
        "",
        "---",
        "",
        "## Failure Distribution",
        "",
        "| Rank | Failure Mode | Category | Count | % |",
        "|------|-------------|----------|-------|---|",
    ]

    for i, c in enumerate(report.clusters, 1):
        lines.append(
            f"| {i} | **{c.label}** | `{c.category.value}` | {c.size} | {c.percentage:.1f}% |"
        )

    lines += [
        "",
        f"*{report.noise_count} transcript(s) were not assigned to any cluster (noise).*",
        "",
        "---",
        "",
        "## Cluster Details",
        "",
    ]

    for c in report.clusters:
        lines += [
            f"### {c.label}",
            "",
            f"**Category:** `{c.category.value}`  ",
            f"**Size:** {c.size} transcript(s) ({c.percentage:.1f}%)  ",
            "",
            f"> {c.description}",
            "",
        ]
        if c.representative_examples:
            lines.append("**Representative examples:**")
            lines.append("")
            for j, ex in enumerate(c.representative_examples[:2], 1):
                excerpt = ex[:400].replace("\n", " ").strip()
                lines += [
                    f"<details><summary>Example {j}</summary>",
                    "",
                    f"```",
                    excerpt,
                    f"```",
                    "",
                    "</details>",
                    "",
                ]
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# Rich terminal summary
# ──────────────────────────────────────────────

def print_summary(report: AnalysisReport) -> None:
    console.print()
    console.rule("[bold yellow]Analysis Complete[/bold yellow]")
    console.print()

    if not report.clusters:
        console.print(
            "[bold red]No clusters found.[/bold red] "
            f"All {report.total_transcripts} transcript(s) were classified as noise.\n\n"
            "[yellow]Possible causes:[/yellow]\n"
            "  • Too few transcripts (need ≥ 5 for reliable clustering)\n"
            "  • All transcripts are too semantically similar\n"
            "  • All transcripts are too semantically different\n\n"
            "[yellow]Try:[/yellow]\n"
            "  • Adding more transcripts to data/transcripts/\n"
            "  • Running: [bold]python analyze.py run --min-cluster-size 2[/bold]\n"
        )
        return

    table = Table(
        title=f"Failure Mode Distribution  ({report.total_transcripts} transcripts)",
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
    )
    table.add_column("Rank", style="dim", width=5)
    table.add_column("Failure Mode", style="bold")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("%", justify="right", style="green")

    bar_width = 20
    for i, c in enumerate(report.clusters, 1):
        filled = int(c.percentage / 100 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        table.add_row(
            str(i),
            c.label,
            c.category.value,
            str(c.size),
            f"{bar} {c.percentage:.1f}%",
        )

    console.print(table)

    if report.noise_count:
        console.print(
            f"\n[yellow]⚠  {report.noise_count} transcript(s) could not be assigned "
            f"to any cluster and are excluded from the distribution above.[/yellow]"
        )
    console.print()
