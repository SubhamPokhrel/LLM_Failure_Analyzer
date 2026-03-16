"""
Visualizer
----------
Generates visualizations:
  1. Interactive UMAP scatter plot (Plotly HTML)
  2. Failure distribution bar chart (Plotly + static PNG)
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from src.models import AnalysisReport, ClusterResult, EmbeddedTranscript

# Color palette — distinct enough for up to 10 clusters
PALETTE = [
    "#E63946",  # red
    "#2196F3",  # blue
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#FF5722",  # deep orange
    "#607D8B",  # blue-grey
    "#8BC34A",  # light green
    "#F06292",  # pink
]
NOISE_COLOR = "#9E9E9E"


def build_scatter_plot(
    embedded: list[EmbeddedTranscript],
    cluster_results: list[ClusterResult],
    title: str = "LLM Agent Failure Clusters (UMAP)",
) -> go.Figure:
    """
    Return a Plotly Figure with one scatter trace per cluster.
    Noise points (cluster_id = -1) rendered in grey.
    """
    label_map: dict[int, str] = {
        cr.cluster_id: cr.label for cr in cluster_results
    }

    # Group by cluster
    groups: dict[int, list[EmbeddedTranscript]] = {}
    for et in embedded:
        groups.setdefault(et.cluster_id, []).append(et)

    fig = go.Figure()

    for cluster_id, members in sorted(groups.items()):
        is_noise = cluster_id == -1
        label = "Noise" if is_noise else label_map.get(cluster_id, f"Cluster {cluster_id}")
        color = NOISE_COLOR if is_noise else PALETTE[cluster_id % len(PALETTE)]

        hover_texts = []
        for et in members:
            task = et.transcript.task or "(no task)"
            notes = et.transcript.notes or ""
            hover_texts.append(
                f"<b>{et.transcript.id}</b><br>"
                f"Task: {task[:80]}<br>"
                f"Notes: {notes[:80]}"
            )

        fig.add_trace(
            go.Scatter(
                x=[et.umap_x for et in members],
                y=[et.umap_y for et in members],
                mode="markers",
                name=label,
                marker=dict(
                    color=color,
                    size=10,
                    opacity=0.7 if is_noise else 0.9,
                    symbol="x" if is_noise else "circle",
                    line=dict(width=1, color="white"),
                ),
                hovertemplate="%{text}<extra></extra>",
                text=hover_texts,
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis=dict(title="UMAP-1", showgrid=False, zeroline=False),
        yaxis=dict(title="UMAP-2", showgrid=False, zeroline=False),
        plot_bgcolor="#0F1117",
        paper_bgcolor="#0F1117",
        font=dict(color="#E0E0E0"),
        legend=dict(
            bgcolor="rgba(255,255,255,0.05)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
        ),
        height=600,
    )
    return fig


def build_distribution_chart(
    report: AnalysisReport,
    title: str = "Failure Mode Distribution",
) -> go.Figure:
    """Horizontal bar chart of cluster sizes sorted by percentage."""
    clusters = sorted(report.clusters, key=lambda c: c.percentage, reverse=True)
    if not clusters:
        return go.Figure()

    labels = [c.label or f"Cluster {c.cluster_id}" for c in clusters]
    percentages = [c.percentage for c in clusters]
    colors = [PALETTE[c.cluster_id % len(PALETTE)] for c in clusters]

    fig = go.Figure(
        go.Bar(
            x=percentages,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{p:.1f}%" for p in percentages],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(
            title="% of transcripts",
            range=[0, max(percentages) * 1.25],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
        ),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#0F1117",
        paper_bgcolor="#0F1117",
        font=dict(color="#E0E0E0"),
        height=max(300, len(clusters) * 60),
        margin=dict(l=20, r=80, t=60, b=40),
    )
    return fig


def save_all(
    embedded: list[EmbeddedTranscript],
    report: AnalysisReport,
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Save scatter + bar chart to `output_dir`.
    Returns dict of saved file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    scatter = build_scatter_plot(embedded, report.clusters)
    scatter_path = out / "umap_scatter.html"
    scatter.write_html(str(scatter_path))
    saved["scatter_html"] = scatter_path

    bar = build_distribution_chart(report)
    bar_html_path = out / "failure_distribution.html"
    bar.write_html(str(bar_html_path))
    saved["bar_html"] = bar_html_path

    # Try to save static PNGs (requires kaleido)
    try:
        bar_png_path = out / "failure_distribution.png"
        bar.write_image(str(bar_png_path), width=900, height=max(400, len(report.clusters) * 70))
        saved["bar_png"] = bar_png_path

        scatter_png_path = out / "umap_scatter.png"
        scatter.write_image(str(scatter_png_path), width=1000, height=700)
        saved["scatter_png"] = scatter_png_path
    except Exception:
        pass  # kaleido not installed — HTML only

    return saved
