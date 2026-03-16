#!/usr/bin/env python3
"""
analyze.py — CLI for the LLM Agent Failure Mode Analyzer

Usage:
    python analyze.py run
    python analyze.py run --model llama3.2 --embed-model nomic-embed-text
    python analyze.py collect
    python analyze.py embed
    python analyze.py cluster
    python analyze.py report
"""

import sys
from pathlib import Path

import click
from rich.console import Console

console = Console()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

CONFIG_FILE = "config.yaml"


def load_config(overrides: dict) -> "PipelineConfig":  # noqa: F821
    from src.pipeline import PipelineConfig

    cfg_path = Path(CONFIG_FILE)
    if cfg_path.exists():
        cfg = PipelineConfig.from_yaml(cfg_path)
    else:
        cfg = PipelineConfig()

    for k, v in overrides.items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# ──────────────────────────────────────────────
# CLI group
# ──────────────────────────────────────────────

@click.group()
def cli():
    """🔬 LLM Agent Failure Mode Analyzer"""
    pass


# ──────────────────────────────────────────────
# run — full pipeline
# ──────────────────────────────────────────────

@cli.command()
@click.option("--model", default=None, help="Ollama model for cluster labeling")
@click.option("--embed-model", default=None, help="Ollama embedding model")
@click.option("--transcripts-dir", default=None, help="Path to transcripts directory")
@click.option("--output-dir", default=None, help="Output directory for reports")
@click.option("--min-cluster-size", default=None, type=int, help="HDBSCAN min_cluster_size")
@click.option("--umap-neighbors", default=None, type=int, help="UMAP n_neighbors")
@click.option("--force-reembed", is_flag=True, default=False, help="Ignore embedding cache")
@click.option("--config", default=CONFIG_FILE, help="Path to config.yaml")
def run(model, embed_model, transcripts_dir, output_dir, min_cluster_size,
        umap_neighbors, force_reembed, config):
    """Run the full analysis pipeline end-to-end."""
    from src.pipeline import FailureAnalysisPipeline, PipelineConfig

    cfg_path = Path(config)
    cfg = PipelineConfig.from_yaml(cfg_path) if cfg_path.exists() else PipelineConfig()

    overrides = {
        "label_model": model,
        "embed_model": embed_model,
        "transcripts_dir": transcripts_dir,
        "output_dir": output_dir,
        "hdbscan_min_cluster_size": min_cluster_size,
        "umap_neighbors": umap_neighbors,
        "force_reembed": force_reembed,
    }
    for k, v in overrides.items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)

    try:
        pipeline = FailureAnalysisPipeline(cfg)
        pipeline.run()
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise SystemExit(1)


# ──────────────────────────────────────────────
# collect — just parse transcripts
# ──────────────────────────────────────────────

@cli.command()
@click.option("--transcripts-dir", default="data/transcripts", show_default=True)
def collect(transcripts_dir):
    """Parse and validate transcripts without embedding."""
    from src.collector import collect as _collect

    console.rule("[bold]Collecting Transcripts[/bold]")
    transcripts = _collect(transcripts_dir)
    for t in transcripts:
        console.print(
            f"  [green]{t.id}[/green]  "
            f"task={t.task[:60]!r}  "
            f"turns={len(t.turns)}  "
            f"outcome={t.outcome}"
        )


# ──────────────────────────────────────────────
# embed — collect + embed, cache to disk
# ──────────────────────────────────────────────

@cli.command()
@click.option("--transcripts-dir", default="data/transcripts", show_default=True)
@click.option("--embed-model", default="nomic-embed-text", show_default=True)
@click.option("--force", is_flag=True, default=False)
def embed(transcripts_dir, embed_model, force):
    """Collect transcripts and generate embeddings."""
    from src.collector import collect as _collect
    from src.embedder import OllamaEmbedder

    transcripts = _collect(transcripts_dir)
    embedder = OllamaEmbedder(model=embed_model)
    embedded = embedder.embed_all(transcripts, force_reembed=force)
    console.print(f"\n[green]✓[/green] {len(embedded)} embeddings ready.")


# ──────────────────────────────────────────────
# cluster — load cached embeddings + cluster
# ──────────────────────────────────────────────

@cli.command()
@click.option("--min-cluster-size", default=2, show_default=True)
@click.option("--umap-neighbors", default=15, show_default=True)
def cluster(min_cluster_size, umap_neighbors):
    """Load cached embeddings and re-run UMAP + HDBSCAN."""
    console.print("[yellow]This command requires running 'embed' first.[/yellow]")
    console.print("Use [bold]python analyze.py run[/bold] for the full pipeline.")


# ──────────────────────────────────────────────
# report — pretty-print saved report.json
# ──────────────────────────────────────────────

@cli.command()
@click.option("--output-dir", default="reports", show_default=True)
def report(output_dir):
    """Display the saved analysis report in the terminal."""
    import json
    from src.models import AnalysisReport
    from src.reporter import print_summary

    json_path = Path(output_dir) / "report.json"
    if not json_path.exists():
        console.print(f"[red]No report found at {json_path}. Run 'analyze.py run' first.[/red]")
        raise SystemExit(1)

    with open(json_path) as f:
        data = json.load(f)
    r = AnalysisReport(**data)
    print_summary(r)


# ──────────────────────────────────────────────
# demo — generate synthetic transcripts and run
# ──────────────────────────────────────────────

@cli.command()
@click.option("--output-dir", default="reports", show_default=True)
@click.option("--n", default=30, show_default=True, help="Number of synthetic transcripts")
def demo(output_dir, n):
    """
    Generate synthetic failure transcripts and run the full pipeline.
    Useful for testing without real data or Ollama.
    """
    from src.demo import run_demo_pipeline
    console.rule("[bold magenta]Demo Mode[/bold magenta]")
    run_demo_pipeline(n=n, output_dir=output_dir)


if __name__ == "__main__":
    cli()
