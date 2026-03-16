"""
Pipeline
--------
End-to-end orchestration of the failure analysis pipeline:
  collect → embed → cluster → label → report → visualize
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from rich.console import Console

from src.collector import collect
from src.clusterer import Clusterer
from src.embedder import OllamaEmbedder, get_embeddings_matrix
from src.labeler import ClusterLabeler
from src.models import AnalysisReport, EmbeddedTranscript
from src.reporter import build_report, print_summary, save_report
from src.visualizer import save_all

console = Console()


@dataclass
class PipelineConfig:
    # Paths
    transcripts_dir: str = "data/transcripts"
    output_dir: str = "reports"
    embeddings_cache: str = "data/embeddings_cache.npy"
    metadata_cache: str = "data/metadata_cache.json"

    # Models
    embed_model: str = "nomic-embed-text"
    label_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 120

    # UMAP
    umap_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"
    umap_random_state: int = 42

    # HDBSCAN
    hdbscan_min_cluster_size: int = 2
    hdbscan_min_samples: int = 1

    # Labeling
    max_examples_per_cluster: int = 3

    # Runtime
    force_reembed: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        cfg = cls()
        ollama = raw.get("ollama", {})
        clustering = raw.get("clustering", {})
        paths = raw.get("paths", {})
        report = raw.get("report", {})

        cfg.embed_model = ollama.get("embed_model", cfg.embed_model)
        cfg.label_model = ollama.get("label_model", cfg.label_model)
        cfg.ollama_base_url = ollama.get("base_url", cfg.ollama_base_url)
        cfg.ollama_timeout = ollama.get("timeout", cfg.ollama_timeout)

        cfg.umap_neighbors = clustering.get("umap_neighbors", cfg.umap_neighbors)
        cfg.umap_min_dist = clustering.get("umap_min_dist", cfg.umap_min_dist)
        cfg.umap_metric = clustering.get("umap_metric", cfg.umap_metric)
        cfg.umap_random_state = clustering.get("umap_random_state", cfg.umap_random_state)
        cfg.hdbscan_min_cluster_size = clustering.get("hdbscan_min_cluster_size", cfg.hdbscan_min_cluster_size)
        cfg.hdbscan_min_samples = clustering.get("hdbscan_min_samples", cfg.hdbscan_min_samples)

        cfg.transcripts_dir = paths.get("transcripts_dir", cfg.transcripts_dir)
        cfg.embeddings_cache = paths.get("embeddings_cache", cfg.embeddings_cache)
        cfg.metadata_cache = paths.get("metadata_cache", cfg.metadata_cache)
        cfg.output_dir = paths.get("output_dir", cfg.output_dir)

        cfg.max_examples_per_cluster = report.get("max_examples_per_cluster", cfg.max_examples_per_cluster)

        return cfg


class FailureAnalysisPipeline:
    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self._embedded: list[EmbeddedTranscript] = []
        self._report: AnalysisReport | None = None

    def run(self) -> AnalysisReport:
        console.rule("[bold blue]LLM Agent Failure Mode Analyzer[/bold blue]")

        # ── Step 1: Collect ──────────────────
        console.print("\n[bold]Step 1 / 5 — Collecting transcripts…[/bold]")
        transcripts = collect(self.cfg.transcripts_dir)
        if not transcripts:
            raise ValueError(f"No transcripts found in {self.cfg.transcripts_dir!r}. "
                             "Add .json, .jsonl, or .txt files.")

        # ── Step 2: Embed ────────────────────
        console.print("\n[bold]Step 2 / 5 — Embedding…[/bold]")
        embedder = OllamaEmbedder(
            model=self.cfg.embed_model,
            base_url=self.cfg.ollama_base_url,
            cache_path=self.cfg.embeddings_cache,
            meta_path=self.cfg.metadata_cache,
            timeout=self.cfg.ollama_timeout,
        )
        embedded = embedder.embed_all(transcripts, force_reembed=self.cfg.force_reembed)
        matrix = get_embeddings_matrix(embedded)

        # ── Step 3: Cluster ──────────────────
        console.print("\n[bold]Step 3 / 5 — Clustering…[/bold]")
        clusterer = Clusterer(
            umap_neighbors=self.cfg.umap_neighbors,
            umap_min_dist=self.cfg.umap_min_dist,
            umap_metric=self.cfg.umap_metric,
            umap_random_state=self.cfg.umap_random_state,
            hdbscan_min_cluster_size=self.cfg.hdbscan_min_cluster_size,
            hdbscan_min_samples=self.cfg.hdbscan_min_samples,
        )
        embedded = clusterer.fit_transform(embedded, matrix)
        cluster_groups = clusterer.cluster_members(embedded)

        # ── Step 4: Label ────────────────────
        console.print("\n[bold]Step 4 / 5 — Labeling clusters with Ollama…[/bold]")
        labeler = ClusterLabeler(
            model=self.cfg.label_model,
            base_url=self.cfg.ollama_base_url,
            timeout=self.cfg.ollama_timeout,
            max_examples=self.cfg.max_examples_per_cluster,
        )
        noise_count = len(cluster_groups.get(-1, []))
        total_clustered = len(embedded) - noise_count
        cluster_results = labeler.label_clusters(cluster_groups, total_clustered)

        # ── Step 5: Report + Visualize ───────
        console.print("\n[bold]Step 5 / 5 — Generating reports & visualizations…[/bold]")
        report = build_report(
            embedded=embedded,
            cluster_results=cluster_results,
            model=self.cfg.label_model,
            embed_model=self.cfg.embed_model,
        )
        report_files = save_report(report, self.cfg.output_dir)
        viz_files = save_all(embedded, report, self.cfg.output_dir)

        self._embedded = embedded
        self._report = report

        # ── Terminal summary ─────────────────
        print_summary(report)

        all_files = {**report_files, **viz_files}
        console.print("[bold green]✓ Output files:[/bold green]")
        for name, path in all_files.items():
            console.print(f"  {name:15} → [cyan]{path}[/cyan]")
        console.print()

        return report

    @property
    def embedded(self) -> list[EmbeddedTranscript]:
        return self._embedded

    @property
    def report(self) -> AnalysisReport | None:
        return self._report
