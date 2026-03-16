"""
Clusterer
---------
Reduces high-dimensional embeddings to 2D with UMAP, then finds
density-based clusters with HDBSCAN.

Small-dataset strategy (n < 20):
  - UMAP neighbors auto-scaled to min(n-1, requested)
  - HDBSCAN min_cluster_size auto-scaled to max(2, n//5)
  - If HDBSCAN produces all-noise, falls back to agglomerative clustering
    on the raw cosine-distance matrix so every transcript is assigned

HDBSCAN is preferred over k-means because:
  - No need to specify k in advance
  - Handles non-convex cluster shapes
  - Labels outliers as noise (cluster_id = -1)
"""

from __future__ import annotations

import numpy as np
from rich.console import Console

try:
    import umap
except ImportError:
    raise ImportError("Run: pip install umap-learn")

try:
    import hdbscan
except ImportError:
    raise ImportError("Run: pip install hdbscan")

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

from src.models import EmbeddedTranscript

console = Console()

# Below this count, switch to more aggressive small-dataset settings
SMALL_DATASET_THRESHOLD = 30


class Clusterer:
    def __init__(
        self,
        # UMAP
        umap_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        umap_metric: str = "cosine",
        umap_random_state: int = 42,
        # HDBSCAN
        hdbscan_min_cluster_size: int = 2,
        hdbscan_min_samples: int = 1,
        hdbscan_metric: str = "euclidean",
    ):
        self.umap_neighbors = umap_neighbors
        self.umap_min_dist = umap_min_dist
        self.umap_metric = umap_metric
        self.umap_random_state = umap_random_state
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        self.hdbscan_metric = hdbscan_metric

        self._reducer: umap.UMAP | None = None
        self._clusterer: hdbscan.HDBSCAN | None = None

    # ──────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────

    def fit_transform(
        self,
        embedded: list[EmbeddedTranscript],
        embeddings_matrix: np.ndarray,
    ) -> list[EmbeddedTranscript]:
        """
        Run UMAP → HDBSCAN on `embedded`.
        For small datasets, automatically falls back to agglomerative
        clustering so that no transcripts are left as noise.

        Returns the same list with cluster_id, umap_x, umap_y populated.
        """
        n = len(embedded)

        if n < 2:
            console.print("[yellow]Only 1 transcript — assigning to cluster 0.[/yellow]")
            embedded[0].cluster_id = 0
            embedded[0].umap_x = 0.0
            embedded[0].umap_y = 0.0
            return embedded

        is_small = n < SMALL_DATASET_THRESHOLD

        # ── UMAP ──────────────────────────────────────────────────────────
        # n_neighbors must be < n; for small datasets use a tighter value
        effective_neighbors = min(self.umap_neighbors, n - 1)
        if is_small:
            effective_neighbors = min(effective_neighbors, max(2, n // 2))

        console.print(
            f"\n[bold]Running UMAP[/bold] "
            f"(n={n}, neighbors={effective_neighbors}, metric={self.umap_metric})"
            + (" [dim][small-dataset mode][/dim]" if is_small else "") + "…"
        )

        self._reducer = umap.UMAP(
            n_components=2,
            n_neighbors=effective_neighbors,
            min_dist=self.umap_min_dist,
            metric=self.umap_metric,
            random_state=self.umap_random_state,
        )
        coords_2d: np.ndarray = self._reducer.fit_transform(embeddings_matrix)

        # ── HDBSCAN ───────────────────────────────────────────────────────
        # For small datasets, use min_cluster_size=2 and min_samples=1
        # to give HDBSCAN the best chance of finding real clusters.
        if is_small:
            effective_min_cluster = 2
            effective_min_samples = 1
        else:
            effective_min_cluster = max(2, self.hdbscan_min_cluster_size)
            effective_min_samples = self.hdbscan_min_samples

        console.print(
            f"[bold]Running HDBSCAN[/bold] "
            f"(min_cluster_size={effective_min_cluster}, min_samples={effective_min_samples})…"
        )

        self._clusterer = hdbscan.HDBSCAN(
            min_cluster_size=effective_min_cluster,
            min_samples=effective_min_samples,
            metric=self.hdbscan_metric,
            cluster_selection_epsilon=0.3 if is_small else 0.0,
        )
        labels: np.ndarray = self._clusterer.fit_predict(coords_2d)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())

        # ── Fallback: all-noise → agglomerative on raw embeddings ─────────
        if n_clusters == 0:
            console.print(
                f"[yellow]HDBSCAN produced 0 clusters ({n_noise} noise points). "
                f"Falling back to agglomerative clustering on raw embeddings.[/yellow]"
            )
            labels = self._agglomerative_fallback(embeddings_matrix, n)
            n_clusters = len(set(labels))
            n_noise = 0

        # ── Partial-noise recovery: reassign noise to nearest cluster ──────
        elif n_noise > 0:
            labels = self._reassign_noise(labels, coords_2d)
            n_noise_after = int((labels == -1).sum())
            if n_noise_after < n_noise:
                console.print(
                    f"[dim]Reassigned {n_noise - n_noise_after} noise point(s) "
                    f"to nearest cluster.[/dim]"
                )
            n_noise = n_noise_after

        # ── Write back ────────────────────────────────────────────────────
        for i, et in enumerate(embedded):
            et.cluster_id = int(labels[i])
            et.umap_x = float(coords_2d[i, 0])
            et.umap_y = float(coords_2d[i, 1])

        console.print(
            f"[green]Found {n_clusters} cluster(s)[/green]"
            + (f", {n_noise} noise point(s)" if n_noise else ", 0 noise points")
        )
        return embedded

    # ──────────────────────────────────────────
    # Fallbacks
    # ──────────────────────────────────────────

    def _agglomerative_fallback(
        self, embeddings_matrix: np.ndarray, n: int
    ) -> np.ndarray:
        """
        Use agglomerative (hierarchical) clustering on the cosine distance
        matrix. Automatically picks k = max(2, sqrt(n)) clusters.
        Every point is assigned — no noise.
        """
        k = max(2, min(int(n ** 0.5), n // 2))
        console.print(
            f"[dim]Agglomerative clustering: k={k} (auto from n={n})[/dim]"
        )
        dist_matrix = cosine_distances(embeddings_matrix)
        agg = AgglomerativeClustering(
            n_clusters=k,
            metric="precomputed",
            linkage="average",
        )
        return agg.fit_predict(dist_matrix)

    @staticmethod
    def _reassign_noise(
        labels: np.ndarray, coords_2d: np.ndarray
    ) -> np.ndarray:
        """
        For each noise point (label == -1), find the nearest non-noise
        centroid in 2D UMAP space and assign it there.
        """
        labels = labels.copy()
        cluster_ids = sorted(set(labels) - {-1})
        if not cluster_ids:
            return labels

        # Compute centroids
        centroids = {
            cid: coords_2d[labels == cid].mean(axis=0)
            for cid in cluster_ids
        }

        for i, lbl in enumerate(labels):
            if lbl == -1:
                pt = coords_2d[i]
                nearest = min(
                    cluster_ids,
                    key=lambda cid: np.linalg.norm(pt - centroids[cid]),
                )
                labels[i] = nearest

        return labels

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    def cluster_members(
        self,
        embedded: list[EmbeddedTranscript],
    ) -> dict[int, list[EmbeddedTranscript]]:
        """Group embedded transcripts by cluster_id."""
        groups: dict[int, list[EmbeddedTranscript]] = {}
        for et in embedded:
            groups.setdefault(et.cluster_id, []).append(et)
        return groups

    @staticmethod
    def cluster_ids(embedded: list[EmbeddedTranscript]) -> list[int]:
        return sorted(set(et.cluster_id for et in embedded))

    @staticmethod
    def centroid(
        members: list[EmbeddedTranscript],
        embeddings_matrix: np.ndarray,
        all_embedded: list[EmbeddedTranscript],
    ) -> int:
        """Return index of the member closest to the cluster centroid."""
        idx_map = {et.transcript.id: i for i, et in enumerate(all_embedded)}
        member_indices = [idx_map[et.transcript.id] for et in members]
        vecs = embeddings_matrix[member_indices]
        centroid_vec = vecs.mean(axis=0)
        dists = np.linalg.norm(vecs - centroid_vec, axis=1)
        return member_indices[int(dists.argmin())]
