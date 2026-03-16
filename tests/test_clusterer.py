"""Tests for src/clusterer.py"""
import numpy as np
import pytest

from src.clusterer import Clusterer
from src.models import EmbeddedTranscript, Transcript


def make_embedded(n: int, dims: int = 16) -> tuple[list[EmbeddedTranscript], np.ndarray]:
    """Create n fake EmbeddedTranscript objects with random embeddings."""
    rng = np.random.default_rng(42)
    matrix = rng.standard_normal((n, dims)).astype(np.float32)
    embedded = [
        EmbeddedTranscript(
            transcript=Transcript(id=f"t{i}", task=f"task {i}"),
            embedding=matrix[i].tolist(),
        )
        for i in range(n)
    ]
    return embedded, matrix


class TestClusterer:
    def test_assigns_cluster_ids(self):
        embedded, matrix = make_embedded(20)
        clusterer = Clusterer(umap_neighbors=5, hdbscan_min_cluster_size=2)
        result = clusterer.fit_transform(embedded, matrix)
        # Every point should have a cluster_id
        for et in result:
            assert isinstance(et.cluster_id, int)

    def test_umap_coordinates_set(self):
        embedded, matrix = make_embedded(10)
        clusterer = Clusterer(umap_neighbors=5, hdbscan_min_cluster_size=2)
        result = clusterer.fit_transform(embedded, matrix)
        for et in result:
            assert isinstance(et.umap_x, float)
            assert isinstance(et.umap_y, float)

    def test_cluster_members_grouping(self):
        embedded, matrix = make_embedded(12)
        clusterer = Clusterer(umap_neighbors=5, hdbscan_min_cluster_size=2)
        embedded = clusterer.fit_transform(embedded, matrix)
        groups = clusterer.cluster_members(embedded)
        # All transcripts accounted for
        total = sum(len(v) for v in groups.values())
        assert total == 12

    def test_single_transcript_no_crash(self):
        embedded, matrix = make_embedded(1)
        clusterer = Clusterer()
        result = clusterer.fit_transform(embedded, matrix)
        assert len(result) == 1
        assert result[0].cluster_id == 0  # fallback

    def test_all_noise_fallback_uses_agglomerative(self):
        """
        When HDBSCAN produces all noise, agglomerative fallback should
        assign every point to a cluster (no -1 labels).
        """
        # Uniform random points — very hard for HDBSCAN to cluster
        rng = np.random.default_rng(99)
        n = 11
        matrix = rng.standard_normal((n, 768)).astype(np.float32)  # realistic embedding dim
        embedded = [
            EmbeddedTranscript(
                transcript=Transcript(id=f"t{i}", task="x"),
                embedding=matrix[i].tolist(),
            )
            for i in range(n)
        ]
        clusterer = Clusterer(umap_neighbors=5, hdbscan_min_cluster_size=2)
        result = clusterer.fit_transform(embedded, matrix)

        # After fallback, no transcript should have cluster_id == -1
        noise = [et for et in result if et.cluster_id == -1]
        assert len(noise) == 0, f"Expected 0 noise after fallback, got {len(noise)}"

        # Should have at least 2 clusters
        cluster_ids = set(et.cluster_id for et in result)
        assert len(cluster_ids) >= 2

    def test_noise_reassigned_to_nearest_cluster(self):
        """Noise points should be absorbed into the nearest cluster."""
        labels = np.array([0, 0, -1, 1, 1, -1])
        coords = np.array([
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],   # near cluster 0
            [5.0, 5.0],
            [5.1, 5.0],
            [5.2, 5.0],   # near cluster 1
        ])
        result = Clusterer._reassign_noise(labels, coords)
        assert result[2] == 0   # was noise, near cluster 0
        assert result[5] == 1   # was noise, near cluster 1
        assert -1 not in result
