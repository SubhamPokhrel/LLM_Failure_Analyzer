"""
Embedder
--------
Embeds transcripts using Ollama's local embedding endpoint.
Results are cached to disk so re-runs skip already-embedded transcripts.

Usage:
    embedder = Embedder(model="nomic-embed-text")
    embedded = embedder.embed_all(transcripts)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from src.models import EmbeddedTranscript, Transcript

console = Console()


class OllamaEmbedder:
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        cache_path: str | Path = "data/embeddings_cache.npy",
        meta_path: str | Path = "data/metadata_cache.json",
        timeout: int = 120,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.cache_path = Path(cache_path)
        self.meta_path = Path(meta_path)
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def embed_all(
        self,
        transcripts: list[Transcript],
        force_reembed: bool = False,
    ) -> list[EmbeddedTranscript]:
        """
        Embed all transcripts.  Returns EmbeddedTranscript list.
        Loads cache from disk and only re-embeds new/missing transcripts.
        """
        cache = self._load_cache() if not force_reembed else {}
        results: list[EmbeddedTranscript] = []
        to_embed: list[Transcript] = []

        for t in transcripts:
            if t.id in cache:
                results.append(
                    EmbeddedTranscript(transcript=t, embedding=cache[t.id])
                )
            else:
                to_embed.append(t)

        if to_embed:
            console.print(
                f"\n[bold]Embedding {len(to_embed)} transcript(s) via Ollama "
                f"[cyan]{self.model}[/cyan]…[/bold]"
            )
            self._check_model_available()
            new_embedded = self._embed_batch(to_embed)
            for et in new_embedded:
                cache[et.transcript.id] = et.embedding
            results.extend(new_embedded)
            self._save_cache(cache)
        else:
            console.print(f"[green]All {len(results)} transcripts loaded from cache.[/green]")

        return results

    def embed_one(self, text: str) -> list[float]:
        """Embed a single string. No caching."""
        return self._call_api(text)

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _embed_batch(self, transcripts: list[Transcript]) -> list[EmbeddedTranscript]:
        embedded = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Embedding…", total=len(transcripts))
            for t in transcripts:
                try:
                    vec = self._call_api(t.to_embed_text())
                    embedded.append(EmbeddedTranscript(transcript=t, embedding=vec))
                except Exception as e:
                    console.print(f"[red]  Failed to embed {t.id}: {e}[/red]")
                progress.advance(task)
                time.sleep(0.05)  # polite rate limit
        return embedded

    def _call_api(self, text: str) -> list[float]:
        resp = self._client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def _check_model_available(self) -> None:
        try:
            resp = self._client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            model_base = self.model.split(":")[0]
            if not any(model_base in m for m in models):
                console.print(
                    f"[yellow]⚠ Model '{self.model}' not found locally. "
                    f"Run: ollama pull {self.model}[/yellow]"
                )
        except httpx.ConnectError:
            raise RuntimeError(
                "Cannot reach Ollama at http://localhost:11434. "
                "Is Ollama running? Try: ollama serve"
            )

    # ──────────────────────────────────────────
    # Cache I/O
    # ──────────────────────────────────────────

    def _load_cache(self) -> dict[str, list[float]]:
        if not self.meta_path.exists():
            return {}
        try:
            with open(self.meta_path) as f:
                meta = json.load(f)
            if self.cache_path.exists():
                vecs = np.load(self.cache_path)
                return {
                    id_: vecs[i].tolist()
                    for i, id_ in enumerate(meta["ids"])
                }
        except Exception as e:
            console.print(f"[yellow]Cache load failed ({e}), re-embedding all.[/yellow]")
        return {}

    def _save_cache(self, cache: dict[str, list[float]]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        ids = list(cache.keys())
        matrix = np.array([cache[i] for i in ids], dtype=np.float32)
        np.save(self.cache_path, matrix)
        with open(self.meta_path, "w") as f:
            json.dump({"ids": ids, "model": self.model}, f)
        console.print(f"[dim]Cache saved: {len(ids)} embeddings → {self.cache_path}[/dim]")


def get_embeddings_matrix(embedded: list[EmbeddedTranscript]) -> np.ndarray:
    """Stack embeddings into an (N, D) numpy matrix."""
    return np.array([e.embedding for e in embedded], dtype=np.float32)
