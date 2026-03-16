"""
Shared data models for the LLM Failure Analyzer.
"""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class FailureCategory(str, Enum):
    AMBIGUITY = "ambiguity_errors"
    CODE_CONVENTION = "code_convention_violations"
    INCOMPLETE = "incomplete_solutions"
    HALLUCINATION = "hallucinated_apis"
    COMMUNICATION = "communication_failure"
    DESIGN = "design_decision_errors"
    CONTEXT_LOSS = "context_loss"
    REASONING = "reasoning_errors"
    UNKNOWN = "unknown"


class Turn(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class Transcript(BaseModel):
    id: str
    task: str = ""
    turns: list[Turn] = Field(default_factory=list)
    outcome: str = "failure"  # "failure" | "partial" | "success"
    notes: str = ""
    source_file: str = ""
    raw_text: str = ""  # full concatenated text for embedding

    def to_embed_text(self) -> str:
        """Return a single string suitable for embedding."""
        if self.raw_text:
            return self.raw_text
        parts = []
        if self.task:
            parts.append(f"TASK: {self.task}")
        for turn in self.turns:
            parts.append(f"{turn.role.upper()}: {turn.content}")
        if self.notes:
            parts.append(f"NOTES: {self.notes}")
        return "\n".join(parts)


class ClusterResult(BaseModel):
    cluster_id: int
    label: str = ""
    description: str = ""
    category: FailureCategory = FailureCategory.UNKNOWN
    transcript_ids: list[str] = Field(default_factory=list)
    size: int = 0
    percentage: float = 0.0
    representative_examples: list[str] = Field(default_factory=list)


class AnalysisReport(BaseModel):
    total_transcripts: int
    clustered_transcripts: int
    noise_count: int
    n_clusters: int
    clusters: list[ClusterResult]
    model_used: str
    embed_model_used: str
    generated_at: str
    summary: str = ""

    def failure_distribution(self) -> dict[str, float]:
        return {c.label or f"Cluster {c.cluster_id}": c.percentage for c in self.clusters}


class EmbeddedTranscript(BaseModel):
    transcript: Transcript
    embedding: list[float]
    cluster_id: int = -1
    umap_x: float = 0.0
    umap_y: float = 0.0
