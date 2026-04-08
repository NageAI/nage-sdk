"""Nage SDK data models."""

from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class STEMMAEntry:
    """One VARVE's contribution to a response."""
    varve: str
    weight: float
    layer: str
    source: Optional[str] = None


@dataclass
class STEMMA:
    """Source attribution for a response."""
    weights: Dict[str, float]
    dominant_layer: str
    dominant_varve: str
    entropy: float
    entries: List[STEMMAEntry] = field(default_factory=list)

    def __repr__(self):
        top = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)[:3]
        parts = ", ".join(f"{k}: {v:.2f}" for k, v in top)
        return f"STEMMA({{{parts}}})"


@dataclass
class KnowledgeSource:
    """A knowledge source that contributed to a response."""
    varve: str
    layer: str
    weight: float
    confidence: float


@dataclass
class ThoughtResponse:
    """Response from client.think()."""
    thought_id: str
    response: str
    stemma: STEMMA
    knowledge: List[KnowledgeSource] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.response

    @property
    def dominant_varve(self) -> str:
        return self.stemma.dominant_varve

    def __repr__(self):
        preview = self.response[:80] + "..." if len(self.response) > 80 else self.response
        return f"ThoughtResponse('{preview}', {self.stemma})"


@dataclass
class ThoughtChunk:
    """One chunk from streaming response."""
    thought_id: str
    delta: str
    done: bool
    response: Optional[str] = None
    stemma: Optional[STEMMA] = None


@dataclass
class LearnResult:
    """Response from client.learn()."""
    varve_id: str
    varve_type: str
    layer: str
    status: str
    training_eta: Optional[int] = None
    message: str = ""


@dataclass
class VARVEHealth:
    """Health info for a single VARVE."""
    varve_id: str
    layer: str
    status: str
    distance: float
    description: str = ""


@dataclass
class KnowledgeState:
    """Full knowledge base state."""
    platform: str
    total_varves: int
    layers: Dict[str, List[VARVEHealth]] = field(default_factory=dict)


@dataclass
class PlatformInfo:
    """Platform metadata."""
    platform_id: str
    model_id: str
    d_model: int
    varves: int
    varve_names: List[str] = field(default_factory=list)
    formula: str = ""
    stratum_tiers: Dict = field(default_factory=dict)
