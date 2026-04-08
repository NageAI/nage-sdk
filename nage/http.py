"""HTTP layer for Nage SDK."""

import json
from typing import Optional, Dict, Any, Iterator
from nage.exceptions import NageError, AuthError, RateLimitError, NotFoundError, ServerError
from nage.models import (
    ThoughtResponse, ThoughtChunk, LearnResult,
    STEMMA, STEMMAEntry, KnowledgeSource,
    KnowledgeState, VARVEHealth, PlatformInfo,
)

DEFAULT_BASE_URL = "https://api.models.nage.ai"
DEFAULT_TIMEOUT = 120.0


def _raise_for_status(status_code: int, body: dict):
    """Raise appropriate exception based on HTTP status."""
    msg = body.get("message", body.get("error", "Unknown error"))
    if status_code == 401:
        raise AuthError(msg, status_code=status_code, body=body)
    elif status_code == 429:
        raise RateLimitError(
            msg, status_code=status_code, body=body,
            limit=body.get("limit"), used=body.get("used"),
        )
    elif status_code == 404:
        raise NotFoundError(msg, status_code=status_code, body=body)
    elif status_code >= 500:
        raise ServerError(msg, status_code=status_code, body=body)
    elif status_code >= 400:
        raise NageError(msg, status_code=status_code, body=body)


def parse_thought_response(data: dict) -> ThoughtResponse:
    """Parse API response into ThoughtResponse."""
    stemma_data = data.get("stemma", {})
    entries = [
        STEMMAEntry(
            varve=e.get("varve", ""),
            weight=e.get("weight", 0.0),
            layer=e.get("layer", ""),
            source=e.get("source"),
        )
        for e in stemma_data.get("entries", [])
    ]
    stemma = STEMMA(
        weights=stemma_data.get("weights", {}),
        dominant_layer=stemma_data.get("dominant_layer", ""),
        dominant_varve=stemma_data.get("dominant_varve", ""),
        entropy=stemma_data.get("entropy", 0.0),
        entries=entries,
    )
    knowledge = [
        KnowledgeSource(
            varve=k.get("varve", ""),
            layer=k.get("layer", ""),
            weight=k.get("weight", 0.0),
            confidence=k.get("confidence", 0.0),
        )
        for k in data.get("knowledge", [])
    ]
    return ThoughtResponse(
        thought_id=data.get("thought_id", ""),
        response=data.get("response", ""),
        stemma=stemma,
        knowledge=knowledge,
        meta=data.get("meta", {}),
    )


def parse_thought_chunk(line: str) -> Optional[ThoughtChunk]:
    """Parse one SSE line into ThoughtChunk."""
    if not line.startswith("data: "):
        return None
    try:
        data = json.loads(line[6:])
    except json.JSONDecodeError:
        return None

    stemma = None
    if data.get("stemma"):
        s = data["stemma"]
        stemma = STEMMA(
            weights=s.get("weights", {}),
            dominant_layer=s.get("dominant_layer", ""),
            dominant_varve=s.get("dominant_varve", ""),
            entropy=s.get("entropy", 0.0),
        )

    return ThoughtChunk(
        thought_id=data.get("thought_id", ""),
        delta=data.get("delta", ""),
        done=data.get("done", False),
        response=data.get("response"),
        stemma=stemma,
    )


def parse_learn_result(data: dict) -> LearnResult:
    return LearnResult(
        varve_id=data.get("varve_id", ""),
        varve_type=data.get("varve_type", ""),
        layer=data.get("layer", ""),
        status=data.get("status", ""),
        training_eta=data.get("training_eta"),
        message=data.get("message", ""),
    )


def parse_knowledge_state(data: dict) -> KnowledgeState:
    layers = {}
    for layer_name, varves in data.get("layers", {}).items():
        layers[layer_name] = [
            VARVEHealth(
                varve_id=v.get("varve_id", ""),
                layer=v.get("layer", ""),
                status=v.get("status", ""),
                distance=v.get("distance", 0.0),
                description=v.get("description", ""),
            )
            for v in varves
        ]
    return KnowledgeState(
        platform=data.get("platform", ""),
        total_varves=data.get("total_varves", 0),
        layers=layers,
    )


def parse_platform_info(data: dict) -> PlatformInfo:
    return PlatformInfo(
        platform_id=data.get("platform_id", ""),
        model_id=data.get("model_id", ""),
        d_model=data.get("d_model", 0),
        varves=data.get("varves", 0),
        varve_names=data.get("varve_names", []),
        formula=data.get("formula", ""),
        stratum_tiers=data.get("stratum_tiers", {}),
    )
