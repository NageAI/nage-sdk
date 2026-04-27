"""
nage — Python SDK
==================
pip install nage

Kullanım:
    import nage
    client = nage.Client("nk_live_...")
    response = client.think("Merhaba!")
    print(response.response)
    print(response.stemma)
"""

__version__ = "0.2.0"
__all__ = [
    # core
    "Client", "AsyncClient",
    # exceptions
    "NageError", "AuthError", "RateLimitError",
    # legacy types
    "STEMMA", "KnowledgeSource", "ThinkResponse",
    # KLM v1 (Phase A+B+C+D)
    "Gamma", "AuditRecord", "AuditExport",
    "ChatCompletion", "ChatChoice", "ChatMessage", "Usage",
    "ModelInfo",
    "VarveHealth", "HealthBand",
]

import json
import time
from dataclasses import dataclass, field
from typing import Optional, Iterator, AsyncIterator

# ── Exceptions ────────────────────────────────────────────────────────────────

class NageError(Exception):
    def __init__(self, message: str, code: Optional[str] = None, status: int = 0):
        super().__init__(message)
        self.code   = code
        self.status = status

class AuthError(NageError):
    pass

class RateLimitError(NageError):
    pass

class ModelError(NageError):
    pass


# ── Response Types ────────────────────────────────────────────────────────────

@dataclass
class STEMMA:
    """VARVE attribution — hangi bilgi katmanı ne kadar katkı sağladı."""
    weights:        dict[str, float]
    dominant_layer: str
    dominant_varve: str
    entropy:        float

    def __repr__(self):
        parts = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
        inner = ", ".join(f"{k}: {v:.2f}" for k, v in parts[:3])
        return f"STEMMA({inner})"

    def top(self, n: int = 3) -> list[tuple[str, float]]:
        return sorted(self.weights.items(), key=lambda x: x[1], reverse=True)[:n]


@dataclass
class KnowledgeSource:
    varve:      str
    layer:      str
    weight:     float
    confidence: float
    cutoff:     Optional[str] = None


@dataclass
class Gamma:
    """KLM γ vector — epistemic fingerprint for a single inference output.

    Cross-cutting primitive defined by SEDIM v1.0 §10 + STRATUM v1.1 §3.
    Tier-aware visibility — fields below populated when the caller's
    STRATUM tier permits ('hidden' returns None for the whole struct).

    Inspect `epistemic_label` for at-a-glance health:
        STABLE      — high confidence + high coherence (clear lineage)
        EVOLVING    — multiple sources, all agreeing
        CONTESTED   — high evidence but conflict between sources
        UNCERTAIN   — low confidence, no strong source
        STALE       — old VARVE dominant (knowledge possibly outdated)
        RAW         — FACIES-only output (no VARVE engaged)
    """
    epistemic_label: str | None = None      # one of 6 above
    warning:         str | None = None      # human-readable, if non-STABLE
    confidence:      float | None = None    # 0.0–1.0
    evidence_score:  float | None = None    # how much VARVE mass contributed
    freshness_score: float | None = None    # decay over time-since-VARVE-update
    coherence_score: float | None = None    # 1 − conflict between active sources
    dominant_source: str | None = None      # e.g. "varve/fehm-tr"
    provenance_map:  dict | None = None     # { source_id: weight, ... }

    def is_stable(self) -> bool:
        return self.epistemic_label == "STABLE"

    def is_contested(self) -> bool:
        return self.epistemic_label == "CONTESTED"


@dataclass
class ThinkResponse:
    thought_id: str
    response:   str
    stemma:     STEMMA
    knowledge:  list[KnowledgeSource] = field(default_factory=list)
    meta:       dict                  = field(default_factory=dict)
    # Phase A: KLM extensions. Present on platforms that ship γ
    # (api.sedim.ai backend post 2026-04-27). Older deployments
    # return them as None.
    gamma:      Optional["Gamma"] = None
    audit_id:   Optional[str]     = None
    audit_ref:  Optional[dict]    = None     # CORE tier only

    def __repr__(self):
        label = self.gamma.epistemic_label if self.gamma else None
        return (
            f"ThinkResponse(thought_id={self.thought_id!r}, "
            f"stemma={self.stemma}, label={label!r})"
        )


# ── Phase B: OpenAI-compat ChatCompletion shape ────────────────────────────

@dataclass
class ChatMessage:
    role:    str
    content: str


@dataclass
class ChatChoice:
    index:         int
    message:       ChatMessage
    finish_reason: str | None = None


@dataclass
class Usage:
    prompt_tokens:     int = 0
    completion_tokens: int = 0
    total_tokens:      int = 0


@dataclass
class ChatCompletion:
    """OpenAI-compatible chat.completion response with KLM extensions.

    Drop-in shape for any OpenAI SDK consumer. The KLM additions
    (gamma, audit_id, audit_ref) are extra fields ignored by strict
    OpenAI parsers but available to STRATUM-aware clients.
    """
    id:        str
    object:    str
    created:   int
    model:     str
    choices:   list[ChatChoice]
    usage:     Usage
    # KLM extensions
    gamma:     Optional["Gamma"] = None
    audit_id:  Optional[str]     = None
    audit_ref: Optional[dict]    = None

    @property
    def text(self) -> str:
        """Convenience: first choice's assistant text."""
        if not self.choices:
            return ""
        return self.choices[0].message.content


@dataclass
class ModelInfo:
    id:              str
    object:          str = "model"
    created:         int = 0
    owned_by:        str = "nage"
    context_window:  int | None = None
    tier_min:        str | None = None     # STRATUM tier required


# ── Phase C: Audit retrieval ───────────────────────────────────────────────

@dataclass
class AuditRecord:
    """A single Knowledge Unit audit record (STRATUM §7.1).

    Returned by `client.audit_get(audit_id)`. Tenant-scoped on the
    server side — passing another tenant's audit_id 404s.
    """
    audit_id:    str
    created_at:  str | None = None
    event_type:  str | None = None
    gamma:       Optional[Gamma] = None
    payload:     dict | None = None         # query_excerpt, dominant_varve, ...
    routing_snapshot: dict | None = None    # CORE tier only
    audit_ref:   dict | None = None         # CORE tier only


@dataclass
class AuditExport:
    """Container for /v1/audit/export tar.gz response.

    Use `.save(path)` to write the archive locally; the SDK does NOT
    decompress automatically because the manifest + signatures should
    be preserved as-shipped for AI Act audit verification.
    """
    content_bytes: bytes
    record_count: int
    format:       str
    filename:     str

    def save(self, path: str) -> str:
        with open(path, "wb") as f:
            f.write(self.content_bytes)
        return path


# ── Phase D: VARVE health ──────────────────────────────────────────────────

@dataclass
class HealthBand:
    healthy_min: float = 0.05
    healthy_max: float = 0.30


@dataclass
class VarveHealth:
    """SEDIM §9.1 health ratio per VARVE.

    `band` ∈ {"healthy", "consolidation_candidate", "drifted"}.
    consolidation_candidate (ρ < 0.05) means the VARVE has converged
    toward FACIES — admin may consolidate it via Governed Consolidation
    (§9.3). drifted (ρ > 0.30) signals over-training; review needed.
    """
    name:         str
    rho:          float
    rho_global:   float
    band:         str
    n_layers:     int
    rho_min:      float | None = None
    rho_max:      float | None = None


@dataclass
class VARVEInfo:
    varve_id:    str
    layer:       str
    status:      str
    description: str
    cutoff:      Optional[str] = None


@dataclass
class KnowledgeInfo:
    platform:     str
    total_varves: int
    layers:       dict[str, list[VARVEInfo]] = field(default_factory=dict)


# ── HTTP Client ───────────────────────────────────────────────────────────────

class _HTTPClient:
    def __init__(self, api_key: str, base_url: str, timeout: int):
        self.api_key  = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout

    def _headers(self) -> dict:
        return {
            "X-Nage-Key":    self.api_key,
            "Content-Type":  "application/json",
            "User-Agent":    f"nage-python/{__version__}",
        }

    def post(self, path: str, body: dict) -> dict:
        import urllib.request
        import urllib.error

        url  = f"{self.base_url}{path}"
        data = json.dumps(body).encode()
        req  = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            self._raise_for_status(e.code, e.read())
        except urllib.error.URLError as e:
            raise NageError(f"Connection error: {e.reason}")

    def get(self, path: str, params: dict = None) -> dict:
        import urllib.request
        import urllib.error
        import urllib.parse

        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)

        req = urllib.request.Request(url, headers=self._headers(), method="GET")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            self._raise_for_status(e.code, e.read())
        except urllib.error.URLError as e:
            raise NageError(f"Connection error: {e.reason}")

    def _raise_for_status(self, status: int, body: bytes) -> None:
        try:
            error = json.loads(body)
            msg   = error.get("message", str(body))
            code  = error.get("error", "unknown_error")
        except Exception:
            msg, code = str(body), "unknown_error"

        if status == 401:
            raise AuthError(msg, code=code, status=status)
        elif status == 429:
            raise RateLimitError(msg, code=code, status=status)
        elif status >= 500:
            raise ModelError(msg, code=code, status=status)
        else:
            raise NageError(msg, code=code, status=status)


# ── Response Parsers ──────────────────────────────────────────────────────────

def _parse_stemma(data: dict) -> STEMMA:
    return STEMMA(
        weights=data.get("weights", {}),
        dominant_layer=data.get("dominant_layer", ""),
        dominant_varve=data.get("dominant_varve", ""),
        entropy=data.get("entropy", 0.0),
    )


def _parse_gamma(data: dict | None) -> Optional[Gamma]:
    """Parse a γ vector if the server returned one.

    Returns None when:
      - server omitted γ entirely (SURFACE tier, or pre-Phase-A backend)
      - data is malformed (defensive: SDK never raises on missing γ)
    """
    if not isinstance(data, dict):
        return None
    return Gamma(
        epistemic_label=data.get("epistemic_label"),
        warning=data.get("warning"),
        confidence=data.get("confidence"),
        evidence_score=data.get("evidence_score"),
        freshness_score=data.get("freshness_score"),
        coherence_score=data.get("coherence_score"),
        dominant_source=data.get("dominant_source"),
        provenance_map=data.get("provenance_map"),
    )


def _parse_think_response(data: dict) -> ThinkResponse:
    stemma = _parse_stemma(data.get("stemma", {}))
    knowledge = [
        KnowledgeSource(
            varve=k.get("varve", ""),
            layer=k.get("layer", ""),
            weight=k.get("weight", 0.0),
            confidence=k.get("confidence", 0.0),
            cutoff=k.get("cutoff"),
        )
        for k in data.get("knowledge", [])
    ]
    return ThinkResponse(
        thought_id=data.get("thought_id", ""),
        response=data.get("response", ""),
        stemma=stemma,
        knowledge=knowledge,
        meta=data.get("meta", {}),
        # Phase A — KLM γ + audit_id when present
        gamma=_parse_gamma(data.get("gamma")),
        audit_id=data.get("audit_id"),
        audit_ref=data.get("audit_ref"),
    )

def _parse_chat_completion(data: dict) -> ChatCompletion:
    """Parse OpenAI-shape chat.completion + KLM extensions."""
    choices = []
    for c in data.get("choices", []):
        msg_data = c.get("message", {}) or {}
        choices.append(ChatChoice(
            index=int(c.get("index", 0)),
            message=ChatMessage(
                role=msg_data.get("role", "assistant"),
                content=msg_data.get("content", ""),
            ),
            finish_reason=c.get("finish_reason"),
        ))
    usage_data = data.get("usage") or {}
    usage = Usage(
        prompt_tokens=int(usage_data.get("prompt_tokens", 0)),
        completion_tokens=int(usage_data.get("completion_tokens", 0)),
        total_tokens=int(usage_data.get("total_tokens", 0)),
    )
    return ChatCompletion(
        id=data.get("id", ""),
        object=data.get("object", "chat.completion"),
        created=int(data.get("created", 0)),
        model=data.get("model", ""),
        choices=choices,
        usage=usage,
        gamma=_parse_gamma(data.get("gamma")),
        audit_id=data.get("audit_id"),
        audit_ref=data.get("audit_ref"),
    )


def _parse_knowledge(data: dict) -> KnowledgeInfo:
    layers = {}
    for layer_name, varves in data.get("layers", {}).items():
        layers[layer_name] = [
            VARVEInfo(
                varve_id=v.get("varve_id", ""),
                layer=v.get("layer", ""),
                status=v.get("status", "unknown"),
                description=v.get("description", ""),
                cutoff=v.get("cutoff"),
            )
            for v in varves
        ]
    return KnowledgeInfo(
        platform=data.get("platform", ""),
        total_varves=data.get("total_varves", 0),
        layers=layers,
    )


# ── Client ────────────────────────────────────────────────────────────────────

class Client:
    """
    Nage API istemcisi.

    Örnek:
        client = nage.Client("nk_live_...")
        response = client.think("Python'da async nedir?")
        print(response.response)
        print(response.stemma)  # STEMMA(MING/coding: 0.62, FEHM/turkish-context: 0.38)
    """

    def __init__(
        self,
        api_key:  str,
        base_url: str = "https://api.nage.ai",
        platform: str = "nage-8b",
        timeout:  int = 120,
    ):
        self.platform = platform
        self._http = _HTTPClient(api_key, base_url, timeout)

    def think(
        self,
        query:       str,
        context:     list[dict] = None,
        platform:    str        = None,
        varve_hint:  str        = None,
        layer_hint:  str        = None,
        max_tokens:  int        = 512,
        temperature: float      = 0.7,
    ) -> ThinkResponse:
        """
        Sorguyu SEDIM üzerinden işle. STEMMA attribution döndürür.

        Args:
            query:       Kullanıcı sorusu
            context:     Önceki mesajlar [{"role": "user", "content": "..."}, ...]
            platform:    nage-8b | nage-14b | nm/fehm | nm/ming
            varve_hint:  Belirli bir VARVE'yi öne çıkar
            layer_hint:  FEHM | MING | CHI | CORTEX
            max_tokens:  Maksimum çıktı token sayısı
            temperature: 0.0 (deterministik) → 2.0 (yaratıcı)

        Returns:
            ThinkResponse — yanıt + STEMMA attribution
        """
        body = {
            "query":       query,
            "platform":    platform or self.platform,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        if context:
            body["context"] = context
        if varve_hint:
            body["varve_hint"] = varve_hint
        if layer_hint:
            body["layer_hint"] = layer_hint

        data = self._http.post("/think", body)
        return _parse_think_response(data)

    def think_stream(
        self,
        query:       str,
        platform:    str   = None,
        max_tokens:  int   = 512,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """
        Streaming inference. Her chunk bir string döndürür.
        Son event STEMMA içerir: [STEMMA]{...}

        Örnek:
            for chunk in client.think_stream("Merhaba"):
                print(chunk, end="", flush=True)
        """
        import urllib.request

        body = {
            "query":       query,
            "platform":    platform or self.platform,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "stream":      True,
        }

        url  = f"{self._http.base_url}/think/stream"
        data = json.dumps(body).encode()
        req  = urllib.request.Request(url, data=data,
                                       headers=self._http._headers(), method="POST")

        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    chunk = line[6:]
                    if chunk == "[DONE]":
                        break
                    yield chunk

    def learn(
        self,
        text:       str,
        domain:     str,
        layer:      str = "CHI",
        varve_type: str = "ephemeral",
        examples:   list = None,
    ) -> dict:
        """
        Teach new knowledge to SEDIM via ApplicationVARVE.

        Args:
            text:       Training text
            domain:     Domain label (e.g. "legal-tr", "medical")
            layer:      FEHM | MING | CHI | CORTEX
            varve_type: "ephemeral" (instant, free) | "flash" (10min)
            examples:   Optional few-shot examples

        Returns:
            LearnResult with varve_id, status, training_eta
        """
        body = {
            "text": text,
            "domain": domain,
            "layer": layer,
            "varve_type": varve_type,
        }
        if examples:
            body["examples"] = examples
        return self._http.post("/learn", body)

    @property
    def knowledge(self) -> KnowledgeInfo:
        """All VARVEs across 4 layers with health status."""
        data = self._http.get("/knowledge", {"platform": self.platform})
        return _parse_knowledge(data)

    def consolidate(self) -> dict:
        """
        Trigger FACIES consolidation.
        Requires LODE or CORE tier.

        FACIES_new = FACIES_old + Σᵢ mean(STEMMAᵢ) · VARVEᵢ
        VARVEs reset to near-zero after consolidation.
        """
        return self._http.post("/knowledge/consolidate", {"platform": self.platform})

    def layers(self) -> dict:
        """Layer definitions: FEHM, MING, CHI, CORTEX."""
        return self._http.get("/knowledge/layers")

    def platform(self) -> dict:
        """Platform metadata: model, varves, formula, STRATUM tiers."""
        return self._http.get("/platform", {"platform": self.platform})

    def health(self) -> dict:
        """Service health check."""
        return self._http.get("/health")

    def session(self):
        """Multi-turn conversation context manager."""
        return _Session(self)

    # ── KLM v1 surface (Phase A+B+C+D) ────────────────────────────────────
    # The /v1/* endpoints are the public STRATUM API. They run alongside
    # the legacy /think — same backend, OpenAI-shape on top, tier-aware
    # γ visibility, audit retrieval, AI-Act export.

    def chat_completion(
        self,
        messages: list[dict] | list[ChatMessage],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        # STRATUM extensions (KLM)
        varve_ids: list[str] | None = None,
        inference_mode: str | None = None,
        end_user_id: str | None = None,
        session_id: str | None = None,
    ) -> ChatCompletion:
        """OpenAI-compatible chat completion via STRATUM /v1/chat/completions.

        Drop-in replacement for OpenAI client.chat.completions.create —
        pass the same `messages` array shape, get back an OpenAI-shape
        ChatCompletion, plus KLM extensions (gamma, audit_id) when the
        caller's STRATUM tier permits.

        Example:
            from nage import Client
            client = Client("nk_live_...", base_url="https://api.sedim.ai")
            r = client.chat_completion(
                messages=[{"role": "user", "content": "merhaba"}],
                model="fehm-8b",
                inference_mode="resonance_gated",   # SEDIM §5.3
            )
            print(r.text)                  # assistant's response
            print(r.gamma.epistemic_label) # STABLE / EVOLVING / ...
            print(r.audit_id)              # for /v1/audit/{id}
        """
        # Allow either dict messages or ChatMessage instances
        body_messages: list[dict] = []
        for m in messages:
            if isinstance(m, ChatMessage):
                body_messages.append({"role": m.role, "content": m.content})
            else:
                body_messages.append({
                    "role":    m.get("role", "user"),
                    "content": m.get("content", ""),
                })

        body: dict = {
            "model":    model or self.platform,
            "messages": body_messages,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        if varve_ids:
            body["varve_ids"] = list(varve_ids)
        if inference_mode:
            body["inference_mode"] = inference_mode
        if end_user_id:
            body["end_user_id"] = end_user_id
        if session_id:
            body["session_id"] = session_id

        data = self._http.post("/v1/chat/completions", body)
        return _parse_chat_completion(data)

    def models_list(self) -> list[ModelInfo]:
        """Available models per STRATUM /v1/models. Tier-aware:
        models above your tier still show up but call-time will 403.
        """
        data = self._http.get("/v1/models")
        return [
            ModelInfo(
                id=m.get("id", ""),
                object=m.get("object", "model"),
                created=int(m.get("created", 0)),
                owned_by=m.get("owned_by", "nage"),
                context_window=m.get("context_window"),
                tier_min=m.get("tier_min"),
            )
            for m in (data.get("data") or [])
        ]

    def audit_get(self, audit_id: str) -> AuditRecord:
        """Retrieve a previous /v1/chat/completions audit Knowledge Unit.

        Tier gate (server-side): LODE+. Tenant-scoped — you can only
        pull audit_ids your org owns; cross-tenant lookup returns 404.
        """
        data = self._http.get(f"/v1/audit/{audit_id}")
        return AuditRecord(
            audit_id=data.get("audit_id", audit_id),
            created_at=data.get("created_at"),
            event_type=data.get("event_type"),
            gamma=_parse_gamma(data.get("gamma")),
            payload=data.get("payload"),
            routing_snapshot=data.get("routing_snapshot"),
            audit_ref=data.get("audit_ref"),
        )

    def audit_export(
        self,
        period_start: str | None = None,
        period_end:   str | None = None,
        format:       str = "ai_act_2024",
    ) -> AuditExport:
        """AI Act audit export via /v1/audit/export.

        Returns the raw tar.gz bytes wrapped in `AuditExport`. Save
        with `.save("audit.tar.gz")` — the SDK does NOT decompress
        because the manifest + per-record signatures must be preserved
        for offline AI Act verification.

        When period_start/end omitted, server defaults to previous
        calendar month.
        """
        body: dict = {"format": format, "include_content_hashes": True}
        if period_start and period_end:
            body["period"] = {"start": period_start, "end": period_end}
        # We need raw bytes, not parsed JSON — fall back to urllib for
        # this one call.
        import urllib.request
        url = f"{self._http.base_url}/v1/audit/export"
        data_bytes = json.dumps(body).encode()
        req = urllib.request.Request(
            url, data=data_bytes,
            headers=self._http._headers(), method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._http.timeout) as resp:
            blob = resp.read()
            cd = resp.headers.get("content-disposition", "")
            filename = "audit-export.tar.gz"
            if "filename=" in cd:
                filename = cd.split("filename=", 1)[1].strip().strip('"')
            count = int(resp.headers.get("X-Audit-Records", "0") or 0)
        return AuditExport(
            content_bytes=blob,
            record_count=count,
            format=format,
            filename=filename,
        )

    def __repr__(self):
        return f"nage.Client(platform={self.platform!r}, base_url={self._http.base_url!r})"


class _Session:
    """Multi-turn conversation with automatic context management."""

    def __init__(self, client: Client):
        self._client = client
        self._context: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._context.clear()

    def think(self, query: str, **kwargs) -> ThinkResponse:
        """Send a message in the session, maintaining conversation history."""
        response = self._client.think(query, context=self._context, **kwargs)
        self._context.append({"role": "user", "content": query})
        self._context.append({"role": "assistant", "content": response.response})
        return response

    @property
    def history(self) -> list[dict]:
        return list(self._context)

    def clear(self) -> None:
        self._context.clear()


# ── Async Client ──────────────────────────────────────────────────────────────

class AsyncClient:
    """
    Async Nage API istemcisi (asyncio).

    Örnek:
        async with nage.AsyncClient("nk_live_...") as client:
            response = await client.think("Merhaba!")
            print(response.stemma)
    """

    def __init__(
        self,
        api_key:  str,
        base_url: str = "https://api.nage.ai",
        platform: str = "nage-8b",
        timeout:  int = 120,
    ):
        self.platform = platform
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    def _headers(self) -> dict:
        return {
            "X-Nage-Key":   self._api_key,
            "Content-Type": "application/json",
            "User-Agent":   f"nage-python/{__version__}",
        }

    async def _get_session(self):
        if self._session is None:
            try:
                import aiohttp
                self._session = aiohttp.ClientSession()
            except ImportError:
                raise NageError("AsyncClient için aiohttp gerekli: pip install aiohttp")
        return self._session

    async def think(
        self,
        query:       str,
        context:     list[dict] = None,
        platform:    str        = None,
        max_tokens:  int        = 512,
        temperature: float      = 0.7,
    ) -> ThinkResponse:
        session = await self._get_session()
        body = {
            "query":       query,
            "platform":    platform or self.platform,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        if context:
            body["context"] = context

        async with session.post(
            f"{self._base_url}/think",
            json=body,
            headers=self._headers(),
            timeout=self._timeout,
        ) as resp:
            data = await resp.json()
            if resp.status >= 400:
                raise NageError(data.get("message", "API hatası"), status=resp.status)
            return _parse_think_response(data)

    async def think_stream(
        self,
        query:       str,
        max_tokens:  int   = 512,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        session = await self._get_session()
        body = {
            "query": query, "platform": self.platform,
            "max_tokens": max_tokens, "temperature": temperature, "stream": True,
        }
        async with session.post(
            f"{self._base_url}/think/stream",
            json=body,
            headers=self._headers(),
        ) as resp:
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    chunk = line[6:]
                    if chunk == "[DONE]":
                        break
                    yield chunk
