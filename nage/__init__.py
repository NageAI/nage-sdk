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

__version__ = "0.1.0"
__all__ = ["Client", "AsyncClient", "NageError", "AuthError", "RateLimitError"]

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
class ThinkResponse:
    thought_id: str
    response:   str
    stemma:     STEMMA
    knowledge:  list[KnowledgeSource] = field(default_factory=list)
    meta:       dict                  = field(default_factory=dict)

    def __repr__(self):
        return f"ThinkResponse(thought_id={self.thought_id!r}, stemma={self.stemma})"


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
