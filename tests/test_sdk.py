"""Nage SDK tests — models, parsing, client, session."""

import json
import pytest
from unittest.mock import MagicMock, patch
from nage import (
    Client, STEMMA, ThinkResponse, KnowledgeSource,
    NageError, AuthError, RateLimitError,
)


# ── Model Tests ──────────────────────────────────────────────────────────

class TestSTEMMA:
    def test_repr(self):
        s = STEMMA(weights={"a": 0.7, "b": 0.3}, dominant_layer="A",
                   dominant_varve="a", entropy=0.6)
        assert "0.70" in repr(s)

    def test_top(self):
        s = STEMMA(weights={"a": 0.1, "b": 0.6, "c": 0.3},
                   dominant_layer="B", dominant_varve="b", entropy=0.9)
        top = s.top(2)
        assert top[0] == ("b", 0.6)
        assert len(top) == 2

    def test_empty_weights(self):
        s = STEMMA(weights={}, dominant_layer="", dominant_varve="", entropy=0.0)
        assert s.top(3) == []


class TestThinkResponse:
    def test_repr(self):
        s = STEMMA(weights={"a": 1.0}, dominant_layer="A",
                   dominant_varve="a", entropy=0.0)
        r = ThinkResponse(thought_id="tht_123", response="Hello", stemma=s)
        assert "tht_123" in repr(r)

    def test_knowledge_list(self):
        s = STEMMA(weights={}, dominant_layer="", dominant_varve="", entropy=0.0)
        k = KnowledgeSource(varve="a", layer="A", weight=0.8, confidence=0.9)
        r = ThinkResponse(thought_id="t", response="", stemma=s, knowledge=[k])
        assert len(r.knowledge) == 1
        assert r.knowledge[0].weight == 0.8


# ── Exception Tests ──────────────────────────────────────────────────────

class TestExceptions:
    def test_nage_error(self):
        e = NageError("test", code="test_code", status=400)
        assert str(e) == "test"
        assert e.code == "test_code"
        assert e.status == 400

    def test_auth_error_is_nage_error(self):
        assert issubclass(AuthError, NageError)

    def test_rate_limit_is_nage_error(self):
        assert issubclass(RateLimitError, NageError)


# ── HTTP Parsing Tests ───────────────────────────────────────────────────

class TestParsing:
    def test_parse_think_response(self):
        from nage import _parse_think_response
        data = {
            "thought_id": "tht_abc",
            "response": "Hello world",
            "stemma": {
                "weights": {"fehm-tr": 0.7, "cortex": 0.3},
                "dominant_layer": "FEHM",
                "dominant_varve": "FEHM/tr",
                "entropy": 0.61,
            },
            "knowledge": [
                {"varve": "fehm-tr", "layer": "FEHM", "weight": 0.7, "confidence": 0.85}
            ],
            "meta": {"platform": "nage-8b"},
        }
        result = _parse_think_response(data)
        assert result.thought_id == "tht_abc"
        assert result.response == "Hello world"
        assert result.stemma.weights["fehm-tr"] == 0.7
        assert result.stemma.dominant_varve == "FEHM/tr"
        assert len(result.knowledge) == 1

    def test_parse_knowledge(self):
        from nage import _parse_knowledge
        data = {
            "platform": "nage-8b",
            "total_varves": 4,
            "layers": {
                "FEHM": [{"varve_id": "FEHM/tr", "layer": "FEHM",
                          "status": "optimal", "description": "test"}]
            },
        }
        result = _parse_knowledge(data)
        assert result.total_varves == 4
        assert "FEHM" in result.layers
        assert result.layers["FEHM"][0].status == "optimal"


# ── Client Tests ─────────────────────────────────────────────────────────

class TestClient:
    def test_init(self):
        c = Client("nk_live_test123", platform="nm/fehm")
        assert c.platform == "nm/fehm"
        assert "nk_live_test123" in c._http.api_key

    def test_repr(self):
        c = Client("nk_live_test123")
        assert "nage.Client" in repr(c)

    def test_think_builds_correct_body(self):
        c = Client("nk_live_test123")
        with patch.object(c._http, "post") as mock_post:
            mock_post.return_value = {
                "thought_id": "tht_1",
                "response": "hi",
                "stemma": {"weights": {}, "dominant_layer": "",
                           "dominant_varve": "", "entropy": 0},
                "knowledge": [],
                "meta": {},
            }
            c.think("hello", max_tokens=256, temperature=0.5)
            call_body = mock_post.call_args[0][1]
            assert call_body["query"] == "hello"
            assert call_body["max_tokens"] == 256
            assert call_body["temperature"] == 0.5

    def test_think_with_varve_hint(self):
        c = Client("nk_live_test123")
        with patch.object(c._http, "post") as mock_post:
            mock_post.return_value = {
                "thought_id": "t", "response": "", "stemma": {
                    "weights": {}, "dominant_layer": "",
                    "dominant_varve": "", "entropy": 0},
                "knowledge": [], "meta": {},
            }
            c.think("q", varve_hint="fehm-tr")
            body = mock_post.call_args[0][1]
            assert body["varve_hint"] == "fehm-tr"

    def test_learn(self):
        c = Client("nk_live_test123")
        with patch.object(c._http, "post") as mock_post:
            mock_post.return_value = {
                "varve_id": "CHI/session-abc",
                "status": "queued",
            }
            result = c.learn("test text", domain="legal", layer="CHI")
            body = mock_post.call_args[0][1]
            assert body["text"] == "test text"
            assert body["domain"] == "legal"

    def test_health(self):
        c = Client("nk_live_test123")
        with patch.object(c._http, "get") as mock_get:
            mock_get.return_value = {"status": "ok"}
            h = c.health()
            assert h["status"] == "ok"


# ── Session Tests ────────────────────────────────────────────────────────

class TestSession:
    def test_session_maintains_context(self):
        c = Client("nk_live_test123")
        with patch.object(c._http, "post") as mock_post:
            mock_post.return_value = {
                "thought_id": "t", "response": "answer",
                "stemma": {"weights": {}, "dominant_layer": "",
                           "dominant_varve": "", "entropy": 0},
                "knowledge": [], "meta": {},
            }
            with c.session() as s:
                s.think("first question")
                assert len(s.history) == 2  # user + assistant

                s.think("follow up")
                assert len(s.history) == 4
                # Second call includes full history at time of call
                second_call_body = mock_post.call_args_list[1][0][1]
                assert "context" in second_call_body

    def test_session_clears_on_exit(self):
        c = Client("nk_live_test123")
        with patch.object(c._http, "post") as mock_post:
            mock_post.return_value = {
                "thought_id": "t", "response": "ok",
                "stemma": {"weights": {}, "dominant_layer": "",
                           "dominant_varve": "", "entropy": 0},
                "knowledge": [], "meta": {},
            }
            session = c.session()
            with session:
                session.think("test")
                assert len(session.history) == 2
            assert len(session.history) == 0


# ── HTTP Client Error Handling ───────────────────────────────────────────

class TestHTTPErrors:
    def test_401_raises_auth_error(self):
        from nage import _HTTPClient
        http = _HTTPClient("bad_key", "https://api.nage.ai", 30)
        with pytest.raises(AuthError):
            http._raise_for_status(401, b'{"error": "invalid_api_key"}')

    def test_429_raises_rate_limit(self):
        from nage import _HTTPClient
        http = _HTTPClient("key", "https://api.nage.ai", 30)
        with pytest.raises(RateLimitError):
            http._raise_for_status(429, b'{"error": "rate_limit"}')

    def test_500_raises_model_error(self):
        from nage import _HTTPClient, ModelError
        http = _HTTPClient("key", "https://api.nage.ai", 30)
        with pytest.raises(ModelError):
            http._raise_for_status(500, b'{"error": "server_error"}')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
