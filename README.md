# nage

Python SDK for [Nage AI](https://nage.ai) — source-attributed intelligence via SEDIM + STRATUM.

```bash
pip install nage
```

## Quick Start (OpenAI-compatible)

```python
import nage

client = nage.Client("nk_live_...", base_url="https://api.sedim.ai")

# Drop-in OpenAI-shape — STRATUM v1.1 §4.3
r = client.chat_completion(
    messages=[{"role": "user", "content": "What is SEDIM?"}],
    model="fehm-8b",
    inference_mode="resonance_gated",   # SEDIM §5.3 — ~30% latency win
)

print(r.text)                           # assistant's reply
print(r.gamma.epistemic_label)          # STABLE / EVOLVING / CONTESTED ...
print(r.gamma.confidence)               # 0.0–1.0
print(r.audit_id)                       # for /v1/audit/{id} retrieval
```

## KLM γ vector (Knowledge Layer Model)

Every response carries a γ vector — the epistemic fingerprint of the answer.
Visibility is tier-stratified per STRATUM v1.1 §3:

| Tier    | γ Detail   | Field set returned |
|---------|------------|--------------------|
| SURFACE | hidden     | (no γ in response) |
| DRIFT   | label      | `epistemic_label`, `warning` |
| VEIN    | confidence | + `confidence`, `dominant_source` |
| LODE    | full       | + `evidence_score`, `coherence_score`, `freshness_score`, `provenance_map` |
| CORE    | complete   | + `audit_ref` (audit_id, frame_id, signed_by) |

```python
g = r.gamma
if g and g.is_contested():
    print(f"⚠ {g.warning}")
    print(f"  sources in conflict: {g.provenance_map}")
```

## Audit retrieval (LODE+ tier)

```python
# Pull the original audit Knowledge Unit for any past response
audit = client.audit_get(r.audit_id)
print(audit.gamma.epistemic_label, audit.gamma.confidence)

# AI Act export — tar.gz of records + signatures + manifest
export = client.audit_export(period_start="2026-04-01", period_end="2026-04-30")
export.save("audit-2026-04.tar.gz")
print(f"exported {export.record_count} records")
```

## Legacy `client.think()` still works

```python
# /think endpoint — unchanged from v0.1
thought = client.think("Türk kahvesi nasıl yapılır?")
print(thought.response)
print(thought.stemma)        # STEMMA(fehm-tr: 0.18, ...)
print(thought.gamma)         # NEW in v0.2 — KLM γ when backend ships it
print(thought.audit_id)      # NEW in v0.2
```

## Models

```python
for m in client.models_list():
    print(f"{m.id:15s}  ctx={m.context_window}  tier_min={m.tier_min}")
# fehm-8b          ctx=32768  tier_min=surface
# cortex-14b       ctx=32768  tier_min=vein
# bilge-14b        ctx=32768  tier_min=vein
```

## API

```python
# Inference
thought = client.think("query", platform="nm/fehm", max_tokens=1024)

# Streaming (SSE)
for chunk in client.think_stream("Explain VARVE architecture"):
    print(chunk, end="", flush=True)

# Multi-turn session
with client.session() as session:
    r1 = session.think("What is a VARVE?")
    r2 = session.think("How does it differ from LoRA?")  # automatic context

# Teach new knowledge
client.learn("Domain-specific text...", domain="legal-tr", layer="FEHM")

# Knowledge state
knowledge = client.knowledge
for layer, varves in knowledge.layers.items():
    print(f"  {layer}: {[v.varve_id for v in varves]}")

# FACIES consolidation (requires LODE+ tier)
client.consolidate()

# Platform info
info = client.platform()
print(info["formula"])  # CENTO = FACIES + Σᵢ STEMMAᵢ(x) · VARVEᵢ
print(info["stratum_tiers"])
```

## Async

```python
import asyncio, nage

async def main():
    async with nage.AsyncClient("nk_live_...") as client:
        thought = await client.think("Hello!")
        print(thought.stemma)

        async for chunk in client.think_stream("Stream test"):
            print(chunk, end="", flush=True)

asyncio.run(main())
```

## STEMMA — Source Attribution

Every response carries its knowledge lineage:

```python
thought.stemma.weights        # {"fehm-tr": 0.73, "cortex": 0.27}
thought.stemma.dominant_varve # "FEHM/tr"
thought.stemma.dominant_layer # "FEHM"
thought.stemma.entropy        # 0.61
thought.stemma.top(2)         # [("fehm-tr", 0.73), ("cortex", 0.27)]
```

## Platforms

| Platform | Focus | Base |
|----------|-------|------|
| `nage-8b` | General purpose | Qwen3-8B |
| `nm/fehm` | Arabic/Turkish | Qwen3-8B |
| `nm/ming` | Code generation | Qwen3-8B |
| `nm/bilge` | ML expertise | Qwen3-14B |
| `nm/cortex` | Reasoning | Qwen3-14B |

## STRATUM Tiers

| Tier | Price | Limit |
|------|-------|-------|
| SURFACE | Free | 1K/day |
| DRIFT | $19/mo | 100K/mo |
| VEIN | $49/mo | 1M/mo |
| LODE | $149/mo | 10M/mo |
| CORE | Custom | Unlimited |

## Links

- [nage.ai](https://nage.ai)
- [nage.ai/platform](https://nage.ai/platform)
- [nage.ai/research](https://nage.ai/research)
- [github.com/NageAI/sedim](https://github.com/NageAI/sedim)

## License

MIT
