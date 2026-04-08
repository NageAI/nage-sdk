# nage

Python SDK for [Nage AI](https://nage.ai) — source-attributed intelligence via SEDIM architecture.

```bash
pip install nage
```

## Quick Start

```python
import nage

client = nage.Client("nk_live_...")

# Think with source attribution
thought = client.think("What is SEDIM?")
print(thought.response)
print(thought.stemma)  # STEMMA(fehm-tr: 0.73, cortex: 0.27)

# Every response knows its source
for varve, weight in thought.stemma.top(3):
    print(f"  {varve}: {weight:.2%}")
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
