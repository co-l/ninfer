# NInfer v3

> Upstream [NInfer](https://github.com/Neroued/ninfer), hardened for production serving.

Maximum single-GPU inference performance for Qwen3.6/3.8-27B and Qwen3.6-35B-A3B, with one job
above all: **staying up and fast under sustained, real-world load** — long agent sessions, cache
overflow, aborts, and dirty caches.

The full build, CLI, artifact, and API documentation lives in the
[upstream repository](https://github.com/Neroued/ninfer). This fork keeps the same `.ninfer`
artifacts, the same binary interface, and the same OpenAI/Anthropic-compatible serving — and adds
the reliability layer on top.

## What this fork adds

- **Deterministic cache planning.** A value-ranked planner decides what the KV cache keeps,
  demotes to a pinned host tier, and evicts — no search, no surprises, no live-session casualties.
- **Dirty-cache stability.** Retention is validated back-to-back on an already-loaded server, no
  restarts: 4 concurrent 150K-token agent sessions (10/10 runs), overflow retention beyond the
  advertised KV capacity, abort re-orientation, and long-context retrieval.
- **Host-tier retention.** Cold weight spills to pinned host KV instead of being destroyed, and is
  restored transparently.
- **Tool-call streaming** across the OpenAI and Anthropic protocols.
- **Deployment scripts** to build and serve on a dedicated GPU box from any workstation.

## Deployment: workstation → GPU box

The launcher machine (no GPU required) talks to the inference box over ssh. Point `.env` at your
box, download the model onto the box, and you're set:

```bash
cp .env.example .env      # set BOX, REMOTE_DIR, ARTIFACT
# download the artifact into models/ on the box (mounted read-only by start.sh):
ssh "$BOX" "hf download neroued/Qwen3.8-27B-nvfp4-NInfer qwen3_8_27b_nvfp4.ninfer \
  --local-dir '$REMOTE_DIR/models'"
./deploy.sh               # sync sources, stop the old server, rebuild the image on the box
./start.sh                # launch ninfer-serve detached; returns once it is listening
./stop.sh                 # stop ninfer-serve
```

`start.sh` serves on port `8000` by default; every knob (KV pool, host tier, speculative decoding,
vision, concurrency) is a `.env` variable documented in `.env.example`.

## Validation

The deployment gate is the [cache-pressure bench suite](https://github.com/co-l/cache-pressure),
run back-to-back on the same server without restarts:

| Bench | Proves |
|---|---|
| `agent-sim` | four concurrent 150K-token agent sessions survive cache pressure |
| `needle-test` | long-context retrieval stays intact (no masked prefill bugs) |
| `cache-pressure` | retention under real overflow pressure, LRU-correct eviction |

## Supported models

| Model | Formats |
|---|---|
| Qwen3.6-27B | `groupwise-int`, `nvfp4` |
| Qwen3.8-27B | `groupwise-int`, `nvfp4` |
| Qwen3.6-35B-A3B | `groupwise-int` (+ DFlash) |

Artifacts and their download links: see the [upstream repo](https://github.com/Neroued/ninfer).

## Repository map

| Path | What it is |
|---|---|
| `deploy.sh` / `start.sh` / `stop.sh` / `deploy-lib.sh` | box deployment scripts |
| `.env.example` | serving configuration reference |
| `AGENTS.md` | maintainer contract for this deployment |

## License

Apache-2.0, same as upstream. See [LICENSE](LICENSE).
