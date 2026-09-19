# AGENTS.md

Operating contract for the maintainer agent in this repository.

## What this is

NInfer v3 — a fork of [NInfer](https://github.com/Neroued/ninfer) hardened for production
serving: deterministic cache planning, host-tier retention, dirty-cache stability. Product,
CLI, artifact, and architecture details live in the [README](README.md) and the upstream repo;
this file covers only what is specific to running and validating this fork.

## Working rules

- Do not commit or push without explicit user approval. Conventional Commits style, types
  consistent with history (`feat`, `fix`, `perf`, `test`, `docs`, `chore`, ...).
- The v3 `.ninfer` artifact is the only product format; upstream conversions of legacy v2 files
  go through `tools/upgrade_ninfer_v2_to_v3.py`.
- Keep the repo free of personal or machine-specific traces — defaults and examples must be
  generic.

## Layout

- `deploy.sh` / `start.sh` / `stop.sh` / `deploy-lib.sh` — workstation → GPU box deployment
- `.env.example` — every serving knob (KV pool, host tier, speculative decoding, vision, ...)
- `README.md` — public face; links upstream for the full manual

## Deployment

The launcher machine (no GPU) drives an inference box over ssh. `BOX` and `REMOTE_DIR` live in
`.env` (precedence: shell env > `.env` > script defaults). Workflow:

```bash
cp .env.example .env       # set BOX, REMOTE_DIR, ARTIFACT
./deploy.sh --dry-run      # show the sync plan only
./deploy.sh                # sync sources + rebuild the serving image on the box
./start.sh                 # launch ninfer-serve detached; returns once listening
./stop.sh                  # stop it
```

`deploy.sh` stops the server before compiling (the serving footprint pins most of the box's RAM).
The box tree at `REMOTE_DIR` is a clean copy of the git-tracked tree (`models/` + `logs/`
preserved); the box only ever runs the committed tree.

Box commands run as a single direct call with a large timeout (`deploy.sh`/`start.sh`/`stop.sh`
≥ 600 s, benches ≥ 3600 s). Never background them and poll with `tail`/`grep`/sleep — the
streamed output is the progress. Prefix Python benches with `env PYTHONUNBUFFERED=1`.

## Validation (the deployment gate)

The harness is the `cache-pressure` Python project (`uvx --from cache-pressure <tool>` from the
workstation). Benches run against the box endpoint and exit non-zero on failure. They must pass
**back-to-back on the same server** — a dirty cache is the norm, never restart to make a bench
pass. (`--base-url http://<box>:8000/v1` implied.)

| Bench | Command | Pass bar |
|---|---|---|
| `agent-sim` | `uvx --from cache-pressure agent-sim --sessions 4 --main-tokens 150000 --sub-tokens 40000` | 4/4 finalize ≥ 98.6% reuse, 116/116 mains, zero `selected_maximal_fallback` |
| `abort-sim` | `uvx --from cache-pressure abort-sim --context-tokens 40000 --thinking-tokens 500 --runs 2` | every run reuses ≥ 95% of the re-prompt and never takes the `root` path |
| `needle-test` | `uvx --from cache-pressure needle-test --lengths 50000,100000,200000` | every length PASS |
| `cache-pressure` | `uvx --from cache-pressure cache-pressure --kv-size 460000 --context-tokens 16000` | every context inside the 40-slot ceiling retained and verified; only the KV overflow shed, LRU oldest-first |

Ground truth comes from the request log (`--request-log-jsonl /logs/requests.jsonl`), not the
API: `usage.prompt_tokens_details.cached_tokens` cannot distinguish a device hit from a host-tier
restore. The log is cumulative — slice by timestamps or line offsets, and exclude the maintainer
agent's own traffic when its model is served by the box.

## Deployment reference

- Box: RTX 5090, ~30 GiB RAM, reachable over ssh (`BOX` in `.env`; the scripts exit with a
  pointer if it is not up).
- Artifact: `models/qwen3_8_27b_nvfp4.ninfer` (Qwen3.8-27B NVFP4), downloaded once on the box.
- Serving flags (as deployed): `--max-context 262144 --kv-capacity 460000 --max-concurrency 4
  --device-state-slots 4 --host-state-slots 32 --host-kv-mib 16384 --max-private-continuations
  128 --max-shared-prefixes 64 --kv-dtype nvfp4 --spec mtp --draft-tokens 4 --lm-head-draft
  --preserve-thinking --vision`.
- 32 host-state slots + 16 GiB host KV are the validated floor (agent-sim 10× dirty). The state
  pool caps retention at ~40 contexts (8 device + 32 host), which is why the `cache-pressure`
  gate uses the 16K-context variant; the 65×8K set needs ≥ 96 slots and does not fit next to the
  16 GiB KV tier on this box.
- DFlash2 (upstream `--spec dflash2`) is deliberately not adopted: the companion tensors cost
  ~26% of the NVFP4 device pool for a small single-request decode gain, and this deployment is
  pool-bound — `--spec mtp` stays the serving spec.

## Sources of truth

Read the README and the upstream docs (build, CLI, serving, performance, architecture) before
inventing your own. In-tree interfaces: `include/ninfer/engine.h` and `include/ninfer/types.h`.
Numerical and performance work follows upstream's methodology (independent oracle per Op, measure
at the claimed scope).
