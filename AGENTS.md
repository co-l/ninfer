# AGENTS.md

These rules apply to the whole repository.

## Objective and scope

Complete the user's explicit deliverable within the applicable product and external contracts.
Choose a coherent solution with functional and numerical correctness, clear ownership, strong
architecture, and maximum performance at the requested scope. Do not sacrifice these goals to
reduce the diff or implementation effort. Evaluate complexity, maintenance cost, and verification
risk as engineering tradeoffs, not reasons to retain a known inferior design.

Before substantial work, identify the deliverable and its completion conditions. Work is relevant
when it completes that deliverable, preserves an applicable contract, resolves a material
uncertainty, or checks a realistic regression. A necessary redesign is in scope; unrelated cleanup,
hardening, compatibility, and benchmark campaigns are not. Address incidental findings when they
block the outcome or are inseparable from the selected implementation.

For analysis or design, deliver the explanation or design. For diagnosis, establish the cause and
supporting evidence; implement a fix when requested. For implementation, complete the selected
design across its affected implementations, callers, tests, tools, and active documentation.

The current product and architecture govern ordinary work. An explicit task may change them;
update the affected contracts and implementation together instead of treating the current design
as an immutable prohibition. Skills provide task-specific methods, not additional deliverables or
approval requirements beyond the user's instructions and the actual execution environment.

## Product and architecture

NInfer is a from-scratch C++/CUDA inference engine for maximum single-GPU performance. It implements
`Qwen3_5ForCausalLM` and `Qwen3_5MoeForCausalLM`; official Qwen3.6/3.8 artifacts and user recipes
use the same architecture, binding and execution path. The implementation targets `sm_120a` and
is tuned on NVIDIA GeForce RTX 5090.

Generation uses one GPU, one resident model, startup-fixed concurrency of one to eight requests,
bounded FIFO ingress, no active-request preemption, and one compact decode batch per round.
Generation and offline CausalScoring use the same public `.ninfer` Engine route. Delivered
capabilities and commands are documented in `README.md`, the product guides, and executable
`--help`. New mathematical architectures, execution platforms, large-scale/preemptive continuous
batching, and priority/QoS require an explicit product change. Another training instance or mixture
of existing representations does not require a checkpoint-specific execution registration.

This is a local, single-owner project with trusted local models, generated artifacts, and
local workflow. Do not derive requirements from a different deployment or trust model.

Keep these ownership boundaries visible when selecting a design:

- v3 `.ninfer` is the only C++ product artifact; CLI, serving, and inference benchmarks use the public
  Engine. NInfer has no Python model-inference route or installed/exported C++ SDK.
- Core owns physical primitives and raw transfers; artifact owns generic framing and
  materialization; Ops own closed mathematical and state-transition implementations.
- Models own fixed mathematics, config interpretation, logical parameter binding, frontend
  semantics and finite execution composition. Immutable Model data owns selected weights and
  resources; native Parameters supply the actual operands to planning and Program execution.
  Program owns mutable state, workspace, context stores and CUDA Graphs. Programs share no mutable
  state or device allocation.
- Converter recipes choose sources, formats, packing and per-input activation permissions. The
  loader validates, uploads and binds the stored representation. Native preparation, resource
  queries and execution enforce actual Op support; there is no whole-artifact capability registry.
- Runtime owns common execution contracts and Engine publication policy; product/serving own input
  acquisition and protocol translation. Model code does not acquire media or own transport.

Detailed model/runtime responsibilities and source ownership are defined in
[Engine architecture](docs/maintainer/engine-architecture.md). Read the relevant boundary before
changing it. Prefer explicit implementations for supported architectures. Do not introduce generic model
graphs, family base classes, plugin discovery, string-driven execution, hidden device allocation,
runtime weight repacking, or placeholders for hypothetical targets without a product requirement.

## Change consistency

Project-owned APIs, CLIs, Python tools, fixtures, reports, formats, and documentation do not preserve
backward compatibility. When replacing behavior, remove superseded aliases, fallbacks, transition
branches, and their tests within the affected contract. Leave unrelated paths alone.

Advertised OpenAI and Anthropic protocol behavior is an external contract. Changes update the
affected schema tests and serving documentation together.

Keep stable requirements in their existing active reference. Temporary plans are useful only for
active work; remove them when completed or abandoned. Maintain one current authority rather than
parallel `final`, `v2`, or `new-design` documents.

## Verification and completion

Select evidence to support the changed behavior and material claims. Tests should protect supported
observable behavior, mathematical or state semantics, and realistic regressions, including plausible
boundary failures that have not occurred yet. Avoid tests that merely mirror implementation,
freeze private file/class organization, or increase coverage numbers.

For numerical changes, identify represented public inputs, the independent mathematical oracle,
semantic cast/quantization/state boundaries, output criteria, and relevant real model shapes. Each
floating-point Op uses a naive FP32/FP64 oracle; exact transforms/codecs use an exact oracle. Packed
inputs are independently decoded with their stored scales. Qualify production routes directly
against that oracle, not another kernel or plausible model output. Private arithmetic need not
reproduce unfused materializations unless an intermediate is an observable semantic boundary.
[Op development](docs/maintainer/op-development.md) defines the full qualification contract.

Measure performance at the claimed scope. An Op microbenchmark establishes an Op result, not an
end-to-end improvement. Use whole-inference profiling when an in-scope end-to-end attribution is
unresolved; use kernel profiling when an identified kernel question can change the decision. Reuse
applicable evidence and stop collecting once the relevant alternatives can be distinguished.

Choose the affected checks, rather than running this table as a checklist:

| Change | Typical evidence |
|---|---|
| Documentation | affected links/references and `git diff --check` |
| C++ runtime/API | affected build targets and behavioral tests |
| Python tooling | Python 3.11 `py_compile` and affected tests |
| Artifact framing/binding/conversion | affected contract tests; real artifact when semantics require it |
| CUDA mathematics | independent oracle at relevant shapes and route boundaries |
| Memory or lifetime | affected execution; sanitizer for a concrete lifetime question |
| Performance | measurement at the claimed scope; profiling only for unresolved attribution |
| Serving | affected schema tests and observable request/stream behavior |

Record the target, relevant hardware/toolchain, workload or command, and summarized result needed
to interpret a material claim. Hashes, clean worktrees, full command transcripts, raw report
inventories, and exact probabilistic outputs are not default requirements. Use exact comparison for
exact outputs, and appropriate numerical or behavioral criteria otherwise. State checks that could
not run and their implications.

Finish when the deliverable is usable, applicable contracts are satisfied, material claims have
sufficient evidence, relevant checks pass or their limitations are clear, and no known in-scope
issue blocks use. Expand or repeat verification only for new changes, failures, or unresolved risks
that could change the result. Supporting work is not an independent completion objective.

## Reference navigation

Read the authority relevant to the current decision; this is not a mandatory reading list.

| Decision | Entry point |
|---|---|
| Product capabilities and exact commands | `README.md`, executable `--help`; `docs/cli.md`, `docs/serving.md`, `docs/perplexity.md` |
| Execution, model/runtime ownership, scheduling, transactions, graphs | `docs/maintainer/engine-architecture.md` |
| Context resources, checkpoints, replicas; physical KV | `docs/maintainer/resource-scheduling-and-context-cache.md`; `docs/maintainer/paged-kv-cache.md` |
| Artifact, layout, codec, conversion, or model mathematics | model/artifact references and conversion guide linked from `docs/README.md` |
| Op contracts, implementation ownership, numerical/performance qualification | `docs/maintainer/op-development.md` |
| Test/benchmark commands and published performance | `tests/README.md`, `bench/README.md`, `docs/performance.md` |
| In-tree C++ interface | `include/ninfer/engine.h`, `include/ninfer/types.h` |

[Documentation map](docs/README.md) routes to narrower authorities when needed.

## Local operations

Use `cmake --build <build-dir> -j` by default. Adjust parallelism when actual resource pressure
causes failures or interferes with the task, and briefly explain why.

## How to validate against the real deployment

The end-to-end truth is the engine serving live on the RTX 5090 box; the
measurement harness is the **`cache-pressure`** project at `../cache-pressure`
(a `uv` Python project — run tools with `uv run <tool>`, never `uvx --from .`
while developing, per its own AGENTS.md). All four benches run from the
workstation against the box endpoint and exit non-zero on failure. They run
**back-to-back on the same server, without restarts**: a dirty cache is the
norm, and a change is validated only when **all four pass dirty on the same
image**.

| Bench | Command (`--base-url http://192.168.1.238:8000/v1` implied) | What it proves | Pass bar |
|---|---|---|---|
| `agent-sim` | `uv run agent-sim --sessions 4 --main-tokens 150000 --sub-tokens 40000` | four concurrent 150K-token agent sessions survive cache pressure: main-continuation reuse and finalize survival | 4/4 finalize ≥ 98.6% reuse, 116/116 mains, zero `selected_maximal_fallback` (ground truth via `--ninfer-log`) |
| `abort-sim` | `uv run abort-sim --context-tokens 40000 --thinking-tokens 500 --runs 2` | a mid-thinking client abort loses no processed prefix: re-orientation re-sends the captured partial turn and must reuse it | every run reuses ≥ 95% of the re-prompt and never takes the `root` path (see `../5090/SESSION-2026-09-13-abort-repro.md`) |
| `needle-test` | `uv run needle-test --lengths 50000,100000,200000` | end-to-end long-context retrieval stays intact (hidden secret codes at depth) — the cache is not masking a broken prefill | every length PASS |
| `cache-pressure` | `uv run cache-pressure --kv-size 460000` | retention under real overflow pressure: hydrate `ceil(kv-size/8001)+5` ~8K contexts (63 at 460K), re-send in reverse order | 100% retained (63/63 at 460K), ≈ 108% of the advertised capacity — on a cache already holding other dead content (no restart) |

Remote work runs by direct call (agent behaviour):

- **Direct call, never background.** Every box command — `./deploy.sh`,
  `./start.sh`, `./stop.sh`, and each bench — is a single `run_command`
  with a large timeout (`./deploy.sh`/`./start.sh`/`./stop.sh`:
  ≥ 600000 ms; benches: ≥ 3600000 ms). The call blocks and its output
  streams **live** in the session: that is both the progress view and the
  wait — the tool returns only when the command exits, carrying the full
  transcript + exit code.
- **No `tail`, no `grep`, no ssh-peeking, no sleep-polling.** Do not launch
  things via `background_process` and then poll the box with `sleep` +
  `tail`/`grep` to track progress, and do not interrupt a running call to
  "check" on it. The streamed output IS the progress. (The only `tail` in
  the repo lives inside `start.sh`, whose job is to return once
  `listening on` appears — the agent never tails.)
- Prefix `env PYTHONUNBUFFERED=1` for Python benches (e.g. `env
  PYTHONUNBUFFERED=1 uv run agent-sim ...`): Python block-buffers stdout
  when piped, so without it the stream arrives in ~8 KiB chunks, hiding
  progress. `agent-sim` also prints one per-turn line per completed turn
  when the TTY view is off (see `../cache-pressure/AGENTS.md`).
- Announce the command + expected duration before launching; on return, read
  the exit code and summary for the verdict (a bench exits non-zero on
  failure). A hung command blocks until the timeout — size it to the job.

Hygiene (results are only comparable when these hold):

- **Dirty state is the norm — never restart to make a bench pass.** A correct
  cache evicts the oldest *dead* content to make room for a new working set,
  so every bench must pass on a cache already occupied by prior work (a bench
  that only passes cold is hiding an engine defect — the host tier
  accumulating dead content and evicting the wrong things — not a hygiene
  requirement). The historical "fresh server required" caveat on retention
  (2/65 after agent-sim) was exactly that defect and is superseded by this
  criterion.
- **Ground truth comes from the request log**, not the API:
  `usage.prompt_tokens_details.cached_tokens` cannot distinguish a device hit
  from a RAM-tier restore. The box writes `--request-log-jsonl
  /logs/requests.jsonl` (host path `/home/conrad/dev/nicefox-5090-prod/logs/requests.jsonl`,
  appended, cumulative across runs — slice by timestamps or line offsets)
  with `computed_prefill_tokens`,
  `prefix_cache_hit_tokens`, `prefix_reuse_path`. Pass `--ninfer-log`
  (agent-sim) or fetch the tail and `abort-sim --annotate run.json
  --ninfer-log <tail>` for the ground-truth reuse class (`hit` / `partial` /
  `miss`).
- `--salt` gives deterministic A/B (identical planned inputs; model-generated
  replies still differ, so cross-run prefixes are not byte-identical).
- `abort-sim`: the probe re-prompt uses `max_tokens: 1` because the engine
  treats `max_tokens: 0` as a no-op that runs no prefill at all
  (`Engine::submit` short-circuit; the Anthropic endpoint rejects it as
  `cache_prewarm_not_supported`).
- `needle-test`: keep lengths under the served `max_model_len` (the default
  50K–450K sweep overflows this deployment's 262144 context — pass
  `--lengths` as above).
- **The maintainer agent may run on the box itself.** If the agent's own
  model is served by the box (e.g. this repo's `qwen3.8-27b`), the agent's
  inference appears in the request log and its context occupies the cache:
  the box is never idle while the agent generates, its own requests must be
  excluded from bench analysis (slice by the bench's own timestamps/salts),
  and a cache-overflowing bench run will evict the agent's own context
  (its next turn re-prefills). Run benches from a session that does not
  share the box's serving instance.
- One engine server runs on the box at a time (ninfer or vLLM); the launch
  scripts stop the other.

### The deployment (box `gaming_pc`, 192.168.1.238)

| | |
|---|---|
| hardware | RTX 5090 (32 GiB), 30 GiB RAM; box must be ON — wake with `../5090/poweron-gaming-pc.sh` (WoL + 400 W cap) |
| source tree | `/home/conrad/dev/nicefox-5090-prod` (plain tree, **not** a git repo; a clean copy of this repository — git-tracked files only, deployed by `./deploy.sh` — plus the box-local `models/` + `logs/`; the cache-fix planner work is committed on the `nicefox` remote, no patch files) |
| artifact | `models/qwen3_8_27b_nvfp4.ninfer` (21.5 GiB, Qwen3.8-27B NVFP4) |
| image | `localhost/ninfer:local` (Dockerfile: CUDA 13.1.2-devel/Ubuntu 24.04, `podman build --jobs 4` with ninja parallelism via `BUILD_PARALLEL`, default 16; measured: compile peaks at ~7 GiB RAM and takes ~4 min — CPU-bound, not RAM-bound) |
| container | `ninfer-serve` via `./start.sh` (launch) / `./stop.sh` (stop) in this repo (rootful podman, GPU 0, ports 8000, mounts `models/` ro + `logs/`); env overrides `NINFER_CONCURRENCY` (4), `NINFER_KV_CAPACITY` (460000), `NINFER_KV_DTYPE` (nvfp4), `NINFER_DEVICE_STATE_SLOTS` (4), `NINFER_VISION` (1) |
| request log | `/home/conrad/dev/nicefox-5090-prod/logs/requests.jsonl` |
| health | `curl -sf localhost:8000/health` on the box (model load is seconds; serve log shows `engine ready`) |

Serving flags (as deployed): `--max-context 262144 --kv-capacity 460000
--max-concurrency 4 --max-pending-requests 16 --pending-timeout-ms 600000
--device-state-slots 4 --host-state-slots 96 --host-kv-mib 12288
--max-private-continuations 128 --max-shared-prefixes 64 --kv-dtype nvfp4
--spec mtp --draft-tokens 4 --lm-head-draft --preserve-thinking --vision`.
Sweep-derived floors in `ninfer-serve.sh`'s header: 460K is the VRAM-safe
device pool (hard limit ≈ 469K at C=4), 96 host state slots and 12 GiB host
KV are mandatory for finalize survival, `--max-private-continuations 128` is
required by the 65×8K retention set, and no container memory limit is set on
purpose (the pinned state+KV footprint needs the whole box).

### Build and deploy flow

The box image is built from this repository's sources only. The box tree is a
clean copy of this repo (git-tracked files, `--delete` sync) plus the
box-local `models/` and `logs/`; there is no patch layer anymore — the
cache-fix planner work is committed on the `nicefox` remote.

1. Change the source and commit on `main` (the box only ever deploys the
   committed tree).
2. Deploy: `./deploy.sh` — rsyncs the tracked sources to
   `gaming_pc:/home/conrad/dev/nicefox-5090-prod/`, stops `ninfer-serve`
   (the serving footprint pins ~28 GiB; the compile alone is only ~7 GiB, but
   server + build together exceed the box's 30 GiB), then builds
   `localhost/ninfer:local` on the box
   (`sudo podman build --jobs 4 -t ninfer:local .` with
   `--build-arg BUILD_PARALLEL=$PARALLEL`, default 16 → ~4 min).
   `./deploy.sh --dry-run` shows the sync plan without building. Env
   overrides: `BOX`, `REMOTE_DIR`, `IMAGE`, `PARALLEL`. Box must be on
   (`ssh gaming_pc`), else the script exits with a pointer to
   `../5090/poweron-gaming-pc.sh`.
3. Launch: `./start.sh` — launches `ninfer-serve` detached on the box and
   returns once the log shows `listening on` (never tails forever). `./stop.sh`
   stops it. Both live in this repo and embed the full serving recipe; the
   box must be on (`ssh gaming_pc`) and port 8000 free (only one engine serves
   at a time — stop vLLM/SGLang first if they own port 8000). Wait for
   `engine ready` / `/health` OK.
4. Validate from the workstation: the cache-pressure bench suite must pass
   back-to-back on the same server, no restart — `agent-sim`, `needle-test`,
   `cache-pressure` (see the table above; `abort-sim` is a known-open bug,
   SESSION-2026-09-13-abort-repro.md, not part of the deploy gate).

## Commits

Use the selected Python 3.11 interpreter explicitly. On this machine it is
`/home/neroued/miniconda3/envs/py311/bin/python`; the default shell's `python3` may be a different
version. Use `python3` only after selecting the maintainer environment or checking its version.
Normal resources are `build/`, `out/qwen3_6_27b.ninfer`, its `.conversion.json` report, and
`profiles/ncu/`, `profiles/nsys/`, `profiles/bench/`; the local toolchain is CUDA 13.1.
Select model artifacts by explicit path, never glob order, modification time, or unqualified
“latest”. Source checkpoints and large artifacts are prerequisites; download or regenerate them
only when that work is in scope. Install or upgrade dependencies only when the task needs it.

Create commits only when requested. Use Conventional Commit subjects with concise lowercase types
such as `feat`, `fix`, `perf`, `bench`, `test`, `build`, `refactor`, `docs`, or `chore`.
