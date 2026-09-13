# AGENTS.md

These rules apply to the whole repository.

## Governing objective

Complete the user's explicit deliverable within the applicable product contract. For the declared
product target and requested outcome, choose the technically strongest coherent solution. Optimize
for architectural integrity, clear ownership, functional and numerical correctness, and maximum
relevant performance. Never optimize a solution for a small diff, few changed files, low
implementation effort, short-term simplicity, backward compatibility, or preservation of a
superseded internal path. Make every affected implementation, test, tool, and active authority
consistent with the selected design.

Correctness, performance, tests, profiling, documentation, provenance, cleanup, and tooling are
means to the requested outcome, not independent objectives. Do not let supporting work replace,
delay, or materially enlarge the requested deliverable.

## Responding to user corrections

When the user points out an error in the agent's execution or reasoning, never reply with the
formulaic opening “你说得对，……” ("you're right, ..."). This phrase and cosmetic variants of it
are strictly prohibited in correction responses because they sound reflexive and insincere. Do not
replace it with another generic agreement such as “确实如此”, “完全正确”, or “好问题”. Instead,
state the specific mistake directly, explain its concrete effect when relevant, and say what has
been or will be changed. Keep the response proportionate; do not add performative apology or praise.

When choosing between possible work, use this order:

1. respect applicable product and external-contract constraints;
2. satisfy the user's explicit deliverable and acceptance criteria;
3. preserve functional and numerical correctness of supported behavior;
4. choose the strongest architecture and clearest ownership for the declared product model;
5. maximize performance at the scope relevant to the task;
6. gather only the evidence and provenance needed to support the result.

The product and architecture described here are the current contract for ordinary work. A task may
explicitly change that contract; when it does, update the affected implementation, tests, and active
authorities consistently rather than treating the current description as an immutable prohibition.

## Scope control

Before substantial work, determine the requested output, the behavior or decision it must support,
and the conditions under which it is complete. This is an execution discipline, not a requirement
to create a separate planning artifact.

Work is in scope only when it:

- directly contributes to the requested deliverable;
- is necessary to preserve an applicable product, semantic, or external contract;
- resolves uncertainty that could materially change the result; or
- checks a realistic regression introduced by the change.

An architectural redesign, cross-cutting refactor, or replacement of an existing path is in scope
when it is necessary to deliver the strongest solution for the requested outcome. Do not use scope
control as a reason to ship an inferior patch. Do not expand into unrelated audits, cleanup,
hardening, compatibility work, benchmark campaigns, or documentation projects. General engineering
preferences, possible future scenarios, and concerns outside the declared product model do not
create requirements by themselves.

Handle incidental findings proportionally:

- address them when they block the requested outcome or make it materially incorrect;
- include them when they are inseparable from a coherent implementation;
- otherwise leave them unchanged and mention them only when they are useful to the user.

For analysis, review, or design work, the requested explanation or design artifact is the
deliverable; experiments and code inspection serve only to resolve material questions. For
implementation work, implement the selected design completely across its affected boundaries,
remove the superseded project-owned path, and validate its supported observable behavior. For
diagnosis, establish the cause and supporting evidence without turning the task into an unrequested
fix or redesign.

## Evidence, provenance, and completion

Select evidence from the claim or decision it supports. The availability of a tool, test suite,
artifact, or profiler does not make its use necessary. Prefer representative evidence over
exhaustive evidence, and do not repeat an experiment unless the previous result is invalid or
inconclusive, or the new result could change a live decision.

Verification must match the semantic contract: use exact comparison for exact formats and
transformations, and numerical or behavioral criteria for floating-point and probabilistic work.
Do not substitute final-output plausibility for verification of an operator or state transition.

Record only the provenance needed to interpret a material result. By default, this is the relevant
target, hardware/toolchain, workload or command, and summarized outcome. Fixed hashes, clean
worktrees, full command transcripts, raw profiler inventories, byte-identical regeneration, and
exact probabilistic outputs are not validity requirements unless a concrete contract or the user
requires them.

Stop when:

- the requested deliverable exists;
- applicable contracts are satisfied;
- material claims have sufficient evidence;
- relevant checks pass, or their limitations are stated clearly; and
- no known in-scope issue prevents the result from being used.

Do not continue merely to eliminate all uncertainty, collect more metrics, complete a process loop,
improve descriptive provenance, investigate unrelated observations, or make working notes
exhaustive. The final result should lead with the deliverable, key decisions, relevant verification,
and material limitations. Raw logs, experiment diaries, exhaustive command histories, hashes, and
intermediate artifacts are excluded unless requested or themselves the deliverable.

## Current product contract

NInfer is a from-scratch C++/CUDA inference engine for maximum single-GPU inference performance on
a small set of explicitly registered checkpoint artifacts. The supported identities are
`qwen3.6-27b/groupwise-int`, `qwen3.6-27b/nvfp4`, `qwen3.8-27b/groupwise-int`,
`qwen3.8-27b/nvfp4`, and `qwen3.6-35b-a3b/groupwise-int`. The current implementation is compiled
for `sm_120a` and tuned and measured on NVIDIA GeForce RTX 5090. All identities execute Text,
image/video Vision, MTP, prefix reuse, CLI, OpenAI/Anthropic serving, and measurement through the
same public `.ninfer` Engine route; the 35B-A3B target additionally supports DFlash for both Text
and image/video Vision prompts.

The current workload is one GPU and one resident model instance with a startup-fixed one to eight
active requests. The Engine forms one compact decode batch at every round boundary and uses bounded
FIFO ingress with no request preemption. Large-scale or preemptive continuous batching, priority/QoS
scheduling, additional checkpoint targets, and retargeting the implementation to another execution
platform are outside the current product. This is a local, single-owner project. Registered models,
generated artifacts, and the local workflow are trusted.
Requirements derived from a different workload, trust model, or deployment model are out of scope
until that product contract is explicitly changed.

The 27B and 35B-A3B execution packages are peer compile-time Variants of one identity-free Qwen3.6
family runtime. The family owns the shared `SequencePlan<Variant>`, `RequestPlan<Variant>`, and
`Program<Variant>` algorithms; frontend and output semantics; Text/Vision/speculative schedules;
state transactions; workspace composition; and CUDA Graph capture/replay mechanics. Each package
separately owns its registered artifact identities and bindings, immutable model view,
dimensions/storage facts, three closed execution-leaf families, graph frontier data, and Program
instance bytes. No mutable state or device allocation is shared between Programs, neither package
is defined as a delta from the other, and there is no runtime family selection or target-dependent
branch inside family scheduling. All artifacts embed the same six frontend resources, and a
prepared prompt carries no exact-target tag.

## Engineering priorities

Prioritize functional correctness, architectural quality, clear ownership, direct code, and maximum
requested inference performance. Change size, implementation difficulty, short-term simplicity,
and backward compatibility for project-owned contracts are not quality criteria. Low maintenance
cost may distinguish otherwise equivalent designs, but it never justifies worse architecture or
performance. Generality, defensive hardening, formal completeness, broad compatibility, and test
coverage are not goals by themselves.

Prefer explicit target-specific implementation over framework-like abstraction. Do not add generic
model graphs, family base classes, plugin discovery, string-driven execution, hidden device
allocation, runtime weight repacking, or placeholders for hypothetical models or hardware unless an
explicitly changed product contract requires them.

## Sources of truth

Read only current authorities relevant to a live decision in the task. The following list is a
routing map, not a mandatory reading list:

- `README.md` and executable `--help`: delivered capabilities and exact commands;
- `docs/README.md`: public documentation map;
- `docs/cli.md`: CLI input, output, sampling, MTP, and runtime options;
- `docs/serving.md`: OpenAI/Anthropic HTTP behavior;
- `docs/performance.md`: published performance methodology and results;
- `docs/maintainer/engine-architecture.md`: Gateway/Frontend/Engine/Runtime boundaries, execution
  ownership, request/response/continuation lifecycles, admission, scheduling, output transactions,
  batched execution, and CUDA Graph semantics;
- `docs/maintainer/resource-scheduling-and-context-cache.md`: resource selection and accounting,
  continuation/checkpoint ownership, materialization transactions, and Device/Host replica policy;
- `docs/maintainer/paged-kv-cache.md`: shared KV capacity, page ownership, retention, physical
  layouts, and paged consumer contracts;
- `docs/maintainer/artifact-container.md`, `storage-layouts.md`, and `tensor-formats.md`:
  generic `.ninfer` contracts;
- `docs/maintainer/qwen3.6-27b-artifact.md`, `qwen3.8-27b-artifact.md`, and
  `qwen3.6-35b-a3b-artifact.md`: exact target inventories, conversion, and binding;
- `docs/maintainer/qwen3.6-27b-model.md` and `qwen3.6-35b-a3b-model.md`: exact model mathematics,
  dimensions, and state semantics;
- `docs/maintainer/op-development.md`: Op admission, contracts, implementation ownership,
  qualification, and performance evidence rules;
- `include/ninfer/engine.h` and `include/ninfer/types.h`: in-tree C++ product interface.

Do not survey unrelated references for completeness. Read additional documents only when they
govern a live decision in the current task.

## Product and ownership boundaries

These boundaries govern ordinary implementation work. An explicit architecture task may revise
them, but must update the corresponding active authorities and affected implementation together.

- `.ninfer` is the only C++ product artifact. Do not add extension detection, compatibility shims,
  or a second product lane.
- `include/ninfer/engine.h` and `include/ninfer/types.h` are the opaque Engine interface used by
  in-tree applications and owning host values. NInfer does not currently install or export a C++
  SDK. `include/ninfer/ops/` contains repository-internal semantic Op contracts.
- `src/core` owns device primitives, tensors/views, checked layouts, arenas, graph RAII, physical
  KV-cache containers, and raw transfer mechanisms.
- `src/artifact` owns generic `.ninfer` framing, descriptors, binding primitives, and
  materialization. It has no checkpoint execution semantics.
- `src/ops` owns every semantically closed Op implementation, including fused, fixed-shape, and
  device-specialized paths. Op ownership follows the mathematical or state-transition contract,
  not its first model caller or demonstrated cross-target reuse.
- `src/targets/qwen3_6` owns only the Qwen3.6-family invariants shared by the 27B and 35B-A3B
  targets: tokenizer/template and output semantics, media preprocessing and MRoPE prompt
  construction, owning prepared-prompt/output-session types, semantic weight-view schemas, passive
  Vision definitions, and the fixed
  planning/Program/Text/Vision/speculative/state/workspace/CUDA-Graph algorithms. It has no target
  identity, registry entry, artifact binder, target leaf
  implementation, or storage for a live Program instance.
- `src/targets/<package>` owns its registered checkpoint identities, storage profiles, binder,
  `LoadedModel`, configuration, populated family model-view values and private leaf payloads,
  diagnostics, graph frontier values, and exactly three execution-leaf families: attention
  projection, GDN projection/control, and post-mixer. It aliases and instantiates the family
  runtime types; it does not own a copied Program, Text/Vision/speculative schedule, workspace
  composition, state transaction, or graph-capture algorithm. Leaf Ops remain implemented under
  `src/ops`.
- `src/runtime` owns common contracts, generated-token transaction/publication policy, and the
  public Engine PIMPL. It does not own model mathematics or target state.
- `src/media/decode` consumes already-owned bytes. URL/path/data acquisition belongs to
  `src/product/media_acquire`, CLI, or serving and is not linked into a target.
- `src/product/prompt_input` owns the shared product-side JSON/message-to-owning-input adapter.
- `src/serve` owns protocol translation and transport. CLI, server, and benchmark call only the
  public Engine for inference.
- `tools/convert/<target>` owns target-private artifact inventories, source recipes, conversion,
  and converter-side payload verification. NInfer maintains no Python model-inference route.

## Compatibility and document lifecycle

Project-owned C++ APIs, CLIs, Python tools, fixtures, reports, formats, and active documentation do
not preserve backward compatibility. When a task replaces project-owned behavior, remove the
obsolete aliases, fallbacks, transition branches, and tests in the affected contract instead of
maintaining two paths. Do not turn that rule into unrelated repository-wide cleanup.

The advertised OpenAI and Anthropic protocol surfaces are real external contracts. A change to
their behavior must update the affected schema tests and serving documentation together.

Integrate stable requirements into the existing active reference. Use a temporary dated plan only
when active work genuinely needs one; a plan is not a substitute for the requested deliverable.
Remove completed or abandoned plans instead of retaining a historical documentation tree. Do not
create parallel `final`, `v2`, or `new-design` references.

## Numerical correctness

When a task changes numerical behavior or makes a numerical claim, identify the mathematical
oracle, represented public inputs, explicit semantic cast/quantization/state boundaries, output
criterion, and real model shapes relevant to that claim. If a route's private precision or
reduction profile matters to the evidence, describe it as an implementation profile rather than a
semantic requirement. Apply exact, tolerance-based, or behavioral comparison according to the
actual semantic contract.

Every floating-point Op has one independent naive FP32/FP64 mathematical oracle; exact transforms
and codecs have one independent exact oracle. The oracle evaluates the complete logical formula
from the represented public inputs and, for packed weights, decodes the signed code with the exact
stored scale. It does not copy a production kernel's staging casts, reduction tree, workspace dtype,
or another implementation's output.

The oracle does not prescribe a production arithmetic path. Unless an intermediate value is an
observable Op output, explicit Cast/quantize/dequantize result, registered codec value, or specified
persistent state, kernels may choose the natural intermediate precision, instruction operands,
reduction association, workspace representation, and kernel decomposition for their route. A fused
kernel is neither required to reproduce an unfused BF16 materialization nor forbidden from using a
lower-precision intermediate when that is the natural qualified implementation. Every production
route is checked directly against the same oracle with a criterion appropriate to its output and
implementation profile; pairwise implementation parity is supplementary evidence only.

Where relevant to the changed behavior, account for numeric-format decode, BF16 fusion order, FP32
GDN state, BF16/INT8 KV, MTP accept/commit state, arena lifetime, and CUDA Graph address stability.
This is a risk map, not a checklist for every numerical task.

## Performance work

Define a performance claim at the level where it matters: operator, schedule, request phase, or
end-to-end inference. Measure that level directly when practical. An isolated microbenchmark can
support an operator-level claim but does not establish an end-to-end improvement.

Use whole-inference profiling when end-to-end attribution remains unresolved. Use kernel profiling
only after a relevant kernel has been identified and a kernel-level answer could materially change
the current design or implementation decision. Do not collect additional profiling data once the
relevant alternatives can be distinguished and the requested claim has adequate support.

Retain concise context sufficient to interpret a reported result: relevant hardware/toolchain,
artifact identity at the descriptive level, workload or command, and summarized measurements. Raw
reports and fixed repository or artifact hashes are not required by default.

## Tests and verification

Add or retain a test only when it protects supported observable behavior or a realistic regression:
numerical kernel/model correctness, `.ninfer` framing/binding, external schema/report behavior, a
small real integration route, GPU lifetime, or a reproduced bug. Do not add tests for coverage,
private file/class shape, getters/constructors, deleted compatibility, source-string scans,
hypothetical failures, or test ceremony.

Run a focused set of checks sufficient to support the changed behavior and its material claims.
The following are typical choices, not a cumulative checklist:

| Change | Relevant evidence |
|---|---|
| documentation | affected active-link/stale-reference review and `git diff --check` |
| C++ runtime/API | affected explicit targets and meaningful tests |
| Python tooling | `py_compile` and affected Python tests |
| `.ninfer` reader/converter/binder | affected contract tests and a real artifact when semantics require it |
| CUDA math | independent numerical oracle at relevant shapes |
| memory/lifetime | the affected execution; sanitizer only for a concrete lifetime risk |
| performance | measurement at the claimed scope; attribution tools only when needed |
| serving | affected OpenAI/Anthropic schema tests and observable request/stream behavior |

Do not replace weak verification with low-value tests. State clearly when a relevant check could not
run and why.

## Local environment

Use unrestricted build-tool parallelism for repository compilation. Invoke CMake builds as
`cmake --build <build-dir> -j`; do not supply a numeric job limit such as `-j2` or `-j32`.

These are conventional project resources, not a checklist of resources every task must use:

| Purpose | Path |
|---|---|
| repository | current checkout |
| Python 3.11 | `python3` in the selected maintainer environment |
| BF16 source checkpoint | explicit local checkpoint directory |
| product artifact | `out/qwen3_6_27b.ninfer` |
| conversion report | `out/qwen3_6_27b.ninfer.conversion.json` |
| normal build | `build/` |
| profiler output | `profiles/ncu/`, `profiles/nsys/`, `profiles/bench/` |
| hardware/toolchain | RTX 5090, `sm_120a`, CUDA 13.1 |

Use the selected Python 3.11 interpreter explicitly. Do not install or upgrade dependencies unless
the task requires it. Never select an artifact by glob, modification time, or an unqualified
“latest” name. Large artifacts, source checkpoints, and profiler outputs are local prerequisites;
do not download or regenerate them unless that work is in scope.

```bash
PYTHON=python3
MODEL=/path/to/Qwen3.6-27B
NINFER_WEIGHTS=out/qwen3_6_27b.ninfer
```

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

Create a commit only when the user requests one. Use Conventional Commit-style subjects, for
example:

```text
feat(engine): cut over the registered target to native artifacts
```

Use concise lowercase types consistent with repository history (`feat`, `fix`, `perf`, `bench`,
`test`, `build`, `refactor`, `docs`, `chore`).
