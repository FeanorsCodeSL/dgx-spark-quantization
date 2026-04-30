# Nemotron-3-Nano-Omni-30B-A3B-Reasoning — AWQ-INT4 Quantization + 3-way Eval on DGX Spark

## Goal

Two outputs, both targeting
[`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16)
on a single DGX Spark (GB10 / SM121a, 128 GiB unified memory):

1. **A vLLM-loadable AWQ-INT4 GEMM artifact** (data-free RTN, AutoAWQ
   format) — smallest disk (~22–25 GiB target), vision + audio towers
   preserved at fp16 for multimodal serving.
2. **A three-way head-to-head eval** (bf16 baseline vs our AWQ vs
   NVIDIA's published NVFP4 build) on the same battery used for
   `runs/qwen3.6-35b-distill/` (GSM8K full chat-CoT, full MMLU,
   ARC-Challenge full), with a `REPORT.md` carrying signed Δ-vs-bf16
   deltas and a decision rule for which build to ship under which
   constraints.

**Why no FP8 build in this plan.** NVIDIA already publishes
[`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8)
(per-tensor E4M3, FP8 KV cache, ~32.8 GB). Reproducing it on Spark would
consume hours without producing a meaningfully different artifact — use
NVIDIA's directly when FP8 is the target. AWQ-INT4 is the only quantized
build we generate; the FP8 column in the REPORT is *card-cited only*,
not measured.

**Why include NVFP4 in the eval but not generate it.** NVIDIA also
publishes
[`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4)
(NVFP4 experts + FP8 mamba/attention + BF16 encoders, ~20.9 GB). NVFP4
vLLM kernel support on SM121a (GB10) is unverified — the eval phase
resolves it empirically. If the kernels work, NVFP4 becomes the third
column in the head-to-head and the user can pick AWQ-INT4 vs NVFP4 with
measured numbers. If the kernels don't work, we record the verbatim
loader failure and the REPORT falls back to a 2-way table — that
information is itself useful (it tells future Spark users not to waste
time on NVFP4).

The procedure mirrors the Qwen3.5-MoE AWQ work, but the **base architecture
is fundamentally different**: Nemotron-Nano-Omni is a `NemotronH` Mamba2 ⨉
Transformer-attention ⨉ MoE hybrid wrapped in a multimodal class with vision
(CRADIO v2-H) and audio (Parakeet) encoders. The hybrid layer pattern, the
Mamba2 SSM blocks, and the relu² MoE-expert MLPs all need targeted recipe
adjustments — but the Spark-side scaffolding (cgroup memory cap, Docker vLLM
serve, lm-eval driver) is reusable as-is.

---

## References

Read these first when resuming this plan in a new session — the live context
window starts empty and the recipe choices below are derived from these
files.

### Generic framework (reusable across runs)

- `CLAUDE.md` — project policy: env-var-configured recipes, atomic save,
  `device_map="cpu"`, RSS guard, KV-cache parity at eval time, etc.
- `docs/repo-layout.md` — what goes in `tools/` vs `docs/` vs `runs/<slug>/`.
- `docs/adding-a-run.md` — step-by-step for starting a new run from
  `templates/run/`. Phase 0 below is essentially this checklist.
- `docs/schemes/awq-gemm.md` — generic AutoAWQ GEMM playbook (asymmetric
  per-group of 128, `[0,4,1,5,2,6,3,7]` pack order, fp16 scales rounded
  through fp16 in quant math, unfused MoE expert layout on disk).
- `tools/run_under_memcap.sh` — `systemd-run --scope` cgroup wrapper.
  Defaults `MemoryMax=112G`, `MemoryHigh=100G`, `MemorySwapMax=0`. Override
  via env (`MEMORY_MAX=64G ...`).
- `tools/serve_vllm_docker.sh` — wraps the known-good
  `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:20260415` image. Always
  passes `--trust-remote-code`; this is required for Nemotron's custom
  `modeling_nemotron_h.py` to load.
- `tools/run_eval_full.sh` — drives `lm-eval==0.4.11` against a running vLLM
  endpoint: GSM8K full chat-CoT, full MMLU (57 subtasks), ARC-Challenge
  full. Refuses to run if `/v1/models` doesn't report the expected
  `--served-model-name`.
- `requirements.txt` — `torch>=2.5`, `transformers>=4.46`, `safetensors>=0.4.5`,
  `compressed-tensors>=0.7.1`, `lm-eval==0.4.11`, etc.
- `HUGGINGFACE_PUBLISHING.md` — upload pipeline for the artifact dirs.

### Sibling run (the worked precedent)

- `runs/qwen3.6-35b-distill/PLAN.md` — historical execution plan for
  Qwen3.5-MoE 35B; many of the Spark-specific guardrails (cgroup wrapper,
  Docker vLLM, `device_map="cpu"`, RSS hard-stop) are reused verbatim.
- `runs/qwen3.6-35b-distill/REPORT.md` — three-way eval (bf16 MMLU 0.834 /
  GSM8K 0.945; AWQ −2.73 pp MMLU, −0.61 pp GSM8K strict, vision tower
  preserved). The two-way (bf16 vs AWQ) subset of this REPORT is the
  structural target for the Nemotron REPORT in Phase 5.
- `runs/qwen3.6-35b-distill/recipes/awq_gemm.py` — shard-streaming
  data-free RTN AWQ packer with per-expert unfusing. Direct ancestor of
  the Nemotron AWQ recipe.
- `runs/qwen3.6-35b-distill/recipes/inspect_modules.py` — CPU-only module-
  tree dump used to choose the `ignore` list. Will be adapted for Nemotron
  (different module class names + Mamba blocks + vision/audio).

### Source model (read these in Phase 1 before writing recipes)

- HF model card:
  <https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16>
- `config.json` (full snapshot captured during planning):
  - Outer class: `NemotronH_Nano_Omni_Reasoning_V3` (multimodal wrapper,
    `auto_map.AutoModelForCausalLM = "modeling.NemotronH_Nano_Omni_Reasoning_V3"`).
  - Inner LM class: `NemotronHForCausalLM` under `llm_config`
    (`auto_map.AutoModelForCausalLM = "modeling_nemotron_h.NemotronHForCausalLM"`,
    `model_type: "nemotron_h"`).
  - 52 hidden layers, `hybrid_override_pattern`
    `"MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"` →
    23 Mamba2 layers (`M`) + 23 MoE-MLP layers (`E`) + 6 attention layers (`*`).
  - `hidden_size=2688`; `intermediate_size=1856` (dense MLP);
    `moe_intermediate_size=1856` per routed expert;
    `moe_shared_expert_intermediate_size=3712`;
    `n_routed_experts=128`, `num_experts_per_tok=6`, `n_shared_experts=1`,
    `topk_group=1`, `n_group=1`/`n_groups=8`.
  - Attention (GQA): `num_attention_heads=32`, `num_key_value_heads=2`,
    `head_dim=128` (so Q is `[2688 → 4096]`, K/V are `[2688 → 256]`),
    `rope_theta=10000`.
  - Mamba2: `mamba_num_heads=64`, `mamba_head_dim=64`, `ssm_state_size=128`,
    `conv_kernel=4`, `expand=2`, `chunk_size=128`,
    `time_step_min=0.001`, `time_step_max=0.1`, `mamba_hidden_act="silu"`.
  - MLP activation: `mlp_hidden_act: "relu2"` (squared ReLU). The MoE block
    likely uses `up + down` (no `gate_proj`) under relu²; **must be
    confirmed by Phase 1 module inspection** before writing recipes.
  - Vision (`vision_config`): RADIOModel CRADIO v2-H, `vit_hidden_size=1280`,
    `projector_hidden_size=20480`, `force_image_size=512`, `patch_size=16`,
    `separate_video_embedder=true`, `max_num_patches=13312`.
  - Audio (`sound_config`): Parakeet, `hidden_size=1024`, `num_attention_heads=8`,
    `num_hidden_layers=24`, `intermediate_size=4096`,
    `subsampling_factor=8`, `num_mel_bins=128`, `sampling_rate=16000`.
  - `vocab_size=131072`, `tie_word_embeddings=false`.
  - `torch_dtype: bfloat16`. 17 safetensors shards, ~66 GB on disk.
- HF file listing (for reference): 17 `model-NNNNN-of-00017.safetensors`,
  plus `model.safetensors.index.json`, `config.json`, `configuration.py`,
  `configuration_nemotron_h.py`, `configuration_radio.py`, `modeling.py`,
  `modeling_nemotron_h.py`, `audio_model.py`, `processing.py`,
  `processing_utils.py`, `image_processing.py`, `video_processing.py`,
  `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`,
  `generation_config.json`, `preprocessor_config.json`.
- vLLM serving notes from the card:
  - vLLM ≥ 0.20.0 required.
  - `--trust-remote-code` mandatory.
  - `--reasoning-parser nemotron_v3` for chat-CoT.
  - Optional `--tool-call-parser qwen3_coder`, `--enable-auto-tool-choice`.
  - Multimodal serve flags: `--media-io-kwargs '{"video":{"fps":2,"num_frames":256}}'`,
    `--video-pruning-rate 0.5`, `--allowed-local-media-path /`.
- Existing nvidia-published quantized variants:
  - `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8` (per-tensor E4M3
    weights except router/lm_head, fp8 KV cache, ~32.8 GB). **Generation
    out of scope** (NVIDIA already publishes it; we don't re-derive). Card
    claims "<1 pp loss vs BF16 across 9 multimodal benchmarks" — REPORT
    cites this verbatim rather than measuring.
  - `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` (NVFP4 experts,
    FP8 mamba/attention, BF16 encoders, ~20.9 GB). **Generation out of
    scope, eval IN scope** (Phase 5). Useful both as a measured comparison
    point in the REPORT and as a policy reference for which modules NVIDIA
    themselves keep dense — our AWQ recipe should be at least that
    conservative: keep mamba + attention + encoders dense, quantize only
    the bulk-of-params routed experts.

---

## Build & run

- **Containerized:** mixed.
  - **Quantization recipes** are plain Python invoked from the project venv
    (`/home/sergio/git/dgx-spark-quantization/.venv/`) — no container.
    They MUST be wrapped with `tools/run_under_memcap.sh` so any allocator
    runaway is bounded by the cgroup `MemoryMax=112G` rather than by the
    kernel OOM-killing the box.
  - **vLLM serving** runs in the known-good
    `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:20260415` Docker image
    via `tools/serve_vllm_docker.sh`. Bare-metal vLLM is not assumed to
    work on aarch64 / SM121a.
  - **Eval** is `lm-eval==0.4.11` from the project venv driven by
    `tools/run_eval_full.sh`, hitting the Docker vLLM endpoint over HTTP.
- **Python env:**
  ```bash
  cd /home/sergio/git/dgx-spark-quantization
  source .venv/bin/activate
  ```
- **Quantize command (AWQ, shard-streaming so SRC_DIR is local):**
  ```bash
  export SRC_DIR="$PWD/hf-cache/hub/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/snapshots/<sha>"
  export DST_DIR="$PWD/artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4"
  tools/run_under_memcap.sh python runs/nemotron-3-nano-omni-30b-a3b/recipes/awq_gemm.py
  ```
- **Recipe self-test:** the AWQ recipe runs its `roundtrip_test()`
  automatically at startup before touching any source shards. Failures
  abort before any output is written.
- **Serve a quantized artifact:**
  ```bash
  tools/serve_vllm_docker.sh "$PWD/artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4" \
    --kv-cache-dtype fp8_e4m3 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --reasoning-parser nemotron_v3 \
    --served-model-name nemotron-omni-awq
  ```
- **Eval against a running endpoint:**
  ```bash
  tools/run_eval_full.sh nemotron-omni-awq \
    "$PWD/artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4" \
    "$PWD/runs/nemotron-3-nano-omni-30b-a3b/results/awq_full"
  ```
- **No test suite, no CI** in this repo. Verification at the recipe level is
  the in-file `--selftest` (synthetic tensors); verification at the artifact
  level is the live `vllm serve` smoke + the lm-eval battery.

---

## Execution model — orchestrator + transient agents

Per the orchestrator workflow in `~/.claude/CLAUDE.md` ("Executing a
plan"), this plan is driven by an orchestrator (the top-level Claude
Code session) that spawns short-lived implementation + verification
agents per phase. The orchestrator does **not** run quantization,
serve vLLM, or write recipes itself — those happen inside transient
agents whose tool output and intermediate diffs are discarded the
moment they return their report. The orchestrator's own context only
ever holds:

- this plan document (the durable artifact);
- per-phase agent reports (terse summaries the orchestrator writes
  back into the plan as task-checkbox updates);
- user decisions on verification findings.

**Per-phase loop** (already specified in `~/.claude/CLAUDE.md`, repeated
here for reference, not redefinition):

1. Orchestrator reads this plan and re-reads every path under
   **References** before touching anything.
2. Orchestrator spawns an implementation agent (cold context). Prompt
   contains: this plan's path, the phase heading, the phase's full task
   list, the **References** list verbatim, the relevant **Build & run**
   commands, the verification criteria the agent must satisfy, and an
   instruction to report what it changed / ran / deviated.
3. Orchestrator records the report and ticks completed tasks `[x]` in
   this document.
4. Orchestrator spawns a verification agent (cold context, separate from
   the impl agent). Prompt contains: plan path, phase heading,
   files/areas the impl agent touched, the verification criteria. The
   agent **runs the documented commands and inspects the code** rather
   than trusting the impl agent's summary, then returns a severity-
   tagged finding list.
5. Orchestrator surfaces findings to the user with proposed fixes.
6. Orchestrator spawns a fix agent (only on user-approved findings),
   then re-runs the verification agent on the fixed areas.
7. On user sign-off, orchestrator flips this phase's `Status:` to
   `completed` and moves to the next phase **in the same session** — no
   `/compact` needed because the heavy work was in the transient agents.

**Sizing.** Every phase below is intentionally scoped so a single impl
agent and a single verification agent can complete it without context
bloat. If a phase grows past that during execution, the orchestrator
splits it (e.g. introduces a Phase 2a / 2b) rather than accepting bloat.

**File-touch scope per agent** (orchestrator enforces by listing
allowed paths in each agent prompt):

- Phase 0: `runs/nemotron-3-nano-omni-30b-a3b/**`, top-level `README.md`.
- Phase 1: `runs/nemotron-3-nano-omni-30b-a3b/**`, `hf-cache/**` (downloads).
- Phase 2: `runs/nemotron-3-nano-omni-30b-a3b/**`,
  `artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4/**`.
- Phase 2b: `runs/nemotron-3-nano-omni-30b-a3b/**`,
  `artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4/**`,
  `requirements.txt`, `tools/serve_vllm_docker.sh`.
- Phase 3: `runs/nemotron-3-nano-omni-30b-a3b/results/bf16_full/**`.
- Phase 4: `runs/nemotron-3-nano-omni-30b-a3b/results/awq_full/**`.
- Phase 5: `runs/nemotron-3-nano-omni-30b-a3b/results/nvfp4_full/**`,
  `hf-cache/**` (NVFP4 download).
- Phase 6: `runs/nemotron-3-nano-omni-30b-a3b/REPORT.md`,
  `artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4/README.md`,
  `docs/schemes/awq-gemm.md`, top-level `README.md`.

---

## Phase 0 — Run skeleton & slug

**Status:** completed
**Kind:** mixed

Lay down the per-run directory tree so subsequent phases write into well-
known paths. No model download yet.

### Tasks

- [x] Choose slug `nemotron-3-nano-omni-30b-a3b` (lowercase, dotted-decimal
      for the size tag, matches the convention in
      `runs/qwen3.6-35b-distill/`).
- [x] `cp -r templates/run runs/nemotron-3-nano-omni-30b-a3b/` to lay down
      the skeleton (`README.md`, `REPORT.md`, `recipes/`, `results/`).
- [x] Fill in `runs/nemotron-3-nano-omni-30b-a3b/README.md` with the model
      card pointer, the architecture summary from the **References**
      section above (52 layers / 23 M + 23 E + 6 attn / 128 routed experts
      top-6 / GQA 32-Q-2-KV / Mamba2 SSM / RADIO vision + Parakeet audio),
      and a status line dated 2026-04-29 (today; the plan originally said
      2026-04-28).
- [x] Add a row to the top-level `README.md`'s Runs index pointing at the
      new run.
- [x] Artifact directory name fixed to
      `Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4` (matches the
      proposed HF repo name; folder == repo string for ease).
- [x] In the same README, link to the upstream-published variants and
      explicitly note that FP8 / NVFP4 are out of scope because NVIDIA
      already publishes them.

**Impl agent deviation:** the bash example in the run README uses `SHA` (uppercase
placeholder) instead of `<sha>` so the no-`<...>`-placeholders verification grep
stays clean. Phase 1 replaces `SHA` with the real snapshot hash.

### Verification (all must pass to mark the phase done)

- [x] `ls runs/nemotron-3-nano-omni-30b-a3b/` shows
      `README.md REPORT.md recipes/ results/`.
- [x] `runs/nemotron-3-nano-omni-30b-a3b/README.md` does **not** contain
      any `<...>` placeholder text from the template.
- [x] Top-level `README.md` Runs index lists this slug with a one-line
      summary.
- [x] `git status` shows the new tree as untracked; nothing else has moved.

---

## Phase 1 — Module-tree inspection & scheme decisions

**Status:** completed
**Kind:** logic

Before writing the AWQ recipe, dump the actual `nn.Module` tree of the
bf16 source so the `modules_to_not_convert` list is derived from real
names — not guesses against the config.

### Tasks

- [x] Pre-cache the source weights into `$PWD/hf-cache` (do not let the
      recipes pull them mid-run; one network failure mid-quantize wastes
      hours). **Used `hf download` (the deprecated `huggingface-cli` is
      gone in `huggingface_hub>=1.0`); `--cache-dir` puts snapshots at
      `hf-cache/models--<org>--<name>/snapshots/<sha>/` (no `hub/`
      prefix).** With `HF_HUB_ENABLE_HF_TRANSFER=1` both downloads
      finished in ~2 minutes total.
      Captured `SRC_DIR =
      hf-cache/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/snapshots/d15962741057ae3a07147df504060e9f0838224e/`
      (62 GiB, 17 shards). Recorded in run README.
- [x] Also pre-cache the NVFP4 build. Captured `NVFP4_DIR =
      hf-cache/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4/snapshots/889396e9cebaefdb69a469afc7bd111660f78eff/`
      (21 GiB, 3 shards). Recorded in run README.
- [x] Adapt `runs/qwen3.6-35b-distill/recipes/inspect_modules.py` →
      `runs/nemotron-3-nano-omni-30b-a3b/recipes/inspect_modules.py`.
      Used `MODEL_ID` env (not `BF16_MODEL`), expanded `interesting`
      keywords, added per-class histogram + layer-container probe + MoE
      expert-layout probe + vision/audio dump. Required
      `attn_implementation="eager"` (FA2 not installed; inspection is
      module-walk only, no forward).
- [x] Ran it under cgroup guard. `module_inspection.txt` is **3,451
      lines**.
- [x] Documented findings in run README (`## Module-tree inspection
      (Phase 1 findings)`). **Key results that drive Phase 2:**
  - **Layer container:** `language_model.backbone.layers` (NOT
    `language_model.model.layers`). 52 `NemotronHBlock` entries.
  - **MoE expert layout: UNFUSED, ungated.** Each routed expert is
    `NemotronHMLP` with only `up_proj (1856, 2688)` + `down_proj
    (2688, 1856)` — there is **no `gate_proj`** (relu² is ungated).
    Phase 2's recipe **does not need** the per-expert unfusing or
    `gate_up_proj` half-split that the qwen3 recipe carries; quantize
    2-D weights directly.
  - **Router gate:** `mixer.gate.weight` (NOT `mlp.gate.weight`).
  - **LM attention naming:** `mixer.{q,k,v,o}_proj.weight` (NOT
    `self_attn.*`).
  - **Audio→LM bridge:** `sound_projection.{linear1,linear2}` (caught
    by added `projection` substring in skip list).
  - **5,888 quantizable weights** = 23 MoE layers × 128 experts × 2
    projections (out of 6,355 total `nn.Linear` modules).
- [x] Wrote AWQ skip policy to
      `runs/nemotron-3-nano-omni-30b-a3b/recipes/_classify.py` —
      `should_quantize(name, shape)` separated from the recipe so Phase
      2's `awq_gemm.py` and the Phase 1 test both import it. Final skip
      substrings (case-insensitive): `lm_head`, `embed_tokens`,
      `embedding`, `embed`, `norm`, `layernorm`, `rmsnorm`, `vision`,
      `radio`, `vision_model`, `vision_tower`, `image_proj`, `video`,
      `sound`, `audio`, `parakeet`, `audio_encoder`, `projector`,
      `projection`, `mamba`, `ssm`, `in_proj`, `out_proj`, `dt_proj`,
      `conv1d`, `self_attn`, `q_proj`, `k_proj`, `v_proj`, `o_proj`,
      `shared_expert`. `SKIP_ENDSWITH = ("mixer.gate.weight",
      "mlp.gate.weight", "block_sparse_moe.gate.weight")`.
      `LAYER0_RE = .*\.layers\.0\..*`.

**Tooling additions (impl agent deviations):** the venv now has `pytest
9.0.3`, `hf_transfer 0.1.9`, `timm 1.0.26`, `open_clip_torch 3.3.0`,
`librosa 0.11.0`. The last three are mandatory for `trust_remote_code`
to import the RADIO ViT and Parakeet feature extractor; Phase 2 / 3
will need them too.

### Verification (all must pass to mark the phase done)

- [x] `runs/nemotron-3-nano-omni-30b-a3b/results/module_inspection.txt`
      exists, **3,451 lines**, mentions `Mamba`, `Linear`, `RADIO`,
      `Parakeet` (1,646 keyword hits combined).
- [x] Run README documents (a) layer-container path
      (`language_model.backbone.layers`), (b) per-layer-type Linear
      inventory (16-row table), (c) fused/unfused expert layout
      (UNFUSED, ungated — `mixer.experts.<j>.{up_proj,down_proj}` only),
      (d) the AWQ skip policy, each with concrete dotted names from
      `module_inspection.txt`.
- [x] **Automated test passes:**
      `pytest runs/nemotron-3-nano-omni-30b-a3b/recipes/test_module_classify.py -v`
      → **60 PASSED** (32 synthetic + 25 real-name-from-inspection + 3
      invariants). Verification agent independently spot-checked 19
      additional real names against `should_quantize` — all 19 verdicts
      match the documented policy. The classifier helper lives in
      `recipes/_classify.py` (separated from `awq_gemm.py` so Phase 2
      and the test can both import without circular dependency); Phase
      2's recipe will `from _classify import should_quantize`.

---

## Phase 2 — AWQ-INT4 GEMM recipe + smoke (AutoAWQ packed format) — BLOCKED

**Status:** blocked-superseded-by-2b
**Kind:** logic

> **Outcome (2026-04-29):** the AutoAWQ-packed artifact builds correctly
> (5,888 quantized + 1,461 copied tensors, 21.55 GiB) but **does not serve
> in vLLM on Spark** — none of the four kernel paths (`awq_marlin`, `awq`,
> `awq + --dtype float16`, `moe_wna16`) handle ungated relu² MoE. Root
> cause is at the kernel level (vLLM's `FusedMoE.make_expert_params_mapping`
> hard-wires `up_proj → w13` slot expecting SwiGLU 2× intermediate; our
> ungated relu² weights produce silent garbage on `awq_marlin`, hard
> assertions on `moe_wna16`). BF16 baseline through the same image is
> CLEAN (391 + Paris correct), so the image / chat template / reasoning
> parser are all fine. Recipe and artifact are kept as evidence; eval is
> deferred. **Phase 2b takes over** with a different on-disk format
> (compressed-tensors W4A16 via `llm-compressor`) that *does* route
> through a working ungated-MoE loader path — same path NVIDIA's NVFP4
> uses, and the path stelterlab successfully shipped for the LM-only
> base model.

Adapt `runs/qwen3.6-35b-distill/recipes/awq_gemm.py` →
`runs/nemotron-3-nano-omni-30b-a3b/recipes/awq_gemm.py`. Same shard-
streaming RTN packer, same `[0,4,1,5,2,6,3,7]` pack order, but the skip
policy is Nemotron-specific (the AWQ build keeps Mamba, attention,
vision, audio, layer-0, shared-expert, router, lm_head all dense — the
bulk of compression comes from quantizing only the 128 routed expert
MLPs across the 23 MoE layers).

### Tasks

- [ ] Copy the Qwen3 AWQ recipe as the starting point. Replace the
      `QUANTIZATION_CONFIG` block's `modules_to_not_convert` list with the
      Nemotron AWQ skip list from Phase 1.
- [ ] **`should_quantize` rewrite:** the Qwen3 version blanket-skipped
      `linear_attn`, `self_attn`, `shared_expert`, `mlp.gate`, `layers.0.`,
      `mtp`. For Nemotron, blanket-skip `mamba`, `ssm`, `in_proj`,
      `dt_proj`, `out_proj`, `conv1d`, `self_attn`, `shared_expert`,
      `mlp.gate` (router-only — must distinguish from `mlp.gate_proj` if
      that exists), `model.layers.0.`, `vision_model` / `radio` /
      `vision_tower`, `sound_model` / `audio_encoder` / `parakeet`,
      `*projector*`, `lm_head`, `embed_tokens`, `*norm*`. Add unit
      coverage in `test_module_classify.py`.
- [ ] **MoE expert handling:** if Phase 1 confirms fused 3-D, mirror
      Qwen3's split:
  - `experts.gate_up_proj` → for each expert, slice the first half along
    out_features as `gate_proj`, second half as `up_proj` (only if the
    architecture actually has gating; with relu² it may be a single
    `up_proj` of shape `[intermediate, hidden]` — Phase 1 settles this).
  - `experts.down_proj` → 256 per-expert `down_proj`. (Wait — Nemotron
    has 128 routed experts, not 256. Adapt the `NUM_EXPERTS = 128`
    constant.)
  - If unfused: per-expert tensors are already in source shards; just
    quantize them as 2-D with the standard `[out, in]` path.
- [ ] **`NUM_EXPERTS = 128`** (was 256 for Qwen3).
- [ ] **Pass-through file copy list:** add `preprocessor_config.json`,
      `processing.py`, `processing_utils.py`, `image_processing.py`,
      `video_processing.py`, `audio_model.py`, `configuration.py`,
      `configuration_nemotron_h.py`, `configuration_radio.py`,
      `modeling.py`, `modeling_nemotron_h.py`, `__init__.py` — the
      multimodal serving path needs the custom Python files in the
      artifact dir because vLLM `--trust-remote-code` will import them.
- [ ] **`config.json` rewrite:** start from the source `config.json`,
      add `quantization_config` (the AWQ block), keep `architectures: ["NemotronH_Nano_Omni_Reasoning_V3"]`
      (we want multimodal serving to work for the AWQ build).
- [ ] Run the recipe's `roundtrip_test()` first by importing and calling
      it directly:
      ```bash
      python -c 'import sys; sys.path.insert(0, "runs/nemotron-3-nano-omni-30b-a3b/recipes"); import awq_gemm; awq_gemm.roundtrip_test()'
      ```
      Must pass before launching the full run.
- [ ] Run the full quantization under cgroup guard (peak host RAM stays
      around one-shard-worth, ~5 GiB — `MEMORY_MAX=64G` is plenty;
      defaulting to 112G is fine):
      ```bash
      tools/run_under_memcap.sh \
        bash -lc '
          source .venv/bin/activate
          export SRC_DIR="$PWD/hf-cache/hub/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/snapshots/<sha>"
          export DST_DIR="$PWD/artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4"
          python runs/nemotron-3-nano-omni-30b-a3b/recipes/awq_gemm.py \
            2>&1 | tee runs/nemotron-3-nano-omni-30b-a3b/results/awq_quant.log
        '
      ```
      Expected disk size: ~20–25 GiB (NVFP4 reference is 20.9 GB; AWQ-INT4
      with more dense components than NVFP4 should land 22–25 GiB).
- [ ] Smoke-validate by serving and hitting the endpoint, with
      multimodal flags off but trust-remote-code on:
      ```bash
      tools/serve_vllm_docker.sh \
        "$PWD/artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4" \
        --kv-cache-dtype fp8_e4m3 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.85 \
        --reasoning-parser nemotron_v3 \
        --served-model-name nemotron-omni-awq
      ```
      `vllm serve` should auto-detect `quant_method: awq` and pick the
      AWQ kernel path. Then:
      ```bash
      curl -s http://localhost:8000/v1/chat/completions \
        -H 'content-type: application/json' \
        -d '{"model":"nemotron-omni-awq",
             "messages":[{"role":"user","content":"Solve: 17 * 23 = ?  Briefly."}],
             "max_tokens":128, "temperature":0}' | jq .
      ```
      Confirm the response contains a reasoning block (whatever
      `nemotron_v3` emits) and the numeric answer 391.
- [ ] Optional: smoke-test multimodal by sending one image (data URL).
      Not required for the eval phase but useful to confirm the vision
      tower survived dense.

### Verification (all must pass to mark the phase done)

- [x] **Roundtrip test passes:** `roundtrip OK (max-err=0.02178)` at
      `group_size=64`.
- [x] **Classification test passes:** `pytest ... -v` → 63 PASSED
      (60 prior + 3 new mlp1 cases).
- [x] Quantization log shows **`n_quantized=5888, n_copied=1461`** —
      exactly 23 MoE × 128 experts × 2 projections (up + down). All
      Mamba2/attention/vision/audio/projectors/embeds/layer 0 dense.
- [x] Artifact dir has 6 shards / 21.55 GiB / 19,125 keys /
      `quantization_config.quant_method == "awq"`, all 13 custom `.py`
      and 6 aux files copied.
- [ ] **vLLM smoke FAILED — root cause: kernel does not support relu²
      MoE.** Diagnostic complete; the failure is at the vLLM kernel
      level, not in our recipe. Empirical sequence (all in
      `results/awq_serve_v{2,3,4}.log`):
      - **`awq_marlin` (auto-selected)** → loads but produces degenerate
        looping. Marlin layout assumes SwiGLU semantics; with our
        ungated relu² weights it interprets garbage.
      - **`--quantization awq` (force vanilla AutoAWQ)** → vLLM rejects:
        `torch.bfloat16 is not supported for quantization method awq`.
      - **`--quantization awq --dtype float16`** → loads but crashes at
        warmup with a dtype mismatch in the RADIO video embedder
        (vision weights are bf16; vLLM upcast the engine to fp16 but
        not the dense vision Linears).
      - **`--quantization moe_wna16`** → hard-asserts at warmup with
        `AssertionError: Only SiLU activation is supported, not
        MoEActivation.RELU2_NO_MUL.` (vllm/.../moe_wna16.py:375).
      - **BF16 baseline through the same image** → CLEAN (391 + Paris
        both correct in `results/bf16_smoke.txt`). Confirms image /
        parser / template / model class are all fine; only the
        quantized-MoE kernel path is broken.
      The plan's Risks section anticipated this exact case ("vLLM 0.20.0
      isn't in the pinned Docker image" — current image runs
      `0.19.1rc1.dev322`). Resolution requires moving off this image
      tag; the recipe and artifact are not at fault.

      **Resolution candidates** (orchestrator surfaced to user before
      proceeding):
      1. Upgrade the Docker image to `:latest` (different digest from
         `:20260415`, likely includes the moe_wna16 relu² support that
         the model card's "vLLM ≥ 0.20.0" requirement implies).
      2. Skip our AWQ build entirely; pivot to evaluating NVIDIA's
         published NVFP4 (Phase 5) and using the BF16 baseline (Phase 3)
         as the only two columns in the REPORT. AWQ recipe is shipped
         as code; serving deferred until a relu²-aware vLLM kernel
         lands.
      3. Eval our AWQ via transformers/AutoAWQ direct loader (no vLLM)
         — much slower but functional; outside this repo's tooling
         scope.
- [x] **Deviation logged:** `--gpu-memory-utilization` reduced from the
      plan's 0.85 to **0.55** because Spark GB10 reports
      "Memory-Usage: Not Supported" to nvidia-smi → vLLM's free-memory
      check rejects 0.85. Phase 3/4/5 must use ≤ 0.55 for the AWQ
      artifact and ≤ 0.45 for the bf16 baseline (62 GiB weights).

---

## Phase 2b — AWQ-INT4 via llm-compressor (compressed-tensors W4A16)

**Status:** pending
**Kind:** logic

Re-engineer the AWQ build to produce a `compressed-tensors`
`pack-quantized` W4A16 artifact via `llm-compressor==0.9.0.1`. This is
the on-disk format vLLM actually has a working ungated-MoE loader for —
the same loader path NVIDIA's NVFP4 build uses, and the path
[stelterlab](https://huggingface.co/stelterlab/NVIDIA-Nemotron-3-Nano-30B-A3B-AWQ)
successfully shipped on the **LM-only** base (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`).
That base shares **every NemotronH config knob** with the LM portion of
our Omni-Reasoning target (`mlp_hidden_act=relu2`, 128 routed experts,
`moe_intermediate_size=1856`, hybrid pattern, hidden_size=2688) — only
the multimodal wrapper + RADIO + Parakeet + `mlp1` projector are added
on top, all of which we already know how to ignore.

### References (read these first)

- stelterlab AWQ artifact: <https://huggingface.co/stelterlab/NVIDIA-Nemotron-3-Nano-30B-A3B-AWQ>
- stelterlab recipe.yaml (verbatim source): <https://huggingface.co/stelterlab/NVIDIA-Nemotron-3-Nano-30B-A3B-AWQ/raw/main/recipe.yaml>
- stelterlab artifact `config.json` (target `quantization_config` shape): <https://huggingface.co/stelterlab/NVIDIA-Nemotron-3-Nano-30B-A3B-AWQ/raw/main/config.json>
- stelterlab `model.safetensors.index.json` (target on-disk key shape — `experts.<j>.up_proj.weight_packed` / `weight_scale` / `weight_shape`, no `gate_proj`): <https://huggingface.co/stelterlab/NVIDIA-Nemotron-3-Nano-30B-A3B-AWQ/raw/main/model.safetensors.index.json>
- llm-compressor docs: <https://github.com/vllm-project/llm-compressor> (AWQModifier reference)
- Phase 1's classifier (`runs/nemotron-3-nano-omni-30b-a3b/recipes/_classify.py`) — reuse the substring/endswith policy for the ignore list expansion.

### Tasks

- [ ] **Tooling:** install `llm-compressor==0.9.0.1` into the project
      venv (`.venv`). Confirm it imports cleanly:
      `python -c 'import llmcompressor; print(llmcompressor.__version__)'`.
      Add the pin to `requirements.txt` (do not remove `autoawq` — keep
      Phase 2a's recipe runnable as evidence).
- [ ] **New recipe:** write
      `runs/nemotron-3-nano-omni-30b-a3b/recipes/awq_compressed_tensors.py`.
      Driver pattern (compressed-tensors-style, NOT AutoAWQ-style):
      - Read `SRC_DIR` (BF16 source snapshot) + `DST_DIR` (artifact out).
      - Atomic save: refuse if `DST_DIR` is non-empty; write into
        `<DST_DIR>.tmp.<pid>/` and rename. Same policy as Phase 2a.
      - `from transformers import AutoModelForCausalLM, AutoTokenizer`
        with `trust_remote_code=True`, `device_map="cpu"`,
        `torch_dtype=torch.bfloat16`.
      - Build the AWQModifier recipe in Python (mirroring stelterlab's
        YAML but with the `language_model.` prefix added to every
        smooth-layer path). See **Recipe shape** below.
      - Pick a calibration dataset:
        `open-platypus` (general reasoning instruction corpus, ~25k
        samples) — sample 256 prompts, max_seq_len=2048. Reasoning
        prompts are in-domain for this reasoning model.
      - `from llmcompressor import oneshot; oneshot(model=model,
         dataset=ds, recipe=recipe, max_seq_length=2048,
         num_calibration_samples=256, output_dir="<DST_DIR>.tmp.<pid>")`.
      - After save, copy through the multimodal `.py` files and aux
        files exactly like Phase 2a's recipe (13 custom .py + 6 aux:
        tokenizer, chat_template, generation_config, preprocessor_config,
        special_tokens_map, tokenizer_config). Phase 2a's
        `awq_gemm.py` already has the pass-through logic — copy it.
      - Rename `<DST_DIR>.tmp.<pid>/` → `<DST_DIR>/` atomically.
- [ ] **Recipe shape (build in Python):**
      ```python
      from llmcompressor.modifiers.awq import AWQModifier
      recipe = AWQModifier(
          targets=["Linear"],
          ignore=[
              "lm_head",
              "re:.*language_model\\.embed_tokens$",
              "re:.*language_model\\.backbone\\.norm_f$",
              "re:.*mixer\\.gate$",          # MoE router
              "re:.*\\.layers\\.0\\..*",     # layer-0 dense (parity with Phase 2a)
              "re:^vision.*", "re:.*radio.*", "re:.*vision_model.*",
              "re:^audio.*", "re:.*sound.*", "re:.*parakeet.*", "re:.*audio_encoder.*",
              "re:.*mlp1.*",                 # vision projector
              "re:.*projector.*", "re:.*projection.*",
              "re:.*shared_expert.*",
              # mamba inner Linears (skip everything except expert MLPs)
              "re:.*mixer\\.in_proj$", "re:.*mixer\\.out_proj$",
              "re:.*mixer\\.dt_proj$",
              # attention (kept dense — only 6 attention layers, marginal compression)
              "re:.*mixer\\.q_proj$", "re:.*mixer\\.k_proj$",
              "re:.*mixer\\.v_proj$", "re:.*mixer\\.o_proj$",
          ],
          scheme="W4A16",            # 4-bit weights, 16-bit activations
          group_size=64,             # 1856 = 64 * 29; 2688 = 64 * 42
          symmetric=True,
          observer="minmax",
          duo_scaling=True,
          n_grid=20,
          mappings=[
              # Smooth-layer mappings for the 6 attention layers
              # (paths use the Omni `language_model.` prefix)
              {"smooth_layer": f"re:.*language_model\\.backbone\\.layers\\.{i}\\.norm$",
               "balance_layers": [
                   f"re:.*language_model\\.backbone\\.layers\\.{i}\\.mixer\\.q_proj$",
                   f"re:.*language_model\\.backbone\\.layers\\.{i}\\.mixer\\.k_proj$",
                   f"re:.*language_model\\.backbone\\.layers\\.{i}\\.mixer\\.v_proj$",
               ]}
              for i in [5, 12, 19, 26, 33, 42]
          ],
      )
      ```
      Note: the **6 attention layer indices** (5, 12, 19, 26, 33, 42)
      are the `*` positions in `hybrid_override_pattern`. Confirm by
      grepping `module_inspection.txt` for `mixer.q_proj` and listing
      the layer indices that match. If the actual positions differ from
      the list above, the smooth-layer mappings will silently no-op (no
      regex match) and AWQ will fall back to per-layer min/max — still
      functional, just suboptimal. **The recipe must include a startup
      assertion that prints how many smooth-layer mappings actually
      matched a module — fail loud if the count is < 6.**
- [ ] **MoE expert ignore audit:** the recipe above does NOT explicitly
      ignore expert MLPs — that's intentional, since experts are the
      only modules we *want* quantized. Re-running the classifier from
      Phase 1 against the planned ignore patterns must yield exactly
      the **5,888 expected quantized tensors** (23 MoE layers × 128
      experts × 2 projections). Add a dry-run mode to the recipe
      (`--dry-run`) that walks the model, applies the ignore regexes,
      and prints `would_quantize=N, would_skip=M` BEFORE calibration.
      Abort if `would_quantize != 5888`.
- [ ] **Self-test:** add `--selftest` flag (synthetic-tensor roundtrip)
      that builds a tiny 2-layer model with a single mock MoE block,
      runs llm-compressor oneshot on it, asserts the output dir
      contains `weight_packed`/`weight_scale`/`weight_shape` keys for
      the experts and dense `weight` for the embeddings/router. Must
      pass before the full run touches any source shards.
- [ ] **Clear the prior artifact:** `rm -rf
      artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4/` (the
      Phase 2a AutoAWQ artifact will be overwritten — its diagnostic
      value is captured in the plan + the recipe code, not the bytes).
- [ ] **Run the full quantization** under cgroup guard, detached so
      bash timeouts don't kill it:
      ```bash
      tools/run_under_memcap.sh \
        bash -lc '
          source .venv/bin/activate
          export SRC_DIR="$PWD/hf-cache/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/snapshots/d15962741057ae3a07147df504060e9f0838224e"
          export DST_DIR="$PWD/artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4"
          nohup setsid python runs/nemotron-3-nano-omni-30b-a3b/recipes/awq_compressed_tensors.py \
            > runs/nemotron-3-nano-omni-30b-a3b/results/awq_ct_quant.log 2>&1 &
          disown
          echo "pid=$!"
        '
      ```
      Wall-clock estimate: 1–3 hours (calibration walks 256 samples
      through the bf16 model on CPU + then per-layer AWQ search).
      Watch RSS via `psutil` log lines in the recipe — should stay
      under ~80 GiB.
- [ ] **Smoke-validate by serving** on the newer image (`:20260428` —
      Phase 2a confirmed BF16 serves cleanly there):
      ```bash
      tools/serve_vllm_docker.sh \
        "$PWD/artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4" \
        --kv-cache-dtype fp8_e4m3 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.55 \
        --reasoning-parser nemotron_v3 \
        --served-model-name nemotron-omni-awq-ct
      ```
      Expected: `vllm serve` auto-detects `compressed-tensors` and
      loads via the working ungated-MoE path. Then the same 4 probes
      Phase 2a used (17×23=391; capital of France; capital of Spain;
      2+2). All four must produce the correct answer with non-degenerate
      generation (no looping). Save to
      `runs/nemotron-3-nano-omni-30b-a3b/results/awq_ct_smoke.txt`.
- [ ] **Update `tools/serve_vllm_docker.sh`** to default to
      `:20260428` (the newer image where BF16 serves cleanly). Note
      the change in a one-line comment at the top of the script. (Phase
      2a's `:20260415` reference in this plan stays — it's history.)

### Recipe shape — open questions to resolve in implementation

These are decisions the impl agent should make based on what stelterlab
actually does + what `llm-compressor`'s API supports:

- Whether to use `AWQModifier` directly or wrap in
  `oneshot(recipe=[AWQModifier(...)])`. stelterlab's YAML uses the
  `oneshot` shape — Python equivalent is `oneshot(recipe=AWQModifier(...))`
  or pass a `Recipe` object.
- Whether `model.embed_tokens` is at `language_model.embed_tokens` or
  `language_model.backbone.embed_tokens` — Phase 1's
  `module_inspection.txt` settles this; grep for `embed_tokens`.
- Whether the smooth-layer norm path is `.norm` or `.input_layernorm`
  — same source, grep for the actual attribute name on the
  attention-bearing layers.

### Verification (all must pass to mark the phase done)

- [ ] **Tooling check:** `python -c 'import llmcompressor; print(llmcompressor.__version__)'`
      prints `0.9.0.1`.
- [ ] **Self-test passes:** `python runs/.../awq_compressed_tensors.py --selftest`
      exits 0 and prints `selftest OK`.
- [ ] **Dry-run preflight:** the recipe's dry-run mode prints
      `would_quantize=5888, would_skip=<expected>` and the ignore-list
      coverage matches Phase 1's classifier exactly. If the count is
      off, the recipe aborts before calibration.
- [ ] **Smooth-layer match count:** recipe startup log contains
      `smooth_layer mappings matched: 6/6 attention layers`.
- [ ] **Quantization log shows successful completion:** final RSS,
      total wall-clock, `quantized=5888 (or matches dry-run preflight),
      copied=<N>, shards=<M>, total=<size> GiB`. No tracebacks.
- [ ] **Artifact dir structure:** has 5–7 shards, ~20–25 GiB,
      `model.safetensors.index.json` present.
- [ ] **`config.json` shape:** `quantization_config.format ==
      "pack-quantized"`, `quant_method == "compressed-tensors"`,
      `config_groups.group_0.weights.{num_bits=4, group_size=64,
      symmetric=true, strategy="group", observer="minmax"}`,
      `targets=["Linear"]`, `ignore` includes `lm_head`. The
      `architectures` field stays `["NemotronH_Nano_Omni_Reasoning_V3"]`.
- [ ] **Index keys match stelterlab's shape:** `model.safetensors.index.json`
      contains `language_model.backbone.layers.<i>.mixer.experts.<j>.up_proj.weight_packed`
      (and `.weight_scale`, `.weight_shape`) and `.down_proj.*` triplets
      for every (MoE-layer, expert) pair. Crucially: NO `gate_proj` keys
      anywhere in expert paths, NO fused `gate_up_proj` keys. Mamba
      `in_proj`/`out_proj` keys exist with regular `.weight` (NOT
      `.weight_packed`) — proves the Mamba-skip ignore worked.
- [ ] **Multimodal pass-through:** all 13 custom `.py` files + 6 aux
      files present in artifact dir (parity with Phase 2a artifact list,
      verified via `ls`).
- [ ] **vLLM smoke test passes — non-degenerate output on all 4 probes:**
      - `17 * 23 = ?` → response contains `391`.
      - `What is the capital of France?` → response contains `Paris`.
      - `What is the capital of Spain?` → response contains `Madrid`.
      - `What is 2+2?` → response contains `4`.
      All without looping or degenerate token streams. Save full
      transcript to `results/awq_ct_smoke.txt`.
- [ ] **Loader path confirmed:** `docker logs` from the smoke serve
      includes a line indicating `compressed-tensors` was selected
      (look for `quantization=compressed-tensors` in the resolved
      args, or `Using CompressedTensors...` in the loader output).
      Save the relevant log excerpt to `results/awq_ct_serve.log`.

---

## Phase 3 — BF16 baseline full eval

**Status:** pending
**Kind:** logic

The Δ-vs-bf16 deltas in the REPORT are only trustworthy when the bf16
baseline is measured under the **same** vLLM / KV-cache settings as the
AWQ run. Run this **before** the AWQ eval (i.e. between Phase 2 and
Phase 4) so any vLLM-loader or eval-harness friction surfaces on the
known-good bf16 weights instead of being mistaken for an AWQ defect.

### Tasks

- [ ] Serve the bf16 source (from the local `hf-cache` snapshot) under the
      Docker vLLM, pinning `--kv-cache-dtype fp8_e4m3` to match what the
      AWQ run will use. Memory will be tight (66 GB weights + KV + CUDA
      overhead vs 128 GB unified RAM), so use
      `--max-model-len 4096 --gpu-memory-utilization 0.7`:
      ```bash
      tools/serve_vllm_docker.sh \
        "$PWD/hf-cache/hub/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/snapshots/<sha>" \
        --kv-cache-dtype fp8_e4m3 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.7 \
        --reasoning-parser nemotron_v3 \
        --served-model-name nemotron-omni-bf16
      ```
- [ ] In another terminal, run the eval:
      ```bash
      tools/run_eval_full.sh nemotron-omni-bf16 \
        "$PWD/hf-cache/hub/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/snapshots/<sha>" \
        "$PWD/runs/nemotron-3-nano-omni-30b-a3b/results/bf16_full"
      ```
      Wall-clock estimate from Qwen3 precedent: 2.5–3.5 hours.
- [ ] Tear down vLLM (`docker rm -f` if backgrounded; otherwise Ctrl-C the
      foreground container).

### Verification (all must pass to mark the phase done)

- [ ] `runs/nemotron-3-nano-omni-30b-a3b/results/bf16_full/` contains
      `gsm8k/`, `mmlu/`, `arc_challenge/` subdirs, each with a
      `results_*.json` summary file. The top-level `run.log` shows
      `rc=0` for all three tasks.
- [ ] `gsm8k` `exact_match strict` ≥ 0.85 (sanity check — Nemotron-Nano-Omni
      is a strong reasoning model; if we see < 0.85 something is wrong with
      the chat template or the reasoning parser).
- [ ] `mmlu` overall `acc` ≥ 0.70 (sanity floor for a well-trained 30 B
      MoE; if it drops below, the eval harness is mis-configured).
- [ ] `arc_challenge` `acc` and `acc_norm` are within typical 0.4–0.7
      band.
- [ ] Compare the saved `samples_*.jsonl` log of the first 5 GSM8K samples
      with manual reading — confirm the `<think>` block is being
      recognized by `--reasoning-parser nemotron_v3` (i.e., the response
      content body is the *answer*, not the raw `<think>` tags).

---

## Phase 4 — AWQ full eval

**Status:** pending
**Kind:** logic

### Tasks

- [ ] Serve the AWQ artifact:
      ```bash
      tools/serve_vllm_docker.sh \
        "$PWD/artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4" \
        --kv-cache-dtype fp8_e4m3 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.85 \
        --reasoning-parser nemotron_v3 \
        --served-model-name nemotron-omni-awq
      ```
- [ ] Run the eval:
      ```bash
      tools/run_eval_full.sh nemotron-omni-awq \
        "$PWD/artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4" \
        "$PWD/runs/nemotron-3-nano-omni-30b-a3b/results/awq_full"
      ```
- [ ] Tear down vLLM.

### Verification (all must pass to mark the phase done)

- [ ] `results/awq_full/{gsm8k,mmlu,arc_challenge}/results_*.json` exist;
      `run.log` shows rc=0 for each.
- [ ] Sanity-bound deltas vs Phase 3: `|Δ MMLU| ≤ 6 pp`, `|Δ GSM8K
      strict| ≤ 4 pp`, `|Δ ARC-C acc| ≤ 5 pp`. AWQ's published precedent
      on Qwen3 was −2.73 pp MMLU; a Nemotron AWQ delta beyond 6 pp
      indicates something in the recipe is wrong (most likely the skip
      list is too aggressive about quantizing, or layer 0 wasn't
      preserved).

---

## Phase 5 — NVFP4 eval (NVIDIA's official artifact)

**Status:** pending
**Kind:** logic

Run the same eval battery against NVIDIA's published NVFP4 build to give
the REPORT a third comparison point. NVFP4 vLLM kernel support on SM121a
(GB10) is unverified — this phase resolves it empirically. Two acceptable
outcomes: (a) it loads and we get measured numbers; (b) it doesn't load
and we record the verbatim loader failure for posterity. Either way the
REPORT in Phase 6 has more information than it would without this phase.

### Tasks

- [ ] Confirm `NVFP4_DIR` from Phase 1's pre-cache is populated (~21 GB
      under `hf-cache/hub/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4/snapshots/<sha>/`).
      If missing, download:
      ```bash
      huggingface-cli download nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 \
        --cache-dir "$PWD/hf-cache"
      ```
- [ ] Attempt to serve under Docker vLLM, matching the KV-cache and
      `max-model-len` from Phase 3 / Phase 4 so the deltas are
      apples-to-apples:
      ```bash
      tools/serve_vllm_docker.sh \
        "$NVFP4_DIR" \
        --kv-cache-dtype fp8_e4m3 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.85 \
        --reasoning-parser nemotron_v3 \
        --served-model-name nemotron-omni-nvfp4
      ```
      Watch `docker logs -f` for kernel-not-implemented /
      unsupported-dtype / NVFP4-not-registered errors during model load.
- [ ] **Branch on outcome:**
  - **(a) Model loads.** Smoke-test with the same trivial chat-completion
    prompt used in Phase 2 (`17 * 23 = ?`). Confirm `391` in the
    response. Then run the full eval:
    ```bash
    tools/run_eval_full.sh nemotron-omni-nvfp4 \
      "$NVFP4_DIR" \
      "$PWD/runs/nemotron-3-nano-omni-30b-a3b/results/nvfp4_full"
    ```
  - **(b) Model fails to load.** Capture the full container error
    (start-to-crash) into
    `runs/nemotron-3-nano-omni-30b-a3b/results/nvfp4_loader_failure.txt`
    with at least 30 lines of context (the relevant Python traceback
    plus the surrounding vLLM init log lines). Write a short note in
    the run README documenting that NVFP4 isn't currently supported on
    Spark with the pinned image. **Do not try to fix vLLM** — that's
    out of scope; just record the gap.
- [ ] Tear down vLLM (`docker rm -f` if backgrounded; otherwise Ctrl-C
      the foreground container).

### Verification (all must pass to mark the phase done)

Two acceptable outcomes; the verification agent must establish which
one applies and confirm the corresponding evidence is in place.

**Outcome (a) — NVFP4 loads and evaluates:**
- [ ] `results/nvfp4_full/{gsm8k,mmlu,arc_challenge}/results_*.json`
      exist; `run.log` shows rc=0 for each.
- [ ] Smoke-test response contained `391`.
- [ ] Sanity-bound deltas vs Phase 3: `|Δ MMLU| ≤ 6 pp`,
      `|Δ GSM8K strict| ≤ 4 pp`, `|Δ ARC-C acc| ≤ 5 pp`. NVIDIA's
      card claims "<1 pp loss vs BF16 across 9 multimodal benchmarks";
      anything beyond ~3 pp on a single text task suggests a serving-
      config issue (KV-cache dtype, chat template, reasoning-parser
      mismatch), not an NVFP4-quality issue — diagnose before ticking
      the phase complete.

**Outcome (b) — NVFP4 doesn't load on Spark:**
- [ ] `results/nvfp4_loader_failure.txt` exists with ≥ 30 lines of
      context including the verbatim Python traceback or vLLM error
      message.
- [ ] Run README has a one-paragraph note documenting the gap and the
      pinned vLLM image tag at the time of the test.
- [ ] No `results/nvfp4_full/` subdirectory exists (clean state — the
      REPORT in Phase 6 will know to fall back to a 2-way table).

---

## Phase 6 — REPORT, model card, scheme-doc updates

**Status:** pending
**Kind:** mixed

Write the per-run REPORT, the AWQ artifact README (the model card that
ships to Hugging Face inside the artifact dir), and update
`docs/schemes/awq-gemm.md` with any Nemotron-specific gotchas worth
lifting into the generic playbook.

The REPORT shape branches on Phase 5's outcome — `(a)` 3-way table when
NVFP4 evaluated successfully, `(b)` 2-way table when NVFP4 didn't load on
Spark.

### Tasks

- [ ] Fill `runs/nemotron-3-nano-omni-30b-a3b/REPORT.md` using
      `templates/run/REPORT.md` as the skeleton and
      `runs/qwen3.6-35b-distill/REPORT.md` as the structural reference.
      Sections required:
      - TL;DR table:
        - **3-way (Phase 5 outcome a):** bf16 / AWQ-INT4 / NVFP4 ×
          bits / disk / GSM8K / MMLU / ARC-C / Δ vs bf16. AWQ row is
          our build; NVFP4 row is NVIDIA's official.
        - **2-way (Phase 5 outcome b):** bf16 / AWQ-INT4 only, plus a
          callout cell in NVFP4's place reading "did not load on Spark
          with vLLM `<image-tag>` — see
          `results/nvfp4_loader_failure.txt`".
      - One-paragraph headline interpretation.
      - Architecture context: 52 layers / 23 M / 23 E / 6 attn /
        128 routed-top-6 / GQA / Mamba2 / RADIO+Parakeet, with the
        what-to-quantize-vs-leave-alone table. Quote the actual module
        inventory from Phase 1.
      - **Existing official builds:** one short paragraph linking
        `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8` and `-NVFP4`,
        explaining the FP8 row is *card-cited only* (we don't re-derive
        FP8 — NVIDIA already publishes it) and pointing at the NVFP4
        results section (or the loader-failure file in outcome b).
      - BF16 baseline: full eval table, wall-clock, memory-pressure note.
      - AWQ build: why-this-scheme paragraph, implementation note (relu²
        expert split, layer-0/vision/audio preservation, MoE
        fused-vs-unfused choice from Phase 1), settings table, what was
        stripped vs kept, smoke result, full eval table with per-domain
        MMLU breakdown + best/worst subtasks + wall-clock breakdown.
      - **NVFP4 row** (outcome a): a parallel "NVFP4 (NVIDIA official)"
        section quoting NVIDIA's published settings (NVFP4 experts, FP8
        mamba/attention, BF16 encoders) and our measured eval table.
        Make it clear this is a *measurement of NVIDIA's artifact*, not
        a recipe of ours. **OR** (outcome b): an "NVFP4 attempted but
        not supported on Spark" section quoting the loader error and
        the pinned vLLM image tag at test time.
      - Head-to-head delta table:
        - **3-way:** bf16 / AWQ / NVFP4 × every metric × signed Δ vs
          bf16, σ-tagged.
        - **2-way fallback:** same table without the NVFP4 column.
      - Decision rule: when to ship our AWQ vs use NVIDIA's FP8 vs
        NVIDIA's NVFP4 — based on size, multimodal preservation,
        hardware, measured deltas. (Outcome b: drop the NVFP4 column.)
      - Disk-space accounting.
      - Out-of-scope list.
      - Roadmap checklist.
      - File index.
- [ ] Write the AWQ artifact README at
      `artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4/README.md`
      with YAML frontmatter (license matching nvidia's NVIDIA Open Model
      Agreement, `base_model:
      nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`,
      `base_model_relation: quantized`, `quantized_by: <user>`, tags
      `[awq, autoawq, quantization, vllm, nemotron, mamba, moe,
      multimodal]`). Body: one-page summary of the build, settings
      table, eval deltas (vs bf16, plus a one-line "for context, NVFP4
      lands at ..." or "NVFP4 didn't load on Spark"), suggested vLLM
      serve command (with `--reasoning-parser nemotron_v3`,
      `--trust-remote-code`, multimodal flags), link to REPORT on
      GitHub.
- [ ] Update `docs/schemes/awq-gemm.md` with any Nemotron-specific
      gotchas worth surfacing into the generic playbook. Likely
      candidates: how to handle Mamba inner Linears (always skip;
      conv1d isn't Linear but in_proj / out_proj / dt_proj are); the
      "two-encoder multimodal preserve" pattern (vision + audio both
      kept dense, both have their custom `.py` files copied through);
      the relu²-MoE fused-expert split. **Don't speculatively
      generalize** — only lift gotchas that have actually surfaced in
      two distinct runs.
- [ ] Update the top-level `README.md` Runs index entry for this slug
      with the measured numbers (TL;DR row pulled from the REPORT —
      include NVFP4 in the row only if Phase 5 outcome was (a)).

### Verification (all must pass to mark the phase done)

- [ ] `runs/nemotron-3-nano-omni-30b-a3b/REPORT.md` exists and renders
      correctly: every numeric cell is filled in (no `?` left from the
      template), every Δ-vs-bf16 cell carries either a σ-comparison
      interpretation or "within stderr" text, and the TL;DR table cross-
      checks against the per-build "Full eval" tables.
- [ ] REPORT shape matches Phase 5's outcome: 3-way table iff
      `results/nvfp4_full/` exists with full eval JSONs; 2-way fallback
      with loader-failure callout iff `results/nvfp4_loader_failure.txt`
      exists. Verification agent must check the file system, not just
      the REPORT prose.
- [ ] No `<...>` placeholder text remains in the AWQ artifact's
      `README.md`. YAML frontmatter parses (test:
      `python -c "import yaml; yaml.safe_load(open(p).read().split('---')[1])"`).
- [ ] All cross-document links resolve: `git grep -E '\[.*\]\(\.\..*\.md\)'
      runs/nemotron-3-nano-omni-30b-a3b/` returns only paths that exist.
- [ ] **New automated check (rendering integrity):** add
      `runs/nemotron-3-nano-omni-30b-a3b/recipes/test_report_consistency.py`
      that:
      - Parses the REPORT's TL;DR table and the per-build full-eval tables.
      - For each measured build (bf16, AWQ, and NVFP4 if outcome a),
        asserts TL;DR cell == per-build cell (so they can't drift).
      - For each measured quantized build, asserts the Δ row's signed
        delta == TL;DR(quant) − TL;DR(bf16) within ±0.01 pp.
      - In outcome b, asserts the NVFP4 cell in the TL;DR is the
        documented "did not load" callout text rather than a number.
      Run it with:
      `pytest runs/nemotron-3-nano-omni-30b-a3b/recipes/test_report_consistency.py -v`.
      Must pass.
- [ ] Top-level `README.md` Runs index shows the run with the headline
      numbers (MMLU bf16 vs AWQ delta; +NVFP4 if outcome a).

---

## Risks and rollback

- **Mamba2 quantization disaster.** If the ignore policy misses a single
  Mamba inner Linear, the SSM dynamics blow up and quality collapses
  catastrophically (not gracefully). Mitigation: Phase 1's
  `test_module_classify.py` enumerates Mamba inner names explicitly; the
  recipe's `should_quantize` defaults to *skip* unless the regex match
  affirmatively says quantize.
- **vLLM 0.20.0 isn't in the pinned Docker image.** The card says vLLM
  ≥ 0.20.0 is required for Nemotron support (`--reasoning-parser
  nemotron_v3` is new). The pinned image is
  `dgx-vllm-eugr-nightly-tf5:20260415` — verify at the start of Phase 2
  that `vllm --version` inside the container is ≥ 0.20.0; if not,
  upgrade the image tag and document the new tag in
  `tools/serve_vllm_docker.sh`.
- **Custom-code import order.** `trust_remote_code=True` runs `modeling.py`
  → which imports `modeling_nemotron_h.py` → which imports `audio_model.py`
  and the RADIO files. The AWQ build keeps every encoder dense, but the
  artifact dir still needs every custom `.py` copied through so vLLM can
  import them at startup. Phase 2 lists the full pass-through file set.
- **State-dict prefix unknown.** vLLM's `hf_to_vllm_mapper` for
  `NemotronH_Nano_Omni_*` may not exist in the cached image, in which
  case loading fails with `KeyError` on a prefix like
  `language_model.model.layers.0...`. Mitigation: Phase 3 serves the
  bf16 source through the Docker image first — that smoke-tests the
  loader on known-good weights, tells us whether the AWQ recipe needs
  to prefix its saved keys, and lets us fix the loader gap (or upgrade
  the vLLM image) before re-quantizing.
- **17-shard download time.** ~66 GB at typical home upstream is 2–3
  hours. Pre-cache in Phase 1 during the inspector run so it doesn't
  block Phase 2. The NVFP4 build (~21 GB, also pre-cached in Phase 1)
  adds another ~30 min but pulling it now means Phase 5 starts
  immediately.
- **NVFP4 kernels may not work on SM121a.** Phase 5 is structured around
  this: outcome (a) gives us a third measured column; outcome (b) gives
  us a documented loader failure. Either is acceptable — the only
  failure mode this plan rejects is "tried, didn't work, didn't tell
  anyone." The orchestrator must surface the outcome to the user
  explicitly before moving on to Phase 6 so the REPORT shape is chosen
  with full information.

---

## Out of scope

- **Generating an FP8 build.** NVIDIA already publishes
  `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8`. Reproducing it on
  Spark would consume hours without producing a meaningfully different
  artifact. Use NVIDIA's directly when FP8 is the target. The REPORT
  cites NVIDIA's card claims about the FP8 build but does not measure
  them.
- **Generating an NVFP4 build.** NVIDIA also publishes the NVFP4
  variant. We don't re-derive it; we *do* eval it (Phase 5) — that's
  cheap and gives the REPORT a third measured column when the kernels
  work on Spark.
- **Multimodal eval (vision/audio benchmarks).** The eval battery here is
  the same text-only one used for Qwen3.5-MoE — the AWQ build
  *preserves* multimodal capability but we don't measure it. Open as a
  follow-up later.
- **HumanEval** (sandboxed Docker required; out of scope per repo policy).
- **Long-context probe** (RULER / NIAH ≥ 32 K).
- **Calibrated AWQ** (real activation-aware salience pass with
  calibration data). Data-free RTN is the baseline; calibration is a
  follow-up if AWQ deltas exceed ~3 pp on MMLU.
- **Fixing vLLM NVFP4-kernel support if Phase 5 hits outcome (b).** We
  document the gap and move on — patching vLLM kernels is a separate
  project.
- **Publishing to Hugging Face.** Once the REPORT is signed off, the
  artifact upload is a separate step driven by
  `HUGGINGFACE_PUBLISHING.md` — not part of this plan.
