# Nemotron-3-Nano-Omni-30B-A3B-Reasoning — AWQ-INT4 Quantization + 3-way Eval on DGX Spark

## Goal

Two outputs, both targeting
[`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16)
on a single DGX Spark (GB10 / SM121a, 128 GiB unified memory):

1. **A vLLM-loadable AWQ-INT4 artifact** (llm-compressor calibrated AWQ,
   compressed-tensors W4A16 `pack-quantized` format) — smallest disk
   (~21–22 GiB), vision + audio towers preserved dense for multimodal
   serving.
2. **A three-way head-to-head eval** (our AWQ vs NVIDIA's published
   NVFP4 build vs bf16 baseline) on the same battery used for
   `runs/qwen3.6-35b-distill/` (GSM8K full chat-CoT, full MMLU,
   ARC-Challenge full), with a `REPORT.md` carrying signed Δ-vs-bf16
   deltas and a decision rule for which build to ship under which
   constraints.

**Current outcome (2026-04-30).** The AWQ artifact now exists and serves
successfully in vLLM:

- Artifact:
  `artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4/`
  (6 safetensors shards, ~21.34 GiB final payload / 22G on disk).
- Working recipe:
  `runs/nemotron-3-nano-omni-30b-a3b/recipes/awq_compressed_tensors.py`.
- Format: `quant_method="compressed-tensors"`,
  `format="pack-quantized"`, W4A16, `group_size=64`, symmetric
  group-wise 4-bit expert weights.
- vLLM smoke passed on
  `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:20260428` with
  `--kv-cache-dtype fp8_e4m3`, `--max-model-len 8192`,
  `--gpu-memory-utilization 0.55`, and `--reasoning-parser nemotron_v3`.
  The final `/v1/chat/completions` smoke returned `2+2 equals 4.`
- Full AWQ/NVFP4/bf16 evals are still pending. The plan below keeps those
  later phases as future work.

The first successful compressed-tensors run almost failed at the same
place as earlier attempts: the default `llmcompressor.oneshot(...,
output_dir=..., save_compressed=True)` path finishes calibration but then
spikes memory during Transformers shard serialization. The working recipe
therefore runs oneshot with `output_dir=None, save_compressed=False`,
compresses the calibrated inner LM in memory with `ModelCompressor`, and
streams a bounded custom merge of compressed LM tensors plus dense
multimodal tensors into final safetensors shards. Peak RSS during the
successful run was about 94.75 GiB, under the Spark memory cap.

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
- Prior AWQ scheme docs from sibling runs are background only; the working
  Nemotron artifact uses compressed-tensors W4A16.
- `tools/run_under_memcap.sh` — `systemd-run --scope` cgroup wrapper.
  Defaults `MemoryMax=112G`, `MemoryHigh=100G`, `MemorySwapMax=0`. Override
  via env (`MEMORY_MAX=64G ...`).
- `tools/serve_vllm_docker.sh` — now defaults to the validated
  `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:20260428` image. Always
  passes `--trust-remote-code`; this is required for Nemotron's custom
  `modeling_nemotron_h.py` to load. The older `:20260415` image is kept
  only as historical context; it was not used for the final working
  artifact.
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
  structural target for the Nemotron REPORT in Phase 6.
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
    scope, eval IN scope** (Phase 4). Useful both as a measured comparison
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
  - **vLLM serving** runs in the validated
    `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:20260428` Docker image
    via `tools/serve_vllm_docker.sh`. Bare-metal vLLM is not assumed to
    work on aarch64 / SM121a.
  - **Eval** is `lm-eval==0.4.11` from the project venv driven by
    `tools/run_eval_full.sh`, hitting the Docker vLLM endpoint over HTTP.
- **Python env:**
  ```bash
  cd /home/sergio/git/dgx-spark-quantization
  source .venv/bin/activate
  ```
- **Quantize command (working AWQ compressed-tensors recipe):**
  ```bash
  export SRC_DIR="$PWD/hf-cache/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/snapshots/d15962741057ae3a07147df504060e9f0838224e"
  export DST_DIR="$PWD/artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4"
  tools/run_under_memcap.sh python runs/nemotron-3-nano-omni-30b-a3b/recipes/awq_compressed_tensors.py
  ```
- **Recipe self-test:** run
  `python runs/nemotron-3-nano-omni-30b-a3b/recipes/awq_compressed_tensors.py --selftest`
  before touching source shards. The current recipe prints `selftest OK`.
- **Serve a quantized artifact:**
  ```bash
  tools/serve_vllm_docker.sh "$PWD/artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4" \
    --kv-cache-dtype fp8_e4m3 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.55 \
    --reasoning-parser nemotron_v3 \
    --served-model-name nemotron-omni-awq-ct
  ```
- **Eval against a running endpoint:**
  ```bash
  tools/run_eval_full.sh nemotron-omni-awq-ct \
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
  splits it into smaller numbered phases rather than accepting bloat.

**File-touch scope per agent** (orchestrator enforces by listing
allowed paths in each agent prompt):

- Phase 0: `runs/nemotron-3-nano-omni-30b-a3b/**`, top-level `README.md`.
- Phase 1: `runs/nemotron-3-nano-omni-30b-a3b/**`, `hf-cache/**` (downloads).
- Phase 2: `runs/nemotron-3-nano-omni-30b-a3b/**`,
  `artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4/**`,
  `requirements.txt`, `tools/serve_vllm_docker.sh`.
- Phase 3: `runs/nemotron-3-nano-omni-30b-a3b/results/awq_full/**`.
- Phase 4: `runs/nemotron-3-nano-omni-30b-a3b/results/nvfp4_full/**`,
  `hf-cache/**` (NVFP4 download).
- Phase 5: `runs/nemotron-3-nano-omni-30b-a3b/results/bf16_full/**`.
- Phase 6: `runs/nemotron-3-nano-omni-30b-a3b/REPORT.md`,
  `artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4/README.md`,
  `docs/schemes/awq-compressed-tensors.md`, top-level `README.md`.

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
      `should_quantize(name, shape)` separated from the recipe so the
      compressed-tensors recipe and the Phase 1 test both import it. Final skip
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
to import the RADIO ViT and Parakeet feature extractor; the quantization
and serving phases need them too.

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
      `recipes/_classify.py` (separated from the quantization driver so
      Phase 2 and the test can both import without circular dependency).

---

## Phase 2 — AWQ-INT4 via llm-compressor (compressed-tensors W4A16)

**Status:** completed
**Kind:** logic

Re-engineer the AWQ build to produce a `compressed-tensors`
`pack-quantized` W4A16 artifact via `llm-compressor`. This is the
on-disk format vLLM actually has a working ungated-MoE loader for — the
same general compressed-tensors loader family used by NVIDIA's quantized
artifacts, and the path
[stelterlab](https://huggingface.co/stelterlab/NVIDIA-Nemotron-3-Nano-30B-A3B-AWQ)
successfully shipped on the **LM-only** Nemotron base.

**Outcome (2026-04-30): completed.** The final artifact is
`artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4/`.
It is vLLM-loadable, uses compressed-tensors W4A16, preserves the
multimodal wrapper files and dense multimodal tensors, and passed a live
OpenAI-compatible `/v1/chat/completions` smoke.

The critical implementation deviation from the original compressed-tensors
design:
the recipe cannot let `llmcompressor.oneshot` save the compressed model
itself. Earlier attempts completed calibration/compression and then died
at `Writing model shards: 0%`, which was a serializer memory spike. The
working flow is:

1. Load the full Omni model on CPU with `trust_remote_code=True` and
   `torch_dtype=torch.bfloat16`.
2. Temporarily free the multimodal modules during calibration to reduce
   host RSS; the final dense multimodal tensors are copied from the
   source shards later.
3. Run `oneshot(..., output_dir=None, save_compressed=False)`.
4. Compress the calibrated inner LM in memory with
   `compressed_tensors.ModelCompressor`.
5. Capture the compressed LM `state_dict` references and stream them,
   prefixed as `language_model.*`, through the recipe's custom bounded
   safetensors sharder.
6. Copy dense multimodal/source tensors and tokenizer/custom-code files
   into the same final artifact.
7. Atomically rename `<DST_DIR>.tmp.<pid>/` to `<DST_DIR>/`.

The second critical fix was in `quantization_config.ignore`: the first
vLLM load of the completed artifact failed with
`KeyError: 'layers.0.mixer.in_proj.weight'`. The artifact had correctly
kept Mamba/attention/shared-expert modules dense, but vLLM remaps
`backbone` to `model` internally and does not remap `re:` ignore entries.
The recipe now emits ignore regexes for all relevant prefixes:
`language_model.backbone`, `language_model.model`, `backbone`, and
`model`. After updating the artifact config with those expanded regexes,
vLLM loaded the model successfully.

### References (read these first)

- stelterlab AWQ artifact: <https://huggingface.co/stelterlab/NVIDIA-Nemotron-3-Nano-30B-A3B-AWQ>
- stelterlab recipe.yaml (verbatim source): <https://huggingface.co/stelterlab/NVIDIA-Nemotron-3-Nano-30B-A3B-AWQ/raw/main/recipe.yaml>
- stelterlab artifact `config.json` (target `quantization_config` shape): <https://huggingface.co/stelterlab/NVIDIA-Nemotron-3-Nano-30B-A3B-AWQ/raw/main/config.json>
- stelterlab `model.safetensors.index.json` (target on-disk key shape — `experts.<j>.up_proj.weight_packed` / `weight_scale` / `weight_shape`, no `gate_proj`): <https://huggingface.co/stelterlab/NVIDIA-Nemotron-3-Nano-30B-A3B-AWQ/raw/main/model.safetensors.index.json>
- llm-compressor docs: <https://github.com/vllm-project/llm-compressor> (AWQModifier reference)
- Phase 1's classifier (`runs/nemotron-3-nano-omni-30b-a3b/recipes/_classify.py`) — reuse the substring/endswith policy for the ignore list expansion.

### Tasks

- [x] **Tooling:** install `llmcompressor` into the project venv and add
      `llmcompressor>=0.9.0` to `requirements.txt`.
- [x] **New recipe:** write
      `runs/nemotron-3-nano-omni-30b-a3b/recipes/awq_compressed_tensors.py`.
      Final driver pattern (compressed-tensors-style, with custom bounded
      shard writing):
      - Read `SRC_DIR` (BF16 source snapshot) + `DST_DIR` (artifact out).
      - Atomic save: refuse if `DST_DIR` is non-empty; write into
        `<DST_DIR>.tmp.<pid>/` and rename.
      - `from transformers import AutoModelForCausalLM, AutoTokenizer`
        with `trust_remote_code=True`, `device_map="cpu"`,
        `torch_dtype=torch.bfloat16`.
      - Build the AWQModifier recipe in Python, with expanded ignore
        regexes covering both HF/source and vLLM-runtime module prefixes.
      - Pick a calibration dataset:
        `open-platypus` (general reasoning instruction corpus, ~25k
        samples) — sample 256 prompts, max_seq_len=2048. Reasoning
        prompts are in-domain for this reasoning model.
      - Run `oneshot` with `output_dir=None, save_compressed=False`.
      - Compress the calibrated inner LM in memory via
        `ModelCompressor.from_pretrained_model(inner).compress_model(inner)`.
      - Stream compressed LM tensors plus dense multimodal tensors into
        custom safetensors shards. This avoids the OOM-prone Transformers
        serializer path.
      - Copy through the tokenizer/aux files and all custom `.py` files
        required by `trust_remote_code`.
      - Rename `<DST_DIR>.tmp.<pid>/` → `<DST_DIR>/` atomically.
- [x] **Recipe shape (build in Python):**
      ```python
      from llmcompressor.modifiers.awq import AWQModifier
      recipe = AWQModifier(
          targets=["Linear"],
          ignore=[
              "lm_head",
              "re:.*language_model\\.embed_tokens$",
              "re:.*language_model\\.backbone\\.norm_f$",
              "re:.*mixer\\.gate$",          # MoE router
              "re:.*\\.layers\\.0\\..*",     # layer-0 dense
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
      The final recipe uses this policy but expands each dense-module
      regex across the source and vLLM-visible prefixes:
      `language_model.backbone`, `language_model.model`, `backbone`, and
      `model`. Without this expansion vLLM constructs quantized Mamba
      modules and then fails when loading dense `in_proj.weight` /
      `out_proj.weight` tensors.
      Note: the **6 attention layer indices** (5, 12, 19, 26, 33, 42)
      are the `*` positions in `hybrid_override_pattern`. Confirm by
      grepping `module_inspection.txt` for `mixer.q_proj` and listing
      the layer indices that match. If the actual positions differ from
      the list above, the smooth-layer mappings will silently no-op (no
      regex match) and AWQ will fall back to per-layer min/max — still
      functional, just suboptimal. **The recipe must include a startup
      assertion that prints how many smooth-layer mappings actually
      matched a module — fail loud if the count is < 6.**
- [x] **MoE expert ignore audit:** the recipe above does NOT explicitly
      ignore expert MLPs — that's intentional, since experts are the
      only modules we *want* quantized. Re-running the classifier from
      Phase 1 against the planned ignore patterns must yield exactly
      the **5,888 expected quantized tensors** (23 MoE layers × 128
      experts × 2 projections). Add a dry-run mode to the recipe
      (`--dry-run`) that walks the model, applies the ignore regexes,
      and prints `would_quantize=N, would_skip=M` BEFORE calibration.
      Abort if `would_quantize != 5888`. Final dry-run printed
      `Linear modules total=6005 would_quantize=5888 would_skip=117`.
- [x] **Self-test:** add `--selftest` flag (synthetic-tensor roundtrip)
      that builds a tiny 2-layer model with a single mock MoE block,
      runs llm-compressor oneshot on it, asserts the output dir
      contains `weight_packed`/`weight_scale`/`weight_shape` keys for
      the experts and dense `weight` for the embeddings/router. Must
      pass before the full run touches any source shards. Final self-test
      passed and printed `selftest OK`.
- [x] **Clear the prior artifact:** `rm -rf
      artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4/`
      before writing the final compressed-tensors artifact.
- [x] **Run the full quantization** under cgroup guard:
      ```bash
      tools/run_under_memcap.sh \
        bash -lc '
          source .venv/bin/activate
          export SRC_DIR="$PWD/hf-cache/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/snapshots/d15962741057ae3a07147df504060e9f0838224e"
          export DST_DIR="$PWD/artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4"
          python runs/nemotron-3-nano-omni-30b-a3b/recipes/awq_compressed_tensors.py \
            2>&1 | tee -a runs/nemotron-3-nano-omni-30b-a3b/results/awq_ct_quant.log
        '
      ```
      Final log:
      `runs/nemotron-3-nano-omni-30b-a3b/results/awq_ct_quant.log`.
      Peak RSS reached about 94.75 GiB during custom shard merge.
- [x] **Smoke-validate by serving** on the validated image (`:20260428`):
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
      loads via the working ungated-MoE path. Final smoke used
      `/v1/models` plus a chat completion prompt,
      `Answer in one sentence: what is 2+2?`, and returned
      `2+2 equals 4.`. Save to
      `runs/nemotron-3-nano-omni-30b-a3b/results/awq_ct_smoke.txt`.
- [x] **Update `tools/serve_vllm_docker.sh`** to default to
      `:20260428` (the newer image where BF16 serves cleanly). Note
      the change in a one-line comment at the top of the script.

### Resolved implementation details

- The recipe passes an `AWQModifier` object directly to `oneshot`, with
  `output_dir=None` and `save_compressed=False`.
- The embedding path stays dense and is handled by the broad embedding /
  lm-head ignore policy plus dense source-tensor merge.
- Smooth-layer mappings matched all 6 attention layers and all 23 MoE
  layers. The log records `smooth layer mappings matched 6/6 attention
  layers, 23/23 MoE layers`.
- vLLM requires ignore regexes that match both source/HF names
  (`language_model.backbone.*`) and runtime names
  (`language_model.model.*`). This is now encoded in the recipe and in
  the artifact `config.json`.

### Verification (all must pass to mark the phase done)

- [x] **Tooling check:** `python -c 'import llmcompressor; print(llmcompressor.__version__)'`
      imports successfully from the project venv.
- [x] **Self-test passes:** `python runs/.../awq_compressed_tensors.py --selftest`
      exits 0 and prints `selftest OK`.
- [x] **Dry-run preflight:** the recipe's dry-run mode prints
      `Linear modules total=6005 would_quantize=5888 would_skip=117`.
      The count matches Phase 1's expected 23 MoE layers × 128 routed
      experts × 2 projections.
- [x] **Smooth-layer match count:** recipe startup log contains
      `smooth layer mappings matched 6/6 attention layers, 23/23 MoE layers`.
- [x] **Quantization log shows successful completion:** final log shows
      `DONE. files=27 total=21.34 GiB final RSS=92.07 GiB`, with no
      traceback. Custom merge wrote 6 shards and reported
      `merge: total 18019 LM keys + 1106 MM keys -> 6 shards`.
- [x] **Artifact dir structure:** has 6 shards, ~21.34 GiB payload
      / 22G on disk,
      `model.safetensors.index.json` present.
- [x] **`config.json` shape:** `quantization_config.format ==
      "pack-quantized"`, `quant_method == "compressed-tensors"`,
      `config_groups.group_0.weights.{num_bits=4, group_size=64,
      symmetric=true, strategy="group", observer="minmax"}`,
      `targets=["Linear"]`, `ignore` includes `lm_head`. The
      `architectures` field stays `["NemotronH_Nano_Omni_Reasoning_V3"]`.
- [x] **Index keys match stelterlab's shape:** `model.safetensors.index.json`
      contains `language_model.backbone.layers.<i>.mixer.experts.<j>.up_proj.weight_packed`
      (and `.weight_scale`, `.weight_shape`) and `.down_proj.*` triplets
      for every (MoE-layer, expert) pair. Crucially: NO `gate_proj` keys
      anywhere in expert paths, NO fused `gate_up_proj` keys. Mamba
      `in_proj`/`out_proj` keys exist with regular `.weight` (NOT
      `.weight_packed`) — proves the Mamba-skip ignore worked.
- [x] **Multimodal pass-through:** all custom `.py` files + aux
      files present in artifact dir, verified via `ls`.
- [x] **vLLM smoke test passes — non-degenerate output:**
      `/v1/models` returned the served model, and `/v1/chat/completions`
      with `max_tokens=128` returned `2+2 equals 4.` with
      `finish_reason="stop"`. The first `max_tokens=32` request produced
      reasoning only and stopped by length; increasing to 128 resolved
      it. Full transcript saved to `results/awq_ct_smoke.txt`.
- [x] **Loader path confirmed:** `docker logs` from the smoke serve
      includes a line indicating `compressed-tensors` was selected
      (`quantization=compressed-tensors`,
      `Using CompressedTensorsWNA16MarlinMoEMethod`, and
      `Using Marlin backend for WNA16 MoE`). Successful serve log:
      `results/awq_ct_serve_smoke_retry.log`.

---

## Phase 3 — AWQ full eval

**Status:** pending
**Kind:** logic

### Tasks

- [ ] Serve the AWQ artifact:
      ```bash
      tools/serve_vllm_docker.sh \
        "$PWD/artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4" \
        --kv-cache-dtype fp8_e4m3 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.55 \
        --reasoning-parser nemotron_v3 \
        --served-model-name nemotron-omni-awq-ct
      ```
- [ ] Run the eval:
      ```bash
      tools/run_eval_full.sh nemotron-omni-awq-ct \
        "$PWD/artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4" \
        "$PWD/runs/nemotron-3-nano-omni-30b-a3b/results/awq_full"
      ```
- [ ] Tear down vLLM.

### Verification (all must pass to mark the phase done)

- [ ] `results/awq_full/{gsm8k,mmlu,arc_challenge}/results_*.json` exist;
      `run.log` shows rc=0 for each.
- [ ] After Phase 5 completes, sanity-bound deltas vs bf16:
      `|Δ MMLU| ≤ 6 pp`, `|Δ GSM8K
      strict| ≤ 4 pp`, `|Δ ARC-C acc| ≤ 5 pp`. AWQ's published precedent
      on Qwen3 was −2.73 pp MMLU; a Nemotron AWQ delta beyond 6 pp
      indicates something in the recipe is wrong (most likely the skip
      list is too aggressive about quantizing, or layer 0 wasn't
      preserved).

---

## Phase 4 — NVFP4 eval (NVIDIA's official artifact)

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
      under `hf-cache/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4/snapshots/889396e9cebaefdb69a469afc7bd111660f78eff/`).
      If missing, download:
      ```bash
      huggingface-cli download nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 \
        --cache-dir "$PWD/hf-cache"
      ```
- [ ] Attempt to serve under Docker vLLM, matching the KV-cache and
      `max-model-len` from Phase 3 and the later Phase 5 bf16 baseline
      so the deltas are
      apples-to-apples:
      ```bash
      tools/serve_vllm_docker.sh \
        "$NVFP4_DIR" \
        --kv-cache-dtype fp8_e4m3 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.55 \
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
- [ ] After Phase 5 completes, sanity-bound deltas vs bf16:
      `|Δ MMLU| ≤ 6 pp`,
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

## Phase 5 — BF16 baseline full eval

**Status:** pending
**Kind:** logic

Measure the bf16 baseline last, after the two smaller quantized artifacts.
The Δ-vs-bf16 deltas in the REPORT are only trustworthy when the baseline
uses the same vLLM image, KV-cache dtype, model length, prompts, and eval
harness as the AWQ and NVFP4 runs.

### Tasks

- [ ] Serve the bf16 source (from the local `hf-cache` snapshot) under the
      Docker vLLM, pinning `--kv-cache-dtype fp8_e4m3` to match the
      quantized runs. Memory will be tight (66 GB weights + KV + CUDA
      overhead vs 128 GB unified RAM), so use
      `--max-model-len 4096 --gpu-memory-utilization 0.45`:
      ```bash
      tools/serve_vllm_docker.sh \
        "$PWD/hf-cache/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/snapshots/d15962741057ae3a07147df504060e9f0838224e" \
        --kv-cache-dtype fp8_e4m3 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.45 \
        --reasoning-parser nemotron_v3 \
        --served-model-name nemotron-omni-bf16
      ```
- [ ] In another terminal, run the eval:
      ```bash
      tools/run_eval_full.sh nemotron-omni-bf16 \
        "$PWD/hf-cache/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/snapshots/d15962741057ae3a07147df504060e9f0838224e" \
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

## Phase 6 — REPORT, model card, scheme-doc updates

**Status:** pending
**Kind:** mixed

Write the per-run REPORT and the AWQ artifact README (the model card that
ships to Hugging Face inside the artifact dir). The generic compressed-tensors
AWQ scheme doc now lives at
`docs/schemes/awq-compressed-tensors.md`; keep Nemotron-specific eval results
in this run's REPORT/model card.

The REPORT shape branches on Phase 4's outcome — `(a)` 3-way table when
NVFP4 evaluated successfully, `(b)` 2-way table when NVFP4 didn't load on
Spark.

### Tasks

- [ ] Fill `runs/nemotron-3-nano-omni-30b-a3b/REPORT.md` using
      `templates/run/REPORT.md` as the skeleton and
      `runs/qwen3.6-35b-distill/REPORT.md` as the structural reference.
      Sections required:
      - TL;DR table:
        - **3-way (Phase 4 outcome a):** AWQ-INT4 / NVFP4 / bf16 ×
          bits / disk / GSM8K / MMLU / ARC-C / Δ vs bf16. AWQ row is
          our build; NVFP4 row is NVIDIA's official.
        - **2-way (Phase 4 outcome b):** AWQ-INT4 / bf16 only, plus a
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
      - BF16 baseline: full eval table, wall-clock, memory-pressure note.
      - Head-to-head delta table:
        - **3-way:** AWQ / NVFP4 / bf16 × every metric × signed Δ vs
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
      `[awq, compressed-tensors, quantization, vllm, nemotron, mamba, moe,
      multimodal]`). Body: one-page summary of the build, settings
      table, eval deltas (vs bf16, plus a one-line "for context, NVFP4
      lands at ..." or "NVFP4 didn't load on Spark"), suggested vLLM
      serve command (with `--reasoning-parser nemotron_v3`,
      `--trust-remote-code`, multimodal flags), link to REPORT on
      GitHub.
- [x] Added generic compressed-tensors AWQ scheme doc at
      `docs/schemes/awq-compressed-tensors.md`, covering vLLM runtime-prefix
      ignore regexes, keeping Mamba/attention/shared experts dense, group-size
      constraints, and custom bounded shard writing to avoid the Transformers
      serializer memory spike.
- [ ] Update the top-level `README.md` Runs index entry for this slug
      with the measured numbers (TL;DR row pulled from the REPORT —
      include NVFP4 in the row only if Phase 4 outcome was (a)).

### Verification (all must pass to mark the phase done)

- [ ] `runs/nemotron-3-nano-omni-30b-a3b/REPORT.md` exists and renders
      correctly: every numeric cell is filled in (no `?` left from the
      template), every Δ-vs-bf16 cell carries either a σ-comparison
      interpretation or "within stderr" text, and the TL;DR table cross-
      checks against the per-build "Full eval" tables.
- [ ] REPORT shape matches Phase 4's outcome: 3-way table iff
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
- **vLLM image compatibility.** The original `:20260415` image could
  serve BF16 but did not provide a working quantized path for this relu²
  MoE model. The validated image is now
  `dgx-vllm-eugr-nightly-tf5:20260428`, which reports vLLM
  `0.20.1rc1.dev23+gde3da0b97.d20260428` and successfully loads the
  compressed-tensors artifact. Keep future evals on that image unless
  there is an explicit reason to retest.
- **Custom-code import order.** `trust_remote_code=True` runs `modeling.py`
  → which imports `modeling_nemotron_h.py` → which imports `audio_model.py`
  and the RADIO files. The AWQ build keeps every encoder dense, but the
  artifact dir still needs every custom `.py` copied through so vLLM can
  import them at startup. Phase 2 lists the full pass-through file set.
- **vLLM state-dict / ignore-prefix mismatch.** This happened once:
  the first compressed-tensors artifact loaded far enough to read the
  shards but then failed with `KeyError:
  'layers.0.mixer.in_proj.weight'`. Root cause was vLLM's
  `backbone`→`model` runtime mapping combined with compressed-tensors
  regex ignores that only mentioned the HF/source prefix. Mitigation is
  implemented in `awq_compressed_tensors.py`: emit ignore regexes for
  `language_model.backbone`, `language_model.model`, `backbone`, and
  `model`.
- **17-shard download time.** ~66 GB at typical home upstream is 2–3
  hours. Pre-cache in Phase 1 during the inspector run so it doesn't
  block Phase 2. The NVFP4 build (~21 GB, also pre-cached in Phase 1)
  adds another ~30 min but pulling it now means Phase 4 starts
  immediately.
- **NVFP4 kernels may not work on SM121a.** Phase 4 is structured around
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
  variant. We don't re-derive it; we *do* eval it (Phase 4) — that's
  cheap and gives the REPORT a third measured column when the kernels
  work on Spark.
- **Multimodal eval (vision/audio benchmarks).** The eval battery here is
  the same text-only one used for Qwen3.5-MoE — the AWQ build
  *preserves* multimodal capability but we don't measure it. Open as a
  follow-up later.
- **HumanEval** (sandboxed Docker required; out of scope per repo policy).
- **Long-context probe** (RULER / NIAH ≥ 32 K).
- **Further AWQ recipe tuning.** The shipped compressed-tensors artifact
  already uses calibrated llm-compressor AWQ. Further tuning, such as
  different calibration corpora/sample counts or selective dense expert
  layers, is a follow-up only if full eval deltas justify it.
- **Fixing vLLM NVFP4-kernel support if Phase 4 hits outcome (b).** We
  document the gap and move on — patching vLLM kernels is a separate
  project.
- **Publishing to Hugging Face.** Once the REPORT is signed off, the
  artifact upload is a separate step driven by
  `HUGGINGFACE_PUBLISHING.md` — not part of this plan.
