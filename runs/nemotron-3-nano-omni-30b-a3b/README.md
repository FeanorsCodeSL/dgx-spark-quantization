# Nemotron-3-Nano-Omni-30B-A3B-Reasoning — quantization run

**Base model**: [`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16)
30B-A3B (active 3B) NemotronH Mamba2 ⨉ Transformer-attention ⨉ MoE hybrid,
multimodal (vision + audio), reasoning fine-tune. 17 safetensors shards,
~66 GB on disk. License: NVIDIA Open Model Agreement.

**Hardware**: NVIDIA DGX Spark, GB10 / SM121a, 128 GiB unified memory.

**Status**: in-progress — last updated 2026-04-29

> Driven by [`docs/plans/nemotron-3-nano-omni-30b-quantization.md`](../../docs/plans/nemotron-3-nano-omni-30b-quantization.md).
> All recipe / eval / report decisions live there.

---

## TL;DR

| build | bits | disk | MMLU | GSM8K (strict) | ARC-C | Δ MMLU vs bf16 |
|---|---|---|---|---|---|---|
| bf16 baseline | 16 | ~66 GiB | ? | ? | ? | — |
| AWQ-INT4 GEMM (ours, multimodal) | 4 | ~22–25 GiB | ? | ? | ? | ? |
| NVFP4 (NVIDIA official, eval only) | 4 (NVFP4 experts) | ~21 GiB | ? | ? | ? | ? |

Plan: ship our AWQ-INT4 GEMM build (smallest disk with multimodal preserved);
cite NVIDIA's FP8 / NVFP4 builds for users who want the official paths.
Numbers filled in Phase 6.

For full numbers, settings, and the head-to-head, see
[`REPORT.md`](./REPORT.md).

---

## Schemes used

- [`docs/schemes/awq-gemm.md`](../../docs/schemes/awq-gemm.md) — data-free
  RTN AWQ-INT4 GEMM. Model-specific deviations (Mamba inner Linears kept
  dense, layer-0 + shared-expert + router-gate + vision + audio + projectors
  all kept dense; only the 128 routed experts across the 23 MoE layers get
  4-bit packed) are documented in `recipes/awq_gemm.py` and in the
  `REPORT.md`.

FP8 and NVFP4 are not generated here — see
[Existing official builds](#existing-official-builds) below.

## Existing official builds

NVIDIA already publishes two quantized variants of this model. We don't
reproduce them; we cite/eval them so the head-to-head in `REPORT.md` covers
the realistic shipping options end-to-end.

- [`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8)
  — per-tensor E4M3, FP8 KV cache, ~32.8 GB. Generation **out of scope**
  (use NVIDIA's directly when FP8 is the target). REPORT cites NVIDIA's
  card claims for context but does not measure.
- [`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4)
  — NVFP4 experts + FP8 mamba/attention + BF16 encoders, ~20.9 GB.
  Generation **out of scope**. Eval **in scope** in Phase 5 — NVFP4 vLLM
  kernel support on SM121a is unverified, so the phase resolves it
  empirically.

## What's in this run

| path | content |
|---|---|
| [`REPORT.md`](./REPORT.md) | full quant report with deltas |
| [`recipes/`](./recipes/) | model-specific quantizer drivers (AWQ-INT4 GEMM) |
| [`results/`](./results/) | `results_*.json` + `run.log` per full eval, plus `module_inspection.txt` and quant logs |

## Architecture (from config.json)

- Outer class `NemotronH_Nano_Omni_Reasoning_V3` (multimodal wrapper);
  inner LM class `NemotronHForCausalLM` under `llm_config`;
  `model_type: nemotron_h`. `trust_remote_code=True` is required at load
  time; the artifact dir must carry the source's custom `.py` files.
- 52 hidden layers, `hybrid_override_pattern`
  `"MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"` →
  23 Mamba2 (`M`) + 23 MoE-MLP (`E`) + 6 attention (`*`).
- `hidden_size=2688`; dense `intermediate_size=1856`;
  `moe_intermediate_size=1856` per routed expert;
  `moe_shared_expert_intermediate_size=3712`;
  `n_routed_experts=128`, `num_experts_per_tok=6`, `n_shared_experts=1`.
- Attention (GQA): `num_attention_heads=32`, `num_key_value_heads=2`,
  `head_dim=128`, `rope_theta=10000`. So Q is `[2688 → 4096]`,
  K/V are `[2688 → 256]`.
- Mamba2: `mamba_num_heads=64`, `mamba_head_dim=64`, `ssm_state_size=128`,
  `conv_kernel=4`, `expand=2`, `chunk_size=128`,
  `time_step_min=0.001`, `time_step_max=0.1`, `mamba_hidden_act=silu`.
- MLP activation: `mlp_hidden_act: relu2` (squared ReLU). MoE expert MLP
  layout TBD by Phase 1 (likely `up + down` rather than `gate + up + down`
  due to relu²).
- Vision (`vision_config`): RADIOModel CRADIO v2-H,
  `vit_hidden_size=1280`, `projector_hidden_size=20480`,
  `force_image_size=512`, `patch_size=16`,
  `separate_video_embedder=true`, `max_num_patches=13312`.
- Audio (`sound_config`): Parakeet, `hidden_size=1024`,
  `num_attention_heads=8`, `num_hidden_layers=24`,
  `intermediate_size=4096`, `subsampling_factor=8`,
  `num_mel_bins=128`, `sampling_rate=16000`.
- `vocab_size=131072`, `tie_word_embeddings=false`. `torch_dtype: bfloat16`.

## Module-tree inspection (Phase 1 findings)

Live inspection of the bf16 source on CPU (`recipes/inspect_modules.py`,
3,451-line dump in [`results/module_inspection.txt`](./results/module_inspection.txt)).
The plan's config-derived numbers are confirmed; the *names* of every
module are now pinned, which is what the AWQ recipe (Phase 2) compiles
its skip policy from.

**Layer container path:** `language_model.backbone.layers` (52 entries,
`NemotronHBlock` each).  *Not* `language_model.model.layers` — the
NemotronH wrapper uses `backbone`.

**Per-layer structure:** every block has `norm` (`NemotronHRMSNorm`) plus a
polymorphic `mixer`:

| `hybrid_override_pattern` token | `mixer` class | inner Linears (dotted name) | shape |
|---|---|---|---|
| `M` (23×) Mamba2 | `NemotronHMamba2Mixer` | `mixer.in_proj` | `(10304, 2688)` |
|  |  | `mixer.out_proj` | `(2688, 4096)` |
|  |  | `mixer.conv1d` (Conv1d, kernel 4, groups 6144) | `(6144, 1, 4)` |
|  |  | `mixer.norm` (`Zamba2RMSNormGated`) | rank-1 |
| `E` (23×) MoE-MLP | `NemotronHMoE` | `mixer.gate` (`NemotronHTopkRouter`) | `(128, 2688)` |
|  |  | `mixer.experts.<j>.up_proj` for `j` in `[0..127]` | `(1856, 2688)` |
|  |  | `mixer.experts.<j>.down_proj` for `j` in `[0..127]` | `(2688, 1856)` |
|  |  | `mixer.shared_experts.up_proj` | `(3712, 2688)` |
|  |  | `mixer.shared_experts.down_proj` | `(2688, 3712)` |
| `*` (6×) Attention | `NemotronHAttention` | `mixer.q_proj` | `(4096, 2688)` |
|  |  | `mixer.k_proj` (GQA narrow) | `(256, 2688)` |
|  |  | `mixer.v_proj` (GQA narrow) | `(256, 2688)` |
|  |  | `mixer.o_proj` | `(2688, 4096)` |

Per-class top-30 from `module_inspection.txt`: `6,355 Linear`,
`2,967 NemotronHMLP` (= 23 layers × (128 routed + 1 shared) experts),
`2,967 ReLUSquaredActivation`, `52 NemotronHBlock`,
`23 NemotronHMamba2Mixer`, `23 NemotronHMoE`, `23 NemotronHTopkRouter`,
`6 NemotronHAttention`, `24 ParakeetEncoderBlock`, `2 RADIOModel`.

**MoE expert layout — UNFUSED, no `gate_proj`.** Each routed expert is a
distinct `NemotronHMLP` whose only Linears are `up_proj` and `down_proj`.
**There is no `gate_proj`** — the relu² (squared-ReLU) activation is
un-gated, so the MLP is just `down_proj(relu²(up_proj(x)))`.  Compared to
the qwen3.5-MoE 3-D fused layout (`experts.gate_up_proj` `[E, 2I, H]`),
Nemotron is per-expert 2-D, which means the AWQ recipe **does not need
the per-expert unfusing step** the qwen3 recipe carries.

Total expert-related parameters: 5,934 = 23 layers × (128 routed × 2 +
shared × 2) = 23 × (256 + 2) = 5,934. ✓

**Top-level multimodal modules:**

| dotted path | class | role |
|---|---|---|
| `language_model` | `NemotronHForCausalLM` | the LM (backbone + lm_head) |
| `language_model.backbone` | `NemotronHModel` | embeddings + 52 NemotronHBlock |
| `language_model.backbone.embeddings` | `Embedding(131072, 2688)` | input token embedding |
| `language_model.lm_head` | `Linear(131072, 2688)` | output head — **untied** despite `tie_word_embeddings=false` |
| `vision_model` | `RADIOModel` | CRADIO v2-H ViT (32 blocks, `Block`/`Attention`/`Mlp`) |
| `sound_encoder` | `SoundEncoder` (Parakeet-style) | 24 `ParakeetEncoderBlock` |
| `sound_projection` | `SoundProjection` | audio→LM bridge: `linear1` (4096←1024) + `SquaredReLU` + `linear2` (2688←4096) + `RMSNorm` |

There is no separate `vision_projector` — the RADIO output is consumed by
the LM directly via the model's `_get_visual_embeddings` path (custom
`modeling.py`).

**AWQ skip policy** (`recipes/_classify.py::should_quantize`).  Mirrors
NVIDIA's NVFP4 keep-dense policy (mamba + attention + encoders dense;
only routed experts get aggressive bits).  Quantize **iff** the tensor:

- has rank 2 or 3, AND
- ends with `.weight`, `.gate_up_proj`, `.up_proj`, `.gate_proj`, or
  `.down_proj`, AND
- does NOT end with `mixer.gate.weight` / `mlp.gate.weight` /
  `block_sparse_moe.gate.weight` (router gates), AND
- does NOT contain (case-insensitive substring): `lm_head`, `embed_tokens`,
  `embedding`, `embed`, `norm`, `layernorm`, `rmsnorm`, `vision`, `radio`,
  `vision_model`, `vision_tower`, `image_proj`, `video`, `sound`, `audio`,
  `parakeet`, `audio_encoder`, `projector`, `projection`, `mamba`, `ssm`,
  `in_proj`, `out_proj`, `dt_proj`, `conv1d`, `self_attn`, `q_proj`,
  `k_proj`, `v_proj`, `o_proj`, `shared_expert`, AND
- does NOT match `.*\.layers\.0\..*` (layer-0 conservative preservation —
  layer 0 happens to be Mamba in this model so this is doubly-redundant
  but the rule stays for portability).

The only weights this lets through are
`language_model.backbone.layers.<i>.mixer.experts.<j>.{up,down}_proj.weight`
for `i ∈ {1, 3, 6, 8, 10, 13, …}` (the 23 MoE layers, none of them
layer 0) and `j ∈ [0..127]` — i.e. **23 × 128 × 2 = 5,888 routed-expert
weights** out of 6,355 total Linears.

The classifier and its policy are unit-tested in
[`recipes/test_module_classify.py`](./recipes/test_module_classify.py)
with 32 synthetic cases + 25 real names extracted from the inspection
output (60 assertions total).  Run:

```bash
source .venv/bin/activate
pytest runs/nemotron-3-nano-omni-30b-a3b/recipes/test_module_classify.py -v
```

## Reproducing

**SRC_DIR** (filled in Phase 1): `$PWD/hf-cache/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/snapshots/d15962741057ae3a07147df504060e9f0838224e`
**NVFP4_DIR** (filled in Phase 1): `$PWD/hf-cache/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4/snapshots/889396e9cebaefdb69a469afc7bd111660f78eff`

> Note: `--cache-dir <DIR>` puts snapshots at `<DIR>/models--<org>--<name>/...`
> (no `hub/` sub-prefix, unlike when `HF_HOME` is the controlling env var).
> The paths above reflect the *actual* on-disk layout from the Phase 1
> downloads.

```bash
# 1. Pre-cache the bf16 source and the NVFP4 build (Phase 1).
#    `huggingface-cli` is deprecated; use `hf` (huggingface_hub ≥ 0.26).
HF_HUB_ENABLE_HF_TRANSFER=1 hf download nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
  --cache-dir "$PWD/hf-cache" --max-workers 8
HF_HUB_ENABLE_HF_TRANSFER=1 hf download nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 \
  --cache-dir "$PWD/hf-cache" --max-workers 8

# 2. AWQ quantize (Phase 2). Output goes under artifacts/ (gitignored).
export SRC_DIR="$PWD/hf-cache/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/snapshots/d15962741057ae3a07147df504060e9f0838224e"
export DST_DIR="$PWD/artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4"
tools/run_under_memcap.sh python runs/nemotron-3-nano-omni-30b-a3b/recipes/awq_gemm.py

# 3. Serve & smoke (Phase 2 verification).
tools/serve_vllm_docker.sh "$DST_DIR" \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --reasoning-parser nemotron_v3 \
  --served-model-name nemotron-omni-awq

# 4. Eval (Phases 3, 4, 5).
tools/run_eval_full.sh nemotron-omni-awq "$DST_DIR" \
  "$PWD/runs/nemotron-3-nano-omni-30b-a3b/results/awq_full"
```

## Publishing

The artifact directories are uploaded to Hugging Face per
[`HUGGINGFACE_PUBLISHING.md`](../../HUGGINGFACE_PUBLISHING.md).

---

## AWQ build (Phase 2 outputs)

**Artifact**: `artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4/`
**Recipe**: [`recipes/awq_gemm.py`](./recipes/awq_gemm.py) (uses [`recipes/_classify.py`](./recipes/_classify.py) for the skip policy)
**Total size**: 21.55 GiB across 6 shards
**Quantized tensors**: 5,888 (= 23 MoE layers × 128 experts × 2 projections — `up_proj` + `down_proj`)
**Copied dense**: 1,461 (everything else: Mamba2, attention, vision, audio, projectors, layer 0, router, lm_head, embeds, norms, shared experts)
**Quantization wall-clock**: ~7 min (pid 81819, scope `run-rb567669de3f2415f92910cedabc9c08b.scope`, peak RSS 22.4 GiB during shard 8)
**vLLM cold-start**: 220 s (3 min 40 s) on `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:20260415` with `--gpu-memory-utilization 0.55`

### Group-size deviation

Nemotron's `moe_intermediate_size = 1856 = 64 × 29` is not divisible by the
canonical AWQ-GEMM group size of 128. Each routed-expert `down_proj` has
`in_features=1856`, so per-group quant requires `group_size ∈ {32, 64}`.
We use **`group_size = 64`** — the largest value that divides both 1856 and
2688 (the up_proj `in_features`). The plan asserted vLLM's AWQ-GEMM kernel
supports `{32, 64, 128}`; **smoke-test results below cast doubt on whether
gs=64 is actually correctly handled by vLLM's `moe_wna16` path on this
arch — needs verification before re-issuing.**

### `mlp1` postscript

Phase 1's keyword-filtered module inspection missed the top-level vision
projector `mlp1` (an `nn.Sequential` at the wrapper level — RADIO→LM
bridge, `mlp1.{1,3}.weight`). The Phase 2 preflight tally caught the
discrepancy (5,890 vs expected 5,888) **before** the full quantization ran;
`mlp1` was added to both `_classify.SKIP_SUBSTRINGS_CI` and the recipe's
`modules_to_not_convert`. Lesson for future runs: the inspect script's
keyword filter alone misses modules outside the conventional `vision_*` /
`*_projector` namings — add an explicit "top-level direct children" probe
to `inspect_modules.py`.

### `--gpu-memory-utilization` deviation

Plan called for `--gpu-memory-utilization 0.85`, but Spark's GB10 reports
~76.88 GiB of 121.69 GiB free at serve time (the rest is held by
unidentified consumers — `nvidia-smi` shows "Memory-Usage: Not Supported"
on this unified-memory arch, so we can't enumerate them). vLLM refuses to
start when target utilization exceeds free memory. Lowered to **0.55**
(~67 GiB target) to fit. The eval phases (3–5) will need the same
adjustment.

### Quantization config (in artifact `config.json`)

```json
{
  "bits": 4,
  "group_size": 64,
  "quant_method": "awq",
  "version": "gemm",
  "zero_point": true,
  "modules_to_not_convert": [
    "lm_head", "embed_tokens", "embedding",
    "norm", "layernorm", "rmsnorm",
    "vision", "radio", "vision_model", "vision_tower", "image_proj", "video",
    "sound", "audio", "parakeet", "audio_encoder",
    "projector", "projection", "mlp1",
    "mamba", "ssm", "in_proj", "out_proj", "dt_proj", "conv1d",
    "self_attn", "q_proj", "k_proj", "v_proj", "o_proj",
    "shared_expert",
    "mixer.gate", "mlp.gate",
    ".layers.0."
  ]
}
```

### Smoke test

```bash
tools/serve_vllm_docker.sh "$PWD/artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4" \
  --kv-cache-dtype fp8_e4m3 --max-model-len 8192 --gpu-memory-utilization 0.55 \
  --reasoning-parser nemotron_v3 --served-model-name nemotron-omni-awq
```

**Verdict: ✗ FAILED — model produces degenerate looped output on every prompt.**

| probe | prompt | response | expected |
|---|---|---|---|
| chat /v1/chat/completions | `Solve: 17 * 23 = ?  Briefly.` | `17 * 23 = 17 * 23 = ...` × 32 reps in `reasoning` field, `content: null`, `finish_reason: length` | `391` |
| chat /v1/chat/completions | `Hello` | `1\n2\n3\n4\n5\n6\n...\n24\n2` (counting digits, in `reasoning` field) | a greeting |
| bare /v1/completions | `The capital of France is` | ` a capital of France, but it is a capital of France, but it is a capital of France` | `Paris` |

The model has lost all semantic capability — failure is at the weights /
loader level, not specific to the math prompt or the reasoning parser.
Full failure record: `results/awq_smoke.txt`.

**vLLM version observed**: `0.19.1rc1.dev322+g03f8d3a54.d20260415.cu132` —
**below the model card's required vLLM ≥ 0.20.0**. The plan's Risks
section explicitly flagged this: `--reasoning-parser nemotron_v3` is
*accepted* by 0.19.1.dev (no error), but Nemotron-Nano-Omni multimodal
+ AWQ may not be fully implemented in this dev build. **This is the
single most likely root cause** — pull a newer Spark vLLM image
(≥ 0.20.0) and re-run the smoke before assuming the recipe is broken.

**Possible root causes** (must investigate before re-issuing):

1. vLLM's AWQ-GEMM kernel may not actually support `group_size=64` for the
   Nemotron MoE arch via `moe_wna16` — could be silently producing garbage.
2. AWQ pack order `[0,4,1,5,2,6,3,7]` may not match vLLM's reverse-order
   for unfused per-expert AWQ tensors (the qwen3 precedent used fused
   `gate_up_proj` + `down_proj` 3-D layouts; Nemotron's per-expert 2-D
   layout may need a different convention).
3. `modules_to_not_convert` substring matching in vLLM's loader may not
   behave identically to recipe's `_classify.should_quantize` — vLLM might
   quantize a critical Linear that the recipe wrote dense, or expect packed
   AWQ in a slot where we wrote bf16.
4. State-dict key naming mismatch — vLLM's `hf_to_vllm_mapper` for
   `NemotronH_Nano_Omni_Reasoning_V3` may rewrite expert names; on-disk
   names may not align with what the loader expects, causing weights to
   land in wrong slots.

**Suggested next diagnostic**: serve the BF16 source through this same
vLLM image first (Phase 3 will do this anyway). If BF16 also loops,
something is wrong with vLLM/parser. If BF16 is clean, the AWQ recipe is
the bug — narrow further by re-quantizing with `group_size=32` (also
divides 1856) to test cause #1.
