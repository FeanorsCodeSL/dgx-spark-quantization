# DGX Spark Quantization Plan: Spark-Specific AutoAWQ INT4 + FP8 Baseline

> **Historical document.** This is the planning blueprint that was written *before*
> the work started, when the project lived in a flat working tree called
> "Project Distill". Path references here ("Project Distill", absolute paths
> under `/home/sergio/...`) reflect that original layout. Within this
> framework repo the equivalents are:
>
> | original | now |
> |---|---|
> | `Project Distill/quantize_qwen36_distilled_fp8.py` | `runs/qwen3.6-35b-distill/recipes/fp8_dynamic.py` |
> | `Project Distill/quantize_awq_gemm_streaming.py` | `runs/qwen3.6-35b-distill/recipes/awq_gemm.py` |
> | `Project Distill/run_eval_full.sh` | `tools/run_eval_full.sh` |
> | `Project Distill/scripts/run_quantize_safe.sh` | `tools/run_under_memcap.sh` |
> | `Project Distill/scripts/serve_vllm_docker.sh` | `tools/serve_vllm_docker.sh` |
> | `Project Distill/eval_results/<run>/...` | `runs/qwen3.6-35b-distill/results/<run>/...` |
>
> See [`REPORT.md`](./REPORT.md) for the actual outcome (with three-way
> bf16 / FP8 / AWQ deltas).

This is the Spark-specific execution plan for:

```text
lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled
```

It incorporates the constraints already observed on this DGX Spark:

```text
Bare-metal vLLM in the venv is not assumed to work on aarch64 / SM121a.
vLLM serving should use the known-good Docker image.
Do not run risky BF16 device_map="auto" validation loads.
Use the existing Project Distill venv.
Use the existing Project Distill HF cache.
Run AutoRound behind a cgroup memory guard.
Treat AutoAWQ as the primary INT4 export path.
Treat native AutoRound as fallback.
```

---

## 0. Fixed local assumptions

Project root:

```bash
export PROJECT_ROOT="/home/sergio/git/Project Distill"
```

Existing virtual environment:

```bash
export VENV="$PROJECT_ROOT/.venv"
```

Existing Hugging Face cache:

```bash
export HF_HOME="$PROJECT_ROOT/hf-cache"
```

Original BF16 Hugging Face model:

```bash
export BF16_MODEL="lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled"
```

Existing FP8 checkpoint:

```bash
export FP8_MODEL="$PROJECT_ROOT/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-FP8-Dynamic"
```

Primary INT4 AutoAWQ output:

```bash
export AWQ_HQ_MODEL="$PROJECT_ROOT/qwen36-35b-distill-autoround-awq-int4-highquality"
```

Smoke-test INT4 AutoAWQ output:

```bash
export AWQ_SMOKE_MODEL="$PROJECT_ROOT/qwen36-35b-distill-autoround-awq-int4-smoke"
```

Native AutoRound fallback output:

```bash
export AUTOROUND_NATIVE_MODEL="$PROJECT_ROOT/qwen36-35b-distill-autoround-int4"
```

Known-good vLLM Docker image:

```bash
export VLLM_IMAGE="ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:20260415"
```

Set these at the start of every terminal session:

```bash
export PROJECT_ROOT="/home/sergio/git/Project Distill"
export VENV="$PROJECT_ROOT/.venv"
export HF_HOME="$PROJECT_ROOT/hf-cache"
export BF16_MODEL="lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled"
export FP8_MODEL="$PROJECT_ROOT/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-FP8-Dynamic"
export AWQ_HQ_MODEL="$PROJECT_ROOT/qwen36-35b-distill-autoround-awq-int4-highquality"
export AWQ_SMOKE_MODEL="$PROJECT_ROOT/qwen36-35b-distill-autoround-awq-int4-smoke"
export AUTOROUND_NATIVE_MODEL="$PROJECT_ROOT/qwen36-35b-distill-autoround-int4"
export VLLM_IMAGE="ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:20260415"

cd "$PROJECT_ROOT"
source "$VENV/bin/activate"
```

---

## 1. Execution strategy

Do **not** launch the high-quality multi-hour INT4 job first.

Use this sequence:

```text
1. Activate the known-good Project Distill venv.
2. Verify torch/CUDA in the existing venv without reinstalling torch.
3. Create a cgroup-guarded quantization wrapper.
4. Create a Docker vLLM serving wrapper.
5. Verify AutoRound CLI flags on the installed version.
6. Inspect model module names with CPU-only loading under the cgroup guard.
7. Decide whether router / lm_head / DeltaNet / linear-attention exclusions are needed.
8. Run a tiny AutoRound -> AutoAWQ smoke quantization under the cgroup guard.
9. Try loading the smoke output with the known-good vLLM Docker image.
10. If smoke passes, run the high-quality AutoRound -> AutoAWQ job under the cgroup guard.
11. Validate the high-quality output with Docker vLLM.
12. If AutoAWQ fails, try native AutoRound as fallback.
```

The target high-quality recipe is:

```text
bits:       4
group_size: 128
iters:      1000
nsamples:   512
format:     auto_awq
```

This is slower, but not automatically much more OOM-prone than the smaller recipe. Memory pressure is controlled mainly by:

```text
batch size
sequence length
model placement/offload
low_gpu_mem_usage
gradient accumulation
other active GPU/CPU-memory workloads
```

On DGX Spark, the most important pressure control is the cgroup wrapper, because unified memory makes normal GPU-memory reporting less reliable.

---

## 2. Do not reinstall torch blindly

Use the existing environment:

```bash
cd "$PROJECT_ROOT"
source "$VENV/bin/activate"
export HF_HOME="$PROJECT_ROOT/hf-cache"
```

Check the current torch build:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

Do **not** run:

```bash
pip install -U torch
```

A blank torch install may pull CPU-only wheels.

If a package needs to be added, avoid disturbing torch. First inspect current packages:

```bash
python -m pip freeze | tee pip-freeze-before-autoround.txt
```

Check AutoRound availability:

```bash
which auto-round || true
auto-round --help | tee autoround_help.txt
```

If AutoRound is missing, install it cautiously:

```bash
python -m pip install -U auto-round
```

Then immediately re-check torch:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

If torch changes or CUDA disappears, stop and restore the known-good venv.

---

## 3. Cgroup guard for AutoRound

This wrapper reproduces the harness that finally produced a successful FP8 run after several Spark hangs. Properties used:

```text
MemoryAccounting=yes      account this scope's memory
MemoryMax=112G            hard cap; kernel kills the scope's processes before host OOM
MemoryHigh=100G           soft cap; kernel reclaims aggressively past this point
MemorySwapMax=0           no swap-out; failures fail fast instead of paging the host
TasksMax=infinity         disable the default user-scope task ceiling that PyTorch threading can hit
```

`OOMScoreAdjust=1000` was tried in the FP8 run and rejected by user-scope `systemd-run`; it is intentionally not in the wrapper. If `systemctl --user` is not available, the wrapper falls back to `sudo systemd-run --scope`.

Create the wrapper:

```bash
mkdir -p "$PROJECT_ROOT/scripts"
cat > "$PROJECT_ROOT/scripts/run_quantize_safe.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

MEMORY_MAX="${MEMORY_MAX:-112G}"
MEMORY_HIGH="${MEMORY_HIGH:-100G}"
MEMORY_SWAP_MAX="${MEMORY_SWAP_MAX:-0}"
TASKS_MAX="${TASKS_MAX:-infinity}"

if [[ $# -eq 0 ]]; then
  echo "Usage: MEMORY_MAX=112G MEMORY_HIGH=100G $0 <command> [args...]" >&2
  exit 2
fi

if ! command -v systemd-run >/dev/null 2>&1; then
  echo "systemd-run not found; refusing to run without cgroup guard on DGX Spark." >&2
  exit 1
fi

COMMON_PROPS=(
  -p "MemoryAccounting=yes"
  -p "MemoryMax=${MEMORY_MAX}"
  -p "MemoryHigh=${MEMORY_HIGH}"
  -p "MemorySwapMax=${MEMORY_SWAP_MAX}"
  -p "TasksMax=${TASKS_MAX}"
)

if systemctl --user show >/dev/null 2>&1; then
  exec systemd-run --user --scope --collect "${COMMON_PROPS[@]}" "$@"
else
  exec sudo systemd-run --scope --collect "${COMMON_PROPS[@]}" "$@"
fi
SH

chmod +x "$PROJECT_ROOT/scripts/run_quantize_safe.sh"
```

Sanity test:

```bash
MEMORY_MAX=112G MEMORY_HIGH=100G "$PROJECT_ROOT/scripts/run_quantize_safe.sh" bash -lc 'echo cgroup wrapper ok'
```

Verify the cgroup is actually applied on a live process:

```bash
MEMORY_MAX=112G MEMORY_HIGH=100G "$PROJECT_ROOT/scripts/run_quantize_safe.sh" bash -lc '
  cat /proc/self/cgroup
  echo "---"
  for f in memory.max memory.high memory.swap.max pids.max; do
    p="/sys/fs/cgroup$(awk -F: "{print \$3}" /proc/self/cgroup)/$f"
    echo "$f: $(cat "$p" 2>/dev/null || echo missing)"
  done
'
```

Expected: `memory.max=120259084288` (≈112G), `memory.high≈107374182400` (100G), `memory.swap.max=0`, `pids.max=max`.

Use this wrapper for:

```text
module inspection that loads BF16 weights on CPU
AutoRound smoke quantization
AutoRound high-quality quantization
native AutoRound fallback quantization
```

Do not run AutoRound directly on the Spark without this guard.

### 3.1 Allocator knob for quantization runs

The successful FP8 run also used:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This reduces fragmentation when allocating large tensors against unified memory. Every quantization shell in §10/§13/§14/§16/§19 exports it before running `auto-round`.

---

## 4. Docker wrapper for vLLM serving

Create a wrapper so every vLLM serve command uses the known-good image.

```bash
mkdir -p "$PROJECT_ROOT/scripts"
cat > "$PROJECT_ROOT/scripts/serve_vllm_docker.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model-path-or-hf-id> [vllm args...]" >&2
  exit 2
fi

MODEL="$1"
shift

PROJECT_ROOT="${PROJECT_ROOT:-/home/sergio/git/Project Distill}"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/hf-cache}"
VLLM_IMAGE="${VLLM_IMAGE:-ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:20260415}"

docker run --rm \
  --gpus all \
  --ipc=host \
  --network host \
  --mount "type=bind,source=${PROJECT_ROOT},target=${PROJECT_ROOT}" \
  -w "$PROJECT_ROOT" \
  -e "HF_HOME=${HF_HOME}" \
  -e "TRANSFORMERS_CACHE=${HF_HOME}" \
  -e "HF_HUB_CACHE=${HF_HOME}/hub" \
  "$VLLM_IMAGE" \
  vllm serve "$MODEL" \
    --trust-remote-code \
    "$@"
SH

chmod +x "$PROJECT_ROOT/scripts/serve_vllm_docker.sh"
```

From now on, do **not** use bare:

```bash
vllm serve ...
```

Use:

```bash
"$PROJECT_ROOT/scripts/serve_vllm_docker.sh" "$MODEL_PATH" ...
```

The wrapper runs in the foreground with `--rm` so Ctrl-C exits cleanly. To background a long-lived serve and inspect logs later, swap `--rm` for `-d --name vllm_qwen36_int4` and tail with `docker logs -f vllm_qwen36_int4`. Tear down with `docker rm -f vllm_qwen36_int4`.

---

## 5. Skip risky BF16 GPU validation

Do not run a separate BF16 validation with:

```python
device_map="auto"
```

That call pattern has already hung the Spark during FP8 work.

Instead, use only lightweight metadata/tokenizer checks and the CPU-only module inspector in the scouting phase.

Lightweight check:

```bash
python - <<'PY'
from transformers import AutoConfig, AutoTokenizer
import os

model_id = os.environ["BF16_MODEL"]

cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print("config class:", cfg.__class__.__name__)
print("architectures:", getattr(cfg, "architectures", None))
print("tokenizer class:", tok.__class__.__name__)
PY
```

---

## 6. Validate the existing FP8 checkpoint with Docker vLLM

Start conservatively:

```bash
"$PROJECT_ROOT/scripts/serve_vllm_docker.sh" "$FP8_MODEL" \
  --dtype auto \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.80
```

In another terminal:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$FP8_MODEL"'",
    "messages": [
      {"role": "user", "content": "Explain in one sentence what quantization does."}
    ],
    "max_tokens": 80,
    "temperature": 0
  }'
```

If stable, test:

```bash
--max-model-len 32768
```

On Spark unified memory, `--gpu-memory-utilization` is a soft runtime hint. The cgroup wrapper is the real protection for quantization jobs. vLLM serving should still be tested conservatively.

---

## 7. AutoRound CLI scouting

Do not assume CLI syntax across AutoRound versions.

Run:

```bash
auto-round --version || true
auto-round --help | tee "$PROJECT_ROOT/autoround_help.txt"
```

Search for relevant flags:

```bash
grep -iE "auto_awq|format|bits|group|iters|nsamples|fp_layers|exclude|low_gpu_mem|batch|gradient|seq" "$PROJECT_ROOT/autoround_help.txt" || true
```

Confirm the installed version supports, or has equivalents for:

```text
--bits
--group_size
--iters
--nsamples
--format auto_awq
```

Trust-remote-code: in v0.10.2 there is no `--trust_remote_code` flag; trust is on by default and the inverse is `--disable_trust_remote_code`. Do not pass `--trust_remote_code`; it will fail argparse.

Also check whether it supports an exclusion/full-precision flag such as:

```text
--fp_layers   (alias of --ignore_layers in v0.10.2)
```

And — important on Spark unified memory — look for device / placement flags such as:

```text
--device              (e.g. cpu, cuda, cuda:0)
--device_map          (auto, balanced, cpu, etc.)
--low_gpu_mem_usage   (off-loads weights between calibration steps)
--batch_size
--gradient_accumulate_steps
```

If a `--device cpu` or `--low_gpu_mem_usage` knob exists, plan to use it for the smoke run and the high-quality run. AutoRound's default of pulling everything to GPU is the same call pattern that hung the box during FP8 work; the cgroup wrapper will catch a runaway, but avoiding it is better than recovering from it.

If names differ, use the installed version's help output as the source of truth.

---

## 8. CPU-only module inspection under cgroup guard

Create the inspector:

```bash
cat > "$PROJECT_ROOT/inspect_qwen_modules.py" <<'PY'
from transformers import AutoModelForCausalLM
import os

MODEL_ID = os.environ["BF16_MODEL"]

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map={"": "cpu"},
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

interesting = []
linear_count = 0

for name, module in model.named_modules():
    cls_name = module.__class__.__name__
    cls = cls_name.lower()
    lname = name.lower()

    if cls_name == "Linear":
        linear_count += 1

    if any(x in lname or x in cls for x in [
        "router",
        "gate",
        "expert",
        "moe",
        "lm_head",
        "norm",
        "deltanet",
        "delta",
        "linear_attn",
        "linear_attention",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]):
        interesting.append((name, cls_name))

print("Total Linear modules:", linear_count)

print("\nInteresting modules:")
for name, cls_name in interesting[:3000]:
    print(f"{name} :: {cls_name}")

print("\nLast 100 interesting modules:")
for name, cls_name in interesting[-100:]:
    print(f"{name} :: {cls_name}")
PY
```

Run it under cgroup guard:

```bash
MEMORY_MAX=112G MEMORY_HIGH=100G "$PROJECT_ROOT/scripts/run_quantize_safe.sh" bash -lc '
set -euo pipefail
export PROJECT_ROOT="/home/sergio/git/Project Distill"
export VENV="$PROJECT_ROOT/.venv"
export HF_HOME="$PROJECT_ROOT/hf-cache"
export BF16_MODEL="lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled"
cd "$PROJECT_ROOT"
source "$VENV/bin/activate"
python "$PROJECT_ROOT/inspect_qwen_modules.py" | tee "$PROJECT_ROOT/module_inspection.txt"
'
```

Review:

```bash
grep -iE "lm_head|router|deltanet|linear_attn|linear_attention|moe|expert|gate" "$PROJECT_ROOT/module_inspection.txt" | head -300
```

---

## 9. Exclusion policy before smoke quantization

### 9.0 Findings from §8 module inspection (2026-04-27)

The BF16 model loads as `Qwen3_5MoeTextModel` (not the multimodal class), with 351 Linear modules organized as:

```text
lm_head                                        :: Linear   (skipped by default unless --quant_lm_head)
model.layers.{i}.self_attn.{q,k,v,o}_proj      :: Linear   (10 full-attn layers: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39)
model.layers.{i}.linear_attn.{in_proj_a,in_proj_b,in_proj_qkv,in_proj_z,out_proj} :: Linear (30 DeltaNet layers, all others)
model.layers.{i}.mlp.shared_expert.{gate_proj,up_proj,down_proj}                  :: Linear (all 40 layers)
model.layers.{i}.mlp.shared_expert_gate                                           :: Linear (all 40 layers)
model.layers.{i}.mlp.gate                      :: Qwen3_5MoeTopKRouter  (NOT Linear -> auto-skipped)
model.layers.{i}.mlp.experts                   :: Qwen3_5MoeExperts     (NOT Linear -> auto-skipped; this is the 256-expert FUSED module)
```

Counts: 10*4 + 30*5 + 40*3 + 40*1 + 1 = 351. Matches inspection.

### 9.1 Critical architectural caveat: fused experts

`Qwen3_5MoeExperts` is a custom fused module. The 256 expert weights are stored as 3D tensors (e.g., `experts.gate_up_proj` of shape `[256, ...]`), not as 256 individual `nn.Linear`. AutoRound walks `nn.Linear`. Therefore the smoke run will reveal one of two outcomes:

```text
A) AutoRound + AutoAWQ skip the experts entirely. The output is "INT4" only on attention + shared_expert + DeltaNet (~10% of the 35B params). The bulk of the model stays BF16. The model still loads but is not actually small.
B) AutoRound has special handling for fused MoE (via --shared_layers or trust_remote_code-driven hooks) and quantizes the fused tensor as a single target.
```

The smoke run resolves which outcome holds. Decide whether to proceed to HQ based on the smoke output's on-disk size and weight-class breakdown, NOT just on whether AutoAWQ wrote files.

### 9.2 Exclusion decisions for the smoke run

```text
lm_head:        not needed in --fp_layers; AutoRound default keeps it full-precision.
router:         mlp.gate is class Qwen3_5MoeTopKRouter (not Linear) -> auto-skipped. No manual exclusion needed.
DeltaNet:       these ARE Linears and will be quantized by default. Do NOT exclude in smoke; observe quality. Reconsider for HQ if smoke output produces obvious garbage.
shared_expert.gate_proj: legitimate MLP gate projection. Do NOT exclude.
shared_expert_gate: small Linear for mixing the shared expert; quantizable but tiny. Do NOT exclude.
```

Smoke run: omit `--fp_layers` entirely. Use AutoRound defaults.

### 9.3 Original guidance (kept for reference)

Start with these assumptions:

```text
lm_head should stay full precision.
explicit router modules should stay full precision.
explicit DeltaNet / linear-attention custom blocks should stay full precision if they contain Linears that AutoRound would otherwise touch.
norms are usually not quantized by Linear-only walkers, but verify.
do not blindly exclude every name containing "gate".
```

Important distinction:

```text
router/gating logic for expert selection: exclude if present as Linear/router modules.
MLP gate projection: often a normal quantizable Linear; do not exclude just because the name contains "gate".
```

Do not manually assign different bit widths to fused MoE/QKV layers.

If the installed AutoRound supports `--fp_layers`, build an exclusion string from the actual names found in `module_inspection.txt`.

Example only:

```bash
export AUTOROUND_FP_LAYERS="lm_head,.*router.*,.*deltanet.*,.*linear_attn.*,.*linear_attention.*"
```

If the installed version does not support `--fp_layers`, omit it and rely on AutoRound's defaults for the smoke test.

---

## 10. Tiny AutoRound -> AutoAWQ smoke quantization

This is not for quality. It tests:

```text
AutoRound can walk the architecture.
AutoAWQ export completes.
State-dict keys are loadable.
Docker vLLM can load the result.
No immediate NaNs/crashes.
```

Run the smoke job under the cgroup guard.

> Note: `iters=1 nsamples=4` is intentionally minimal. If AutoRound enforces a higher minimum and exits early, bump to `iters=10 nsamples=32` and re-run. This is purely about exercising the pipeline, not quality.

Without explicit fp_layers:

```bash
MEMORY_MAX=112G MEMORY_HIGH=100G "$PROJECT_ROOT/scripts/run_quantize_safe.sh" bash -lc '
set -euo pipefail
export PROJECT_ROOT="/home/sergio/git/Project Distill"
export VENV="$PROJECT_ROOT/.venv"
export HF_HOME="$PROJECT_ROOT/hf-cache"
export BF16_MODEL="lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled"
export AWQ_SMOKE_MODEL="$PROJECT_ROOT/qwen36-35b-distill-autoround-awq-int4-smoke"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_ROOT"
source "$VENV/bin/activate"

auto-round \
  --model "$BF16_MODEL" \
  --bits 4 \
  --group_size 128 \
  --iters 1 \
  --nsamples 4 \
  --format auto_awq \
  --output_dir "$AWQ_SMOKE_MODEL"
'
```

With explicit `--fp_layers`, if supported:

```bash
MEMORY_MAX=112G MEMORY_HIGH=100G "$PROJECT_ROOT/scripts/run_quantize_safe.sh" bash -lc '
set -euo pipefail
export PROJECT_ROOT="/home/sergio/git/Project Distill"
export VENV="$PROJECT_ROOT/.venv"
export HF_HOME="$PROJECT_ROOT/hf-cache"
export BF16_MODEL="lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled"
export AWQ_SMOKE_MODEL="$PROJECT_ROOT/qwen36-35b-distill-autoround-awq-int4-smoke"
export AUTOROUND_FP_LAYERS="lm_head,.*router.*,.*deltanet.*,.*linear_attn.*,.*linear_attention.*"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_ROOT"
source "$VENV/bin/activate"

auto-round \
  --model "$BF16_MODEL" \
  --bits 4 \
  --group_size 128 \
  --iters 1 \
  --nsamples 4 \
  --format auto_awq \
  --fp_layers "$AUTOROUND_FP_LAYERS" \
  --output_dir "$AWQ_SMOKE_MODEL"
'
```

If the CLI rejects `--fp_layers`, remove it and use the installed version's equivalent if one exists.

---

## 11. Inspect smoke output key structure

```bash
find "$AWQ_SMOKE_MODEL" -maxdepth 1 -type f -print
```

Inspect safetensor keys:

```bash
python - <<'PY'
from safetensors import safe_open
import glob
import os

model_dir = os.environ["AWQ_SMOKE_MODEL"]
paths = glob.glob(os.path.join(model_dir, "*.safetensors"))

print("safetensors files:", len(paths))

if paths:
    with safe_open(paths[0], framework="pt") as f:
        keys = list(f.keys())
        print("First 100 keys:")
        for k in keys[:100]:
            print(k)
PY
```

Check whether keys use the prefix structure expected by the model class loaded by vLLM, for example:

```text
model.language_model.layers...
```

versus:

```text
model.layers...
```

If the prefix differs from what worked during FP8, use the same rekeying approach from the FP8 work before launching the full AutoAWQ job.

---

## 12. Validate smoke output with Docker vLLM

Use Docker vLLM, not bare vLLM.

```bash
"$PROJECT_ROOT/scripts/serve_vllm_docker.sh" "$AWQ_SMOKE_MODEL" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.80
```

In another terminal:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$AWQ_SMOKE_MODEL"'",
    "messages": [
      {"role": "user", "content": "Say OK if you can read this."}
    ],
    "max_tokens": 16,
    "temperature": 0
  }'
```

Proceed only if:

```text
Docker vLLM starts.
The model loads.
The API responds.
No NaN/crash/kernel error appears.
The output is at least structurally sane.
```

The smoke output is not expected to be high quality.

---

## 13. High-quality AutoRound -> AutoAWQ quantization

Only run this after the smoke path passes.

Without explicit fp_layers:

```bash
MEMORY_MAX=112G MEMORY_HIGH=100G "$PROJECT_ROOT/scripts/run_quantize_safe.sh" bash -lc '
set -euo pipefail
export PROJECT_ROOT="/home/sergio/git/Project Distill"
export VENV="$PROJECT_ROOT/.venv"
export HF_HOME="$PROJECT_ROOT/hf-cache"
export BF16_MODEL="lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled"
export AWQ_HQ_MODEL="$PROJECT_ROOT/qwen36-35b-distill-autoround-awq-int4-highquality"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_ROOT"
source "$VENV/bin/activate"

auto-round \
  --model "$BF16_MODEL" \
  --bits 4 \
  --group_size 128 \
  --iters 1000 \
  --nsamples 512 \
  --format auto_awq \
  --output_dir "$AWQ_HQ_MODEL"
'
```

With explicit fp_layers, if the smoke test confirmed support and need:

```bash
MEMORY_MAX=112G MEMORY_HIGH=100G "$PROJECT_ROOT/scripts/run_quantize_safe.sh" bash -lc '
set -euo pipefail
export PROJECT_ROOT="/home/sergio/git/Project Distill"
export VENV="$PROJECT_ROOT/.venv"
export HF_HOME="$PROJECT_ROOT/hf-cache"
export BF16_MODEL="lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled"
export AWQ_HQ_MODEL="$PROJECT_ROOT/qwen36-35b-distill-autoround-awq-int4-highquality"
export AUTOROUND_FP_LAYERS="lm_head,.*router.*,.*deltanet.*,.*linear_attn.*,.*linear_attention.*"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_ROOT"
source "$VENV/bin/activate"

auto-round \
  --model "$BF16_MODEL" \
  --bits 4 \
  --group_size 128 \
  --iters 1000 \
  --nsamples 512 \
  --format auto_awq \
  --fp_layers "$AUTOROUND_FP_LAYERS" \
  --output_dir "$AWQ_HQ_MODEL"
'
```

If this OOMs, do not immediately reduce quality. First try memory controls.

---

## 14. If the high-quality AutoAWQ run OOMs

First check for other workloads:

```bash
nvidia-smi
ps aux --sort=-rss | head -30
```

Stop unrelated heavy jobs.

Then try lower memory pressure while preserving the high-quality target. Only use flags supported by the installed AutoRound CLI.

Potential low-memory version:

```bash
MEMORY_MAX=112G MEMORY_HIGH=100G "$PROJECT_ROOT/scripts/run_quantize_safe.sh" bash -lc '
set -euo pipefail
export PROJECT_ROOT="/home/sergio/git/Project Distill"
export VENV="$PROJECT_ROOT/.venv"
export HF_HOME="$PROJECT_ROOT/hf-cache"
export BF16_MODEL="lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled"
export AWQ_HQ_MODEL="$PROJECT_ROOT/qwen36-35b-distill-autoround-awq-int4-highquality-lowmem"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_ROOT"
source "$VENV/bin/activate"

auto-round \
  --model "$BF16_MODEL" \
  --bits 4 \
  --group_size 128 \
  --iters 1000 \
  --nsamples 512 \
  --batch_size 1 \
  --gradient_accumulate_steps 8 \
  --low_gpu_mem_usage \
  --format auto_awq \
  --output_dir "$AWQ_HQ_MODEL"
'
```

If the installed AutoRound version rejects any of these flags, remove unsupported flags and retry.

Step down only in this order:

```text
1. Keep iters=1000 and nsamples=512; add low-memory flags if supported.
2. Keep iters=1000; reduce nsamples from 512 to 256.
3. Keep nsamples=512; reduce seqlen only if you explicitly set a large seqlen.
4. Use iters=200 and nsamples=256 only as a fallback.
```

Fallback medium-quality command:

```bash
MEMORY_MAX=112G MEMORY_HIGH=100G "$PROJECT_ROOT/scripts/run_quantize_safe.sh" bash -lc '
set -euo pipefail
export PROJECT_ROOT="/home/sergio/git/Project Distill"
export VENV="$PROJECT_ROOT/.venv"
export HF_HOME="$PROJECT_ROOT/hf-cache"
export BF16_MODEL="lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled"
export AWQ_HQ_MODEL="$PROJECT_ROOT/qwen36-35b-distill-autoround-awq-int4-medium"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_ROOT"
source "$VENV/bin/activate"

auto-round \
  --model "$BF16_MODEL" \
  --bits 4 \
  --group_size 128 \
  --iters 200 \
  --nsamples 256 \
  --format auto_awq \
  --output_dir "$AWQ_HQ_MODEL"
'
```

---

## 15. Validate high-quality AutoAWQ with Docker vLLM

```bash
"$PROJECT_ROOT/scripts/serve_vllm_docker.sh" "$AWQ_HQ_MODEL" \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.80
```

In another terminal:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$AWQ_HQ_MODEL"'",
    "messages": [
      {"role": "user", "content": "Explain quantization in one paragraph."}
    ],
    "max_tokens": 256,
    "temperature": 0
  }'
```

If stable, test:

```bash
--max-model-len 32768
```

---

## 16. Native AutoRound fallback

Use this only if AutoAWQ export or vLLM loading fails and cannot be fixed with exclusions or rekeying.

```bash
MEMORY_MAX=112G MEMORY_HIGH=100G "$PROJECT_ROOT/scripts/run_quantize_safe.sh" bash -lc '
set -euo pipefail
export PROJECT_ROOT="/home/sergio/git/Project Distill"
export VENV="$PROJECT_ROOT/.venv"
export HF_HOME="$PROJECT_ROOT/hf-cache"
export BF16_MODEL="lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled"
export AUTOROUND_NATIVE_MODEL="$PROJECT_ROOT/qwen36-35b-distill-autoround-int4"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_ROOT"
source "$VENV/bin/activate"

auto-round \
  --model "$BF16_MODEL" \
  --bits 4 \
  --group_size 128 \
  --iters 1000 \
  --nsamples 512 \
  --format auto_round \
  --output_dir "$AUTOROUND_NATIVE_MODEL"
'
```

Serve fallback model:

```bash
"$PROJECT_ROOT/scripts/serve_vllm_docker.sh" "$AUTOROUND_NATIVE_MODEL" \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.80
```

---

## 17. Comparison tests

Compare:

```text
FP8_DYNAMIC baseline
AutoAWQ INT4 high-quality
Native AutoRound INT4 fallback, if produced
```

Use deterministic generation:

```text
temperature: 0
top_p: 1
max_tokens: 256
```

Suggested prompts:

```text
1. Explain quantization in one paragraph.
2. Write a Python function to parse a JSONL file and count records.
3. Solve: if a system has 4 GPUs and each processes 23 tokens/sec, how many tokens in 15 minutes?
4. Give a concise architecture plan for a desktop chat app with local model inference.
5. A bat and ball cost $1.10 total. The bat costs $1 more than the ball. What does each cost?
```

Create test script:

```bash
cat > "$PROJECT_ROOT/test_vllm_model.py" <<'PY'
import os
import sys
from openai import OpenAI

model = sys.argv[1]

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

prompts = [
    "Explain quantization in one paragraph.",
    "Write a Python function to parse a JSONL file and count records.",
    "If a system has 4 GPUs and each processes 23 tokens/sec, how many tokens in 15 minutes?",
    "Give a concise architecture plan for a desktop chat app with local model inference.",
    "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. What does each cost?",
]

for i, prompt in enumerate(prompts, 1):
    print(f"\n--- Prompt {i} ---")
    print(prompt)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        top_p=1,
        max_tokens=256,
    )
    print(resp.choices[0].message.content)
PY
```

Install OpenAI client if missing, without touching torch:

```bash
python -m pip install -U openai
```

Run after serving a model:

```bash
python "$PROJECT_ROOT/test_vllm_model.py" "$AWQ_HQ_MODEL"
```

---

## 18. Pass/fail checklist before full run

Do not run the full high-quality job unless these are true:

```text
Existing .venv torch still has CUDA.
HF_HOME points to Project Distill/hf-cache.
AutoRound CLI help confirms required flags or equivalents.
CPU-only module inspection completed under cgroup guard.
Potential exclusions have been decided from actual module names.
Smoke AutoRound -> AutoAWQ completed under cgroup guard.
Smoke output keys look compatible with the model loader.
Smoke output loads in Docker vLLM.
Smoke output responds to a trivial prompt.
```

---

## 19. Summary commands

After environment exports:

```bash
cd "$PROJECT_ROOT"
source "$VENV/bin/activate"
export HF_HOME="$PROJECT_ROOT/hf-cache"
```

Run smoke:

```bash
MEMORY_MAX=112G MEMORY_HIGH=100G "$PROJECT_ROOT/scripts/run_quantize_safe.sh" bash -lc '
set -euo pipefail
export PROJECT_ROOT="/home/sergio/git/Project Distill"
export VENV="$PROJECT_ROOT/.venv"
export HF_HOME="$PROJECT_ROOT/hf-cache"
export BF16_MODEL="lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled"
export AWQ_SMOKE_MODEL="$PROJECT_ROOT/qwen36-35b-distill-autoround-awq-int4-smoke"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROJECT_ROOT"
source "$VENV/bin/activate"
auto-round \
  --model "$BF16_MODEL" \
  --bits 4 \
  --group_size 128 \
  --iters 1 \
  --nsamples 4 \
  --format auto_awq \
  --output_dir "$AWQ_SMOKE_MODEL"
'
```

Serve smoke:

```bash
"$PROJECT_ROOT/scripts/serve_vllm_docker.sh" "$AWQ_SMOKE_MODEL" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.80
```

Run high-quality:

```bash
MEMORY_MAX=112G MEMORY_HIGH=100G "$PROJECT_ROOT/scripts/run_quantize_safe.sh" bash -lc '
set -euo pipefail
export PROJECT_ROOT="/home/sergio/git/Project Distill"
export VENV="$PROJECT_ROOT/.venv"
export HF_HOME="$PROJECT_ROOT/hf-cache"
export BF16_MODEL="lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled"
export AWQ_HQ_MODEL="$PROJECT_ROOT/qwen36-35b-distill-autoround-awq-int4-highquality"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROJECT_ROOT"
source "$VENV/bin/activate"
auto-round \
  --model "$BF16_MODEL" \
  --bits 4 \
  --group_size 128 \
  --iters 1000 \
  --nsamples 512 \
  --format auto_awq \
  --output_dir "$AWQ_HQ_MODEL"
'
```

Serve high-quality:

```bash
"$PROJECT_ROOT/scripts/serve_vllm_docker.sh" "$AWQ_HQ_MODEL" \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.80
```
