# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A reusable **framework** for quantizing open-weight models on a single
NVIDIA DGX Spark (GB10 / SM121a, 128 GiB unified memory) and shipping
vLLM-loadable artifacts to Hugging Face. The framework lives in
`tools/` + `docs/`; each *base model* gets its own `runs/<slug>/`
subdirectory with model-specific recipes, eval results, and report.

## Common commands

```bash
# Set up env (Python 3.11+; PyTorch 2.5+)
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt

# Quantize. Recipes read MODEL_ID / SAVE_DIR (sometimes SRC_DIR / DST_DIR) from env.
# Always wrap with the cgroup memory cap so a runaway alloc can't take the box down.
export MODEL_ID="<org>/<base-model>"
export SAVE_DIR="$PWD/artifacts/<Artifact-Name>"
tools/run_under_memcap.sh python runs/<slug>/recipes/<scheme>.py

# Serve via vLLM (Docker; bare-metal vLLM isn't yet stable on aarch64 / SM121a).
tools/serve_vllm_docker.sh "$PWD/artifacts/<Artifact-Name>" \
  --quantization compressed-tensors --kv-cache-dtype fp8_e4m3 \
  --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --served-model-name <served-name>

# Eval (GSM8K chat-CoT + full MMLU + ARC-C) against the running vLLM endpoint.
tools/run_eval_full.sh <served-name> "$PWD/artifacts/<Artifact-Name>" \
  "$PWD/runs/<slug>/results/<scheme>_full"

# Recipe-level sanity check (synthetic tensors, no model download):
python runs/<slug>/recipes/fp8_dynamic.py --selftest

# Start a new run from the template:
cp -r templates/run runs/<new-slug>
```

There is no test suite, lint config, or CI in this repo. Recipes have an
in-file `--selftest` for fast synthetic-tensor verification.

## Architecture

The split between **generic** and **per-run** is the load-bearing design:

- `tools/` — model-agnostic shell scripts. `serve_vllm_docker.sh` wraps
  the community DGX-Spark vLLM image; `run_under_memcap.sh` enforces a
  cgroup v2 `MemoryMax` (default 112 GiB, no swap) via `systemd-run`;
  `run_eval_full.sh` drives `lm-evaluation-harness == 0.4.11` against
  a live vLLM endpoint.
- `docs/schemes/<name>.md` — portable knowledge about a quantization
  scheme: what to quantize / leave alone, the canonical
  `quantization_config` block, common gotchas. **When implementing a
  new scheme, write this doc while doing it** so the next run inherits
  the playbook.
- `runs/<slug>/` — one base model per slug. Contains `recipes/<scheme>.py`
  (the model-specific quantizer), `results/<scheme>_full/` (eval JSONs +
  `run.log`), `REPORT.md` (the writeup with bf16 / FP8 / AWQ deltas),
  and `HF_PREVIEW_*.md` (canonical model-card source for each artifact).
- `artifacts/` — gitignored output dirs (the safetensors bytes).
  Subdirectory name should equal the proposed HF repo name so the
  on-disk folder and HF repo are referred to by the same string.
- `templates/run/` — skeleton; `cp -r` to start a new run.

**The rule of thumb:** if a file would apply unchanged to a different
base model, it goes in `tools/` or `docs/`. Otherwise it goes under
`runs/<slug>/`. Don't speculatively generalize across runs — wait until
3+ runs reveal a real pattern, then lift into `runs/_shared/`.

## Recipes — what they do

Each recipe is a self-contained driver script for one (model, scheme)
pair. They share a few hard rules learned the painful way:

- **Configure via env vars** (`MODEL_ID`, `SAVE_DIR`, `HF_CACHE`, or for
  some recipes `SRC_DIR`/`DST_DIR`). Never hardcode paths — keeps
  recipes portable and re-runnable on any machine.
- **Quantize on CPU** (`device_map="cpu"`). On Spark, GPU and CPU share
  unified memory, so "cpu" is not slower than "auto" — it just stops
  accelerate from creating a parallel mapping that gets accounted twice.
- **Watch RSS** (`psutil.Process().memory_info().rss`). The FP8 recipe
  hard-stops at ~105 GiB before the cgroup SIGKILLs the scope, so you
  get a Python traceback rather than a silent kill.
- **Avoid `llmcompressor.oneshot` for large MoE.** Its calibration path
  unfuses fused 3-D experts into thousands of `nn.Linear` modules while
  holding refs to the originals — the FP8 recipe sidesteps it for this
  reason. Walk modules manually and quantize in place.
- **Atomic save.** Write into `<SAVE_DIR>.tmp.<pid>/`, then rename.
  Refuse to write into a non-empty `SAVE_DIR` (mixing shards from a
  prior partial run produces silently inconsistent checkpoints).

## Loader-layout pitfalls (vLLM compressed-tensors / moe_wna16)

These are the failure modes that have actually bitten this repo:

- **MoE expert key naming differs by scheme.** vLLM's compressed-tensors
  loader expects **fused 3-D** `experts.gate_up_proj` / `experts.down_proj`
  for FP8_DYNAMIC; the AWQ / `moe_wna16` path expects **unfused per-expert**
  `experts.<i>.gate_proj` / `up_proj` / `down_proj`. Get the
  scheme/loader pairing right *at save time* — by the time vLLM rejects
  the artifact you've already quantized.
- **Multimodal wrapper class but text-only artifact.** vLLM still runs
  the multimodal init path (loading the image processor) even when no
  vision module is present. Copy `processor_config.json` from the base
  HF cache into the artifact dir, and keep `architectures` set to the
  multimodal class.
- **Key prefix.** When the LM is wrapped (e.g.
  `Qwen3_5MoeForConditionalGeneration` → `self.language_model = ...`),
  every state-dict key needs the `language_model.` prefix. Same for the
  `ignore` list inside `quantization_config`.
- **KV cache dtype must match across compared runs.** A bf16 baseline
  with `--kv-cache-dtype auto` vs an FP8/AWQ run with `fp8_e4m3` makes
  the delta uninterpretable. Always pin identical `--kv-cache-dtype`
  across all runs in one comparison.

## Memory cap wrapper

`tools/run_under_memcap.sh` defaults: `MemoryMax=112G`, `MemoryHigh=100G`,
`MemorySwapMax=0`. Override per-invocation: `MEMORY_MAX=64G MEMORY_HIGH=56G
tools/run_under_memcap.sh ...`. Requires Linux with cgroup v2 + `systemd-run`
(any modern Ubuntu / Debian / Fedora). Refuses to run without it — the
intent is that no quantizer ever runs unbounded on this hardware.

## Eval hygiene

Always also eval the bf16 baseline under the same vLLM/KV-cache settings
as the quantized builds — the "Δ vs bf16" deltas are only trustworthy
when measured under identical conditions. Decision rules in REPORTs
treat anything inside ±1σ as a tie; always quote ± stderr alongside any
score delta.

## Artifacts and git

`artifacts/*` is gitignored except `artifacts/README.md`. Quantized
`*.safetensors` are gitignored everywhere as a defensive belt-and-suspenders
(`*.safetensors.index.json` is allowed — small JSON). Eval `samples_*.jsonl`
files are gitignored; `run.log` and `results_*.json` are committed.
