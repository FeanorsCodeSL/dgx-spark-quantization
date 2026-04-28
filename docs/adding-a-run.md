# Adding a new run

A "run" is one base model → one or more quantized artifacts → an eval
report. This guide is the recipe for starting a new one.

> **Mental model.** The repo is a *framework* for repeated quantization work.
> Per-run dirs are leaves; everything reusable (scheme reference, eval driver,
> serve wrapper, publishing guide) lives outside `runs/`. When in doubt about
> where something belongs: if it would apply unchanged to a different model,
> it goes in `tools/` or `docs/`. If it's specific to *this* base, it goes
> under `runs/<run>/`.

---

## 1. Pick a slug and copy the template

Choose a short, lowercase slug for the base model. Conventions:

- Use the model's distinctive name segment, not the org or full HF id.
  `qwen3.6-35b-distill` ✓; `lordx64-Qwen3.6-...` ✗.
- Stick to `[a-z0-9.-]+`; no spaces, no underscores in the dirname.
- Keep it short — it'll show up in artifact names too.

```bash
SLUG="<your-slug>"          # e.g. llama4-70b-instruct
cp -r templates/run "runs/${SLUG}"
```

You now have `runs/<slug>/` with placeholder `README.md` and `REPORT.md` and
empty `recipes/` + `results/` subdirs.

---

## 2. Document the source model

Open `runs/<slug>/README.md` (the run's index). Fill in:

- Base model HF id and link.
- Architecture class (e.g. `LlamaForCausalLM`, `Qwen3_5MoeForConditionalGeneration`).
- Param count, active params (for MoE), context window.
- License (this determines what you're allowed to redistribute).
- Any architectural quirks that'll affect quantization choices —
  multimodality, MoE-fused vs unfused experts, linear attention, MTP head,
  unusual layer 0 behavior, custom RoPE.

The point of this section: when future-you adds a sibling quantization six
months later, this file tells you what's true about the model so you don't
re-derive it from `config.json`.

---

## 3. Pick the schemes

The two schemes documented in this repo today:

- [`docs/schemes/fp8-dynamic.md`](./schemes/fp8-dynamic.md) — 2× compression,
  effectively lossless on most architectures. Native vLLM. Requires modern
  FP8-capable hardware to be efficient.
- [`docs/schemes/awq-gemm.md`](./schemes/awq-gemm.md) — 4× compression,
  ~1–3 pp MMLU loss with data-free RTN. Preserves multimodal stack at fp16
  by default.

Future schemes worth adding when you do them: AutoRound INT4, GPTQ, NVFP4,
MXFP4, GGUF→safetensors transcoding. When you implement one for the first
time, write a corresponding `docs/schemes/<name>.md` *while doing it* so the
next run inherits the playbook.

---

## 4. Write the recipe(s)

Each scheme gets its own `runs/<slug>/recipes/<scheme>.py`. The recipe is
the *driver script* — the bit that knows what's specific about this model:

- which modules to quantize / leave alone (the `ignore` or
  `modules_to_not_convert` list)
- whether MoE experts are fused or unfused on disk
- whether the model is wrapped in a multimodal class needing
  `language_model.` prefix rewriting
- whether to strip vision tower / MTP head before save
- the canonical save layout the runtime expects

Cross-reference `runs/qwen3.6-35b-distill/recipes/` for a worked-out pattern.
Many of the helper functions there (per-channel FP8 quant, AWQ pack/unpack,
shard-streaming I/O) are copy-pasteable — extract them into a shared module
under `runs/_shared/` if you find yourself doing it three times.

> **Don't speculatively generalize.** Write the recipe specific to this
> model. After three or four runs, real abstraction patterns will surface
> and you can lift them into shared code with confidence. Generalizing
> upfront from one example is how recipes accumulate dead branches.

### Configuration

Recipes should read paths from environment variables, not hardcode them:

```python
MODEL_ID = os.environ.get("MODEL_ID", "<default>")
HF_CACHE = os.environ.get("HF_CACHE", os.path.expanduser("~/.cache/huggingface"))
SAVE_DIR = os.environ["SAVE_DIR"]   # required
```

That keeps the recipe portable across machines and lets you re-run someone
else's recipe by just exporting the right env vars.

---

## 5. Quantize

Run each recipe under the cgroup memory cap so a runaway allocator can't
take down the box. Output goes under `artifacts/<artifact-name>/`, where
the artifact name typically matches the proposed HF repo name:

```bash
export MODEL_ID="<base>"
export SAVE_DIR="$PWD/artifacts/<artifact-name>"   # e.g. .../<Base-Hyphenated>-FP8-Dynamic

tools/run_under_memcap.sh \
  python "runs/${SLUG}/recipes/<scheme>.py"
```

The HF source model is auto-downloaded into `~/.cache/huggingface/`
(set `HF_HOME` to override). No input directory is needed in the repo.

`tools/run_under_memcap.sh` defaults to `MemoryMax=112G`,
`MemoryHigh=100G`, no swap. Override with env vars if you're on a smaller
/ larger box:

```bash
MEMORY_MAX=64G MEMORY_HIGH=56G \
  tools/run_under_memcap.sh ...
```

Everything under `artifacts/` is gitignored except the explanatory
`artifacts/README.md`.

---

## 6. Smoke-test the artifact in vLLM

Always serve and hit the live endpoint before running a 3-hour eval:

```bash
tools/serve_vllm_docker.sh "$PWD/<artifact-dir>" \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --served-model-name <served-name> \
  --reasoning-parser qwen3      # or remove for non-reasoning models

# In another terminal:
curl -s http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "<served-name>", "messages":[{"role":"user","content":"Hello"}]}' \
  | jq .
```

If the model loads and responds sensibly, proceed. If it fails at load,
common diagnostics:

- `Can't load image processor` — copy `processor_config.json` from the base
  HF cache into the artifact dir.
- `architectures mismatch` — set `config.json:architectures` to whatever
  vLLM expects for this model class.
- `key not found` — your save layout doesn't match the loader's expected
  naming. Check the loader's `hf_to_vllm_mapper`.

---

## 7. Run the eval

```bash
tools/run_eval_full.sh \
  <served-name> \
  "$PWD/<artifact-dir>" \
  "$PWD/runs/${SLUG}/results/<scheme>_full"
```

This runs GSM8K (full, chat-templated CoT), full MMLU (57 subtasks), and
ARC-Challenge (full) against the running vLLM endpoint. ~1.5–3 hours wall
clock.

**Repeat** for each artifact (FP8, AWQ, …) and **always** also for the bf16
baseline — without a baseline number measured under the same conditions, the
"Δ vs bf16" deltas in your report aren't trustworthy.

For the bf16 baseline, point vLLM at the cached source snapshot directory
or the HF id directly:

```bash
tools/serve_vllm_docker.sh "<base-model-hf-id>" \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.7 \
  --served-model-name <served-name>-bf16
```

> **Match the KV cache dtype across runs.** If the AWQ run uses
> `--kv-cache-dtype fp8_e4m3` and the bf16 run uses `auto`, your delta is
> contaminated by KV-quant noise, not just weight precision. Always pin
> identical KV-cache dtype across all runs in a single comparison.

---

## 8. Write the REPORT

Open `runs/<slug>/REPORT.md` (the placeholder is in the template). Fill in:

- TL;DR table: bf16 baseline + each quantized build, with absolute scores
  and Δ vs bf16.
- Architecture context: what you chose to quantize and why (link to the
  scheme reference docs for the *general* rationale; here, document the
  *model-specific* choices — what was stripped, what was kept, what was
  unusual).
- Per-build "Settings" tables (the canonical `quantization_config` block).
- Full eval results per build (n, metric, score ± stderr, per-domain
  breakdown for MMLU).
- Three-way head-to-head delta table.
- Decision matrix (when to ship which build).
- Roadmap of what's next.

Template skeleton lives in `templates/run/REPORT.md`.

---

## 9. Update the top-level README

The repo's top-level README has a "Runs" index. Add an entry for your new
run with a one-line summary and link.

---

## 10. Publish to Hugging Face

The artifact dirs are **the things you ship to HF**, not the run dir.
Follow [`HUGGINGFACE_PUBLISHING.md`](../HUGGINGFACE_PUBLISHING.md) — it walks
through the publishing pipeline (account, token, repo creation, large-folder
upload, model card metadata for the *quantization-of* relationship,
licenses, troubleshooting).

The model card lives **inside the artifact dir** (`<artifact-dir>/README.md`)
and gets uploaded with the safetensors. Use the existing artifact READMEs
in this repo as a starting template for the YAML frontmatter.

---

## 11. Commit

```bash
git add runs/<slug>/ docs/schemes/<new-scheme>.md
git commit -m "Add <slug> run (<schemes>)"
```

The artifact dirs are gitignored, so they won't accidentally get pushed.

---

## Common gotchas

- **vLLM rejects the artifact at load.** Most likely the on-disk key naming
  doesn't match what the loader expects. Read the loader's
  `hf_to_vllm_mapper` for the model class, then verify your save layout
  matches.
- **Multimodal class but text-only build.** vLLM still runs the multimodal
  init path (loading the image processor) even when no vision module is
  present. Copy `processor_config.json` from the base into the artifact dir.
- **MoE expert key naming.** Some loaders want fused 3-D
  (`experts.gate_up_proj` `[E, 2I, H]`); others want unfused per-expert
  (`experts.<i>.gate_proj`). FP8 wants fused; AWQ wants unfused. Get the
  scheme/loader pairing right before save.
- **bf16 baseline OOM on tight memory.** Use `--max-model-len 4096` and
  `--gpu-memory-utilization 0.7` to leave room for KV cache + CUDA overhead.
  See `docs/dgx_spark_notes.md` for the worked numbers (when that doc is
  added).
- **Stderr-comparison hygiene.** Always quote ± stderr alongside any score
  delta. A "−0.5 pp drop" can be ±0.4 pp stderr — i.e., noise. Decision
  rules in this repo treat anything inside ±1 σ as a tie.
