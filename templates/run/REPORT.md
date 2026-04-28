# <BASE_MODEL_SLUG> — Quantization Report

> Copy and fill in. Delete the `*<text>*` placeholder hints as you go.

**Base model**: [`<org>/<name>`](https://huggingface.co/<org>/<name>)
*(N B parameters / X B active for MoE, architecture, distill source if any)*

**Hardware**: *(e.g. NVIDIA DGX Spark, GB10 / SM121a, 128 GiB unified memory)*

**Goal**: produce vLLM-loadable quantizations and measure the quality delta
against bf16 on reasoning + knowledge benchmarks.

This doc covers *(N quantized builds)* plus the bf16 baseline, all evaluated
on the same battery (GSM8K + full MMLU + ARC-Challenge) on YYYY-MM-DD.

---

## TL;DR

| | **bf16 (base)** | **<build A>** | **<build B>** |
|---|---|---|---|
| Bits / weight | 16 | 8 | 4 |
| Activation precision | 16 | 8 dynamic per-token | 16 |
| Calibration | n/a | none | none (data-free RTN) |
| Disk size | ~? GiB | ~? GiB | ~? GiB |
| GSM8K (full 1,319) | ? | ? | ? |
| MMLU (full 14,042) | ? | ? | ? |
| ARC-Challenge (full 1,172) | ? | ? | ? |
| Δ MMLU vs bf16 | — | ? | ? |

**Headline**: *(one paragraph — what to ship and why)*

---

## Architecture context

*(What the model is. Layer count, hidden size, MoE config, attention type,
multimodal stack. The "what to quantize / leave alone" decisions in each
recipe rest on this.)*

---

## Build A — *(scheme name)*

Artifact: [`<artifact-dir>/`](../../<artifact-dir>/)
Per-artifact model card: [`<artifact-dir>/README.md`](../../<artifact-dir>/README.md)
Quantizer: [`recipes/<recipe>.py`](./recipes/<recipe>.py)
Generic scheme reference: [`docs/schemes/<scheme>.md`](../../docs/schemes/<scheme>.md)

### Why this scheme for this model

*(Specific reasons: distill signal lives in attention LoRA → FP8 keeps it /
multimodal stack must survive → AWQ at fp16 vision / etc.)*

### Implementation notes

*(Anything tricky: memory-budget workarounds, save-time key rewrites, vLLM
loader quirks, fused-vs-unfused expert decisions.)*

### Settings

| key | value |
|---|---|
| ... | ... |

### What was stripped vs kept vs quantized

- **Stripped**: *(if any — e.g. vision tower, MTP head)*
- **Quantized**: *(which Linear modules)*
- **Kept dense**: *(router gates, linear-attn, layer 0, lm_head, …)*

### Smoke result

*(Live serve verification. Did the model load? Did `<think>` blocks survive
on a reasoning prompt?)*

```bash
vllm serve <artifact-dir> --quantization ...
```

### Full eval (YYYY-MM-DD, total wall-clock H h M min)

| benchmark | n | metric | score |
|---|---|---|---|
| GSM8K (5-shot CoT, chat) | 1,319 | exact_match strict | ? ± ? |
| GSM8K (5-shot CoT, chat) | 1,319 | exact_match flexible | ? ± ? |
| MMLU overall (5-shot, raw MC) | 14,042 | acc | ? ± ? |
| MMLU social sciences | — | acc | ? |
| MMLU other | — | acc | ? |
| MMLU STEM | — | acc | ? |
| MMLU humanities | — | acc | ? |
| ARC-Challenge (raw MC) | 1,172 | acc | ? ± ? |
| ARC-Challenge (raw MC) | 1,172 | acc_norm | ? ± ? |

*(Best/worst MMLU subtasks, wall-clock breakdown, anything notable.)*

---

## Build B — *(other scheme)*

*(Same structure as Build A.)*

---

## BF16 baseline — full eval (YYYY-MM-DD)

*(Same structure. Note any memory-pressure adjustments needed —
`--max-model-len`, `--gpu-memory-utilization`. Always pin the same KV-cache
dtype as the quantized runs so the deltas are weight-only.)*

---

## Three-way head-to-head: bf16 vs <A> vs <B>

Same harness, same prompts, same `temperature=0`, same KV-cache dtype.
Deltas signed; positive means the quantized build *beat* bf16 on that metric.

| benchmark | metric | **bf16** | **<A>** | Δ A | **<B>** | Δ B |
|---|---|---|---|---|---|---|
| GSM8K | exact_match strict | ? | ? | ? | ? | ? |
| MMLU | overall acc | ? | ? | ? | ? | ? |
| ... | | | | | | |

**What the deltas say**: *(the analysis paragraphs. Be honest about what's
inside ± stderr — that's noise, not a signal.)*

---

## Side-by-side recipe table

| dimension | <A> | <B> |
|---|---|---|
| Format | ... | ... |
| ... | ... | ... |

---

## Decision matrix (with measured numbers)

| if you care about... | pick | why |
|---|---|---|
| Maximum quality preservation | ... | ... |
| Smallest disk | ... | ... |
| Multimodal capability | ... | ... |

**Default recommendation**: *(which build to ship)*

---

## Out of scope

- *(things you considered but didn't do, with rationale)*

---

## Roadmap checklist

- [x] *(done items)*
- [ ] *(remaining items)*

---

## File index

- `recipes/<scheme>.py` — quantizer
- `results/<scheme>_full/` — eval JSONs + run.log
- `<artifact-dir>/` — quantized weights (gitignored, uploaded to HF)
