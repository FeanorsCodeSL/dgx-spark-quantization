# Repo layout

```
dgx-spark-quantization/
├── README.md                          framework overview, quick start, runs index
├── HUGGINGFACE_PUBLISHING.md          generic HF publishing guide
├── LICENSE                            Apache 2.0 (covers code in this repo)
├── requirements.txt                   pinned Python deps (incl. hf CLI + hf_transfer)
├── .gitignore
│
├── docs/
│   ├── adding-a-run.md                onboarding playbook for a new model
│   ├── repo-layout.md                 this file
│   └── schemes/                       per-scheme reference (portable across models)
│       ├── fp8-dynamic.md
│       └── awq-gemm.md                (add more as schemes are implemented)
│
├── tools/                             generic, model-agnostic tooling
│   ├── serve_vllm_docker.sh           DGX-Spark vLLM container launcher
│   ├── run_under_memcap.sh            systemd-run cgroup wrapper (kernel SIGKILL on overshoot)
│   └── run_eval_full.sh               GSM8K + MMLU + ARC-C battery driver
│
├── templates/
│   └── run/                           skeleton — copy when starting a new run
│       ├── README.md                  per-run index template
│       ├── REPORT.md                  full quant-report template
│       ├── recipes/                   (empty; .gitkeep)
│       └── results/                   (empty; .gitkeep)
│
├── runs/                              one subdir per (base model)
│   └── qwen3.6-35b-distill/           the first instance
│       ├── README.md                  this run's index
│       ├── REPORT.md                  full quant report (bf16 / FP8 / AWQ deltas)
│       ├── PLAN.md                    historical planning blueprint
│       ├── HF_PREVIEW_FP8.md          canonical model-card source for the FP8 artifact
│       ├── HF_PREVIEW_AWQ.md          canonical model-card source for the AWQ artifact
│       ├── recipes/
│       │   ├── fp8_dynamic.py         model-specific FP8 driver
│       │   ├── awq_gemm.py            model-specific AWQ driver
│       │   └── inspect_modules.py     architecture-discovery helper
│       └── results/
│           ├── README.md              what's here, how to reproduce
│           ├── awq_full/              results_*.json + run.log
│           ├── fp8_full/
│           └── bf16_full/
│
├── artifacts/                         all quantized model outputs (gitignored)
│   ├── README.md                      (committed) explains the convention
│   ├── Qwen3.6-...-FP8-Dynamic/       FP8 artifact, ready to upload to HF
│   └── Qwen3.6-...-AWQ-INT4/          AWQ artifact, ready to upload to HF
│
├── hf-cache/                          (optional) local HF download cache (gitignored)
└── .venv/                             local Python env (gitignored)
```

---

## What lives where

### "Generic" vs "per-run" — the rule of thumb

If a file would apply unchanged to a different base model, it goes in
`tools/` or `docs/`. If it's specific to *this* base, it goes under
`runs/<run>/`.

| concern | location | reason |
|---|---|---|
| Eval driver (lm-eval against vLLM) | `tools/` | model-agnostic |
| vLLM container launcher | `tools/` | depends only on Docker + vLLM image |
| cgroup memory wrapper | `tools/` | depends only on systemd-run |
| Scheme reference (FP8 dynamic, AWQ-GEMM, ...) | `docs/schemes/` | portable across architectures |
| Onboarding guide | `docs/adding-a-run.md` | model-agnostic |
| HF publishing guide | `HUGGINGFACE_PUBLISHING.md` | model-agnostic |
| The actual quantizer for one model | `runs/<run>/recipes/` | architecture-specific |
| One model's eval results | `runs/<run>/results/` | per-run |
| One model's writeup | `runs/<run>/REPORT.md` | per-run |
| Quantized model bytes (safetensors) | top level / gitignored | uploaded to HF, not GitHub |

### Why per-run instead of per-(model, scheme)

A run typically produces multiple sibling quantizations (one base → FP8 *and*
AWQ, with a shared bf16 baseline). They share the architecture-discovery
work, the eval battery, and the comparative report. Grouping by base model
keeps that natural unit together.

Naming convention: `runs/<base-model-slug>/` where slug is short, lowercase,
matches what users would search for (e.g. `qwen3.6-35b-distill`,
`llama4-70b-instruct`).

### Why artifacts under `artifacts/` (and not per-run)

Three reasons:

1. **One discoverable location.** Newcomers cloning the repo find every
   quantized output in one predictable place, not scattered across
   `runs/<each-slug>/`.
2. **Transient by design.** Artifacts exist only between quantization and
   HF upload. The "permanent" record is the HF repo, not the on-disk
   directory in this tree.
3. **Clean diff surface.** `.gitignore` is one line (`artifacts/*` with a
   `!artifacts/README.md` exception) instead of a per-run pattern that
   has to be remembered for every new run.

The artifact subdirectory name should match the proposed Hugging Face
repo name — typically `<Base-Model-Hyphenated>-<Scheme-Tag>`, e.g.
`Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-FP8-Dynamic`. That way
the local folder and the HF repo are referred to by the same string.

---

## When the layout should change

### Adding a new scheme

1. Implement it in a `runs/<run>/recipes/<new_scheme>.py` for one model
   first.
2. **While** doing it, write `docs/schemes/<new-scheme>.md` capturing the
   portable knowledge: when to pick this scheme, the canonical
   `quantization_config` block, what to quantize / leave alone, the
   common implementation gotchas. The next run inherits this.
3. Add the scheme to the "Schemes available" section of the top-level
   README.

### Adding a new run

See [`docs/adding-a-run.md`](./adding-a-run.md). The short version: copy
`templates/run/` to `runs/<new-slug>/`, fill in.

### Lifting shared recipe code

After 2–3 runs, real shared patterns will surface — per-channel FP8 quant,
AWQ pack/unpack, shard-streaming I/O, atomic save dirs. **At that point**
(not before), extract them into `runs/_shared/` or a small package directory
imported by recipes.

Until then, copy-pasting between recipes is fine. Three similar copies are
cheaper than a premature abstraction that ends up wrong for the fourth case.
