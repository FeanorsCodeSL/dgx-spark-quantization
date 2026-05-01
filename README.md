# dgx-spark-quantization

A reusable framework for quantizing open-weight models on a single
**NVIDIA DGX Spark** (GB10 / SM121a, 128 GiB unified memory) and shipping
the artifacts to Hugging Face — vLLM-loadable, eval-validated, and
documented end-to-end.

> Maintained by **[FeanorsCode](https://feanorscode.com)** ·
> Org: **[github.com/FeanorsCodeSL](https://github.com/FeanorsCodeSL)**.

When you find a bf16 (or GGUF) model that's missing a vLLM-loadable
quantization, you copy the template, write the model-specific recipe,
quantize, eval, and ship — without re-deriving the framework.

---

## What's in here

- **Generic, reusable tooling** — eval driver, vLLM container launcher,
  cgroup memory-cap wrapper.
- **Per-scheme reference docs** — what FP8 dynamic, AWQ-GEMM, and
  compressed-tensors AWQ are, what to quantize / leave alone, where the
  pitfalls are.
- **One subdirectory per quantized model** ("a *run*") — model-specific
  recipes, eval results, and writeup.
- **A copy-and-fill template** for adding the next run.
- **An end-to-end Hugging Face publishing guide** that covers everything
  from token to upload to model-card metadata.

---

## Quick start

```bash
# 1. Clone and set up a clean Python env
git clone https://github.com/FeanorsCodeSL/dgx-spark-quantization
cd dgx-spark-quantization
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Pick a base model and an existing scheme. The base auto-downloads
#    from Hugging Face into ~/.cache/huggingface (or $HF_HOME if set).
SLUG="qwen3.6-35b-distill"            # use the existing example, or copy templates/run
SCHEME="fp8_dynamic"                  # one of: fp8_dynamic, awq_gemm, awq_compressed_tensors, ...
ARTIFACT="My-Awesome-Model-FP8-Dynamic"

# 3. Quantize. Output goes under artifacts/ (gitignored).
export MODEL_ID="<org>/<base-model>"
export SAVE_DIR="$PWD/artifacts/${ARTIFACT}"
tools/run_under_memcap.sh \
  python "runs/${SLUG}/recipes/${SCHEME}.py"

# 4. Serve via vLLM (Docker; the launcher wraps the DGX-Spark image).
tools/serve_vllm_docker.sh "$PWD/artifacts/${ARTIFACT}" \
  --quantization compressed-tensors --kv-cache-dtype fp8_e4m3 \
  --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --served-model-name "${ARTIFACT}"

# 5. Eval. Writes results JSONs + run.log into the run's results dir.
tools/run_eval_full.sh \
  "${ARTIFACT}" \
  "$PWD/artifacts/${ARTIFACT}" \
  "$PWD/runs/${SLUG}/results/${SCHEME}_full"

# 6. Publish to Hugging Face — see HUGGINGFACE_PUBLISHING.md.
hf auth login                 # one-time
hf upload-large-folder \
  <your-user>/${ARTIFACT} \
  ./artifacts/${ARTIFACT} \
  --repo-type model --num-workers 8
```

The `tools/run_under_memcap.sh` wrapper requires Linux with cgroup v2 +
systemd-run (any modern Ubuntu / Debian / Fedora).

---

## Schemes available

| scheme | bits | calibration | runtime | reference |
|---|---|---|---|---|
| **FP8 W8A8 dynamic** (compressed-tensors) | 8 | none | vLLM native | [`docs/schemes/fp8-dynamic.md`](./docs/schemes/fp8-dynamic.md) |
| **AWQ-INT4 GEMM** (data-free RTN) | 4 | none | vLLM via `moe_wna16` / AutoAWQ | [`docs/schemes/awq-gemm.md`](./docs/schemes/awq-gemm.md) |
| **AWQ-INT4 W4A16** (compressed-tensors) | 4 | AWQ calibration | vLLM `compressed-tensors` | [`docs/schemes/awq-compressed-tensors.md`](./docs/schemes/awq-compressed-tensors.md) |

Add a new scheme by writing a `docs/schemes/<name>.md` reference and a
recipe under `runs/<run>/recipes/<name>.py`. Likely future additions:
AutoRound INT4, GPTQ, NVFP4, MXFP4, GGUF→safetensors transcoding.

---

## Runs

| run | base model | schemes | status | report |
|---|---|---|---|---|
| [`qwen3.6-35b-distill`](./runs/qwen3.6-35b-distill/) | [`lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled`](https://huggingface.co/lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled) | FP8 dynamic, AWQ-INT4 GEMM | done (2026-04-27) | [REPORT](./runs/qwen3.6-35b-distill/REPORT.md) |
| [`nemotron-3-nano-omni-30b-a3b`](./runs/nemotron-3-nano-omni-30b-a3b/) | [`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16) | AWQ-INT4 W4A16 compressed-tensors | done (2026-05-01) | [REPORT](./runs/nemotron-3-nano-omni-30b-a3b/REPORT.md) |

### Headline (qwen3.6-35b-distill)

| build | bits | disk | MMLU | GSM8K (strict) | Δ MMLU vs bf16 |
|---|---|---|---|---|---|
| bf16 baseline | 16 | ~67 GiB | 0.8341 | 0.9447 | — |
| FP8 W8A8 dynamic (text-only) | 8 | ~35 GiB | 0.8332 | 0.9447 | **−0.09 pp** (within stderr) |
| AWQ-INT4 GEMM (multimodal) | 4 | ~24 GiB | 0.8068 | 0.9386 | −2.73 pp |

FP8 effectively lossless. AWQ trades MMLU for disk + multimodal preservation.

### Headline (nemotron-3-nano-omni-30b-a3b)

| build | bits | disk | MMLU | GSM8K (strict) | ARC-C | Delta MMLU vs bf16 |
|---|---|---|---|---|---|---|
| bf16 baseline | 16 | ~66 GiB | 0.7150 | 0.7900 | 0.5239 / 0.5631 norm | — |
| AWQ-INT4 W4A16 compressed-tensors | 4 | ~22 GiB | 0.6904 | 0.7983 | 0.5247 / 0.5589 norm | -2.46 pp |
| NVFP4 (NVIDIA official, eval only) | 4 | ~21 GiB | 0.7124 | 0.7589 | 0.5230 / 0.5401 norm | -0.26 pp |

AWQ is the local publishable artifact; NVFP4 is NVIDIA's official artifact
measured for comparison. AWQ multimodal smoke passed for image/video on
the pinned vLLM image; audio passed after adding the missing vLLM audio
decode dependencies (`av` + `soundfile`) before server startup.

---

## Repo layout

```
.
├── README.md                              this file
├── HUGGINGFACE_PUBLISHING.md              generic HF publishing guide
├── LICENSE                                Apache 2.0 (covers code in this repo)
├── requirements.txt                       pinned Python deps (incl. hf CLI + hf_transfer)
├── .gitignore
│
├── docs/
│   ├── adding-a-run.md                    onboarding playbook for a new model
│   ├── repo-layout.md                     detailed structural walkthrough
│   └── schemes/                           per-scheme reference (portable across models)
│       ├── fp8-dynamic.md
│       ├── awq-gemm.md
│       └── awq-compressed-tensors.md
│
├── tools/                                 generic, model-agnostic tooling
│   ├── serve_vllm_docker.sh
│   ├── run_under_memcap.sh
│   └── run_eval_full.sh
│
├── templates/
│   └── run/                               skeleton — `cp -r` when starting a new run
│       ├── README.md
│       ├── PLAN.md
│       └── REPORT.md
│
├── runs/
│   ├── qwen3.6-35b-distill/               completed reference run
│   │   ├── README.md
│   │   ├── PLAN.md
│   │   ├── REPORT.md                      full quant report with deltas
│   │   ├── recipes/                       model-specific quantizers
│   │   └── results/                       per-build eval JSONs + run.log
│   └── nemotron-3-nano-omni-30b-a3b/      completed Nemotron Omni run
│       ├── README.md
│       ├── PLAN.md
│       ├── REPORT.md
│       ├── recipes/
│       └── results/
│
└── artifacts/                             quantized model outputs (gitignored)
    ├── README.md                          (committed) explains the convention
    └── <Base-Hyphenated>-<Scheme>/        one per HF-bound artifact
```

The quantized **artifact directories** (the safetensors shards) live under
`artifacts/` and are gitignored — they're for upload to Hugging Face, not
GitHub. Naming each one to match its proposed HF repo name keeps the
on-disk folder and the HF repo referable by the same string.

---

## Hardware / software prerequisites

This repo was developed on:

- **DGX Spark**, single node, GB10 / SM121a, Ubuntu 24.04 aarch64,
  128 GiB unified memory.
- **vLLM** via Docker (the bare-metal vLLM build is not yet stable on
  aarch64 / SM121a). The launcher in
  [`tools/serve_vllm_docker.sh`](./tools/serve_vllm_docker.sh) wraps the
  community pre-built image
  [`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker)
  (MIT) by default.

Most of the *scripts* are generic — recipes are pure CPU
(`device_map="cpu"` for FP8, shard-streaming for AWQ) and need only:

- Python 3.11+, PyTorch 2.5+, `transformers`, `safetensors`, `psutil`,
  `compressed-tensors`, `huggingface_hub`.
- `lm-evaluation-harness == 0.4.11` for the eval battery.

All pinned in [`requirements.txt`](./requirements.txt). The Hugging Face
CLI (`hf`) installs as part of `huggingface_hub[cli,hf_transfer]` and is
required only at upload time.

The cgroup memory wrapper requires Linux with cgroup v2 + systemd-run
(any modern Ubuntu / Debian / Fedora).

---

## Adding a new run

Short version:

```bash
SLUG="<your-slug>"                              # e.g. llama4-70b-instruct
cp -r templates/run "runs/${SLUG}"
# Then: edit runs/${SLUG}/README.md and write recipes/<scheme>.py.
```

Full step-by-step playbook: [`docs/adding-a-run.md`](./docs/adding-a-run.md).

---

## Publishing to Hugging Face

Each artifact directory under `artifacts/` ships to Hugging Face as a
quantization-of the base model. The HF model card (YAML frontmatter with
`base_model` + `base_model_relation: quantized`) lives **inside the
artifact dir** alongside the safetensors and gets uploaded with them.

Step-by-step: [`HUGGINGFACE_PUBLISHING.md`](./HUGGINGFACE_PUBLISHING.md).

---

## License

- Code in this repo: Apache 2.0 (see [`LICENSE`](./LICENSE)).
- Quantized model artifacts inherit their base model's license. Always
  check the base model card before redistributing on HF.

---

## About FeanorsCode

[FeanorsCode](https://feanorscode.com) is a small engineering company.
We publish quantized open-weight models to Hugging Face under
[`feanorscode`](https://huggingface.co/feanorscode) and maintain the
infrastructure code that produces them under
[`FeanorsCodeSL`](https://github.com/FeanorsCodeSL) on GitHub.

If you use this framework for your own quantizations, we'd love to hear
about it — open an issue on the repo with what you built.

---

## Acknowledgements

- [`Qwen team`](https://huggingface.co/Qwen) for the Qwen3.5-MoE
  architecture used in the first run.
- [`lordx64`](https://huggingface.co/lordx64) for the Claude-4.7-Opus
  reasoning-distilled base used as the first run's input.
- [Lin et al.](https://arxiv.org/abs/2306.00978) for the AWQ method.
- [`casper-hansen`](https://github.com/casper-hansen/AutoAWQ) for AutoAWQ.
- [`QuantTrio/Qwen3.6-35B-A3B-AWQ`](https://huggingface.co/QuantTrio/Qwen3.6-35B-A3B-AWQ)
  as the AWQ-GEMM layout reference.
- [`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker) for
  the community pre-built DGX-Spark vLLM image.
- [`vllm-project`](https://github.com/vllm-project) for vLLM,
  `compressed-tensors`, and `llmcompressor`.
- [EleutherAI](https://github.com/EleutherAI/lm-evaluation-harness) for
  `lm-evaluation-harness`.
