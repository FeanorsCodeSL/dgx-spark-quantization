# <BASE_MODEL_SLUG> — Quantization Plan

> Copy this template to `runs/<slug>/PLAN.md` and keep model-specific
> execution notes here. Generic scheme behavior belongs in `docs/schemes/`;
> final measured results belong in `REPORT.md`.

## Goal

- Base model: `<org>/<name>`
- Target artifact(s): `<artifact names>`
- Eval battery: GSM8K, MMLU, ARC-Challenge, plus any model-specific tasks.

## Phase 0 — Run Skeleton

- [ ] Choose slug and copy `templates/run/`.
- [ ] Fill `README.md` with source model, hardware, intended schemes, and
      links.
- [ ] Add the run to the top-level `README.md` Runs table.

## Phase 1 — Architecture Inspection

- [ ] Cache the source model locally.
- [ ] Inspect module names, tensor shapes, and custom-code requirements.
- [ ] Decide what to quantize and what to keep dense.
- [ ] Record findings in `README.md` and save raw output under `results/`.

## Phase 2 — Quantize

- [ ] Implement the model-specific recipe under `recipes/`.
- [ ] Run any self-tests or dry-runs.
- [ ] Run full quantization under `tools/run_under_memcap.sh`.
- [ ] Smoke-serve the artifact with `tools/serve_vllm_docker.sh`.

## Phase 3 — Eval Quantized Artifact(s)

- [ ] Serve each artifact with pinned vLLM settings.
- [ ] Run `tools/run_eval_full.sh`.
- [ ] Save results under `results/<scheme>_full/`.

## Phase 4 — Eval Baseline

- [ ] Serve bf16 with matching vLLM settings where practical.
- [ ] Run the same eval battery.
- [ ] Save results under `results/bf16_full/`.

## Phase 5 — Report And Model Card

- [ ] Fill `REPORT.md` with measured numbers and deltas.
- [ ] Write artifact README/model card if publishing.
- [ ] Update top-level `README.md` status and headline numbers.
