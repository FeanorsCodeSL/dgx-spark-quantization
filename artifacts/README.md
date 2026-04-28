# `artifacts/`

Quantized model outputs land here. Each subdirectory is one Hugging Face
model repo's worth of files (safetensors shards + `config.json` +
`tokenizer.*` + the model card `README.md`).

**Everything in this directory except this `README.md` is gitignored** —
artifacts are 24–80 GiB each and ship to Hugging Face, not GitHub.

---

## Recipe convention

Recipes write into `artifacts/<artifact-name>/`. The artifact name is
typically the proposed Hugging Face repo name (so the folder name and the
HF repo name match):

```bash
SAVE_DIR="$PWD/artifacts/<artifact-name>" \
  tools/run_under_memcap.sh \
  python runs/<run-slug>/recipes/<scheme>.py
```

Naming pattern: `<Base-Model-Hyphenated>-<Scheme-Tag>`, e.g.
`Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-FP8-Dynamic`.
Long but searchable — users looking for a quantization of `Foo` will
search for `Foo` in HF and pick the matching repo by scheme tag.

## Publishing

Each subdirectory uploads to a separate HF model repo. See
[`../HUGGINGFACE_PUBLISHING.md`](../HUGGINGFACE_PUBLISHING.md) for the
end-to-end pipeline (account, write token, large-folder upload, model
card metadata, troubleshooting).

The artifact's model card lives **inside** the artifact dir as
`README.md`. It carries the YAML frontmatter that links the new HF
repo back to the base model under "Quantizations" — `base_model`,
`base_model_relation: quantized`, `quantized_by`. Don't move the card
out of the artifact dir.
