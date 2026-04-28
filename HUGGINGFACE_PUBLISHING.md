# Publishing quantized artifacts to Hugging Face

Generic publishing pipeline for any artifact directory produced by a run
in this repo. The pipeline is the same regardless of base model or scheme;
only the dirnames and model-card metadata change per artifact.

Each artifact directory should already contain a fully-formed model card
(`README.md` with YAML frontmatter declaring `base_model` /
`base_model_relation: quantized` / `quantized_by`). Once uploaded, the HF UI
links the new repo back to the source model and tags it as a quantization.

> **Worked example** at the end of this doc walks through publishing the
> two artifacts produced by [`runs/qwen3.6-35b-distill`](./runs/qwen3.6-35b-distill/)
> (`Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-FP8-Dynamic/` and `Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-AWQ-INT4/`).
> If this is your first time, follow that example first; the placeholder
> sections below are the generic recipe to apply on later runs.

---

## 0. What you'll end up with

For each artifact you upload, one new HF model repo:

```
https://huggingface.co/<your-user-or-org>/<repo-name>
```

It will:

- Show up under **Quantizations** on the base model's HF page.
- Be loadable directly with `vllm serve <repo>` or `from_pretrained` (after
  installing the matching loader for the scheme — e.g. `compressed-tensors`
  for FP8, AutoAWQ-loader for AWQ).
- Carry tags (`fp8`, `awq`, `quantization`, …) that surface it in HF search
  and filters.

> **Tip — name repos so they're searchable.** Users searching for "a
> quantization of X" will look for repo names containing X. Pattern that
> works:
>
> ```
> <Base-Name-Hyphenated>-<Scheme-Tag>
> e.g. Qwen3.6-35B-A3B-Distill-FP8-Dynamic
>      Llama4-70B-Instruct-AWQ-INT4
> ```

---

## 1. Prerequisites

### Hugging Face account & token

1. Sign up at [huggingface.co/join](https://huggingface.co/join) if you don't
   already have an account.
2. Create a write-scoped access token:
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) →
   **New token** → Type **Write**.
3. Decide on a repo namespace. Either your **personal user** (`<your-user>`)
   or an **organization** you own. If you want an org, create one at
   [huggingface.co/organizations/new](https://huggingface.co/organizations/new)
   first.

### CLI install

```bash
pip install -U "huggingface_hub[cli,hf_transfer]"
```

`hf_transfer` (the Rust-accelerated multipart uploader) is *important* — at
24–35 GiB per artifact, single-threaded uploads will take many hours.

### Log in (one-time)

```bash
hf auth login
# Paste your write token when prompted.
# Or set HF_TOKEN=hf_xxx in your shell before commands.
```

Verify:

```bash
hf auth whoami
# Should print your HF username.
```

---

## 2. Pre-flight: prepare the artifact directory

For each artifact directory you're about to upload, do this checklist.

### 2.1. Confirm the file inventory

A clean artifact dir contains only what loaders need:

```text
<artifact-dir>/
├── README.md                          model card (YAML frontmatter + body)
├── config.json                        with the right `quantization_config` block
├── generation_config.json             (optional — not always present)
├── chat_template.jinja                (if the model uses one)
├── tokenizer.json
├── tokenizer_config.json
├── processor_config.json              (multimodal models — vLLM looks for it)
├── model.safetensors.index.json
└── model-NNNNN-of-MMMMM.safetensors   (multiple shards, ~5 GiB each)
```

```bash
ls -lh <artifact-dir>/
```

If you see anything else (`*.tmp`, `*.FAILED`, leftover `.pt` snapshots,
ad-hoc test files), **delete it before uploading** — those files would get
pushed unless the upload command is told otherwise.

### 2.2. Verify the YAML frontmatter

Open the artifact's `README.md` and confirm the top-of-file frontmatter is
intact. The exact tags depend on the scheme; the *required* keys are
`license`, `base_model`, `base_model_relation: quantized`, and
`quantized_by`. Example shape (FP8 build):

```yaml
---
license: <match-the-base-model>     # check the base model card
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- <arch-family>                     # e.g. llama, qwen3, mistral
- <scheme>                          # e.g. fp8 or awq
- <runtime-format>                  # e.g. compressed-tensors or autoawq
- quantization
- vllm
base_model: <org>/<base-name>       # full HF id of the source model
base_model_relation: quantized
quantized_by: <your-hf-username>
inference: false
---
```

The two metadata keys that make HF link your repo back to the base as a
*quantization* are:

| key | value |
|---|---|
| `base_model` | the full HF id of the source model |
| `base_model_relation` | `quantized` |

If those are wrong, your repo will appear as a standalone model rather than
showing up under **Quantizations** on the base model's page.

> **Match the base model's license.** The base model determines what
> license you're allowed to redistribute under. Check the base's HF card
> and copy its license string. Don't default to `apache-2.0` if the base
> uses Llama-3 / Gemma / a custom commercial-use clause.

### 2.3. Fix cross-links inside the artifact README

If the artifact `README.md` references files outside the artifact dir
(e.g. `../REPORT.md` or `../../docs/...`), those links will 404 on HF — HF
only sees the artifact dir, not its sibling files in the framework repo.

Two ways to fix:

**(a) Point at the public GitHub repo** (recommended once your framework
repo is pushed):

```bash
sed -i 's|\.\./REPORT\.md|https://github.com/<your-gh-user>/dgx-spark-quantization/blob/main/runs/<slug>/REPORT.md|g' \
  <artifact-dir>/README.md
```

**(b) Copy the report into the artifact dir** so it ships alongside:

```bash
cp runs/<slug>/REPORT.md <artifact-dir>/REPORT.md
sed -i 's|\.\./REPORT\.md|./REPORT.md|g' <artifact-dir>/README.md
```

Pick one and commit (or just run it before the upload — the change only
matters at upload time).

---

## 3. Create the empty HF repo

Web UI ([huggingface.co/new](https://huggingface.co/new)) or CLI:

```bash
hf repos create <repo-name> --type model
```

If you're publishing under an org, add `--organization <org-name>`.

By default repos are public. If you want it private during testing, append
`--private` and flip to public later in the web UI.

---

## 4. Upload

**Use `hf upload-large-folder`**, not the older single-file or
small-folder uploaders. The large-folder uploader chunks files into multipart
uploads, runs them in parallel, and resumes cleanly if the connection drops.

Before the upload, turn on the fast Rust transfer backend:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
```

```bash
hf upload-large-folder \
  <your-user>/<repo-name> \
  ./<artifact-dir> \
  --repo-type model \
  --num-workers 8
```

Wall-clock budget at ~50 MB/s upstream:

| artifact size | rough upload time |
|---|---|
| ~10 GiB |  4–6 min |
| ~24 GiB |  8–10 min |
| ~35 GiB | 12–15 min |
| ~70 GiB | 25–30 min |

**The uploader is resumable.** If your connection drops or you ctrl-C, just
rerun the same command — it will skip files that already match on the remote.

### What gets ignored automatically

`upload-large-folder` respects a built-in ignore list (`.git/`,
`__pycache__/`, `.DS_Store`, etc.). To exclude something explicitly, use
`--exclude '<glob>'`, e.g. to skip a stray test file:

```bash
hf upload-large-folder ... --exclude 'samples_*.jsonl'
```

---

## 5. Verify

In a browser:

1. Go to `https://huggingface.co/<your-user>/<repo-name>`.
2. Confirm the **model card** rendered correctly (title, tables, sections).
   If the YAML frontmatter is malformed, the page will show the raw YAML
   instead of a card.
3. Confirm the **Files and versions** tab lists the safetensors shards.
4. Hover the **right-hand sidebar**:
   - "Base model" links to the source.
   - The license badge matches what's in your YAML.
5. Click through to the base model's page → **Quantizations** section. Your
   repo should appear there within ~minutes (HF re-indexes on a short cron).

### Programmatic loading sanity check

```bash
# FP8 builds — vLLM + compressed-tensors loader.
vllm serve <your-user>/<repo-name> \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 4096

# AWQ builds — vLLM auto-detects `quant_method: awq` from config.json.
vllm serve <your-user>/<repo-name> \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 4096
```

(Add `--reasoning-parser <name>` for reasoning models.)

If the model loads and a `/v1/chat/completions` request returns a sensible
response, you're done.

---

## 6. Iterating on the model card after upload

The README on HF is a regular file in the repo — edit it via the web UI or
push a new version:

```bash
# Edit <artifact-dir>/README.md locally, then:
hf upload \
  <your-user>/<repo-name> \
  <artifact-dir>/README.md \
  README.md
```

(`upload`, singular, is fine for one-off file pushes; reserve
`upload-large-folder` for whole-tree shipments.)

---

## 7. (Optional) Make a "collection" that bundles a run's artifacts

If a run produces multiple artifacts (FP8 + AWQ + ...), group them under a
Hugging Face collection so users see them together with the eval comparison:

1. Visit
   [huggingface.co/new-collection](https://huggingface.co/new-collection).
2. Title: `<Base-Name> — DGX Spark quantizations` (or similar).
3. Description: paste the TL;DR table from the run's `README.md` and link
   to its `REPORT.md` on GitHub.
4. Add each new model repo, plus the base model, as items.

---

## 8. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| Upload stalls / hangs after a few minutes | connection drop, no resume | rerun the same `upload-large-folder` command — it resumes |
| "401 Unauthorized" | token expired or missing scope | regenerate a write token; `hf auth login` again |
| "413 Request Entity Too Large" | not using `upload-large-folder` | switch to `upload-large-folder` |
| Model card shows raw YAML on the page | frontmatter syntax error | run `python -c "import yaml; yaml.safe_load(open('README.md').read().split('---')[1])"`; fix until it parses |
| Repo doesn't appear under "Quantizations" on the base | wrong `base_model` / `base_model_relation` | edit the YAML frontmatter, re-upload `README.md`; re-index can take a few minutes |
| `vllm serve` rejects the FP8 build with "compressed-tensors version" error | loader older than the writer | upgrade `compressed-tensors` to ≥ 0.15 in the serving env |
| `Can't load image processor` on FP8 startup | missing `processor_config.json` (the FP8 build still declares the multimodal class) | confirm `processor_config.json` is in the artifact dir; if not, copy from the base or the AWQ sibling |

---

## 9. Privacy / safety checklist before pushing the button

- [ ] No `.env`, `*.token`, or stray credential files in the artifact dir
      (`grep -rEi 'TOKEN|SECRET|API_KEY' <artifact-dir>/` returns nothing).
- [ ] No `*.tmp`, `*.FAILED`, or stale shards.
- [ ] License in the model card matches what the base model permits.
- [ ] `quantized_by` is your actual HF username (not a placeholder).
- [ ] Repo name doesn't accidentally include the base author's name in a
      misleading way (the base author is credited in the model card; the
      repo is yours).
- [ ] If publishing under an org, you have the right to redistribute the
      base weights (check the base model's license and any usage-restriction
      clauses).

Once those are clean, run the upload command. After it finishes, check the
base model's "Quantizations" tab and confirm your repo showed up.

---

## Worked example: the `qwen3.6-35b-distill` run

Concrete commands for the two artifacts produced by
[`runs/qwen3.6-35b-distill/`](./runs/qwen3.6-35b-distill/). Each artifact's
`README.md` already carries the YAML frontmatter, the
`base_model_relation: quantized` link, embedded eval deltas, and a
`Source code & reproduction` link back to the framework repo — no
pre-flight cross-link fix needed.

The local artifact folder name is kept identical to the proposed HF repo
name so they can be referred to by a single string.

```bash
# Once: prerequisites (only needed the first time)
pip install -U "huggingface_hub[cli,hf_transfer]"
hf auth login            # paste your write token

export HF_HUB_ENABLE_HF_TRANSFER=1

# Create the two HF repos under your namespace.
hf repos create \
  Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-FP8-Dynamic \
  --type model
hf repos create \
  Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-AWQ-INT4 \
  --type model

# Upload (~12 min FP8 / ~9 min AWQ at typical home upstream).
hf upload-large-folder \
  <your-user>/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-FP8-Dynamic \
  ./artifacts/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-FP8-Dynamic \
  --repo-type model --num-workers 8

hf upload-large-folder \
  <your-user>/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-AWQ-INT4 \
  ./artifacts/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-AWQ-INT4 \
  --repo-type model --num-workers 8

# Verify on HF.
xdg-open https://huggingface.co/<your-user>/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-FP8-Dynamic
xdg-open https://huggingface.co/lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled
# ↑ on the base model, scroll to "Quantizations" — both repos should appear.
```

Adapt the artifact dirs / repo names for future runs by following the
per-step instructions above.

---

## Reference

- HF Hub docs: [Uploading models](https://huggingface.co/docs/huggingface_hub/guides/upload)
- HF model card spec: [Model cards](https://huggingface.co/docs/hub/model-cards)
- Frontmatter reference: [Model card metadata](https://huggingface.co/docs/hub/model-cards#model-card-metadata)
- `hf upload-large-folder` source: [`huggingface_hub`](https://github.com/huggingface/huggingface_hub)
