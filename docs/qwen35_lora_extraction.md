# Qwen3.5 / Qwen3.8 LoRA extraction: compatibility and validation

Status: **experimental, architecture tests passing; no real 27B SVD run**.
This fork incorporates upstream arcee-ai/mergekit PR #686 on branch
qwen35-architecture-pr686 with an API-compatibility fix for current WeightInfo.

## What this enables

MergeKit can recognize Qwen3.5-family dense, MoE and multimodal tensor names,
including the 27B model's hybrid linear/full-attention decoder. For a
**same-shape full-precision checkpoint pair**, it can attempt to approximate
the fine-tuning difference with SVD-based LoRA factors.

It does **not** transfer LoRAs from a 9B model to a 27B model. Qwen's two
Image-2.1 PE teachers are Qwen3.5-9B derivatives; Qwen3.8-27B has 64 decoder
layers and hidden size 5120, versus 32 layers and 4096 for the PE teachers.
No architecture registry can turn incompatible weight shapes into a direct
low-rank subtraction.

## Preflight (downloads only small config.json files)

    python mergekit/scripts/qwen35_pair_preflight.py \
      --base Qwen/Qwen3.8-27B \
      --fine-tuned Qwen/Qwen-Image-2.1-PE-T2I

Expected: **BLOCKED** (layer count, hidden size and FFN dimensions differ).

For an extraction that could be shape-compatible, use the matching 9B base:

    python mergekit/scripts/qwen35_pair_preflight.py \
      --base Qwen/Qwen3.5-9B \
      --fine-tuned Qwen/Qwen-Image-2.1-PE-T2I

Repeat for PE-I2I. Passing this config-only preflight does not mean the
checkpoint differences are low-rank or that vision/non-LoRA changes are
preserved. Audit the real tensors and evaluate reconstruction separately.

For Qwen3.8-27B, select a genuine **27B fine-tune of the exact same base**
as the fine-tuned argument. The original Qwen3.8-27B checkpoint and
a full-precision 27B derivative are required; GGUF, GPTQ, EXL3 and OrcaSAQ
quantized artifacts are not substitutes for those extraction sources.

## SVD extraction, after the pair passes

Run from the repository root in an environment with compatible
torch, transformers, PEFT, accelerate, safetensors and mergekit dependencies:

    python -m mergekit.scripts.extract_lora \
      --base-model /path/to/base-bf16 \
      --model /path/to/compatible-finetune-bf16 \
      --out-path /path/to/adapter-r64 \
      --max-rank 64 \
      --device cuda:0

Review the extractor's tensor-selection and module-saving behaviour before
running on multimodal checkpoints. SVD approximation does not automatically
capture all normalization, embedding, convolution, vision or MTP changes.
The resulting adapter may need a separate compatible vision projector and
a loader that supports the particular Qwen3.5 LoRA tensors. Benchmark the
original fine-tune, BF16 base+adapter, and quantized base+adapter separately.

## To put T2I and I2I skills on 27B

1. Use the two 9B PE models as **teachers** to generate validated T2I and
   image-grounded I2I examples. Preserve image references, exact text,
   aspect-ratio decisions and the source model revision.
2. Train **new LoRA adapters on Qwen3.8-27B** using SFT/QLoRA, with the
   vision-capable 27B base for image editing.
3. Evaluate against both teachers and the unmodified 27B with strong prompts.
4. Use the same downstream image renderer for paired image-quality tests.

This is distillation, **not** weight extraction. There is no requirement
to perform full 27B fine-tuning and then SVD-extract the LoRA; direct LoRA
training is more efficient for this objective.

## Environment and testing notes (2026-10-17)

- 12700 Ubuntu: three RTX 3090s, 24 GiB each, initially all idle.
- Root filesystem: 1.8 TB total, only 5.4 GB available at inspection.
- No full PE 9B and Qwen3.8-27B BF16 extraction pair was present in the
  inspected Hugging Face cache. Large downloads were deliberately not started.
- Cherry-picked upstream PR #686 commits and repaired the obsolete
  WeightInfo(is_embed=True) usage for the current API.
- The PR's eight architecture tests passed after the fix. Additional tests
  validate the vocabulary metadata and compatibility preflight.
- Qwen/Qwen3.8-27B config revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
  was instantiated on the meta device: 1184/1184 model tensor names covered;
  the architecture registry also lists 15 extra MTP entries.
- A tiny Qwen3.5 multimodal checkpoint-pair extraction passed on CPU. A
  separate CUDA smoke run is included in the local test suite.
- These tests do not constitute a real 27B weight extraction or serving benchmark.

### Licence

The two Qwen Image-2.1 PE checkpoints use the Qwen Research License.
Review restrictions on commercial use and derivative models before
training a commercial student or distributing derived adapters.

No CI / GitHub Actions are required. Run the tests locally:

    PYTHONPATH=/home/op/qwen38-mergekit-research/.deps-py312 \
      /home/op/miniforge3/envs/qwen-image-21/bin/python -m pytest -q \
      tests/test_qwen35_architecture.py tests/test_qwen35_pair_preflight.py \
      tests/test_qwen35_lora_extraction_smoke.py
