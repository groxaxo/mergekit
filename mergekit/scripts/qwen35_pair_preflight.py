# SPDX-License-Identifier: LGPL-3.0-only
"""Fail-fast Qwen3.5/Qwen3.8 checkpoint compatibility preflight.

This checks model *configuration* only. It does not download weights, compare
tensor contents, or establish that a fine-tune is representable by a LoRA.
Run before mergekit-extract-lora to avoid costly, invalid cross-size attempts.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO_ID = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
TEXT_SHAPE_FIELDS = (
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "vocab_size",
    "layer_types",
    "num_experts",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
)
VISION_SHAPE_FIELDS = (
    "depth",
    "hidden_size",
    "intermediate_size",
    "num_heads",
    "out_hidden_size",
    "patch_size",
    "temporal_patch_size",
    "spatial_merge_size",
    "num_position_embeddings",
    "in_channels",
)


def read_config(source: str) -> dict:
    if source.lower().endswith(".gguf"):
        raise ValueError("GGUF quantized weights are not a full-precision SVD extraction source.")
    path = Path(source).expanduser()
    if path.exists():
        config_path = path / "config.json" if path.is_dir() else path
        if config_path.name != "config.json":
            raise ValueError("Local source must be a model directory or config.json file.")
        return json.loads(config_path.read_text(encoding="utf-8"))
    if not REPO_ID.fullmatch(source):
        raise ValueError("Expected a local model/config.json or Hugging Face owner/repository ID.")
    url = "https://huggingface.co/" + quote(source, safe="/") + "/resolve/main/config.json"
    request = Request(url, headers={"User-Agent": "mergekit-qwen35-pair-preflight/1.0"})
    with urlopen(request, timeout=25) as response:
        data = response.read(2_000_000)
        if len(data) >= 2_000_000:
            raise ValueError("Remote config exceeds the maximum allowed size.")
    return json.loads(data)


def validate_pair(base: dict, finetuned: dict) -> tuple[list[str], list[str]]:
    """Return errors and warnings for direct same-shape weight subtraction."""
    errors: list[str] = []
    warnings: list[str] = []

    def compare(label: str, left, right) -> None:
        if left != right:
            if isinstance(left, list) and isinstance(right, list):
                errors.append(
                    f"{label}: base has {len(left)} entries, fine-tuned has "
                    f"{len(right)}; first values {left[:4]!r} vs {right[:4]!r}"
                )
            else:
                errors.append(f"{label}: base={left!r}, fine-tuned={right!r}")

    compare("model_type", base.get("model_type"), finetuned.get("model_type"))
    compare("architectures", base.get("architectures"), finetuned.get("architectures"))
    for label, config in (("base", base), ("fine-tuned", finetuned)):
        if config.get("quantization_config"):
            errors.append(f"{label} is quantized; extract from high-precision checkpoints.")
        if "qwen3_5" not in str(config.get("model_type", "")).lower():
            errors.append(f"{label} does not declare a Qwen3.5-family model_type.")

    base_text = base.get("text_config", base)
    ft_text = finetuned.get("text_config", finetuned)
    for field in TEXT_SHAPE_FIELDS:
        left, right = base_text.get(field), ft_text.get(field)
        if left is not None or right is not None:
            compare("text_config." + field, left, right)

    base_vision, ft_vision = base.get("vision_config"), finetuned.get("vision_config")
    if bool(base_vision) != bool(ft_vision):
        errors.append("vision_config: only one checkpoint has a vision encoder.")
    elif base_vision and ft_vision:
        for field in VISION_SHAPE_FIELDS:
            left, right = base_vision.get(field), ft_vision.get(field)
            if left is not None or right is not None:
                compare("vision_config." + field, left, right)

    for field in ("mtp_num_hidden_layers", "mtp_use_dedicated_embeddings"):
        left, right = base_text.get(field), ft_text.get(field)
        if left != right:
            warnings.append(
                f"text_config.{field} differs ({left!r} vs {right!r}); "
                "inspect MTP tensor coverage before extraction."
            )

    warnings.append(
        "Config match alone is insufficient: verify tensor names/shapes, tokenizer "
        "alignment, non-LoRA changes, SVD reconstruction error and adapter loading."
    )
    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Local model directory/config.json or HF repo ID")
    parser.add_argument("--fine-tuned", required=True, help="Local model directory/config.json or HF repo ID")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args(argv)
    try:
        base = read_config(args.base)
        finetuned = read_config(args.fine_tuned)
        errors, warnings = validate_pair(base, finetuned)
    except Exception as exc:
        errors, warnings = [f"{type(exc).__name__}: {exc}"], []
    result = {
        "base": args.base,
        "fine_tuned": args.fine_tuned,
        "compatible_for_direct_subtraction": not errors,
        "errors": errors,
        "warnings": warnings,
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("PASS: same-shape config preflight" if not errors else "BLOCKED: incompatible extraction pair")
        for issue in errors:
            print("  ERROR:", issue)
        for issue in warnings:
            print("  NOTE:", issue)
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
