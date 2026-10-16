# SPDX-License-Identifier: LGPL-3.0-only
"""Check MergeKit tensor-name coverage for real Qwen3.8 config without weights."""
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoConfig

from mergekit.architecture import arch_info_for_config
from mergekit.common import get_auto_cls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(args.model)
    architecture = arch_info_for_config(config)
    arch_keys = {w.name for w in architecture.all_weights(config)}
    model_cls = get_auto_cls(config.architectures[0])
    with torch.device("meta"):
        model = model_cls.from_config(config)
    model_keys = set(model.state_dict())
    missing = sorted(model_keys - arch_keys)
    extra = sorted(arch_keys - model_keys)
    result = {
        "model": args.model,
        "config_commit": getattr(config, "_commit_hash", None),
        "architecture": config.architectures,
        "model_tensor_count": len(model_keys),
        "mergekit_tensor_count": len(arch_keys),
        "uncovered_model_tensors": missing,
        "extra_architecture_entries": extra[:25],
        "extra_architecture_entries_count": len(extra),
    }
    print(json.dumps(result, indent=2))
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
