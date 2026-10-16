"""Tiny multimodal Qwen3.5 end-to-end SVD LoRA smoke test (no large weights)."""
import os
import subprocess
import sys
from pathlib import Path

import torch
import pytest
from safetensors.torch import load_file
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
)

from tests.test_qwen35_architecture import _dense_config


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_tiny_qwen35_vlm_rank1_delta_extraction(tmp_path, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    config = _dense_config()
    torch.manual_seed(17)
    base = Qwen3_5ForConditionalGeneration(config)
    fine_tuned = Qwen3_5ForConditionalGeneration(config)
    fine_tuned.load_state_dict(base.state_dict())
    target_name = "model.language_model.layers.0.mlp.gate_proj.weight"
    with torch.no_grad():
        weights = dict(fine_tuned.named_parameters())
        matrix = weights[target_name]
        left = torch.linspace(-0.15, 0.25, matrix.shape[0])
        right = torch.linspace(-0.2, 0.3, matrix.shape[1])
        matrix.add_(torch.outer(left, right))
    expected_delta = (
        dict(fine_tuned.named_parameters())[target_name]
        - dict(base.named_parameters())[target_name]
    ).detach()

    base_path = tmp_path / "base"
    fine_tuned_path = tmp_path / "finetuned"
    out_path = tmp_path / "adapter"
    base.save_pretrained(base_path, safe_serialization=True)
    fine_tuned.save_pretrained(fine_tuned_path, safe_serialization=True)

    env = os.environ.copy()
    cmd = [
        sys.executable, "-m", "mergekit.scripts.extract_lora",
        "--base-model", str(base_path),
        "--model", str(fine_tuned_path),
        "--out-path", str(out_path),
        "--max-rank", "2",
        "--device", device,
        "--embed-lora",
        "--include-regex", r"^model\.language_model\.layers\.0\.mlp\.gate_proj$",
    ]
    result = subprocess.run(
        cmd, cwd=Path(__file__).resolve().parents[1],
        env=env, text=True, capture_output=True, timeout=90
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    tensors = load_file(str(out_path / "adapter_model.safetensors"))
    prefix = "base_model.model.model.language_model.layers.0.mlp.gate_proj"
    a = tensors[prefix + ".lora_A.weight"].float()
    b = tensors[prefix + ".lora_B.weight"].float()
    reconstruction = b @ a
    assert reconstruction.shape == expected_delta.shape
    relative_error = (
        (reconstruction - expected_delta).norm() / expected_delta.norm()
    ).item()
    assert relative_error < 0.002, relative_error
