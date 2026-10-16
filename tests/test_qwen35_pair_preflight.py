"""Unit tests for fail-fast extraction compatibility checks."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "mergekit" / "scripts" / "qwen35_pair_preflight.py"
SPEC = importlib.util.spec_from_file_location("qwen35_pair_preflight", SCRIPT)
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def config(*, layers=32, hidden=4096, intermediate=12288, vision=True, arch=None):
    return {
        "model_type": "qwen3_5",
        "architectures": [arch or "Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "num_hidden_layers": layers,
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_attention_heads": 16 if layers == 32 else 24,
            "num_key_value_heads": 4,
            "vocab_size": 248320,
            "layer_types": ["linear_attention"] * (layers - 1) + ["full_attention"],
        },
        **({"vision_config": {"depth": 24, "hidden_size": 1152, "out_hidden_size": hidden}}
           if vision else {}),
    }


def test_same_shape_pair_passes():
    errors, warnings = preflight.validate_pair(config(), config())
    assert errors == []
    assert warnings


def test_9b_to_27b_rejected_with_shape_reasons():
    errors, _ = preflight.validate_pair(
        config(), config(layers=64, hidden=5120, intermediate=17408)
    )
    assert any("num_hidden_layers" in error for error in errors)
    assert any("hidden_size" in error for error in errors)
    assert any("intermediate_size" in error for error in errors)


def test_vision_mismatch_rejected():
    errors, _ = preflight.validate_pair(config(), config(vision=False))
    assert any("vision_config" in error for error in errors)


def test_architecture_mismatch_rejected():
    errors, _ = preflight.validate_pair(
        config(), config(arch="Qwen3_5ForCausalLM")
    )
    assert any("architectures" in error for error in errors)


def test_quantized_source_rejected():
    ft = config()
    ft["quantization_config"] = {"quant_method": "gptq"}
    errors, _ = preflight.validate_pair(config(), ft)
    assert any("quantized" in error for error in errors)


def test_read_local_config_and_cli(tmp_path, capsys):
    base = tmp_path / "base"
    ft = tmp_path / "ft"
    base.mkdir()
    ft.mkdir()
    (base / "config.json").write_text(json.dumps(config()), encoding="utf-8")
    (ft / "config.json").write_text(json.dumps(config()), encoding="utf-8")
    assert preflight.read_config(str(base))["model_type"] == "qwen3_5"
    assert preflight.main(["--base", str(base), "--fine-tuned", str(ft)]) == 0
    assert "PASS" in capsys.readouterr().out


def test_gguf_source_rejected():
    with pytest.raises(ValueError, match="GGUF"):
        preflight.read_config("Qwen3.8-27B-UD-Q4_K_XL.gguf")
