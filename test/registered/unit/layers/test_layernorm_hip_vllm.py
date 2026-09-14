"""HIP RMSNorm without AITER must call vLLM's 4-arg in-place fused_add_rms_norm.

A 6-arg call is AITER's signature and TypeErrors on gfx1151 (no AITER).
"""

from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRmsNormHipVllmSignature(CustomTestCase):
    def test_residual_path_is_four_arg_inplace(self):
        captured = {}

        def _fused(x, residual, weight, eps):
            captured["nargs"] = 4
            captured["order"] = (x is x, residual is residual)
            x.add_(1)
            residual.add_(1)

        with (
            patch("sglang.srt.layers.layernorm._has_vllm_rms_norm", True),
            patch("sglang.srt.layers.layernorm._use_aiter", False),
            patch(
                "sglang.srt.layers.layernorm.fused_add_rms_norm",
                _fused,
            ),
            patch(
                "sglang.srt.layers.layernorm.is_batch_invariant_mode_enabled",
                return_value=False,
            ),
        ):
            from sglang.srt.layers.layernorm import RMSNorm

            layer = RMSNorm(hidden_size=8, eps=1e-6)
            layer.weight = torch.nn.Parameter(torch.ones(8))
            x = torch.zeros(2, 8)
            residual = torch.zeros(2, 8)
            out, res_out = layer.forward_hip(x, residual)

        self.assertEqual(captured.get("nargs"), 4)
        self.assertIs(out, x)
        self.assertIs(res_out, residual)

    def test_empty_batch_skips_kernel(self):
        fused = MagicMock()
        with (
            patch("sglang.srt.layers.layernorm._has_vllm_rms_norm", True),
            patch("sglang.srt.layers.layernorm._use_aiter", False),
            patch("sglang.srt.layers.layernorm.fused_add_rms_norm", fused),
            patch(
                "sglang.srt.layers.layernorm.is_batch_invariant_mode_enabled",
                return_value=False,
            ),
        ):
            from sglang.srt.layers.layernorm import RMSNorm

            layer = RMSNorm(hidden_size=8, eps=1e-6)
            layer.weight = torch.nn.Parameter(torch.ones(8))
            x = torch.zeros(0, 8)
            residual = torch.zeros(0, 8)
            out, res_out = layer.forward_hip(x, residual)

        fused.assert_not_called()
        self.assertIs(out, x)
        self.assertIs(res_out, residual)


if __name__ == "__main__":
    import unittest

    unittest.main()
