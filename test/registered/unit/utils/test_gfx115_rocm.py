"""gfx1151 (Strix Halo) arch helper and attention defaults.

Guards the RDNA 3.5 split from Instinct: SRT must not default to AITER, and
diffusion must not copy SRT's triton backend.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.utils.common import (
    is_gfx115_supported,
    is_gfx1250_supported,
    is_gfx942_supported,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestIsGfx115Supported(CustomTestCase):
    def tearDown(self):
        is_gfx115_supported.cache_clear()
        is_gfx942_supported.cache_clear()
        is_gfx1250_supported.cache_clear()

    def _with_arch(self, arch: str | None, hip):
        is_gfx115_supported.cache_clear()
        is_gfx942_supported.cache_clear()
        is_gfx1250_supported.cache_clear()
        props = SimpleNamespace(gcnArchName=arch or "")
        with (
            patch("torch.version.hip", hip),
            patch("torch.cuda.get_device_properties", return_value=props),
        ):
            return (
                is_gfx115_supported(),
                is_gfx942_supported(),
                is_gfx1250_supported(),
            )

    def test_strix_halo_is_gfx115_only(self):
        g115, g942, g1250 = self._with_arch("gfx1151", "7.13.0")
        self.assertTrue(g115)
        self.assertFalse(g942)
        self.assertFalse(g1250)

    def test_strix_point_is_gfx115(self):
        g115, _, _ = self._with_arch("gfx1150", "7.13.0")
        self.assertTrue(g115)

    def test_instinct_and_gfx1250_are_not_gfx115(self):
        for arch in ("gfx942", "gfx950", "gfx1250"):
            with self.subTest(arch=arch):
                g115, _, _ = self._with_arch(arch, "7.2.0")
                self.assertFalse(g115)

    def test_non_hip_is_false(self):
        g115, _, _ = self._with_arch("gfx1151", None)
        self.assertFalse(g115)


class TestSrtDefaultAttnGfx115(CustomTestCase):
    def test_mha_defaults_to_triton_on_gfx115(self):
        from sglang.srt.arg_groups.model_override_base import get_default_attn_backend

        platform = MagicMock()
        platform.is_out_of_tree.return_value = False
        platform.is_hopper_with_cuda_12_3 = False
        platform.is_sm100 = False
        platform.is_hip = True
        current = MagicMock()
        current.is_out_of_tree.return_value = False
        model_config = MagicMock()
        model_config.hf_config.architectures = ["LlamaForCausalLM"]
        server_args = SimpleNamespace(
            speculative_algorithm=None,
            speculative_eagle_topk=None,
            page_size=1,
            tp_size=1,
        )
        with (
            patch(
                "sglang.srt.arg_groups.model_override_base.current_platform",
                current,
            ),
            patch(
                "sglang.srt.arg_groups.model_override_base.get_platform",
                return_value=platform,
            ),
            patch(
                "sglang.srt.arg_groups.model_override_base.is_gfx115_supported",
                return_value=True,
            ),
        ):
            self.assertEqual(
                get_default_attn_backend(server_args, False, model_config),
                "triton",
            )

    def test_mha_defaults_to_aiter_on_cdna(self):
        from sglang.srt.arg_groups.model_override_base import get_default_attn_backend

        platform = MagicMock()
        platform.is_out_of_tree.return_value = False
        platform.is_hopper_with_cuda_12_3 = False
        platform.is_sm100 = False
        platform.is_hip = True
        current = MagicMock()
        current.is_out_of_tree.return_value = False
        model_config = MagicMock()
        model_config.hf_config.architectures = ["LlamaForCausalLM"]
        server_args = SimpleNamespace(
            speculative_algorithm=None,
            speculative_eagle_topk=None,
            page_size=1,
            tp_size=1,
        )
        with (
            patch(
                "sglang.srt.arg_groups.model_override_base.current_platform",
                current,
            ),
            patch(
                "sglang.srt.arg_groups.model_override_base.get_platform",
                return_value=platform,
            ),
            patch(
                "sglang.srt.arg_groups.model_override_base.is_gfx115_supported",
                return_value=False,
            ),
        ):
            self.assertEqual(
                get_default_attn_backend(server_args, False, model_config),
                "aiter",
            )


class TestDiffusionDefaultAttnGfx115(CustomTestCase):
    def test_rocm_platform_maps_aiter_to_sdpa_on_gfx115(self):
        from sglang.multimodal_gen.runtime.platforms.interface import (
            AttentionBackendEnum,
        )
        from sglang.multimodal_gen.runtime.platforms.rocm import RocmPlatform

        import torch

        with patch(
            "sglang.srt.utils.common.is_gfx115_supported",
            return_value=True,
        ):
            cls = RocmPlatform.get_attn_backend_cls_str(
                AttentionBackendEnum.AITER, 128, torch.bfloat16
            )
        self.assertIn("sdpa.SDPABackend", cls)

    def test_set_default_attention_backend_sdpa_on_gfx115(self):
        from sglang.multimodal_gen.runtime.platforms.interface import (
            AttentionBackendEnum,
        )
        from sglang.multimodal_gen.runtime.server_args.server_args import (
            ServerArgs,
        )

        args = SimpleNamespace(attention_backend=None)
        with (
            patch(
                "sglang.multimodal_gen.runtime.server_args.server_args.current_platform.is_rocm",
                return_value=True,
            ),
            patch(
                "sglang.srt.utils.common.is_gfx115_supported",
                return_value=True,
            ),
        ):
            ServerArgs._set_default_attention_backend(args)
        self.assertEqual(
            args.attention_backend,
            AttentionBackendEnum.TORCH_SDPA.name.lower(),
        )

    def test_set_default_attention_backend_aiter_on_cdna(self):
        from sglang.multimodal_gen.runtime.platforms.interface import (
            AttentionBackendEnum,
        )
        from sglang.multimodal_gen.runtime.server_args.server_args import (
            ServerArgs,
        )

        args = SimpleNamespace(attention_backend=None)
        with (
            patch(
                "sglang.multimodal_gen.runtime.server_args.server_args.current_platform.is_rocm",
                return_value=True,
            ),
            patch(
                "sglang.srt.utils.common.is_gfx115_supported",
                return_value=False,
            ),
        ):
            ServerArgs._set_default_attention_backend(args)
        self.assertEqual(
            args.attention_backend,
            AttentionBackendEnum.AITER.name.lower(),
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
