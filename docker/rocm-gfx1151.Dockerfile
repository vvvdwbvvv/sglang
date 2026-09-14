# SGLang for AMD Strix Halo (gfx1151 / RDNA 3.5).
#
# Official rocm.Dockerfile GPU_ARCH stages are Instinct (gfx942/gfx950) and
# gfx1250. They do not ship gfx1151 PyTorch/hipBLASLt payloads. This image is a
# thin wrap of a TheRock gfx1151 base: compile sgl-kernel for gfx1151, install
# SRT + diffusion_common, and default to RDNA runtime flags.
#
# Build on the Halo APU (linux/amd64, /dev/kfd). Do not cross-compile from
# another gfx.
#
#   docker build -f docker/rocm-gfx1151.Dockerfile \
#     -t lmsysorg/sglang:dev-rocm-therock-gfx1151 .
#
# Diffusion (primary):
#   docker compose -f docker/compose.rocm-gfx1151.yaml up
#   # or: docker/drun-rocm.sh ... sglang generate --attention-backend torch_sdpa ...
#
# LLM:
#   docker compose -f docker/compose.rocm-gfx1151.yaml --profile llm up

ARG BASE_IMAGE=kyuz0/vllm-therock-gfx1151:stable@sha256:f89c8c689ade28877ade980ba0f29b3142af16c6ebb7f3f285311d38bc81a8a2
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV GPU_ARCH=gfx1151-therock
ENV GPU_ARCH_LIST=gfx1151
ENV PYTORCH_ROCM_ARCH=gfx1151
ENV AMDGPU_TARGET=gfx1151
ENV SGLANG_USE_AITER=0
ENV SGLANG_ROCM_FUSED_DECODE_MLA=0
ENV SGLANG_DIFFUSION_TARGET_DEVICE=rocm
ENV SGLANG_USE_ROCM_VAE=0
ENV SGLANG_DISABLE_CUDNN_CHECK=1
ENV HIP_FORCE_DEV_KERNARG=1
ENV PYTORCH_TUNABLEOP_ENABLED=1
ENV PYTORCH_TUNABLEOP_FILENAME=/root/.tunableop/tunableop_results.csv
ENV HF_HOME=/root/.cache/huggingface
ENV PIP_CONSTRAINT=/etc/sglang/constraints/torch-rocm.txt

WORKDIR /sgl-workspace

RUN mkdir -p /etc/sglang/constraints /root/.tunableop \
    && python3 -c "import torch, torchvision; open('/etc/sglang/constraints/torch-rocm.txt','w').write(f'torch=={torch.__version__}\ntorchvision=={torchvision.__version__}\n')"

# Rust is required to build sglang.srt.rust_extensions during pip install.
ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 --retry 5 --retry-delay 3 --retry-all-errors -sSf https://sh.rustup.rs | sh -s -- -y \
    && rustc --version && cargo --version
ENV CARGO_BUILD_JOBS=4

COPY . /sgl-workspace/sglang

RUN cd /sgl-workspace/sglang/python/sglang/kernels/aot \
    && rm -f pyproject.toml \
    && mv pyproject_rocm.toml pyproject.toml \
    && AMDGPU_TARGET=gfx1151 python3 setup_rocm.py install

# diffusion_hip pulls CUDA-only st_attn/vsa. diffusion_common is the DiT stack
# (diffusers, imageio, ...); cache-dit/peft are optional speed knobs.
RUN cd /sgl-workspace/sglang \
    && cp python/pyproject_other.toml python/pyproject.toml \
    && pip install --no-cache-dir -e "python[srt_hip,diffusion_common]" --no-build-isolation \
    && pip install --no-cache-dir "cache-dit==1.3.0" "peft>=0.18.0,<0.19.0" \
    && (pip cache purge 2>/dev/null || true)

EXPOSE 30000 30010
CMD ["/bin/bash"]
