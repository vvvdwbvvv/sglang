#!/usr/bin/env bash
# Shared docker run wrapper for official SGLang ROCm images.
# Extra arguments are passed to `docker run` after the ROCm host flags.
# Example:
#   docker/drun-rocm.sh -e HF_TOKEN="$HF_TOKEN" \
#     lmsysorg/sglang:v0.5.18-rocm720-mi30x \
#     python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct \
#       --host 0.0.0.0 --port 30000
set -euo pipefail

if [[ ! -e /dev/kfd ]] || [[ ! -e /dev/dri ]]; then
  echo "drun-rocm.sh requires /dev/kfd and /dev/dri (host AMD KMD)." >&2
  exit 1
fi

group_args=(--group-add video)
if getent group render >/dev/null 2>&1; then
  group_args+=(--group-add render)
fi

hf_cache="${HF_HOME:-${HOME}/.cache/huggingface}"
mkdir -p "${hf_cache}"

extra_args=()
# gfx1151 (Strix Halo): persist TunableOp and force RDNA runtime defaults.
# Instinct images keep SGLANG_USE_AITER=1 from the official Dockerfile.
if [[ "${SGLANG_ROCM_ARCH:-}" == "gfx1151" ]] || printf '%s\0' "$@" | grep -qz 'gfx1151'; then
  tunable_dir="${HOME}/.cache/strix-halo-sglang-tunableop"
  mkdir -p "${tunable_dir}"
  extra_args+=(
    -v "${tunable_dir}:/root/.tunableop"
    -e SGLANG_USE_AITER=0
    -e SGLANG_ROCM_FUSED_DECODE_MLA=0
    -e SGLANG_DIFFUSION_TARGET_DEVICE=rocm
    -e SGLANG_USE_ROCM_VAE=0
    -e PYTORCH_TUNABLEOP_ENABLED=1
    -e PYTORCH_TUNABLEOP_FILENAME=/root/.tunableop/tunableop_results.csv
  )
fi

# --network=host and --privileged are required by RDMA. Drop them if you do not need it.
exec docker run -it --rm \
  --network=host \
  --privileged \
  --device=/dev/kfd \
  --device=/dev/dri \
  --ipc=host \
  --shm-size "${SGLANG_ROCM_SHM_SIZE:-32g}" \
  "${group_args[@]}" \
  "${extra_args[@]}" \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v "${hf_cache}:/root/.cache/huggingface" \
  "$@"
