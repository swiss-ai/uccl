#!/usr/bin/env bash
set -euo pipefail

cd /users/anowak/open_source/uccl/ep

export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=9.0
export USE_LIBFABRIC_CXI=1
export UCCL_EP_TRANSPORT=cxi
export FI_PROVIDER=cxi
export FI_CXI_DEFAULT_CQ_SIZE="${FI_CXI_DEFAULT_CQ_SIZE:-131072}"
export FI_CXI_DEFAULT_TX_SIZE="${FI_CXI_DEFAULT_TX_SIZE:-16384}"
export FI_CXI_OFLOW_BUF_SIZE="${FI_CXI_OFLOW_BUF_SIZE:-1073741824}"
export FI_CXI_RDZV_THRESHOLD="${FI_CXI_RDZV_THRESHOLD:-1073741824}"
export FI_CXI_RX_MATCH_MODE="${FI_CXI_RX_MATCH_MODE:-software}"
export FI_MR_CACHE_MONITOR="${FI_MR_CACHE_MONITOR:-userfaultfd}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export UCCL_PROXY_TRACE="${UCCL_PROXY_TRACE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export MASTER_PORT="${MASTER_PORT:-29671}"

master_host="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export MASTER_ADDR="${MASTER_ADDR:-${master_host}}"
export WORLD_SIZE="${SLURM_NTASKS:?SLURM_NTASKS is required}"
export RANK="${SLURM_PROCID:?SLURM_PROCID is required}"
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1

echo "[cxi-gdb] host=$(hostname -f) rank=${RANK}/${WORLD_SIZE} master=${MASTER_ADDR}:${MASTER_PORT}"

build_lock="/iopsstor/scratch/cscs/anowak/uccl_ep_cxi_build.lock"
(
  flock 9
  python3 -m pip install nanobind
  USE_LIBFABRIC_CXI=1 CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0 \
    python3 setup.py install
) 9>"${build_lock}"

python3 -c "import torch; import uccl.ep; print('[cxi-gdb] import ok', torch.cuda.device_count(), flush=True)"

gdb -q -batch \
  -ex "set pagination off" \
  -ex "set confirm off" \
  -ex "run" \
  -ex "thread apply all bt full" \
  --args python3 bench/debug_low_latency_once.py
