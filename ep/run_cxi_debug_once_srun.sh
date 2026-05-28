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
export MASTER_PORT="${MASTER_PORT:-29672}"
export LL_DEBUG_HOOK="${LL_DEBUG_HOOK:-0}"

master_host="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
master_addr="${MASTER_ADDR:-${master_host}}"
node_rank="${SLURM_NODEID:?SLURM_NODEID is required}"
nnodes="${SLURM_NNODES:?SLURM_NNODES is required}"

echo "[cxi-debug-once] host=$(hostname -f) node_rank=${node_rank}/${nnodes} master=${master_addr}:${MASTER_PORT}"

build_lock="/iopsstor/scratch/cscs/anowak/uccl_ep_cxi_build.lock"
(
  flock 9
  python3 -m pip install nanobind
  USE_LIBFABRIC_CXI=1 CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0 \
    python3 setup.py install
) 9>"${build_lock}"

python3 -c "import torch; import uccl.ep; print('[cxi-debug-once] import ok', torch.cuda.device_count(), flush=True)"

torchrun \
  --nnodes="${nnodes}" \
  --nproc_per_node=1 \
  --node_rank="${node_rank}" \
  --master_addr="${master_addr}" \
  --master_port="${MASTER_PORT}" \
  bench/debug_low_latency_once.py
