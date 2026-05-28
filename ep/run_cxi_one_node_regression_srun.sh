#!/usr/bin/env bash
set -euo pipefail

cd /users/anowak/open_source/uccl/ep

export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=9.0
export USE_LIBFABRIC_CXI=1
export FI_PROVIDER=cxi
export NUM_MAX_NVL_PEERS="${NUM_MAX_NVL_PEERS:-4}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"

echo "[cxi-1node] host=$(hostname -f)"
echo "[cxi-1node] gpus=$(python3 -c 'import torch; print(torch.cuda.device_count())')"

build_lock="/iopsstor/scratch/cscs/anowak/uccl_ep_cxi_build.lock"
(
  flock 9
  python3 -m pip install nanobind
  USE_LIBFABRIC_CXI=1 NUM_MAX_NVL_PEERS="${NUM_MAX_NVL_PEERS}" \
    CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0 \
    python3 setup.py install
) 9>"${build_lock}"

python3 -c "import torch; import uccl.ep; print('[cxi-1node] import ok torch', torch.__version__, 'gpus', torch.cuda.device_count(), flush=True)"

echo "[cxi-1node] dual mode smoke"
MASTER_PORT=29681 torchrun \
  --nnodes=1 --node_rank=0 --master_addr="${MASTER_ADDR}" \
  --master_port=29681 --nproc_per_node=4 \
  bench/test_dual_mode.py --num-tokens 8 --hidden 1024 --num-experts 16

echo "[cxi-1node] intranode HT regression"
MASTER_PORT=29682 torchrun \
  --nnodes=1 --node_rank=0 --master_addr="${MASTER_ADDR}" \
  --master_port=29682 --nproc_per_node=4 \
  bench/test_intranode.py --num-tokens 128 --hidden 1024 \
  --num-topk 4 --num-experts 16

echo "[cxi-1node] low-latency local regression"
MASTER_PORT=29683 torchrun \
  --nnodes=1 --node_rank=0 --master_addr="${MASTER_ADDR}" \
  --master_port=29683 --nproc_per_node=2 \
  bench/test_low_latency.py --num-tokens 16 --hidden 7168 \
  --num-topk 4 --num-experts 16 --no-dispatch-use-fp8
