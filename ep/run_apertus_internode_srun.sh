#!/usr/bin/env bash
set -euo pipefail

cd /users/anowak/open_source/uccl/ep

export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=9.0
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME_OVERRIDE:-hsn0}"
export UCCL_SOCKET_IFNAME="${UCCL_SOCKET_IFNAME_OVERRIDE:-hsn0}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-0}"
export UCCL_IB_GID_INDEX="${UCCL_IB_GID_INDEX:-${NCCL_IB_GID_INDEX}}"
export MASTER_PORT="${MASTER_PORT:-29580}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"

master_host="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
master_addr="${MASTER_ADDR:-${master_host}}"
node_rank="${SLURM_NODEID:?SLURM_NODEID is required}"
nnodes="${SLURM_NNODES:?SLURM_NNODES is required}"
nproc_per_node="${NPROC_PER_NODE:-4}"
test_script="${TEST_SCRIPT:-bench/test_internode_simple.py}"

echo "[runner] host=$(hostname -f) node_rank=${node_rank}/${nnodes} master_addr=${master_addr}:${MASTER_PORT}"
echo "[runner] NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} UCCL_SOCKET_IFNAME=${UCCL_SOCKET_IFNAME}"
echo "[runner] NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX} UCCL_IB_GID_INDEX=${UCCL_IB_GID_INDEX}"

echo "[runner] installing nanobind prerequisite"
python3 -m pip install nanobind

echo "[runner] clean/build/install uccl.ep using README/Apertus build path"
build_lock="/iopsstor/scratch/cscs/anowak/uccl_ep_build.lock"
(
  flock 9
  python3 setup.py clean
  CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0 python3 setup.py install
) 9>"${build_lock}"

python3 -c "import torch; import uccl.ep; print('[runner] torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count(), flush=True)"

if [[ "${test_script}" == "bench/test_internode.py" ]]; then
  torchrun \
    --nnodes="${nnodes}" \
    --nproc_per_node="${nproc_per_node}" \
    --node_rank="${node_rank}" \
    --master_addr="${master_addr}" \
    --master_port="${MASTER_PORT}" \
    "${test_script}" \
    --num-tokens=4096 \
    --hidden=7168 \
    --num-topk=8 \
    --num-experts=288 \
    --test-ll-compatibility
else
  torchrun \
    --nnodes="${nnodes}" \
    --nproc_per_node="${nproc_per_node}" \
    --node_rank="${node_rank}" \
    --master_addr="${master_addr}" \
    --master_port="${MASTER_PORT}" \
    "${test_script}"
fi
