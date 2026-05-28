# Repository Agent Instructions

## Development Tools

For Python development, prefer `uv`, `ty`, and `ruff` when they apply.

The main paper/reference for UCCL-EP is **"UCCL-EP: Portable
Expert-Parallel Communication"**, available at
https://arxiv.org/abs/2512.19849. Read and keep this paper's architecture in
mind before making substantial EP changes: GPU kernels enqueue compact
communication commands, CPU proxy threads issue RDMA operations on behalf of
the GPU, and correctness depends on preserving the ordering/completion
semantics expected by the low-latency and high-throughput kernels.

## UCCL EP Build Environment

For references always use the ep/README.md, here you will have all isntructions for compiling and what is this implementation about.

This Apertus container uses the Python environment in `/opt/venv` and the CUDA
toolkit in `/usr/local/cuda`.

Before UCCL EP debugging/build/test work that needs the Apertus environment,
start an interactive Apertus Slurm container from outside any existing Slurm
container:

```bash
srun \
  -A infra01 \
  --time=5:00:00 \
  --reservation=SD-69241-apertus-1-5-0 \
  --environment=apertus2-alps4-temp \
  --container-mounts="${SCRATCH}:${SCRATCH},${HOME}:${HOME},/capstor:/capstor,/iopsstor:/iopsstor" \
  --network=disable_rdzv_get \
  --mpi=pmix \
  --pty bash
```

If the Slurm CLI filter rejects the command, expand the mount variables before
retrying. On this system the working expanded form was:

```bash
srun \
  -A infra01 \
  --time=5:00:00 \
  --reservation=SD-69241-apertus-1-5-0 \
  --environment=apertus2-alps4-temp \
  --container-mounts=/iopsstor/scratch/cscs/anowak:/iopsstor/scratch/cscs/anowak,/users/anowak:/users/anowak,/capstor:/capstor,/iopsstor:/iopsstor \
  --network=disable_rdzv_get \
  --mpi=pmix \
  --pty bash
```

For the UCCL EP Python/CUDA extension, use:

```bash
cd /users/anowak/open_source/uccl/ep
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0 python3 setup.py install
python3 -c "import torch; import uccl.ep"
```

Do not use the capstor CUDA tree for this build path. It is missing headers
needed by the current EP sources, including `cuda_fp8.h`.

## Torchrun Rules

Do not use `torchrun --standalone` for UCCL EP tests in this container. It can
select `localhost:0`, which fails in this environment before UCCL code runs:

```text
c10d: The client socket cannot be initialized to connect to localhost:0
errno: 1 - Operation not permitted
```

Always pass an explicit master address and port for local torchrun tests:

```bash
cd /users/anowak/open_source/uccl/ep
MASTER_ADDR=127.0.0.1 MASTER_PORT=29573 \
torchrun --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=29573 \
  --nproc_per_node=4 \
  bench/test_intranode.py --num-tokens 4096 \
  --hidden 7168 --num-topk 8 --num-experts 256
```

Use `--nproc_per_node` equal to the number of visible GPUs. On the verified
Apertus node, `nvidia-smi` reported four GH200 GPUs, so `--nproc_per_node=4`
was the correct local value.

For internode runs, follow the EP README and set bootstrap/network variables
explicitly, including:

```bash
export NCCL_SOCKET_IFNAME=<interface-for-master-addr>
export UCCL_SOCKET_IFNAME=<same-interface>
export NCCL_IB_GID_INDEX=<gid-index>
export UCCL_IB_GID_INDEX=<same-gid-index>
```

Do not create new Slurm jobs from inside an existing Apertus/Slurm container
unless the user explicitly asks for that.

Do not reuse or attach to an existing Slurm allocation/job for validation or
debugging unless the user explicitly asks for that exact job to be reused.
Always request a fresh allocation for new UCCL EP validation runs, and record
the new job id in the scratchpad and handoff.

## `ucc_ep_Cxi` Handoff

Current UCCL EP/RDMA code paths are verbs-based and expect Linux RDMA-core /
`libibverbs` devices. On Apertus/CXI/Slingshot systems where `fi_info -p cxi`
works but `fi_info -p verbs` and `ibv_devices` do not expose usable devices, the
existing verbs backend is not expected to work for internode traffic.

Observed CXI-only node state on `nid006698`:

```text
fi_info -p cxi
  provider: cxi
  fabric: cxi
  domain: cxi0
  domain: cxi1
  domain: cxi2
  domain: cxi3
  type: FI_EP_RDM
  protocol: FI_PROTO_CXI

fi_info -p verbs
  fi_getinfo: -61 (No data available)

ibv_devices
  device              node GUID
  ------              ----------------
  # no devices listed
```

CXI is exposed through the libfabric `cxi` provider, not through libverbs and
not through libfabric's `verbs` provider. A proper `ucc_ep_Cxi` implementation
should therefore add a libfabric/CXI EP transport path instead of trying to
force CXI through the verbs abstraction.

Suggested direction:

- Detect/enable CXI via libfabric, e.g. `FI_PROVIDER=cxi`, and validate against
  real CXI domains such as `cxi0`-`cxi3` when available.
- Prefer the site's writedata-capable libfabric setup when needed:
  `LIBFABRIC_HOME=/root/libfabric_writedata/install`,
  `LD_LIBRARY_PATH=/root/libfabric_writedata/install/lib:$LD_LIBRARY_PATH`, and
  `FI_CXI_ENABLE_WRITEDATA=1`.
- Keep the existing verbs/RDMA path intact for InfiniBand/RoCE systems.
- Add a separate transport adapter/backend for CXI send/receive, memory
  registration, completion progress, and bootstrap integration.
- Validate with an actual internode CXI run; intranode import/build success is
  not sufficient to prove CXI transport support.
- See `APERTUS_UCCL_EP_BUILD_NOTES.md` for prior build/environment notes and the
  open CXI/libfabric detection question.

## CXI Debugging Methodology

When debugging `ucc_ep_Cxi` on Apertus/Alps, treat UCCL and the CXI provider as
joint suspects. Do not assume a hang or crash is only a UCCL FIFO/proxy bug:
first verify whether the behavior can follow from libfabric/CXI requirements or
provider-specific semantics.

Use the references below while debugging, especially the CXI provider man page
and CSCS Alps libfabric documentation. Relevant CXI checks include:

- Provider/domain state: confirm `FI_PROVIDER=cxi`, `fi_info -p cxi`, visible
  domains such as `cxi0`-`cxi3`, endpoint type `FI_EP_RDM`, and that the
  container is using the host-compatible libfabric/CXI stack.
- Addressing and authorization: ensure endpoint names from `fi_getname` are
  exchanged correctly, inserted into the AV, and that VNI/service allocation is
  compatible across ranks.
- MR mode and keys: CXI requires endpoint-bound scalable MR modes for this path;
  use provider-generated 64-bit keys when requested and verify stale/cached MR
  behavior before blaming UCCL buffer lifetime.
- HMEM/device memory: verify `FI_HMEM` capability, CUDA HMEM MR registration,
  DMABUF/ODP/translation-cache behavior, and whether failures are specific to
  GPU memory versus host memory.
- Progress and completions: CXI uses manual progress, so every posted operation
  must have a matching completion-drain path. Check CQ errors with
  `fi_cq_readerr`, track outstanding operations precisely, and remember that
  completion level matters: transmit completion is not the same as target
  visibility.
- Ordering and fences: validate write-then-atomic assumptions against CXI's
  supported ordering and `FI_FENCE` semantics. Request required capabilities in
  `fi_getinfo` hints, not only at post time.
- Resource limits and flow control: check provider queue sizes, rendezvous and
  triggered-operation limits, overflow/request buffers, and outstanding command
  pressure before assuming kernel-side deadlock.

Preferred debugging order:

1. Reproduce with the smallest libfabric/CXI microbench that uses the same
   operation class: host write, CUDA/HMEM write, atomic, fenced write+atomic,
   or completion polling.
2. Reproduce with the smallest UCCL harness (`bench/debug_low_latency_once.py`)
   before running the full benchmark loop.
3. Add rank/thread logs around proxy init, peer exchange, command dequeue,
   post, CQ poll, FIFO ack, and destroy. Keep logs runtime-gated.
4. If there is a native crash or unexplained background-thread exit, get a
   native backtrace with gdb before making speculative edits.
5. Record every attempt in the active `~/agents-scratchpad/debugging-development`
   session with timestamps, including negative results and exact Slurm job ids.

For UCCL-specific reasoning, keep the UCCL-EP paper architecture in mind: GPU
kernels enqueue compact commands into FIFO channels, CPU proxy threads issue
the network operations, and correctness depends on proxy-managed flow control,
ordering, and completion handling. For CXI-specific reasoning, use the
libfabric/CXI references below as the source of truth for provider behavior.

For high-throughput internode/CXI debugging, also open the local visual
explainer `uccl/UCCL_EP_CXI_HT_EXPLAINER.html`. Treat it as both a human
explainer and an agent code-navigation map. It documents how to travel through
the implementation from the UCCL-EP paper architecture to the concrete code
paths: HT/LL kernels, buffer ownership, D2H FIFO command packing, CPU proxy
posting, CXI completion/barrier semantics, and the hidden 8-local-rank
assumptions that can masquerade as transport bugs.


## References

**UCCL**
- Repo: https://github.com/uccl-project/uccl
- EP README: https://github.com/uccl-project/uccl/tree/main/ep
- Blog (short): https://uccl-project.github.io/posts/uccl-ep/
- Blog (full): https://uccl-project.github.io/posts/uccl-ep-full/
- Paper: https://arxiv.org/abs/2512.19849
- Local CXI/HT/LL code-navigation explainer: `uccl/UCCL_EP_CXI_HT_EXPLAINER.html`

**libfabric**
- Docs: https://ofiwg.github.io/libfabric/
- Repo: https://github.com/ofiwg/libfabric
- CXI provider: https://ofiwg.github.io/libfabric/main/man/fi_cxi.7.html

**CSCS / Alps**
- Libfabric on Alps: https://docs.cscs.ch/software/communication/libfabric/

---
