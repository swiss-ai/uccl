# UCCL EP Build Notes for Apertus CXI

This note records the concrete environment assumptions needed to build `uccl.ep`
inside the Apertus/Slurm container and compares them with the working
`pplx_garden_cxi` setup.

## Status

The README build path works for the Python/CUDA EP extension in this container:

```bash
cd /users/anowak/open_source/uccl/ep
python3 setup.py clean
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0 python3 setup.py install
python3 -c "import torch; import uccl.ep"
```

The install command completed and installed:

```text
/opt/venv/lib/python3.12/site-packages/uccl/ep.abi3.so
```

The import check exited successfully.

## Active Python and CUDA

The active interpreter is the container venv:

```text
/opt/venv/bin/python3
Python 3.12.3
```

Torch is provided by the container image:

```text
torch 2.10.0a0+b558c986e8.nv25.11
torch CUDA 13.0
/usr/local/lib/python3.12/dist-packages/torch
```

Use the CUDA toolkit under `/usr/local/cuda` for UCCL EP:

```text
/usr/local/cuda/bin/nvcc
CUDA 13.0
```

Do not use the capstor CUDA path for this build. It is missing headers needed by
the current UCCL EP sources, notably `cuda_fp8.h`.

## Required Python Dependency

`nanobind` is required by `ep/setup.py` before compilation starts. It was not
present initially in `/opt/venv`; installing it fixed the Python dependency
blocker:

```bash
python3 -m pip install nanobind
```

Verified installed version:

```text
nanobind 2.12.0
/opt/venv/lib/python3.12/site-packages/nanobind
```

## Why `TORCH_CUDA_ARCH_LIST=9.0`

The node exposes GH200 GPUs:

```text
NVIDIA GH200 120GB, compute capability 9.0
```

`ep/setup.py` can auto-detect GH200 and `9.0` when `nvidia-smi` works from the
Python subprocess. In this container it worked during the successful install and
the build summary showed:

```text
Detected GPU compute capability: 9.0
GH200 detected (GPU: NVIDIA GH200 120GB, CPU: aarch64)
Device Arch: 9.0
```

Still, pass `TORCH_CUDA_ARCH_LIST=9.0` explicitly for repeatability. This is the
documented README escape hatch for architecture detection problems and avoids
accidentally inheriting the broad container default:

```text
TORCH_CUDA_ARCH_LIST=8.0 8.6 9.0 10.0 11.0 12.0+PTX
```

## Compile Recipe

Always re-check `/users/anowak/open_source/uccl/ep/README.md` before changing
the EP build or benchmark flow. The README is the source of truth for the EP
build/test entry points.

Do not start a new Slurm job from inside this container.

The CUDA compile rules documented by `ep/README.md` are:

- primary build/install path:

```bash
python setup.py install
```

- optional Makefile path, noted by the README as possibly deprecated:

```bash
make -j install
```

- optional docker wheel path from the repo root:

```bash
bash build.sh cu12 ep --install
```

- import check:

```bash
python -c "import torch; import uccl.ep"
```

- if GPU-memory RDMA registration fails with an `Invalid argument` style error,
  rebuild with `USE_DMABUF=1`;
- EFA support is auto-detected by the existence of `/opt/amazon/efa`, which
  enables the EFA-specific RDMA path with `-DEFA`;
- if architecture auto-detection fails or runtime reports an invalid device
  function, set `TORCH_CUDA_ARCH_LIST` explicitly;
- if compilation gets into a bad state, run `python setup.py clean`.

For this Apertus container, use the README `setup.py` path with the concrete
CUDA/Python environment below.

Minimum environment:

```bash
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=9.0
```

Prerequisite check:

```bash
python3 -c "import torch; print(torch.__version__, torch.version.cuda)"
python3 -c "import nanobind; print(nanobind.__file__)"
which nvcc
nvcc --version
```

If `nanobind` is missing from `/opt/venv`, install it into the active venv:

```bash
python3 -m pip install nanobind
```

Clean and build/install:

```bash
cd /users/anowak/open_source/uccl/ep
python3 setup.py clean
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0 python3 setup.py install
```

Verify the installed extension:

```bash
python3 -c "import torch; import uccl.ep; print('ok')"
```

Expected install target in this container:

```text
/opt/venv/lib/python3.12/site-packages/uccl/ep.abi3.so
```

Do not use this CUDA path for the EP build:

```text
/capstor/scratch/cscs/anowak/envs/apertus2-alps4-temp/cuda
```

That tree exists, but it is missing headers required by the current EP sources,
including `cuda_fp8.h`.

The `make` path exists in the README, but the verified path in this container is
`python3 setup.py install`. The standalone sources under `ep/tests/Makefile`
are not the right validation path here; with CUDA 13 they currently fail on
`cudaDeviceProp::clockRate`.

### EFA/CXI Adaptation Point

The README's EFA note is the main build-rule area that may need an Apertus/CXI
change similar in spirit to `pplx_garden_cxi`.

Current UCCL EP behavior:

```text
/opt/amazon/efa exists -> setup.py/Makefile add -DEFA
/opt/amazon/efa missing -> build without EFA-specific RDMA path
```

PPLX does not use that EFA auto-detection model for Apertus/CXI. It explicitly
targets CXI through libfabric and uses the custom writedata-capable libfabric
environment:

```bash
export LIBFABRIC_HOME=/root/libfabric_writedata/install
export LD_LIBRARY_PATH=/root/libfabric_writedata/install/lib:$LD_LIBRARY_PATH
export FI_CXI_ENABLE_WRITEDATA=1
```

For UCCL EP, the current successful compile does not need PPLX's libfabric
settings because UCCL EP links the verbs path (`ibverbs`, `nl-3`,
`nl-route-3`, `numa`). The future open question is whether Apertus/CXI runtime
support should stay on the existing verbs path, use `USE_DMABUF=1`, or add a
separate CXI/libfabric detection path analogous to PPLX. That should be driven
by a real two-node internode RDMA test, not by the intranode NVLink result.

## Clangd Configuration

The `.clangd` file is editor configuration, not the build recipe. It was reset
to follow the same shape and environment assumptions as:

```text
/users/anowak/open_source/pplx_garden_cxi/.clangd
```

That means `.clangd` intentionally uses the capstor Apertus include tree:

```text
--cuda-path=/capstor/scratch/cscs/anowak/envs/apertus2-alps4-temp/cuda
/capstor/scratch/cscs/anowak/envs/apertus2-alps4-temp/cuda/include
/capstor/scratch/cscs/anowak/envs/apertus2-alps4-temp/cpp-includes/13
/capstor/scratch/cscs/anowak/envs/apertus2-alps4-temp/python-dist-packages/torch/include
```

The only intentional difference from PPLX is the project-local include list.
The PPLX grouped-gemm paths were replaced with UCCL paths:

```text
/users/anowak/open_source/uccl
/users/anowak/open_source/uccl/include
/users/anowak/open_source/uccl/ep/include
/users/anowak/open_source/uccl/ep/src
/users/anowak/open_source/uccl/collective
/users/anowak/open_source/uccl/p2p
```

This creates an important distinction:

- for compiling `uccl.ep`, use `/usr/local/cuda`;
- for `.clangd`, use the PPLX-style capstor include paths.

## PPLX CXI Build and Runtime Assumptions

The sibling project `/users/anowak/open_source/pplx_garden_cxi` is not just
using stock libfabric. Its local scripts point at a custom writedata-capable
libfabric installation:

```bash
export LIBFABRIC_HOME=/root/libfabric_writedata/install
export LD_LIBRARY_PATH=/root/libfabric_writedata/install/lib:$LD_LIBRARY_PATH
export FI_CXI_ENABLE_WRITEDATA=1
```

Relevant files:

```text
/users/anowak/open_source/pplx_garden_cxi/my_build_python.sh
/users/anowak/open_source/pplx_garden_cxi/srun_compile.sh
/users/anowak/open_source/pplx_garden_cxi/test_all2all.sh
/users/anowak/open_source/pplx_garden_cxi/benchmark_scatter_write.sbatch
```

PPLX runtime scripts also carry CXI tuning for the scatter/write benchmarks:

```bash
export FI_CXI_OFLOW_BUF_SIZE=1073741824
export FI_CXI_RDZV_THRESHOLD=1073741824
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=16384
export FI_CXI_DISABLE_HOST_REGISTER=0
export FI_CXI_RX_MATCH_MODE=software
export FI_MR_CACHE_MONITOR=userfaultfd
```

The PPLX Rust provider path explicitly asks libfabric for the CXI provider and
RMA/HMEM capability in `fabric-lib/src/efa/efa_devinfo.rs`:

```text
prov_name = "cxi"
caps = FI_MSG | FI_RMA | FI_HMEM | FI_LOCAL_COMM | FI_REMOTE_COMM
ep type = FI_EP_RDM
mr_mode = FI_MR_ENDPOINT | FI_MR_ALLOCATED | FI_MR_PROV_KEY
```

The PPLX write path in `fabric-lib/src/efa/efa_domain.rs` uses libfabric
`fi_writedata` when the prepared operation has immediate-data flags, and
`fi_writemsg` otherwise. The important point is that PPLX's CXI path depends on
libfabric provider behavior and `FI_CXI_ENABLE_WRITEDATA=1`.

PPLX CUDA/Torch build assumptions also match this container:

```text
/usr/local/cuda/include
/usr/local/cuda/lib64
torch.utils.cmake_prefix_path from python3
```

## PPLX Commit Range From `faa797d`

The requested PPLX reference point is:

```text
faa797df79a70e87a980bcdf867ba4a7ff4c17e3
```

The relevant committed changes after that point are:

```text
b55a329 Update README
7035135 fix gpu/network detection
9cccd42 Debug log changes
93564ea Fix: This commit implements a for overlapping ptrs
55a2694 Fixed error where we were reading the weight values of the next stage (in efa) instead of the current stage ones
2bc6e80 set the correct device number for each worker
75a314d Fix: Paged write was missing the correct change to work with the offset-from-MR-base addressing system
83106a5 Added sh file to run tests
04cca97 fabric-lib: force HMEM CUDA path and propagate cuda_device to worker threads
64fd0f5 Fix: Added a kill gpu processes before running the tests
1a7031d Add scatter_write and paged_write Python bindings to fabric API
5e83a71 Add scatter write benchmark and plot scripts
78452df Added ALPS edf toml file
2464b59 Updated scatter_write sbatch to use the max 16k of the TX size
```

There are many other A2A/debug/test commits in the range, but the commits above
are the ones that change build assumptions, CXI/libfabric behavior, or the
scatter-write benchmark path.

### `7035135`: GPU and Network Detection for CXI

This commit changed the PPLX tests from assuming only
`/sys/class/infiniband_verbs/uverbs*` to a more Apertus/CXI-aware count:

- `torch.cuda.device_count()` is considered for visible GPU count;
- `/sys/class/cxi/cxi*` is counted as the preferred network endpoint source;
- `/sys/class/infiniband_verbs/uverbs*` remains the fallback;
- `PPLX_TEST_NETS_PER_GPU` can override the inferred network count.

For UCCL this maps to a runtime-test concern, not a compile concern: do not
assume EFA-style `uverbs` counts describe the node. On Apertus, CXI devices are
the relevant network endpoints for PPLX's libfabric path. UCCL EP's current path
uses verbs and its own `UCCL_SOCKET_IFNAME`/`UCCL_IB_GID_INDEX` settings from the
README.

### `9cccd42`: Container Build Assumptions

This commit converted `srun_compile.sh` from launching its own `srun` into a
script meant to run inside the already-allocated container. That matches the
current UCCL situation: do not start another Slurm job for compilation.

It also established the core PPLX environment:

```bash
export NCCL_NET="AWS Libfabric"
export NCCL_NET_PLUGIN="ofi"
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/root/libfabric_writedata/install/lib:$LD_LIBRARY_PATH
export RUST_BACKTRACE=full
export FI_CXI_ENABLE_WRITEDATA=1
```

The Python wheel install was changed to avoid isolation and dependency
resolution:

```bash
python3 -m pip install ./dist/*.whl \
  --break-system-packages --force-reinstall --no-build-isolation --no-deps
```

That matters because the container already supplies the intended Torch/CUDA
stack; build isolation can accidentally select the wrong Python/package
environment.

### `93564ea`: GDR Buffer Page Ownership

This commit changed PPLX's GDR allocation to allocate one extra 64 KiB page of
slack before and after the aligned region:

```text
cuMemAlloc(bytesize + 2 * page_size)
align_to(device_ptr + page_size, page_size)
```

The reason was practical: `gdr_pin_buffer` pins whole pages, and an aligned
subregion at the edge of an allocation could overlap pages owned by other CUDA
allocations. That caused unrelated buffers such as padded indices, expert
offsets, and token offsets to overlap/corrupt each other.

This does not directly affect the UCCL EP compile path, but it is relevant when
comparing runtime RDMA/GPU-memory assumptions. PPLX has explicit defensive
padding around GDR-pinned regions; UCCL EP relies on its own CUDA allocation and
verbs memory registration paths.

### `55a2694`: A2A Dispatch Receive Stage Metadata

This commit fixed an A2A kernel bug where the EFA/CXI receive path could read
weight metadata from the next pipeline stage rather than the current one. It
added a per-stage `padded_index` and used that saved value instead of rereading
from shared stage metadata after it may have advanced.

This is a PPLX A2A correctness fix, not a UCCL build requirement. It is useful
context because it shows that some PPLX changes after `faa797d` are kernel
correctness changes rather than environment or CXI-provider setup.

### `2bc6e80`: Correct CUDA Device Per Worker

This commit propagated `cuda_device` through PPLX's `TransferEngineBuilder` into
each worker thread and changed GPU PCI detection to use the current CUDA device
instead of hard-coding device 0.

Key behavior:

```text
cudaGetDevice(&current_device)
cudaGetDeviceProperties(current_device)
cudaSetDevice(cuda_device) in RDMA worker thread
cudaSetDevice(cuda_device) in UVM worker thread
```

`04cca97` later extended this same theme with more HMEM registration debug
context. The transfer to UCCL is conceptual: any runtime test that starts proxy
or worker threads must be careful about CUDA device context. UCCL EP's
`setup.py` build does not create worker threads, so this is not a compile
blocker.

### `75a314d`: CXI Offset-From-MR-Base Addressing

This commit fixed PPLX paged writes for the CXI addressing model:

```text
rma_iov.addr = dst_offset + dst_stride * page_idx
```

The important assumption is that CXI/libfabric uses an offset-from-memory-region
base model: the remote key encodes the base virtual address, and `addr` is an
offset within that registered memory region.

That is PPLX/libfabric-specific. UCCL EP's verbs path uses the address and rkey
semantics from its `RDMAConnectionInfo` and `ibv_*` operations, so this exact
fix is not something to copy into UCCL. It is still important when comparing
PPLX's working CXI behavior against UCCL's verbs behavior.

### `04cca97`: CXI HMEM Registration and CUDA Device Context

This commit changed the PPLX libfabric registration path:

- requested `FI_MR_HMEM` in the CXI `mr_mode`;
- disabled DMA-BUF fallback in `fabric-lib/src/mr.rs` to force the HMEM CUDA
  registration path;
- logged CUDA pointer attributes and `fi_mr_regattr` context;
- propagated `cuda_device` into RDMA worker and UVM worker threads;
- called `cudaSetDevice(cuda_device)` at worker-thread startup.

The worker-thread `cudaSetDevice` part is important. Without it, a thread can
register or interact with a CUDA pointer under the wrong device context.

The current dirty PPLX worktree differs from the commit: `FI_MR_HMEM` is now
commented out again in `fabric-lib/src/efa/efa_devinfo.rs`, with a note that it
seems to be activated already. So the committed history says "force HMEM"; the
current local state says "do not request HMEM explicitly, but still use CXI
HMEM behavior."

### `64fd0f5`: Clean Up Leftover GPU Processes Before Tests

PPLX test scripts added:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

The reason in the commit message is that failed tests sometimes left processes
behind, and later tests then failed randomly. This is relevant for UCCL EP
benchmarks too: before running the README `torchrun` tests, check for leftover
GPU processes if previous runs failed or hung.

### `1a7031d`: Scatter Write API

This commit added Python bindings for scatter writes:

```python
TransferEngine.submit_scatter_writes(
    src_mr,
    dsts: list[ScatterTarget],
    imm_data,
    on_done,
    on_error,
)
```

The Rust binding creates a `ScatterTransferRequest` using:

```text
GroupTransferRouting::AllDomainsShardBytes
```

So the PPLX scatter path is not a generic single write. It shards byte ranges
across all available domains/NICs and optionally sends immediate data. In
`fabric-lib/src/efa/efa_domain.rs`, immediate-data operations are posted through
`fi_writedata`; non-immediate operations use `fi_writemsg`.

### `5e83a71` and `2464b59`: Scatter Write Benchmark

`5e83a71` added the benchmark path that exercises the scatter-write API. The
sbatch script explicitly exports CXI writedata and tuning variables:

```bash
export FI_CXI_ENABLE_WRITEDATA=1
export FI_CXI_OFLOW_BUF_SIZE=1073741824
export FI_CXI_RDZV_THRESHOLD=1073741824
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=32768
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RX_MATCH_MODE=software
export FI_MR_CACHE_MONITOR=userfaultfd
```

`2464b59` corrected the TX queue size:

```bash
export FI_CXI_DEFAULT_TX_SIZE=16384
```

The current dirty PPLX worktree further changes the scatter benchmark:

- account is `infra01`;
- `FI_CXI_DISABLE_HOST_REGISTER=0`;
- `FI_LOG_LEVEL=debug`;
- `CHUNKS_PER_DEST` is propagated into the benchmark;
- telemetry checks are commented out instead of always printed.

That current local dirty state is the most relevant one if reproducing the
latest PPLX experiments, but the committed range explains how the assumptions
arrived.

### `78452df`: Apertus EDF File

This commit added an EDF file for the Apertus image:

```toml
image="jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/apertus-2:alps4-dev-3f34f16a8d1f2fc0"
mounts = ["${SCRATCH}:${SCRATCH}", "${HOME}:${HOME}", "/capstor:/capstor"]
workdir = "${PWD}"
writable = true
PMIX_MCA_psec = "native"
com.hooks.cxi.enabled = "false"
```

The current UCCL container was launched with the named environment
`apertus2-alps4-temp`, but this commit confirms the same broad image/mount/PMIX
shape used by the PPLX work.

## What Transfers to UCCL

Useful assumptions that do transfer:

- Use `/opt/venv/bin/python3`.
- Use Torch from the active Python environment.
- Use `/usr/local/cuda`, not the capstor CUDA tree.
- Use explicit GH200 architecture, `TORCH_CUDA_ARCH_LIST=9.0`.
- Use the same Slurm/container network mode when launching distributed tests:
  `--network=disable_rdzv_get`.

Assumptions that do not directly transfer:

- `FI_CXI_ENABLE_WRITEDATA=1` is a libfabric/CXI provider knob. UCCL EP uses
  verbs (`ibverbs`, `ibv_*`) in its current EP path, not PPLX's libfabric
  `fi_writedata` path.
- `LIBFABRIC_HOME=/root/libfabric_writedata/install` is required for PPLX's
  CXI/libfabric build, but not for compiling UCCL EP with `setup.py`.
- PPLX's scatter-write benchmarks are validating a libfabric write-data path;
  UCCL EP's README tests validate UCCL's DeepEP-compatible EP API over its RDMA
  proxy path.

## UCCL EP Build Details

`ep/setup.py` links against:

```text
ibverbs
nl-3
nl-route-3
numa
CUDA runtime from /usr/local/cuda/lib64
Torch libraries from /usr/local/lib/python3.12/dist-packages/torch/lib
```

The relevant build behavior:

- `/opt/amazon/efa` controls `-DEFA` auto-detection.
- `USE_DMABUF=1` changes GPU memory registration to the DMA-BUF path, useful
  when `nvidia_peermem`, `efa_nv_peermem`, or `ib_peer_mem` are unavailable.
- `USE_INTEL_RDMA_NIC=1` or auto-detected `irdma*` enables the Intel RDMA path.
- `TORCH_CUDA_ARCH_LIST=9.0` generates `sm_90` code for GH200.

For this container, the successful minimal build/install was:

```bash
cd /users/anowak/open_source/uccl/ep
python3 setup.py clean
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0 python3 setup.py install
python3 -c "import torch; import uccl.ep"
```

## Tests and Benchmarks

Detailed scratchpad for this debugging session:

```text
/users/anowak/agents-scratchpad/debugging-development/uccl-compile-apertus-2026-05-26.md
```

The README's documented import test passes:

```bash
python3 -c "import torch; import uccl.ep"
```

The README's documented intranode benchmark passes when launched with an
explicit master address and port. It was run successfully at token counts 128,
4096, and 8192:

```bash
cd /users/anowak/open_source/uccl/ep
MASTER_ADDR=127.0.0.1 MASTER_PORT=29573 \
torchrun --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=29573 \
  --nproc_per_node=4 \
  bench/test_intranode.py --num-tokens 4096 \
  --hidden 7168 --num-topk 8 --num-experts 256
```

The README shows `--nproc_per_node=8`, but this Apertus node currently reports
four GH200 GPUs via `nvidia-smi`, so `4` is the matching local value.

Avoid `torchrun --standalone` in this environment. It selected
`localhost:0`, and c10d failed before UCCL code ran:

```text
The client socket cannot be initialized to connect to localhost:0
errno: 1 - Operation not permitted
```

With the explicit address/port command above, all BF16/FP8 dispatch/combine
checks passed, including the zero-recv-rank case.

`bench/test_dual_mode.py` also passes with explicit rendezvous:

```bash
cd /users/anowak/open_source/uccl/ep
MASTER_ADDR=127.0.0.1 MASTER_PORT=29575 \
torchrun --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=29575 \
  --nproc_per_node=4 \
  bench/test_dual_mode.py
```

It created low-latency and high-throughput buffers at the same time and printed
`[dual-mode] PASS`. This is an existing repo test, but it is not one of the
three README-documented benchmark paths.

### What Intranode Does and Does Not Prove

The intranode test does not validate the real RDMA/NIC data path. In
`bench/test_intranode.py`, the buffer is constructed with:

```python
test_ll_compatibility, num_rdma_bytes = False, 0
Buffer(
    group,
    int(2e9),        # num_nvl_bytes
    num_rdma_bytes,  # 0
    low_latency_mode=False,
    is_intranode=True,
)
```

So this path validates the Python extension, CUDA kernels, NCCL bootstrap,
CUDA IPC/NVLink/local data movement, and high-throughput EP runtime wiring.

The nuance is that proxy objects are still created. `bench/buffer.py` always
calls `initialize_uccl(...)`, which is why the logs show:

```text
Registered proxies for device N (high-throughput mode)
```

That log line alone does not mean RDMA traffic is being exercised. Since
`num_rdma_bytes == 0`, `runtime.set_rdma_buffer(...)` is skipped. The normal
dispatch path only calls `internode_dispatch(...)` when:

```python
self.runtime.get_num_rdma_ranks() > 1
```

On a one-node, four-GPU launch, that condition is not a meaningful validation
of NIC/RDMA behavior.

### Low-Latency and Internode Attempts on One Node

`bench/test_low_latency.py` was attempted on one node with four local ranks:

```bash
cd /users/anowak/open_source/uccl/ep
MASTER_ADDR=127.0.0.1 MASTER_PORT=29577 \
torchrun --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=29577 \
  --nproc_per_node=4 \
  bench/test_low_latency.py --num-tokens=128 \
  --hidden=7168 --num-topk=8 --num-experts=256
```

It initialized low-latency proxies on all four GPUs and entered the first
experiment, then made no further progress before the 240s timeout. This is not
conclusive for the README internode path because the README documents
`test_low_latency.py` as an internode benchmark.

`bench/test_internode_simple.py` was also attempted as an extra smoke test, but
it is not a README-documented path. It exposed a local device-placement problem
on ranks 1-3:

```text
Tensor found on device cuda:0 but backend constrained to cuda:1
Tensor found on device cuda:0 but backend constrained to cuda:2
Tensor found on device cuda:0 but backend constrained to cuda:3
```

Treat that as a separate local smoke-test issue, not as the canonical README
internode result.

`bench/test_internode.py` is also not a clean one-node test here because it
hard-requires `--num-processes=8`. A two-node by four-GPU launch matches that
requirement exactly.

### Required Next Test: Real Internode

The next meaningful validation is a real two-node run. One physical node cannot
validate the remote RDMA/NIC path. With `--nnodes=1`, the code either stays on
local/NVLink/IPC behavior, hits assumptions meant for `num_nodes >= 2`, or
hangs in paths waiting for behavior that only makes sense with remote peers.

Minimum clean shape for this allocation style:

```text
2 nodes x 4 GPUs = 8 total ranks
```

On node 0:

```bash
cd /users/anowak/open_source/uccl/ep
export MASTER_ADDR=<node0_ip_or_hostname>
export MASTER_PORT=29582
export NCCL_SOCKET_IFNAME=<interface-for-master-addr>
export UCCL_SOCKET_IFNAME=<same-interface>
export NCCL_IB_GID_INDEX=<gid-index>
export UCCL_IB_GID_INDEX=<same-gid-index>

torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  bench/test_low_latency.py --num-tokens=128 \
  --hidden=7168 --num-topk=8 --num-experts=256
```

On node 1:

```bash
cd /users/anowak/open_source/uccl/ep
export MASTER_ADDR=<same_node0_ip_or_hostname>
export MASTER_PORT=29582
export NCCL_SOCKET_IFNAME=<interface-for-master-addr>
export UCCL_SOCKET_IFNAME=<same-interface>
export NCCL_IB_GID_INDEX=<gid-index>
export UCCL_IB_GID_INDEX=<same-gid-index>

torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  bench/test_low_latency.py --num-tokens=128 \
  --hidden=7168 --num-topk=8 --num-experts=256
```

Then test normal internode mode:

```bash
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=<0-or-1> \
  --master_addr=$MASTER_ADDR --master_port=29583 \
  bench/test_internode.py --num-processes=8 \
  --num-tokens=4096 --hidden=7168 --num-topk=8 \
  --num-experts=256 --test-ll-compatibility
```

For internode tests, follow the README and set the bootstrap/network variables
explicitly:

```bash
export NCCL_SOCKET_IFNAME=<interface-for-master-addr>
export UCCL_SOCKET_IFNAME=<same-interface>
export NCCL_IB_GID_INDEX=<gid-index>
export UCCL_IB_GID_INDEX=<same-gid-index>
```

Do not use `ep/tests/Makefile` as the primary validation path for this
environment. Those standalone CUDA benchmark sources currently use
`cudaDeviceProp::clockRate`, which fails to compile with the CUDA 13 headers in
this container. The project README points to the Python `setup.py` build and the
`ep/bench/*.py` torchrun benchmarks instead.

## Current Local Repo State

At the time this note was written, no UCCL source change was needed for the
README build/import path. The local worktree already had:

```text
 M pyproject.toml
?? .clangd
?? AGENTS.md
?? APERTUS_UCCL_EP_BUILD_NOTES.md
```

`ep/Makefile` was restored to its original state after investigation; the
working path above uses `ep/setup.py`.

`.clangd` is intentionally based on the PPLX clangd file with only UCCL
project include paths substituted.

## Appendix: Full PPLX Commit List From `faa797d..HEAD`

This is the full ordered commit list inspected in
`/users/anowak/open_source/pplx_garden_cxi`:

```text
b55a329 Update README
3ce62f9 remove debugging codes
52663e2 expose dispatch-time metadata
7758794 more tests
7035135 fix gpu/network detection
9cccd42 Debug log changes
93564ea Fix: This commit implements a for overlapping ptrs
55a2694 Fixed error where we were reading the weight values of the next stage (in efa) instead of the current stage ones
3be9558 Added changes to the tests
83106a5 Added sh file to run tests
a57bd8e Add the debug_poison function
2bc6e80 set the correct device number for each worker
75a314d Fix: Paged write was missing the correct change to work with the offset-from-MR-base addressing system
f868a6b Added changes to test_transfer_engine.py
6ee1f5e Merge branch 'dev' into fix-a2a-devices
2a00018 Added other debug dicts
8446ca0 fix tests
04cca97 fabric-lib: force HMEM CUDA path and propagate cuda_device to worker threads
23995b6 fabric-lib: fix worker join ordering on drop, add ImmCounter/GdrCounter reset
9ca4115 fabric-debug: add --register-only and --register-only-wait subcommands
fde5e6b p2p-all-to-all: add wait_ready/reset_counters and enrich AllToAllDebugState
bb7c4aa a2a-kernels: comment out debug printfs, add clarifying comments, introduce raw_peer
e501cfb p2p-all-to-all: initialize done-flags at construction and expose wait_ready/reset_counters to Python
80d3427 tests: wire up per-rep barrier+reset, gate debug output, update NVL4 configs
c3b57a4 Merge branch 'dev' into fix-a2a-devices
5220273 Merge pull request #2 from henrique/fix-a2a-devices
b85137f Changed TP4-DP2-NVL2 test, and made a new one with the high max num token counts that fails
64fd0f5 Fix: Added a kill gpu processes before running the tests
2ff1367 Make DP token_offset ordering deterministic
bc703b5 Add A2A debug instrumentation and test updates
10db8d1 simple comment
62371ba Add NVTX instrumentation to dispatch send/recv kernel launches
0d239ce Add Megatron all-gather baseline to A2A benchmark for comparison
1a7031d Add scatter_write and paged_write Python bindings to fabric API
5e83a71 Add scatter write benchmark and plot scripts
78452df Added ALPS edf toml file
d00b35c Translated comments
2464b59 Updated scatter_write sbatch to use the max 16k of the TX size
```
