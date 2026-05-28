# Handoff: UCCL EP CXI/libfabric Backend with Native Atomics

Active scratchpad:
`/users/anowak/agents-scratchpad/debugging-development/uccl-cxi-native-atomics-2026-05-27/notes.md`.
Use it for timestamped debugging notes before continuing this work.

## Goal

Add a UCCL EP transport backend for HPE Slingshot/CXI systems that expose the
network through libfabric's `cxi` provider rather than Linux `libibverbs`.

The intended CXI backend should use:

- `fi_write` / `fi_writemsg` for payload movement.
- `fi_atomic` / `fi_atomicmsg` for EP signaling counters.
- `FI_FENCE` on atomic operations that signal completion of preceding writes.
- CUDA HMEM memory registration through `fi_mr_regattr`.
- CXI offset-from-memory-region-base addressing.

This backend should coexist with the current verbs backend. Do not remove or
regress the existing InfiniBand/RoCE/EFA code paths.

## Confirmed target system facts

On the target CXI node, `fi_info -p cxi -v` reports the required broad
capabilities:

```text
protocol: FI_PROTO_CXI
caps: FI_RMA, FI_ATOMIC, FI_WRITE, FI_REMOTE_WRITE, FI_FENCE, FI_HMEM, ...
mr_mode: FI_MR_ALLOCATED, FI_MR_PROV_KEY, FI_MR_ENDPOINT
msg_order: FI_ORDER_RMA_WAW, FI_ORDER_ATOMIC_RAR, FI_ORDER_ATOMIC_RAW,
           FI_ORDER_ATOMIC_WAR, FI_ORDER_ATOMIC_WAW
inject_size: 192
```

The target environment also has valid CXI domains such as `cxi0`-`cxi3` via:

```bash
fi_info -p cxi
```

but does not expose usable verbs devices:

```bash
fi_info -p verbs   # No data available
ibv_devices        # no usable RDMA devices
```

Therefore the current UCCL EP verbs path is not expected to work for internode
traffic on this node.

## Important assumption

Native CXI/libfabric atomics are known to work in the target environment. Do not
spend early implementation effort designing around an EFA-style software-atomic
fallback unless later testing proves it is needed.

The preferred CXI protocol is:

```text
fi_write(payload)
fi_atomic(counter += delta, with FI_FENCE when it signals write completion)
```

not:

```text
fi_writedata(payload/control)
receiver CPU decodes immediate data and performs software atomics
```

`fi_writedata` may still be useful for debugging or simple control messages, but
it should not be the primary EP counter mechanism.

## Why this differs from EFA/PPLX

The sibling project `/users/anowak/open_source/pplx_garden_cxi` is useful as a
libfabric/CXI reference, but its current scatter/write path primarily uses
`fi_writedata` for immediate-data notifications. UCCL EP should not blindly copy
that design for counters because CXI native atomics are available and match
UCCL's existing non-EFA native RDMA atomic model more closely.

PPLX still provides valuable implementation details:

- Explicit provider selection: `prov_name = "cxi"`.
- Endpoint type: `FI_EP_RDM`.
- CQ format: `FI_CQ_FORMAT_DATA`.
- MR mode: `FI_MR_ENDPOINT | FI_MR_ALLOCATED | FI_MR_PROV_KEY`.
- CUDA HMEM registration with `fi_mr_regattr` and `FI_HMEM_CUDA`.
- Endpoint-bound MRs: `fi_mr_bind(mr, ep)` then `fi_control(mr, FI_ENABLE)`.
- CXI remote addresses are treated as offsets from the registered MR base.

## Current UCCL EP transport shape

The current EP transport implementation is verbs-shaped. Key files:

```text
uccl/ep/include/proxy_ctx.hpp
uccl/ep/include/rdma.hpp
uccl/ep/include/rdma_util.hpp
uccl/ep/src/rdma.cpp
uccl/ep/src/proxy.cpp
uccl/ep/src/adaptive_sleeper.cc
uccl/ep/setup.py
uccl/ep/Makefile
```

`ProxyCtx` currently directly stores verbs objects:

```cpp
ibv_context* context;
ibv_pd* pd;
ibv_mr* mr;
ibv_cq* cq;
ibv_cq_ex* cq_ex;
ibv_qp* qp;
ibv_ah* dst_ah;
```

The core proxy flow is:

```text
GPU kernels
  -> D2H/FIFO TransferCmd queues
  -> Proxy::post_gpu_command()
  -> Proxy::post_gpu_commands_mixed()
  -> post_rdma_async_batched()
  -> post_atomic_operations()
  -> CQ polling / completion processing
  -> GPU completion notification
```

The CXI work should introduce a transport abstraction rather than trying to
force libfabric through these verbs types.

## Existing UCCL native atomic path to model

UCCL already has native RDMA atomic logic in the non-EFA verbs path:

```text
uccl/ep/src/rdma.cpp
```

Relevant functions:

```cpp
post_atomic_operations_native_rdma()
post_atomic_operations_fast_mode_native_rdma()
```

These currently use:

```cpp
IBV_WR_ATOMIC_FETCH_AND_ADD
```

against:

```cpp
ctx->remote_atomic_buffer_addr + cmd.req_rptr
ctx->remote_atomic_buffer_rkey
```

The CXI equivalent should be `fi_atomic` / `fi_atomicmsg` with `FI_SUM` on the
known-working 64-bit atomic type.

## CXI backend target mapping

| UCCL concept | Verbs today | CXI/libfabric target |
|---|---|---|
| Provider/device discovery | `ibv_get_device_list` | `fi_getinfo` with `prov_name="cxi"` |
| Endpoint | QP / EFA SRD QP | `FI_EP_RDM` endpoint |
| Completion queue | `ibv_cq` / `ibv_cq_ex` | `fid_cq`, likely `FI_CQ_FORMAT_DATA` |
| Peer address | QPN/LID/GID/PSN | endpoint address from `fi_getname`, inserted with `fi_av_insert` |
| Payload write | `IBV_WR_RDMA_WRITE` | `fi_write` / `fi_writemsg` |
| Counter update | `IBV_WR_ATOMIC_FETCH_AND_ADD` | `fi_atomic` / `fi_atomicmsg`, `FI_SUM` |
| Completion ordering signal | signaled WR / atomic completion | atomic completion; use `FI_FENCE` for write-before-atomic |
| GPU MR registration | `ibv_reg_mr_iova2` / DMA-BUF verbs | `fi_mr_regattr` with `FI_HMEM_CUDA` |
| Remote target address | absolute VA + offset | offset from remote MR base |

## Build-system changes

Add a compile-time option, for example:

```bash
USE_LIBFABRIC_CXI=1
```

`uccl/ep/setup.py` should add:

```python
cxx_flags.append("-DUSE_LIBFABRIC_CXI")
nvcc_flags.append("-DUSE_LIBFABRIC_CXI")
libraries.append("fabric")
```

If `LIBFABRIC_HOME` is set, add:

```text
$LIBFABRIC_HOME/include
$LIBFABRIC_HOME/lib
```

to include/library paths.

Equivalent support should be added to `uccl/ep/Makefile`.

Useful runtime environment for CXI experiments:

```bash
export UCCL_EP_TRANSPORT=cxi
export FI_PROVIDER=cxi
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=16384
export FI_CXI_OFLOW_BUF_SIZE=1073741824
export FI_CXI_RDZV_THRESHOLD=1073741824
export FI_CXI_RX_MATCH_MODE=software
export FI_MR_CACHE_MONITOR=userfaultfd
```

`FI_CXI_ENABLE_WRITEDATA=1` is not required for the native-atomic protocol, but
may be left enabled if `fi_writedata` is used for auxiliary control/debug paths.

## Proposed file layout

Add:

```text
uccl/ep/include/transport.hpp
uccl/ep/include/verbs_transport.hpp
uccl/ep/include/cxi_transport.hpp
uccl/ep/src/verbs_transport.cpp
uccl/ep/src/cxi_transport.cpp
```

A minimal transport interface should cover only what EP needs:

```cpp
enum class TransportKind { Verbs, Cxi };

struct TransportCompletion {
  uint64_t wr_id = 0;
  uint64_t data = 0;
  bool is_write = false;
  bool is_atomic = false;
  bool is_recv_data = false;
  int status = 0;
};

class EpTransport {
 public:
  virtual ~EpTransport() = default;

  virtual void init(...) = 0;
  virtual void register_main_buffer(void* ptr, size_t len, int cuda_device) = 0;
  virtual void register_atomic_buffer(void* ptr, size_t len, int cuda_device) = 0;
  virtual LocalConnInfo local_info() const = 0;
  virtual void connect_peer(int peer, RemoteConnInfo const&) = 0;

  virtual void post_write(int dst_rank, uint64_t wr_id,
                          uint64_t local_offset, uint64_t remote_offset,
                          uint32_t bytes, bool low_latency) = 0;

  virtual void post_atomic_add(int dst_rank, uint64_t wr_id,
                               uint64_t remote_atomic_offset,
                               int64_t value, bool fence) = 0;

  virtual int poll(TransportCompletion* out, int max) = 0;
  virtual void destroy() = 0;
};
```

Do not over-generalize. Keep this interface narrow and driven by current EP use.

## CXI provider initialization

Use PPLX-like hints:

```cpp
hints->caps = FI_RMA | FI_WRITE | FI_ATOMIC | FI_HMEM |
              FI_LOCAL_COMM | FI_REMOTE_COMM;
hints->ep_attr->type = FI_EP_RDM;
hints->fabric_attr->prov_name = strdup("cxi");
hints->domain_attr->mr_mode = FI_MR_ENDPOINT |
                              FI_MR_ALLOCATED |
                              FI_MR_PROV_KEY;
hints->domain_attr->threading = FI_THREAD_DOMAIN;
```

Open:

```text
fi_fabric
fi_domain
fi_cq_open
fi_av_open
fi_endpoint
fi_ep_bind(cq)
fi_ep_bind(av)
fi_enable
fi_getname
```

Use `fi_av_insert` on peer endpoint address blobs exchanged through UCCL's
existing TCP/bootstrap path.

## CXI memory registration

For CUDA/HMEM buffers:

```cpp
int old_device = 0;
cudaGetDevice(&old_device);
cudaSetDevice(cuda_device);

struct iovec iov = {.iov_base = ptr, .iov_len = len};
struct fi_mr_attr attr = {};
attr.iov_count = 1;
attr.access = FI_SEND | FI_RECV | FI_WRITE | FI_READ |
              FI_REMOTE_WRITE | FI_REMOTE_READ;
attr.iface = FI_HMEM_CUDA;
attr.device.cuda = cuda_device;
attr.mr_iov = &iov;

fi_mr_regattr(domain, &attr, 0, &mr);
fi_mr_bind(&mr->fid, &ep->fid, 0);
fi_control(&mr->fid, FI_ENABLE, nullptr);

cudaSetDevice(old_device);
```

For the atomic buffer, use the same MR mode and a compatible access mask. Native
CXI atomics are known to work in the target environment, so design this path
around `fi_atomic`/`fi_atomicmsg` rather than a software fallback.

## CXI write path

Current verbs code computes:

```cpp
remote_addr = ctx->remote_addr + decode_write_offset(cmd.req_rptr, mode);
```

For CXI, use offset-from-MR-base:

```cpp
uint64_t remote_offset = decode_write_offset(cmd.req_rptr, mode);
```

Then:

```cpp
fi_write(ep,
         local_base + local_offset,
         bytes,
         local_desc,
         peer_addr,
         remote_offset,
         remote_key,
         op_context);
```

or use `fi_writemsg` for vector/batched operations.

## CXI atomic path

For EP counters/signals:

```cpp
int64_t delta = value;
```

Use `fi_atomicmsg` when a flag is needed:

```cpp
fi_atomicmsg(ep, &msg, FI_FENCE);
```

Use:

```text
op       = FI_SUM
datatype = known-working 64-bit signed or unsigned type
```

The remote atomic address should also be an offset into the registered atomic
MR:

```cpp
remote_atomic_offset = cmd.req_rptr;
```

For combine, preserve existing semantics:

```cpp
if (get_is_combine(cmd.cmd_type)) {
  value = 1;
}
```

For dispatch, preserve the existing negative-value/counter convention unless a
separate protocol change intentionally removes it.

## Ordering rule

The CXI protocol must ensure:

```text
remote payload write is visible before remote counter update is observed
```

Use `FI_FENCE` on atomics that signal write completion:

```text
fi_write(payload)
fi_atomicmsg(counter += delta, FI_FENCE)
```

Do not rely on implicit provider ordering unless the relevant test already
proves it. The target provider advertises `FI_FENCE`, so use it.

## Completion handling

For CXI, poll with:

```cpp
fi_cq_read(cq, entries, max)
fi_cq_readerr(cq, &err, 0)
```

Map libfabric completions into `TransportCompletion` and reuse UCCL's existing
acked-WR bookkeeping:

```cpp
acked_wrs_.insert(wr_id);
```

For the native atomic path, do not route counter updates through
`remote_process_completions_*()` as `AtomicsImm` software updates. The remote
counter is updated by the NIC/fabric atomic itself.

## Cached notify / combine interaction

Do not rewrite cached notify initially.

Relevant code:

```text
uccl/ep/src/intranode.cu      cached_notify_combine
uccl/ep/src/internode.cu      cached_notify
uccl/ep/src/uccl_ep.cc        intranode_combine / internode_combine calls
```

`cached_notify_combine` and `internode::cached_notify` prepare combine metadata
such as:

```text
send_head
combined_rdma_head
combined_nvl_head
atomic_buffer_ptr head/tail slots
```

The kernels repair negative head entries using reverse scans:

```cpp
if (current_head < 0) {
  head[...] = -last_head - 1;
} else {
  last_head = current_head;
}
```

The CXI backend must keep the atomic buffer layout compatible with these kernels.
`internode::cached_notify` also clears `atomic_buffer_ptr` channel head/tail
slots, so make sure all prior CXI writes/atomics that target those slots have
completed before the next cached notify/clean phase.

## Runtime selection

Add runtime selection, for example:

```bash
export UCCL_EP_TRANSPORT=cxi
```

Suggested behavior:

- default remains existing verbs path;
- `UCCL_EP_TRANSPORT=cxi` requires libfabric/CXI initialization;
- if CXI is requested but cannot be opened, fail loudly;
- do not silently fall back to verbs on CXI-only nodes.

## Validation order

1. Build/import with CXI enabled:

   ```bash
   cd /users/anowak/open_source/uccl/ep
   USE_LIBFABRIC_CXI=1 CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0 \
     python3 setup.py install
   python3 -c "import torch; import uccl.ep"
   ```

2. Standalone CXI microbench:

   ```text
   fi_write payload
   fi_atomic counter += delta with FI_FENCE
   verify counter and payload
   ```

3. Same microbench with CUDA/HMEM memory.

4. Minimal UCCL low-latency internode path.

5. Two-node EP test.

6. Multi-channel / multi-proxy testing.

7. Performance tuning.

## Implementation checklist

- [x] Add build flag `USE_LIBFABRIC_CXI=1`.
- [x] Link `-lfabric` and support `LIBFABRIC_HOME`.
- [x] Add transport abstraction.
- [ ] Move existing verbs implementation behind `VerbsTransport` without behavior change.
- [x] Add `CxiTransport` provider/domain/endpoint/CQ/AV setup.
- [x] Add CXI endpoint address exchange through existing bootstrap.
- [x] Add CUDA HMEM MR registration with endpoint-bound MRs.
- [x] Implement CXI offset-from-MR-base remote addressing.
- [x] Implement `fi_write` payload path.
- [x] Implement `fi_atomicmsg` / `fi_atomic` counter path.
- [x] Use `FI_FENCE` for write-completion-signaling atomics.
- [x] Add CQ polling and error handling.
- [x] Request `FI_HMEM` and `FI_FENCE` in CXI `fi_getinfo` hints.
- [x] Initialize proxy atomic-buffer pointers defensively to `nullptr`.
- [x] Ensure cached notify / atomic buffer cleanup ordering is safe enough for
  current validation: CXI `QUIET` drains outstanding completions before acking,
  write+signaling-atomic commands ack on the fenced atomic completion, and the
  fresh LL/HT reruns below pass after those changes.
- [x] Validate low-latency internode first.
- [x] Debug current 2-node CXI proxy-thread segfault before benchmark work.
- [x] Generalize 2-node high-throughput harness for 4 local ranks.
- [x] Remove the lingering `bench/test_internode.py --num-processes=8`
  default so the benchmark follows `LOCAL_WORLD_SIZE` from `torchrun` unless a
  caller explicitly overrides it.
- [x] Wire minimal CXI normal-mode barrier.
- [x] Fix high-throughput dispatch misaligned-address failure at
  `ep/src/internode.cu:1712`.
- [x] Validate small 2-node CXI high-throughput correctness/tuning with
  `NUM_MAX_NVL_PEERS=4`.
- [x] Validate larger 2-node CXI high-throughput case without
  `CUDA_LAUNCH_BLOCKING=1`: fresh job `2414026` passed on `nid006569` /
  `nid006573` with 2 nodes x 4 ranks, `NUM_MAX_NVL_PEERS=4`,
  `HT_NUM_TOKENS=512`, `HT_HIDDEN=2048`, `HT_NUM_TOPK=4`,
  `HT_NUM_EXPERTS=64`.
- [x] Re-run CXI low-latency regression after high-throughput changes: fresh
  job `2413988` passed on `nid006569` / `nid006573`; both ranks printed
  `All correctness tests passed` and bandwidth output.
- [x] Re-run fresh one-node CXI regression before README-scale benchmarking:
  job `2414067` passed build/import, 4-rank dual-mode, 4-rank small
  intranode high-throughput, and 2-rank local low-latency.
- [x] Re-run fresh one-node full-shape intranode benchmark: job `2414616`
  passed with 4 ranks, 4096 tokens, hidden size 7168, top-k 8, and 288
  experts.
- [x] Re-run fresh two-node low-latency regression: job `2414152` passed on
  `nid006569` / `nid006573`; both ranks printed `All correctness tests passed`.
- [x] Run fresh README-shape CXI/libfabric high-throughput benchmarks with no
  proxy tracing: jobs `2414343`, `2414361`, and `2414395` passed for 2, 4, and
  8 nodes respectively.
- [x] Clean up debug prints / decide final trace knobs before upstreaming:
  decision is to leave the existing library/proxy printfs in place for now.
  They do not affect the measured GPU/network speed path and are useful while
  this backend remains under active CXI bring-up.

Current implementation status:

- `CxiTransport` exists and compiles with `USE_LIBFABRIC_CXI=1`, including
  `fi_getinfo`, `fi_fabric`, `fi_domain`, CQ, AV, endpoint setup, `fi_getname`,
  CUDA/HMEM MR registration, `fi_write`, `fi_atomicmsg(FI_SUM/FI_UINT64)` with
  registered host-side atomic operands, `FI_FENCE` on signaling atomics, and CQ
  polling.
- `RDMAConnectionInfo` carries CXI endpoint-name and 64-bit MR keys under
  `USE_LIBFABRIC_CXI`.
- `UCCL_EP_TRANSPORT=cxi` now routes `Proxy::init_common()` through
  `CxiTransport`: it skips verbs initialization, registers the main and atomic
  buffers with CXI/libfabric, exchanges CXI endpoint metadata through the
  existing TCP bootstrap helpers, inserts peers into the libfabric AV, and
  stores remote CXI keys in per-peer `ProxyCtx` records.
- `Proxy::post_gpu_commands_mixed()` now routes `WRITE`, `ATOMIC`, `QUIET`, and
  normal-mode `BARRIER` commands through the CXI path when
  `UCCL_EP_TRANSPORT=cxi`. The CXI barrier uses native atomics plus the existing
  local shared-memory barrier.
- `Proxy::run_dual()` polls CXI completions and converts them into the existing
  `acked_wrs_` completion path. Sender-only and remote-only proxy modes still
  fail loudly for CXI; CXI currently expects dual mode.
- `Proxy::post_gpu_command()` and `Proxy::post_gpu_commands_mixed()` have
  lightweight runtime tracing gated by `UCCL_PROXY_TRACE=1`. This prints the
  first command batches and RDMA/atomic/quiet/barrier split without affecting
  normal test output.
- Low-latency buffers now force a device allocation through `ep.get_rdma_buffer`
  so same-node low-latency paths can export CUDA IPC handles even on CXI-only
  systems where the old verbs registration probe fails.
- Installed `nanobind` into the active Apertus `/opt/venv`, then verified
  object-level compilation, full CXI-enabled extension install, and import:

  ```bash
  cd /users/anowak/open_source/uccl/ep
  CUDA_PATH=/usr/local/cuda USE_LIBFABRIC_CXI=1 \
    make -B src/cxi_transport.o src/proxy.o
  CUDA_PATH=/usr/local/cuda make -B src/cxi_transport.o src/proxy.o
  USE_LIBFABRIC_CXI=1 CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0 \
    python3 setup.py install
  python3 -c "import torch; import uccl.ep; print('import ok')"
  ```
- The install completed and placed `ep.abi3.so` in
  `/opt/venv/lib/python3.12/site-packages/uccl`.

Resolved `FI_FENCE` issue:

- The earlier `fi_atomicmsg(..., FI_FENCE)` `-FI_EINVAL` failure was due to not
  requesting `FI_FENCE` in `hints->caps`. `FI_FENCE` must be requested during
  `fi_getinfo` capability selection, not only passed as an operation flag.
- `CxiTransport::init()` now includes `FI_FENCE` in `hints->caps`, and
  `CxiTransport::post_atomic_add()` passes `FI_FENCE` when the caller requests a
  fenced signaling atomic.
- The minimal standalone test now also requests `FI_FENCE` in `hints->caps` and
  passes:

  ```bash
  cd /users/anowak/open_source/uccl/ep
  gcc -O2 -g tests/cxi_fence_atomic_min.c -lfabric -o /tmp/cxi_fence_atomic_min
  FI_PROVIDER=cxi /tmp/cxi_fence_atomic_min
  ```

  Result:

  ```text
  dst=0x1234 ctr=10
  ```

Low-latency local hang root cause and fix:

- The one-node 2-rank low-latency hang was not a CXI transport failure. For
  same-node ranks, `internode_ll.cu` expects a CUDA IPC/NVLink path and waits
  on the local `rdma_recv_count` buffer.
- On the CXI-only node, `ep.get_rdma_buffer()` used the old verbs registration
  probe and chose host-pinned fallback memory. Host-pinned memory cannot export
  CUDA IPC handles, so peer `ipc_rdma_base_ptrs` were null.
- Temporary kernel debug prints showed every cross-GPU count send had
  `dst_p2p=0x0`; the kernel fell back to proxy/IBGDA commands, but Python had
  not started proxies because topology was intranode. The receive phase then
  spun forever waiting for peer count signals.
- The fix is to force raw `cudaMalloc` allocation for low-latency RDMA scratch
  buffers:
  - `ep.get_rdma_buffer(num_rdma_bytes, device_index, force_device_alloc=False)`
    accepts a new `force_device_alloc` flag.
  - `bench/buffer.py` passes `low_latency_mode` as `force_device_alloc`.

Existing-test results in current `srun` allocation (`nid007053`):

- The current rebuild after the allocator fix passed install/import:

  ```bash
  cd /users/anowak/open_source/uccl/ep
  USE_LIBFABRIC_CXI=1 CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0 \
    python3 setup.py install
  python3 -c "import torch; import uccl.ep; print('import ok')"
  ```

- Existing construction smoke passed without CXI selection:

  ```bash
  cd /users/anowak/open_source/uccl/ep
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29661 timeout 180s torchrun \
    --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 \
    --master_port=29661 --nproc_per_node=2 \
    bench/test_dual_mode.py --num-tokens 8 --hidden 1024 --num-experts 16
  ```

  Result: `[dual-mode] PASS`.

- Existing high-throughput intranode test passed:

  ```bash
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29662 timeout 240s torchrun \
    --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 \
    --master_port=29662 --nproc_per_node=2 \
    bench/test_intranode.py --num-tokens 64 --hidden 1024 \
    --num-topk 4 --num-experts 16
  ```

  Result: dispatch/combine cases passed, including zero-recv-rank.

- Existing construction smoke also passed with `UCCL_EP_TRANSPORT=cxi`:

  ```bash
  UCCL_EP_TRANSPORT=cxi FI_PROVIDER=cxi \
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29663 timeout 180s torchrun \
    --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 \
    --master_port=29663 --nproc_per_node=2 \
    bench/test_dual_mode.py --num-tokens 8 --hidden 1024 --num-experts 16
  ```

  Result: `[dual-mode] PASS`. Because both ranks were on one node with the
  same OOB IP, this is not proof of CXI internode data movement.

- Existing low-latency test now passes in the one-node 2-rank configuration:

  ```bash
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29660 timeout 300s torchrun \
    --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 \
    --master_port=29660 --nproc_per_node=2 \
    bench/test_low_latency.py --num-tokens 16 --hidden 7168 \
    --num-topk 4 --num-experts 16 --no-dispatch-use-fp8
  ```

  Result: both ranks printed `All correctness tests passed`.

- A smaller low-latency debug harness now also passes in the one-node 2-rank
  configuration:

  ```bash
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29664 timeout 120s \
    torchrun --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 \
    --master_port=29664 --nproc_per_node=2 bench/debug_low_latency_once.py
  ```

  Result: both ranks returned from `event.current_stream_wait()` and printed
  `done`.

- The single-rank low-latency test also passes:

  ```bash
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29654 timeout 120s torchrun \
    --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 \
    --master_port=29654 --nproc_per_node=1 \
    bench/test_low_latency.py --num-tokens 16 --hidden 7168 \
    --num-topk 4 --num-experts 8 --no-dispatch-use-fp8
  ```

  Result: `All correctness tests passed`.

- The standalone CXI host write+atomic smoke still passes:

  ```bash
  gcc -O2 -g tests/cxi_write_atomic_smoke.cpp -lstdc++ -lfabric \
    -o /tmp/cxi_write_atomic_smoke
  FI_PROVIDER=cxi /tmp/cxi_write_atomic_smoke
  ```

  Result: `atomicvalid count=1` and `cxi host write+atomic ok`.

2-node CXI internode status:

- The first 2-node `srun` reached container startup but failed before UCCL
  because `nanobind` was missing. `ep/run_cxi_internode_low_latency_srun.sh`
  now installs `nanobind` under the build lock before `setup.py install`.
- The next 2-node run reached CXI initialization but failed registering the
  atomic buffer MR because the Grace Hopper path used `cudaMallocManaged`.
  `USE_LIBFABRIC_CXI` now allocates the atomic buffer with `cudaMalloc`.
- After that fix, the 2-node run got past main and atomic HMEM MR registration
  on both nodes and initialized CXI endpoints, then segfaulted while Python was
  still sleeping in `initialize_uccl()` before benchmark dispatch/combine work
  began. That points to background proxy-thread startup/idle code rather than a
  posted GPU command.
- `CxiTransport::init()` now requests both `FI_HMEM` and `FI_FENCE` in
  `hints->caps`:

  ```cpp
  FI_RMA | FI_WRITE | FI_ATOMIC | FI_FENCE | FI_REMOTE_WRITE |
      FI_REMOTE_READ | FI_HMEM | FI_LOCAL_COMM | FI_REMOTE_COMM
  ```

- Defensive fix applied after the segfault: `Proxy::atomic_buffer_ptr_` and
  `UcclProxy::atomic_buffer_ptr_` now initialize to `nullptr`. This removes one
  undefined-behavior hazard if any nonzero proxy thread starts before Python has
  called `set_atomic_buffer_ptr()`.
- Native gdb backtrace found the real startup SIGSEGV in
  `Proxy::post_gpu_command()`: the CXI branch in `Proxy::init_common()` returned
  before initializing the `USE_MSCCLPP_FIFO_BACKEND` vectors. The fix initializes
  `fifo_seq_` and `fifo_pending_` before returning from the CXI branch.
- After the FIFO init fix, the full 2-node test no longer crashed but stalled in
  a later reused-buffer correctness sequence. The exact reproducer completed
  BF16 and FP8 `round_scale=False`, then hung at FP8 `round_scale=True`.
- Root cause for the sequence stall: FIFO work request ids are generated as
  `(rb_idx << 32) | fifo_seq_[rb_idx]`, so the first command on ring 0 has a
  valid `wr_id == 0`. `CxiTransport` stores and returns that id through its CQ
  operation context, but `Proxy::poll_cxi_completions()` previously ignored
  completions with id 0. That left the first ring-0 FIFO command permanently in
  `fifo_pending_`, which made one-shot tests pass but eventually starved the
  reused full-test sequence. CXI completion polling now acks every returned
  completion, including `wr_id == 0`.
- Current CXI completion handling also tracks `cxi_outstanding_ops_`, skips CQ
  polling while no CXI work is outstanding, decrements the counter as
  completions arrive, and makes CXI `QUIET` drain all outstanding completions
  before acking the quiet command.
- Validation command:

  ```bash
  srun -A infra01 --time=0:45:00 --nodes=2 --ntasks-per-node=1 \
    --reservation=SD-69241-apertus-1-5-0 \
    --environment=apertus2-alps4-temp \
    --container-mounts=/iopsstor/scratch/cscs/anowak:/iopsstor/scratch/cscs/anowak,/users/anowak:/users/anowak,/capstor:/capstor,/iopsstor:/iopsstor \
    --network=disable_rdzv_get --mpi=pmix \
    bash -lc 'export UCCL_PROXY_TRACE=0; bash ep/run_cxi_internode_low_latency_srun.sh'
  ```

- Result on Slurm job `2411700`, nodes `nid007590` and `nid007592`: both ranks
  completed every correctness case, printed `All correctness tests passed`, and
  completed the bandwidth/profiling section.
- Revalidation after reshaping the `wr_id == 0` ack into a
  `USE_LIBFABRIC_CXI`-guarded branch:
  - CXI rebuild/import passed on Slurm job `2411783`.
  - Two-node sequence reproducer passed on job `2411796`, nodes `nid007005` and
    `nid007006`:

    ```bash
    LL_DEBUG_SEQUENCE=1 UCCL_PROXY_TRACE=0 \
      bash ep/run_cxi_debug_once_srun.sh
    ```

    It completed BF16, FP8, FP8 `round_scale=True`, UE8M0, hook, and zero-copy
    cases.
  - Full two-node low-latency correctness passed again on job `2411803`, nodes
    `nid007484` and `nid007490`; both ranks printed
    `All correctness tests passed` and completed profiling output.
  - One-node high-throughput/intranode regression passed on job `2411833` with
    two local ranks:

    ```bash
    torchrun --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 \
      --master_port=29681 --nproc_per_node=2 \
      bench/test_intranode.py --num-tokens 64 --hidden 1024 \
      --num-topk 4 --num-experts 16
    ```

    It passed BF16/FP8 dispatch/combine cases and zero-recv-rank.
  - Non-CXI compile-only check passed on job `2411851` / rerun after printf
    cleanup: `python3 setup.py install` completed with `USE_LIBFABRIC_CXI`
    unset. Runtime validation on Apertus should still force
    `UCCL_EP_TRANSPORT=cxi` / `FI_PROVIDER=cxi`.
  - CXI build/import was restored after the compile-only check and passed on job
    `2411858`.
- Full internode high-throughput (`bench/test_internode.py`) was not run
  unchanged on the current Apertus allocation because the harness asserts
  `num_local_ranks == 8`, `num_ranks > 8`, and `--num-processes=8`, while the
  verified Apertus nodes expose four GPUs. Treat this as a harness/coverage gap,
  not a CXI pass.

## Current high-throughput CXI status

- Companion visual explainer: `uccl/UCCL_EP_CXI_HT_EXPLAINER.html`. Use it as
  the first human/agent overview for HT layout, FIFO/proxy methodology, CXI
  completion semantics, barrier flow, and the 8-local-rank portability lesson.
- The internode high-throughput harness has now been generalized for the
  two-node/four-GPU Apertus shape:
  - `bench/test_internode.py` no longer requires exactly eight local ranks.
  - `NUM_MAX_NVL_PEERS` can be set at build time; use
    `NUM_MAX_NVL_PEERS=4` for current Apertus 4-GPU nodes.
  - Proxy normal-mode peer filtering now uses `num_ranks / num_nodes` instead
    of hard-coded `MAX_NUM_GPUS`, so ranks connect by same local GPU index on
    4-GPU nodes (`0<->4`, `1<->5`, ...).
- The high-throughput kernel/harness had several implicit dependencies on
  eight local ranks per node. We do not have evidence that eight local ranks are
  required for speed or correctness in the algorithm itself. The likely
  historical reason is that the original DeepSeek/DeepEP deployment shape used
  eight GPUs per node, so some constants and packed loads were hard-coded around
  that server topology. The CXI/Apertus work treats this as a portability bug:
  local NVL peer count should be a build/runtime shape parameter, not an
  algorithmic requirement.
- When debugging future high-throughput failures, always consider hidden
  8-local-rank assumptions before attributing the symptom to CXI/libfabric.
  Examples include raw `uint64_t` loads from per-local-rank bool arrays, rank
  grouping with `MAX_NUM_GPUS`, buffers sized as if there are eight NVL peers,
  and dispatch/combine metadata indexed by a hard-coded local GPU count. These
  bugs can surface as CUDA misaligned-address errors, hangs in proxy/FIFO
  progress, bad peer selection, or incorrect routing metadata even when the CXI
  transport itself is working.
- The previous high-throughput hang at:

  ```text
  [wait_until_cmd_consumed ... cmd:3 ...] waiting slot=0
  ```

  was caused by the CXI path bypassing verbs `quiet()` / `send_barrier()`.
  FIFO `QUIET`/`BARRIER` commands set `*_inflight` but did not assign
  `ctx_.quiet_wr` / `ctx_.barrier_wr`, so `notify_gpu_completion()` could not
  pop the FIFO slot. The fix assigns those WR ids only for
  `use_cxi_transport()`, preserving verbs behavior.
- The next blocker was the explicit CXI BARRIER stub. A minimal CXI normal-mode
  barrier is now wired with a small host MR inside `CxiTransport`:
  non-root node leaders signal root with a fenced CXI atomic add, root releases
  remote node leaders with another atomic add, and local ranks continue to use
  the existing shared-memory `LocalBarrier`.
- A follow-up crash showed CXI `init_common()` returned before mapping
  `ctx_.lb`; CXI init now performs the same local barrier setup as verbs normal
  mode, and CXI destroy unmaps it.
- Latest validation:
  - CXI rebuild/import with `USE_LIBFABRIC_CXI=1 NUM_MAX_NVL_PEERS=4` passed.
  - Two-node high-throughput job `2412166` reached the first BF16
    high-throughput dispatch/hash, then all ranks failed with:

    ```text
    torch.AcceleratorError: CUDA error: misaligned address
    ...
    bench/utils.py:710 in hash_tensor(recv_x)
    ```

  - This is progress beyond the proxy `QUIET` wait and BARRIER crash. The
    current target is CXI normal-mode data/write addressing or alignment. Next
    useful run should add targeted CXI write offset tracing and/or
    `CUDA_LAUNCH_BLOCKING=1`.
  - After that, CXI WRITE completion handling was tightened for normal-mode
    write+counter commands: when a signaling atomic follows a write, the write
    uses an internal WR id and only the fenced atomic uses the FIFO command WR
    id. This avoids popping the GPU FIFO command on write completion before the
    counter atomic completes.
  - Rebuild/import after the write/atomic ordering change passed. Two-node
    high-throughput job `2412203` with `CUDA_LAUNCH_BLOCKING=1` still failed,
    now localized to:

    ```text
    RuntimeError: Failed: CUDA error ... ep/src/internode.cu:1712 'misaligned address'
    ```

    This was caused by remaining raw `uint64_t` loads from the bool
    `is_token_in_rank` routing matrix in the normal-mode dispatch sender. With
    `NUM_MAX_NVL_PEERS=4`, those loads read eight bool entries from a four-peer
    group. The kernel now packs exactly `NUM_MAX_NVL_PEERS` bool values into the
    existing byte layout before broadcasting to `SourceMeta`.
  - Rebuild/import after the routing-pack fix passed. Two-node high-throughput
    job `2412222` with `CUDA_LAUNCH_BLOCKING=1`, `USE_LIBFABRIC_CXI=1`,
    `UCCL_EP_TRANSPORT=cxi`, `FI_PROVIDER=cxi`, and `NUM_MAX_NVL_PEERS=4`
    exited successfully. It completed BF16/FP8 dispatch correctness, async and
    previous-buffer variants, combine testing, and tuning output.
  - A larger follow-up HT validation without `CUDA_LAUNCH_BLOCKING=1` was
    queued as job `2412259` using `--num-tokens 512 --hidden 2048 --num-topk 4
    --num-experts 64`, but no allocation was expected soon. The job was
    cancelled before it started. Continue from this validation step when an
    allocation is available.

## Latest fresh validation, 2026-05-28

Active scratchpad for this debugging session:
`/users/anowak/agents-scratchpad/debugging-development/uccl-cxi-native-atomics-2026-05-27/notes.md`.

All runs below used fresh Slurm jobs; no previous allocation or job was reused.

- One-node CXI regression job `2414067` passed on `nid005425`:
  - CXI build/import with `USE_LIBFABRIC_CXI=1 NUM_MAX_NVL_PEERS=4`.
  - 4-rank `bench/test_dual_mode.py`: `[dual-mode] PASS`.
  - 4-rank small `bench/test_intranode.py`: correctness and tuning passed with
    128 tokens, hidden size 1024, top-k 4, and 16 experts. Best intranode
    tuning results were FP8 dispatch 8.40 GB/s (NVL), 46.39 us; BF16 dispatch
    18.30 GB/s (NVL), 41.29 us; and combine 18.62 GB/s (NVL), 40.58 us.
  - 2-rank local `bench/test_low_latency.py --num-tokens 16 --hidden 7168
    --num-topk 4 --num-experts 16 --no-dispatch-use-fp8`: all correctness
    tests passed.
- Two-node low-latency job `2414152` passed on `nid006569` / `nid006573` with
  `USE_LIBFABRIC_CXI=1`, `UCCL_EP_TRANSPORT=cxi`, and `FI_PROVIDER=cxi`; both
  ranks printed `All correctness tests passed`.
- One-node full-shape intranode benchmark job `2414616` passed on `nid006457`
  with `USE_LIBFABRIC_CXI=1`, `FI_PROVIDER=cxi`, `NUM_MAX_NVL_PEERS=4`, 4
  ranks, 4096 tokens, hidden size 7168, top-k 8, and 288 experts. Correctness,
  async/previous-buffer variants, zero-recv-rank, and tuning completed. Best
  tuning results were FP8 dispatch 346.08 GB/s (NVL), 317.16 us; BF16 dispatch
  392.64 GB/s (NVL), 542.17 us; and combine 364.32 GB/s (NVL), 584.31 us.
- The first README-shape high-throughput attempt, job `2414238`, was
  intentionally cancelled after entering steady proxy traffic because
  `UCCL_PROXY_TRACE=1` produced massive per-command logs. It is not used as
  benchmark data.
- README-shape high-throughput benchmark reruns used
  `HT_NUM_TOKENS=4096`, `HT_HIDDEN=7168`, `HT_NUM_TOPK=8`,
  `HT_NUM_EXPERTS=288`, `NPROC_PER_NODE=4`, `USE_LIBFABRIC_CXI=1`,
  `UCCL_EP_TRANSPORT=cxi`, `FI_PROVIDER=cxi`, `NUM_MAX_NVL_PEERS=4`, and
  `UCCL_PROXY_TRACE=0`. This means 4096 tokens per batch, hidden size 7168,
  top-8 experts per token, 288 total experts, 4 local ranks/GPUs per node, FP8
  dispatch and BF16 dispatch benchmarked separately, and BF16 combine. Each
  GPU/rank has one Slingshot/CXI NIC, and each NIC provides 25 GB/s
  (200 Gb/s) of bandwidth. Results below are the bottleneck RDMA bandwidth and
  transmit latency across the node-local benchmark summaries:

  | Nodes | EP | Job | FP8 dispatch | BF16 dispatch | Combine | Notes |
  |:-----:|:--:|:---:|:-------------:|:--------------:|:-------:|:------|
  | 2 | 8 | `2414343` | 10.42 GB/s, 5794 us | 10.27 GB/s, 11392 us | 10.97 GB/s, 10665 us | `num_topk_groups=2` because only 2 nodes participated |
  | 4 | 16 | `2414361` | 5.93 GB/s, 18314 us | 5.92 GB/s, 35601 us | 6.19 GB/s, 33996 us | `num_topk_groups=4` |
  | 8 | 32 | `2414395` | 4.79 GB/s, 25140 us | 4.85 GB/s, 48132 us | 5.07 GB/s, 46135 us | `num_topk_groups=4` |

- These results have also been added to `ep/README.md` as preliminary
  Apertus/GH200/Slingshot CXI validation data. Keep the scope clear: these
  measurements validate the new libfabric/CXI path over the current 4-GPU-node
  Apertus shape and are not a replacement for the published EFA/InfiniBand
  result tables.
- FIFO CXI microbenchmark validation: commit `0f5299e4` refactors
  `ep/bench/fifo` behind a shared backend API (`FifoProxy` + `FifoBackend`),
  with `PopOnlyBackend` for the original pop-only benchmark and
  `CxiFifoBackend` for CXI post/completion benchmarking. Fresh job `2416038`
  built with `USE_LIBFABRIC_CXI=1` and ran:

  ```bash
  FI_PROVIDER=cxi ./benchmark -x --cxi-bytes 8 --cxi-max-outstanding 256
  ```

  The refactored CXI mode passed on GH200. It posts one 8-byte CXI write per
  GPU FIFO command and pops the FIFO only after CQ completion. Results:

  | FIFO size | 1 thread | 32 threads | 64 threads | 128 threads | 256 threads | 512 threads | 1024 threads |
  |:---------:|:--------:|:----------:|:----------:|:-----------:|:-----------:|:-----------:|:------------:|
  | 2048 | 0.11 Mops/s | 2.20 Mops/s | 2.59 Mops/s | 2.66 Mops/s | 2.74 Mops/s | 2.79 Mops/s | 2.79 Mops/s |
  | 4096 | 0.12 Mops/s | 2.19 Mops/s | 2.58 Mops/s | 2.54 Mops/s | 2.78 Mops/s | 2.80 Mops/s | 2.81 Mops/s |

  The CXI FIFO benchmark therefore plateaus around 2.8 Mops/s completed
  8-byte CXI writes with 4 proxy threads and 256 outstanding writes per proxy.
- All-to-all CXI microbenchmark validation: `ep/bench/alltoall` now has a
  libfabric/CXI backend selected with `--transport cxi` in `launch_bench.py`
  and built with:

  ```bash
  make USE_LIBFABRIC_CXI=1 CUDA_HOME=/usr/local/cuda benchmark
  ```

  The CXI path uses the existing TCP control plane for endpoint-name/key
  exchange, CUDA HMEM MR registration, `FI_EP_RDM`, `fi_write` for each remote
  all-to-all message, and CQ polling for completions. The benchmark now accepts
  runtime message size and traffic pattern arguments through
  `launch_bench.py --msg-size-kb <KB> --pattern <cross|rail>`, so it can match
  the EFA all-to-all sweep shape. The verbs and CXI implementations are now
  behind a small benchmark-local backend API:
  `AllToAllBackend::{setup, begin_round, post_write, poll_send_completions,
  cleanup}`. The shared benchmark runner owns dispatch-table traversal,
  timing, inflight-window accounting, barriers, and throughput reporting, while
  each backend owns endpoint setup, transport-specific post/poll details, and
  cleanup. Fresh compile-only job `2416171` passed.
  Fresh 2-node job `2416196` passed with 4 ranks per node:

  ```bash
  torchrun --nnodes=2 --nproc_per_node=4 ... launch_bench.py --transport cxi
  ```

  Shape: 128 messages, top-k 8, 7168-byte messages, 2 nodes x 4 ranks. Rank 0
  average over the last 30 rounds was 744.88 us and the benchmark reported
  9.85 GB/s. Other rank averages: rank 1 746.49 us, rank 2 733.67 us, rank 3
  745.66 us, rank 4 722.31 us, rank 5 697.17 us, rank 6 716.96 us, rank 7
  710.41 us.

  The EFA-style all-to-all sweep was then run on two Apertus GH200/CXI nodes
  with four ranks per node, one CXI NIC per GPU, and nominal per-NIC bandwidth
  of 25 GB/s (200 Gb/s). `NUM_MSGS=128`, `TOPK=8`, and `WINDOW_SIZE=2048`
  stayed fixed; only payload size and traffic pattern changed. Fresh job
  `2416234` compiled the benchmark, fresh job `2416238` ran the first main
  sweep, and fresh job `2416369` reran the missing 64 KB and 128 KB
  rail-aligned points. Fresh job `2416385` passed post-edit compile
  validation. After the NUMA affinity correction below, fresh job `2416521`
  ran the full cross-rail and rail-aligned sweep again. The current table of
  record is in `ep/bench/alltoall/efa_alltoall_results.md`:

  | Message size | Cross-rail | Rail-aligned |
  | ------------ | ---------- | ------------ |
  | 8 KB | 5.64 GB/s | 5.45 GB/s |
  | 16 KB | 5.48 GB/s | 4.83 GB/s |
  | 32 KB | 5.53 GB/s | 4.93 GB/s |
  | 64 KB | 5.53 GB/s | 4.32 GB/s |
  | 128 KB | 5.57 GB/s | 4.69 GB/s |
  | 256 KB | 5.54 GB/s | 4.77 GB/s |
  | 512 KB | 5.59 GB/s | 5.52 GB/s |
  | 1024 KB | 5.60 GB/s | 5.73 GB/s |
  | 2048 KB | 5.58 GB/s | 5.54 GB/s |
  | 4096 KB | 5.61 GB/s | 5.64 GB/s |

  Backend-API refactor validation: fresh job `2416464` forced a clean rebuild
  with `make clean && make USE_LIBFABRIC_CXI=1 CUDA_HOME=/usr/local/cuda
  benchmark`, then ran a 2-node, 4-rank-per-node CXI smoke benchmark with
  `launch_bench.py --transport cxi --pattern cross --msg-size-kb 8`. Rank 0
  reported 1626.09 us average over the last 30 rounds and 5.16 GB/s.

  NUMA affinity correction: the original all-to-all benchmark inherited the
  p5 heuristic `numa_node = local_rank / 4`, which pinned all Apertus 4-rank
  runs to NUMA node 0. A first attempt to scale by
  `numa_num_configured_nodes()` was also wrong on this container because
  configured NUMA node IDs can be sparse/not allowed, producing invalid nodes
  such as 9, 18, and 27. PPLX Garden's topology code assigns CPU lists to
  GPUs from the detected NUMA topology; for this benchmark's Apertus shape we
  now use the equivalent direct CPU-block mapping over the 288 visible CPUs:
  local rank 0 -> CPUs 0-71, rank 1 -> 72-143, rank 2 -> 144-215, rank 3 ->
  216-287. Fresh job `2416510` forced a clean rebuild and ran the 2-node CXI
  8 KB cross smoke; the printed affinities confirmed those four CPU blocks.
  Rank 0 reported 1516.90 us and 5.53 GB/s.
  Fresh job `2416521` then reran the complete 2-node all-to-all sweep and
  printed the same corrected CPU affinities for every tested payload size.

- Previous testing blocker, now intermittent/resolved for the latest attempts:
  some container `srun` attempts were rejected by Slurm before the script
  started with:

  ```text
  srun: error: cli_filter plugin terminated with error
  ```

  Reusing job `2408897` with `srun --jobid=2408897 --overlap` reaches
  `nid007053`, but those overlap steps do not enter the Apertus container and
  lack the expected Python/CUDA environment.

## Key pitfalls

1. **Do not use verbs addressing semantics in CXI mode.** CXI should use offsets
   from MR base for RMA/atomic targets.
2. **Do not use PPLX's write-data software-atomic path as the primary UCCL
   counter model.** Native atomics work and should be used.
3. **Do not skip CUDA device context setup in proxy threads.** Set the correct
   CUDA device before HMEM registration.
4. **Do not break existing verbs/EFA code.** CXI must be a separate backend.
5. **Do not ignore write-before-atomic ordering.** Use `FI_FENCE` for signaling
   atomics.
6. **Do not allocate low-latency RDMA scratch as host-pinned memory when local
   CUDA IPC is expected.** On CXI-only systems, verbs registration probing is
   not a valid signal for low-latency IPC viability.
