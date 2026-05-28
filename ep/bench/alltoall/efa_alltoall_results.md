# RDMA All-to-All Throughput Results

Both tests keep the same number of total messages sent by each GPU, and the
same max inflight messages sent by each GPU. The difference is only how the
destination rank table is generated.

In this benchmark, a rail is the local GPU/NIC index inside a node. With two
nodes and four ranks per node, ranks `0..3` are node 0 rails `0..3`, and ranks
`4..7` are node 1 rails `0..3`.

`--pattern cross` generates random cross-node destinations. A rank may send to
any rank on another node, and same-node destinations are rejected. For example,
rank `0` can send to ranks `4`, `5`, `6`, or `7`; rank `1` can also send to
ranks `4`, `5`, `6`, or `7`. This exercises traffic that may cross from one
local GPU/NIC rail to a different remote GPU/NIC rail.

`--pattern rail` generates rail-aligned destinations. A rank sends to the same
local rail on a remote node. With two nodes and four ranks per node, rank `0`
sends to rank `4`, rank `1` sends to rank `5`, rank `2` sends to rank `6`, and
rank `3` sends to rank `7`. On more than two nodes, the benchmark rotates
through the remote nodes but preserves the same local rail index.

Message size of 8KB would roughly represent DeepEP/UCCL-EP low-latency performance. 

### On two p5

Based on https://github.com/uccl-project/uccl/commit/0cdfcf5286c0438ecb0df1820a3028558659ff89 (incast scenario).

* Cross-rail all-to-all results (Unit: GB/s): 
    | Message Size (KB) | 2 Nodes (p5) |
    |-------------------|--------------|
    | 8                 | 6.1          |
    | 16                | 8.6          |
    | 32                | 14.1         |
    | 64                | 15.6         |
    | 128               | 15.7         |
    | 256               | 15.3         |
    | 512               | 17.9         |
    | 1024              | 19.0         |
    | 2048              | 19.9         |
    | 4096              | 20.1         |

* Rail-aligned all-to-all results (Unit: GB/s): 
    | Message Size (KB) | 2 Nodes (p5) |
    | ----------------- | ------------ |
    | 8                 | 13.8         |
    | 16                | 21.1         |
    | 32                | 35.6         |
    | 64                | 36.9         |
    | 128               | 41.6         |
    | 256               | 41.9         |
    | 512               | 43.2         |
    | 1024              | 43.5         |
    | 2048              | 43.4         |
    | 4096              | 43.7         |


### On two p5en

Based on https://github.com/uccl-project/uccl/commit/0cdfcf5286c0438ecb0df1820a3028558659ff89 (incast scenario).

We find 6.168 GB/s vs. 14.438 GB/s for cross-rail vs. rail-aligned.

Based on the lastest code (random scenario). 

We find 9.930 GB/s vs. 15.673 GB/s for cross-rail vs. rail-aligned. 

### On two CSCS Apertus GH200 nodes with Slingshot/CXI

Based on the `nk/cxi` libfabric/CXI path. These runs used two nodes, four
ranks per node, one CXI NIC per GPU, and nominal per-NIC bandwidth of
25 GB/s (200 Gb/s). The benchmark keeps `NUM_MSGS=128`, `TOPK=8`, and
`WINDOW_SIZE=2048`, and sweeps only the per-message payload size.

The CXI backend was built with:

```bash
make USE_LIBFABRIC_CXI=1 CUDA_HOME=/usr/local/cuda benchmark
```

The sweep command used:

```bash
torchrun --nnodes=2 --nproc_per_node=4 ... \
  launch_bench.py --transport cxi --pattern <cross|rail> --msg-size-kb <KB>
```

Fresh Slurm jobs: `2416234` for initial compile validation, `2416238` for the
first two-node sweep, `2416369` for a fresh rerun of the missing 64 KB and
128 KB rail-aligned points, `2416385` for post-edit compile validation, and
`2416521` for the full sweep after fixing Apertus CPU affinity. Job `2416988`
is the current full sweep after fixing CXI domain selection so local rank 0
uses `cxi0`, rank 1 uses `cxi1`, rank 2 uses `cxi2`, and rank 3 uses `cxi3`.
The run also pinned local ranks to the four visible 72-core CPU blocks:
local rank 0 -> CPUs 0-71, local rank 1 -> CPUs 72-143, local rank 2 ->
CPUs 144-215, and local rank 3 -> CPUs 216-287.

* Cross-rail all-to-all results (Unit: GB/s):
    | Message Size (KB) | 2 Nodes (GH200/CXI) |
    |-------------------|---------------------|
    | 8                 | 19.59               |
    | 16                | 19.58               |
    | 32                | 20.70               |
    | 64                | 21.44               |
    | 128               | 21.61               |
    | 256               | 22.16               |
    | 512               | 22.08               |
    | 1024              | 22.05               |
    | 2048              | 21.99               |
    | 4096              | 21.96               |

* Rail-aligned all-to-all results (Unit: GB/s):
    | Message Size (KB) | 2 Nodes (GH200/CXI) |
    | ----------------- | ------------------- |
    | 8                 | 16.05               |
    | 16                | 23.00               |
    | 32                | 22.91               |
    | 64                | 23.26               |
    | 128               | 23.08               |
    | 256               | 23.10               |
    | 512               | 23.30               |
    | 1024              | 23.33               |
    | 2048              | 23.33               |
    | 4096              | 23.33               |

### To reproduce

Run [test_alltoall_rail.cpp](./test_alltoall_rail.cpp) and [test_alltoall.cpp](test_alltoall.cpp).
