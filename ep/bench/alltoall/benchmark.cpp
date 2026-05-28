// All-to-All RDMA Benchmark with cross-node dispatch.
// Each GPU sends NUM_MSGS * TOPK messages. Message size and traffic pattern
// are runtime arguments so the same binary can sweep cross-rail and
// rail-aligned cases.

#include <arpa/inet.h>
#ifndef USE_LIBFABRIC_CXI
#include <infiniband/efadv.h>
#include <infiniband/verbs.h>
#else
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_rma.h>
#endif
#include <netinet/in.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include <cuda_runtime.h>
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <sys/socket.h>
#include <unistd.h>

#ifndef NUM_GPUS_PER_NODE
constexpr int NUM_GPUS_PER_NODE = 8;
#endif
constexpr int NUM_NICS_PER_GPU = 1;  // 1 for p6, 2 for p5en
constexpr size_t DEFAULT_MSG_SIZE = 7168;  // 7KB
constexpr int NUM_MSGS = 128;
constexpr int TOPK = 8;
constexpr int WINDOW_SIZE = 2048;
constexpr uint32_t QKEY = 0x11111111u;
constexpr int TCP_PORT_BASE = 18515;

std::mutex barrier_mutex;
std::condition_variable barrier_cv;
std::atomic<int> barrier_count{0};
int barrier_target = 0;
size_t g_msg_size = DEFAULT_MSG_SIZE;
bool g_rail_aligned = false;

// Global TCP socket state for control plane (exchange + barriers)
int control_listenfd = -1;
std::vector<int> control_socks;

#define CUDA_CHECK(cmd)                                             \
  do {                                                              \
    cudaError_t e = (cmd);                                          \
    if (e != cudaSuccess) {                                         \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(e));                               \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

struct RDMAConnectionInfo {
  uint32_t qp_num;
  uint8_t gid[16];
  uint32_t rkey;
  uint64_t addr;
};

#ifdef USE_LIBFABRIC_CXI
struct CxiConnectionInfo {
  uint32_t ep_name_len = 0;
  uint8_t ep_name[256] = {};
  uint64_t mr_key = 0;
  uint64_t addr = 0;
};

void check_fi(int rc, char const* what) {
  if (rc < 0) {
    fprintf(stderr, "%s failed: %s\n", what, fi_strerror(-rc));
    exit(1);
  }
}

struct CxiContext {
  fid_fabric* fabric = nullptr;
  fid_domain* domain = nullptr;
  fid_ep* ep = nullptr;
  fid_cq* cq = nullptr;
  fid_av* av = nullptr;
  fid_mr* mr = nullptr;
  void* gpu_buf = nullptr;
  size_t buf_size = 0;
  CxiConnectionInfo local_info = {};
  std::vector<fi_addr_t> peer_addrs;
  std::vector<CxiConnectionInfo> remote_infos;
};

fid_mr* register_cuda_mr(fid_domain* domain, fid_ep* ep, void* ptr, size_t len,
                         int cuda_device) {
  int old_device = 0;
  CUDA_CHECK(cudaGetDevice(&old_device));
  CUDA_CHECK(cudaSetDevice(cuda_device));

  iovec iov = {};
  iov.iov_base = ptr;
  iov.iov_len = len;
  fi_mr_attr attr = {};
  attr.mr_iov = &iov;
  attr.iov_count = 1;
  attr.access = FI_SEND | FI_RECV | FI_WRITE | FI_READ | FI_REMOTE_WRITE |
                FI_REMOTE_READ;
  attr.iface = FI_HMEM_CUDA;
  attr.device.cuda = static_cast<uint64_t>(cuda_device);

  fid_mr* mr = nullptr;
  int rc = fi_mr_regattr(domain, &attr, 0, &mr);
  CUDA_CHECK(cudaSetDevice(old_device));
  check_fi(rc, "fi_mr_regattr(cuda)");
  check_fi(fi_mr_bind(mr, &ep->fid, 0), "fi_mr_bind");
  check_fi(fi_control(&mr->fid, FI_ENABLE, nullptr), "fi_control(FI_ENABLE)");
  return mr;
}

void cxi_init(CxiContext& cxi, int local_rank) {
  fi_info* hints = fi_allocinfo();
  if (!hints) {
    fprintf(stderr, "fi_allocinfo failed\n");
    exit(1);
  }
  hints->caps = FI_RMA | FI_WRITE | FI_REMOTE_WRITE | FI_REMOTE_READ |
                FI_HMEM | FI_LOCAL_COMM | FI_REMOTE_COMM;
  hints->mode = FI_CONTEXT;
  hints->ep_attr->type = FI_EP_RDM;
  hints->fabric_attr->prov_name = strdup("cxi");
  hints->domain_attr->mr_mode =
      FI_MR_ENDPOINT | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
  hints->domain_attr->threading = FI_THREAD_DOMAIN;

  fi_info* info = nullptr;
  int rc = fi_getinfo(FI_VERSION(1, 15), nullptr, nullptr, 0, hints, &info);
  fi_freeinfo(hints);
  check_fi(rc, "fi_getinfo(cxi)");

  check_fi(fi_fabric(info->fabric_attr, &cxi.fabric, nullptr), "fi_fabric");
  check_fi(fi_domain(cxi.fabric, info, &cxi.domain, nullptr), "fi_domain");
  fi_cq_attr cq_attr = {};
  cq_attr.format = FI_CQ_FORMAT_CONTEXT;
  cq_attr.size = 131072;
  check_fi(fi_cq_open(cxi.domain, &cq_attr, &cxi.cq, nullptr), "fi_cq_open");
  fi_av_attr av_attr = {};
  av_attr.type = FI_AV_MAP;
  check_fi(fi_av_open(cxi.domain, &av_attr, &cxi.av, nullptr), "fi_av_open");
  check_fi(fi_endpoint(cxi.domain, info, &cxi.ep, nullptr), "fi_endpoint");
  check_fi(fi_ep_bind(cxi.ep, &cxi.cq->fid, FI_TRANSMIT | FI_RECV),
           "fi_ep_bind(cq)");
  check_fi(fi_ep_bind(cxi.ep, &cxi.av->fid, 0), "fi_ep_bind(av)");
  check_fi(fi_enable(cxi.ep), "fi_enable");
  fi_freeinfo(info);

  cxi.buf_size = NUM_MSGS * g_msg_size;
  CUDA_CHECK(cudaMalloc(&cxi.gpu_buf, cxi.buf_size));
  CUDA_CHECK(cudaMemset(cxi.gpu_buf, local_rank, cxi.buf_size));
  cxi.mr = register_cuda_mr(cxi.domain, cxi.ep, cxi.gpu_buf, cxi.buf_size,
                            local_rank);
  cxi.local_info.addr = reinterpret_cast<uint64_t>(cxi.gpu_buf);
  cxi.local_info.mr_key = fi_mr_key(cxi.mr);

  size_t ep_name_len = sizeof(cxi.local_info.ep_name);
  check_fi(fi_getname(&cxi.ep->fid, cxi.local_info.ep_name, &ep_name_len),
           "fi_getname");
  cxi.local_info.ep_name_len = static_cast<uint32_t>(ep_name_len);
}

void cxi_destroy(CxiContext& cxi) {
  if (cxi.mr) fi_close(&cxi.mr->fid);
  if (cxi.ep) fi_close(&cxi.ep->fid);
  if (cxi.av) fi_close(&cxi.av->fid);
  if (cxi.cq) fi_close(&cxi.cq->fid);
  if (cxi.domain) fi_close(&cxi.domain->fid);
  if (cxi.fabric) fi_close(&cxi.fabric->fid);
  if (cxi.gpu_buf) cudaFree(cxi.gpu_buf);
}
#endif

#ifndef USE_LIBFABRIC_CXI
struct NicContext {
  ibv_context* ctx = nullptr;
  ibv_pd* pd = nullptr;
  ibv_cq* cq = nullptr;
  ibv_mr* mr = nullptr;
  void* gpu_buf = nullptr;
  size_t buf_size = 0;
};
#endif

#ifndef USE_LIBFABRIC_CXI
struct PeerEndpoint {
  ibv_qp* qp = nullptr;
  ibv_ah* ah = nullptr;
  uint32_t remote_qpn = 0;
  uint32_t remote_rkey = 0;
  uint64_t remote_addr = 0;
};
#endif

// Prepopulated random dispatch table
std::vector<int> generate_dispatch_table(int rank, int world_size,
                                         int local_world_size,
                                         bool rail_aligned) {
  std::vector<int> table(NUM_MSGS * TOPK);
  unsigned int seed = rank * 12345;
  int const node_rank = rank / local_world_size;
  int const num_nodes = std::max(1, world_size / local_world_size);
  int const local_rank = rank % local_world_size;

  for (int i = 0; i < NUM_MSGS * TOPK; i++) {
    if (rail_aligned) {
      int remote_node = (node_rank + 1 + (i % std::max(1, num_nodes - 1))) %
                        num_nodes;
      table[i] = remote_node * local_world_size + local_rank;
    } else {
      int dst = rand_r(&seed) % world_size;
      while (dst / local_world_size == node_rank && num_nodes > 1) {
        dst = rand_r(&seed) % world_size;
      }
      table[i] = dst;
    }
  }
  return table;
}

class AllToAllBackend {
 public:
  virtual ~AllToAllBackend() = default;
  virtual void setup() = 0;
  virtual void begin_round() = 0;
  virtual void post_write(int op_idx, int dst_rank, int msg_id) = 0;
  virtual int poll_send_completions(int max_poll) = 0;
  virtual void cleanup() = 0;
};

#ifndef USE_LIBFABRIC_CXI
void get_gid(ibv_context* ctx, int port, int index, uint8_t* gid) {
  ibv_gid g;
  if (ibv_query_gid(ctx, port, index, &g)) {
    fprintf(stderr, "Failed to query GID\n");
    exit(1);
  }
  memcpy(gid, g.raw, 16);
}

ibv_qp* create_srd_qp(NicContext* nic) {
  ibv_qp_init_attr_ex qp_attr = {};
  efadv_qp_init_attr efa_attr = {};

  qp_attr.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
  qp_attr.send_ops_flags =
      IBV_QP_EX_WITH_RDMA_WRITE | IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM;
  qp_attr.cap.max_send_wr = 2048;
  qp_attr.cap.max_recv_wr = 2048;
  qp_attr.cap.max_send_sge = 1;
  qp_attr.cap.max_recv_sge = 1;
  qp_attr.cap.max_inline_data = 0;
  qp_attr.pd = nic->pd;
  qp_attr.qp_context = nic->ctx;
  qp_attr.sq_sig_all = 0;
  qp_attr.send_cq = nic->cq;
  qp_attr.recv_cq = nic->cq;
  qp_attr.qp_type = IBV_QPT_DRIVER;

  efa_attr.driver_qp_type = EFADV_QP_DRIVER_TYPE_SRD;
  efa_attr.sl = 8;  // Low latency service level

  ibv_qp* qp =
      efadv_create_qp_ex(nic->ctx, &qp_attr, &efa_attr, sizeof(efa_attr));
  if (!qp) {
    perror("Failed to create QP");
    exit(1);
  }

  ibv_qp_attr attr = {};
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = 1;
  attr.qkey = QKEY;
  if (ibv_modify_qp(
          qp, &attr,
          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY)) {
    perror("Failed to modify QP to INIT");
    exit(1);
  }

  attr = {};
  attr.qp_state = IBV_QPS_RTR;
  if (ibv_modify_qp(qp, &attr, IBV_QP_STATE)) {
    perror("Failed to modify QP to RTR");
    exit(1);
  }

  attr = {};
  attr.qp_state = IBV_QPS_RTS;
  attr.rnr_retry = 3;
  if (ibv_modify_qp(qp, &attr,
                    IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_RNR_RETRY)) {
    perror("Failed to modify QP to RTS");
    exit(1);
  }

  return qp;
}

ibv_ah* create_ah(ibv_pd* pd, uint8_t* remote_gid) {
  ibv_ah_attr ah_attr = {};
  ah_attr.port_num = 1;
  ah_attr.is_global = 1;
  memcpy(ah_attr.grh.dgid.raw, remote_gid, 16);

  ibv_ah* ah = ibv_create_ah(pd, &ah_attr);
  if (!ah) {
    perror("Failed to create AH");
    exit(1);
  }
  return ah;
}
#endif

void pin_to_numa(int local_rank) {
  int local_world_size = NUM_GPUS_PER_NODE;
  if (char const* env = std::getenv("LOCAL_WORLD_SIZE")) {
    local_world_size = std::max(1, std::atoi(env));
  }
  unsigned int max_cpus = std::max(1, numa_num_configured_cpus());
  unsigned int cpus_per_rank =
      std::max(1u, max_cpus / static_cast<unsigned int>(local_world_size));
  if (char const* env = std::getenv("UCCL_ALLTOALL_CPUS_PER_NUMA")) {
    cpus_per_rank = std::max(1, std::atoi(env));
  }
  unsigned int cpu_start = static_cast<unsigned int>(local_rank) * cpus_per_rank;
  unsigned int cpu_end = std::min(max_cpus, cpu_start + cpus_per_rank);

  cpu_set_t cpu_mask;
  CPU_ZERO(&cpu_mask);
  for (unsigned int i = cpu_start; i < cpu_end && i < CPU_SETSIZE; i++) {
    CPU_SET(i, &cpu_mask);
  }

  if (sched_setaffinity(0, sizeof(cpu_mask), &cpu_mask) < 0) {
    fprintf(stderr,
            "Warning: Failed to set CPU affinity for local_rank %d CPUs "
            "%u-%u\n",
            local_rank, cpu_start, cpu_end == 0 ? 0 : cpu_end - 1);
  }

  cpu_set_t actual_cpu_mask;
  CPU_ZERO(&actual_cpu_mask);
  if (sched_getaffinity(0, sizeof(actual_cpu_mask), &actual_cpu_mask) == 0) {
    printf("Rank local_rank %d logical NUMA %d requested CPUs %u-%u actual "
           "CPU affinity:",
           local_rank, local_rank, cpu_start, cpu_end == 0 ? 0 : cpu_end - 1);
    for (unsigned int i = 0; i < max_cpus; i++) {
      if (CPU_ISSET(i, &actual_cpu_mask)) {
        printf(" %u", i);
      }
    }
    printf("\n");
  } else {
    fprintf(stderr, "Warning: Failed to read CPU affinity for local_rank %d\n",
            local_rank);
  }

  printf("Rank pinned to logical NUMA %d (local_rank %d)\n", local_rank,
         local_rank);
}

#ifndef USE_LIBFABRIC_CXI
void tcp_control_init(
    int rank, int world_size, char const* master_ip,
    std::vector<RDMAConnectionInfo>& local_info,
    std::vector<std::vector<RDMAConnectionInfo>>& remote_info) {
  remote_info.resize(world_size);
  remote_info[rank] = local_info;

  if (rank == 0) {
    control_listenfd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(control_listenfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(TCP_PORT_BASE);
    addr.sin_addr.s_addr = INADDR_ANY;

    bind(control_listenfd, (sockaddr*)&addr, sizeof(addr));
    listen(control_listenfd, world_size);

    control_socks.resize(world_size, -1);
    for (int i = 1; i < world_size; i++) {
      int sock = accept(control_listenfd, nullptr, nullptr);
      int peer_rank;
      recv(sock, &peer_rank, sizeof(peer_rank), 0);
      remote_info[peer_rank].resize(NUM_NICS_PER_GPU);
      recv(sock, remote_info[peer_rank].data(),
           sizeof(RDMAConnectionInfo) * NUM_NICS_PER_GPU, 0);
      control_socks[peer_rank] = sock;
    }

    // Broadcast all connection info to all ranks
    for (int i = 1; i < world_size; i++) {
      for (int j = 0; j < world_size; j++) {
        if (j == i) continue;
        send(control_socks[i], remote_info[j].data(),
             sizeof(RDMAConnectionInfo) * NUM_NICS_PER_GPU, 0);
      }
    }
  } else {
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(TCP_PORT_BASE);
    inet_pton(AF_INET, master_ip, &addr.sin_addr);

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    int retry_count = 0;
    while (connect(sock, (sockaddr*)&addr, sizeof(addr)) < 0) {
      if (++retry_count > 100) {
        fprintf(stderr,
                "Rank %d: Failed to connect to master after 100 retries\n",
                rank);
        exit(1);
      }
      close(sock);
      sock = socket(AF_INET, SOCK_STREAM, 0);
      usleep(100000);
    }

    send(sock, &rank, sizeof(rank), 0);
    send(sock, local_info.data(), sizeof(RDMAConnectionInfo) * NUM_NICS_PER_GPU,
         0);

    for (int i = 0; i < world_size; i++) {
      if (i == rank) continue;
      remote_info[i].resize(NUM_NICS_PER_GPU);
      recv(sock, remote_info[i].data(),
           sizeof(RDMAConnectionInfo) * NUM_NICS_PER_GPU, 0);
    }

    control_socks.push_back(sock);
  }
}
#endif

void tcp_barrier(int rank, int world_size) {
  if (rank == 0) {
    for (int i = 1; i < world_size; i++) {
      char sync;
      recv(control_socks[i], &sync, 1, 0);
    }
    for (int i = 1; i < world_size; i++) {
      char ack = 1;
      send(control_socks[i], &ack, 1, 0);
    }
  } else {
    char sync = 1;
    send(control_socks[0], &sync, 1, 0);
    char ack;
    recv(control_socks[0], &ack, 1, 0);
  }
}

void tcp_control_cleanup(int rank, int world_size) {
  if (rank == 0) {
    for (int i = 1; i < world_size; i++) {
      if (control_socks[i] >= 0) close(control_socks[i]);
    }
    if (control_listenfd >= 0) close(control_listenfd);
  } else {
    if (!control_socks.empty() && control_socks[0] >= 0) {
      close(control_socks[0]);
    }
  }
}

#ifndef USE_LIBFABRIC_CXI
struct PollResult {
  int send_completions = 0;
  int recv_completions = 0;
};

PollResult poll_cq(ibv_cq* cq, int max_poll) {
  ibv_wc wc[32];
  int poll_budget = std::min(32, max_poll);
  int n = ibv_poll_cq(cq, poll_budget, wc);
  if (n < 0) {
    fprintf(stderr, "Poll CQ error\n");
    exit(1);
  }

  PollResult result;
  for (int i = 0; i < n; i++) {
    if (wc[i].status != IBV_WC_SUCCESS) {
      fprintf(stderr, "WC error: %s\n", ibv_wc_status_str(wc[i].status));
      exit(1);
    }
    if (wc[i].opcode & IBV_WC_RECV) {
      result.recv_completions++;
    } else {
      result.send_completions++;
    }
  }
  return result;
}
#endif

#ifdef USE_LIBFABRIC_CXI
void cxi_tcp_control_init(
    int rank, int world_size, char const* master_ip,
    std::vector<CxiConnectionInfo>& remote_info) {
  remote_info.resize(world_size);

  if (rank == 0) {
    control_listenfd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(control_listenfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(TCP_PORT_BASE);
    addr.sin_addr.s_addr = INADDR_ANY;
    bind(control_listenfd, (sockaddr*)&addr, sizeof(addr));
    listen(control_listenfd, world_size);

    control_socks.resize(world_size, -1);
    for (int i = 1; i < world_size; i++) {
      int sock = accept(control_listenfd, nullptr, nullptr);
      int peer_rank = -1;
      recv(sock, &peer_rank, sizeof(peer_rank), MSG_WAITALL);
      recv(sock, &remote_info[peer_rank], sizeof(CxiConnectionInfo),
           MSG_WAITALL);
      control_socks[peer_rank] = sock;
    }

    for (int i = 1; i < world_size; i++) {
      send(control_socks[i], remote_info.data(),
           sizeof(CxiConnectionInfo) * world_size, 0);
    }
  } else {
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(TCP_PORT_BASE);
    inet_pton(AF_INET, master_ip, &addr.sin_addr);

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    int retry_count = 0;
    while (connect(sock, (sockaddr*)&addr, sizeof(addr)) < 0) {
      if (++retry_count > 100) {
        fprintf(stderr,
                "Rank %d: Failed to connect to master after 100 retries\n",
                rank);
        exit(1);
      }
      close(sock);
      sock = socket(AF_INET, SOCK_STREAM, 0);
      usleep(100000);
    }

    send(sock, &rank, sizeof(rank), 0);
    send(sock, &remote_info[rank], sizeof(CxiConnectionInfo), 0);
    recv(sock, remote_info.data(), sizeof(CxiConnectionInfo) * world_size,
         MSG_WAITALL);
    control_socks.push_back(sock);
  }
}

class CxiAllToAllBackend : public AllToAllBackend {
 public:
  struct WriteCtx {
    fi_context context = {};
  };

  CxiAllToAllBackend(int rank, int local_rank, int world_size,
                     char const* master_ip)
      : rank_(rank),
        local_rank_(local_rank),
        world_size_(world_size),
        master_ip_(master_ip),
        contexts_(WINDOW_SIZE) {}

  void setup() override {
    cxi_init(cxi_, local_rank_);
    remote_info_.resize(world_size_);
    remote_info_[rank_] = cxi_.local_info;
    cxi_tcp_control_init(rank_, world_size_, master_ip_, remote_info_);

    cxi_.peer_addrs.resize(world_size_, FI_ADDR_UNSPEC);
    cxi_.remote_infos = remote_info_;
    for (int r = 0; r < world_size_; ++r) {
      if (r == rank_) continue;
      fi_addr_t addr = FI_ADDR_UNSPEC;
      int rc =
          fi_av_insert(cxi_.av, remote_info_[r].ep_name, 1, &addr, 0, nullptr);
      if (rc != 1) {
        if (rc < 0) check_fi(rc, "fi_av_insert");
        fprintf(stderr, "fi_av_insert inserted no address\n");
        exit(1);
      }
      cxi_.peer_addrs[r] = addr;
    }
  }

  void begin_round() override {}

  void post_write(int, int dst_rank, int msg_id) override {
    auto& ctx = contexts_[next_ctx_++ % contexts_.size()];
    char* local = static_cast<char*>(cxi_.gpu_buf) + msg_id * g_msg_size;
    ssize_t rc =
        fi_write(cxi_.ep, local, g_msg_size, fi_mr_desc(cxi_.mr),
                 cxi_.peer_addrs[dst_rank], msg_id * g_msg_size,
                 remote_info_[dst_rank].mr_key, &ctx.context);
    check_fi(static_cast<int>(rc), "fi_write(cxi alltoall)");
  }

  int poll_send_completions(int max_poll) override {
    fi_cq_entry entries[64] = {};
    int budget = std::min(64, max_poll);
    ssize_t rc = fi_cq_read(cxi_.cq, entries, budget);
    if (rc == -FI_EAGAIN) return 0;
    if (rc < 0) {
      fi_cq_err_entry err = {};
      ssize_t erc = fi_cq_readerr(cxi_.cq, &err, 0);
      if (erc >= 0) {
        fprintf(stderr, "CXI CQ error: %s\n",
                fi_cq_strerror(cxi_.cq, err.prov_errno, err.err_data, nullptr,
                               0));
        exit(1);
      }
      check_fi(static_cast<int>(rc), "fi_cq_read");
    }
    return static_cast<int>(rc);
  }

  void cleanup() override {
    tcp_control_cleanup(rank_, world_size_);
    cxi_destroy(cxi_);
  }

 private:
  int rank_;
  int local_rank_;
  int world_size_;
  char const* master_ip_;
  CxiContext cxi_;
  std::vector<CxiConnectionInfo> remote_info_;
  std::vector<WriteCtx> contexts_;
  size_t next_ctx_ = 0;
};
#endif

#ifndef USE_LIBFABRIC_CXI
class VerbsAllToAllBackend : public AllToAllBackend {
 public:
  VerbsAllToAllBackend(int rank, int local_rank, int world_size,
                       char const* master_ip)
      : rank_(rank),
        local_rank_(local_rank),
        world_size_(world_size),
        master_ip_(master_ip),
        nics_(NUM_NICS_PER_GPU),
        peers_(NUM_NICS_PER_GPU) {
    for (int i = 0; i < NUM_NICS_PER_GPU; i++) {
      peers_[i].resize(world_size_);
    }
  }

  void setup() override {
    int num_devices = 0;
    ibv_device** dev_list = ibv_get_device_list(&num_devices);

    std::vector<ibv_device*> rdmap_devices;
    for (int i = 0; i < num_devices; i++) {
      char const* dev_name = ibv_get_device_name(dev_list[i]);
      if (strncmp(dev_name, "rdmap", 5) == 0) {
        rdmap_devices.push_back(dev_list[i]);
      }
    }

    int nic_base = local_rank_ * NUM_NICS_PER_GPU;

    if (nic_base + NUM_NICS_PER_GPU > static_cast<int>(rdmap_devices.size())) {
      fprintf(stderr, "Not enough rdmap devices: need %d, found %zu\n",
              nic_base + NUM_NICS_PER_GPU, rdmap_devices.size());
      exit(1);
    }

    for (int i = 0; i < NUM_NICS_PER_GPU; i++) {
      auto& nic = nics_[i];
      nic.ctx = ibv_open_device(rdmap_devices[nic_base + i]);
      if (!nic.ctx) {
        fprintf(stderr, "Failed to open device %s (index %d)\n",
                ibv_get_device_name(rdmap_devices[nic_base + i]),
                nic_base + i);
        exit(1);
      }

      nic.pd = ibv_alloc_pd(nic.ctx);
      nic.cq = ibv_create_cq(nic.ctx, 4096, nullptr, nullptr, 0);

      nic.buf_size = NUM_MSGS * g_msg_size;
      CUDA_CHECK(cudaMalloc(&nic.gpu_buf, nic.buf_size));
      CUDA_CHECK(cudaMemset(nic.gpu_buf, rank_, nic.buf_size));

      nic.mr = ibv_reg_mr(nic.pd, nic.gpu_buf, nic.buf_size,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                              IBV_ACCESS_RELAXED_ORDERING);
      if (!nic.mr) {
        perror("Failed to register MR");
        exit(1);
      }
    }
    ibv_free_device_list(dev_list);

    std::vector<RDMAConnectionInfo> local_info(NUM_NICS_PER_GPU);

    for (int i = 0; i < NUM_NICS_PER_GPU; i++) {
      auto& nic = nics_[i];
      ibv_qp* recv_qp = create_srd_qp(&nic);

      auto& info = local_info[i];
      info.qp_num = recv_qp->qp_num;
      get_gid(nic.ctx, 1, 0, info.gid);
      info.rkey = nic.mr->rkey;
      info.addr = (uint64_t)nic.gpu_buf;

      peers_[i][rank_].qp = recv_qp;
    }

    tcp_control_init(rank_, world_size_, master_ip_, local_info, remote_info_);

    for (int i = 0; i < NUM_NICS_PER_GPU; i++) {
      auto& nic = nics_[i];

      for (int r = 0; r < world_size_; r++) {
        if (r == rank_) continue;

        auto& peer = peers_[i][r];
        peer.qp = create_srd_qp(&nic);
        peer.ah = create_ah(nic.pd, remote_info_[r][i].gid);
        peer.remote_qpn = remote_info_[r][i].qp_num;
        peer.remote_rkey = remote_info_[r][i].rkey;
        peer.remote_addr = remote_info_[r][i].addr;
      }

      peers_[i][rank_].ah = create_ah(nic.pd, local_info[i].gid);
      post_initial_recvs(i);
    }
  }

  void begin_round() override {
    recv_completed_.assign(NUM_NICS_PER_GPU, 0);
    recv_slot_.assign(NUM_NICS_PER_GPU, 2048);
  }

  void post_write(int op_idx, int dst_rank, int msg_id) override {
    int nic_idx = op_idx % NUM_NICS_PER_GPU;
    auto& nic = nics_[nic_idx];
    auto& peer = peers_[nic_idx][dst_rank];

    ibv_qp_ex* qpx = ibv_qp_to_qp_ex(peer.qp);
    ibv_wr_start(qpx);

    qpx->wr_id = op_idx;
    qpx->wr_flags = IBV_SEND_SIGNALED;

    ibv_wr_rdma_write_imm(qpx, peer.remote_rkey,
                          peer.remote_addr + msg_id * g_msg_size, op_idx);

    ibv_sge sge = {(uint64_t)nic.gpu_buf + msg_id * g_msg_size, g_msg_size,
                   nic.mr->lkey};
    ibv_wr_set_sge_list(qpx, 1, &sge);
    ibv_wr_set_ud_addr(qpx, peer.ah, peer.remote_qpn, QKEY);

    if (ibv_wr_complete(qpx)) {
      fprintf(stderr, "Failed to post RDMA write\n");
      exit(1);
    }
  }

  int poll_send_completions(int max_poll) override {
    int send_completions = 0;
    for (int i = 0; i < NUM_NICS_PER_GPU; i++) {
      auto result = poll_cq(nics_[i].cq, max_poll);
      send_completions += result.send_completions;
      recv_completed_[i] += result.recv_completions;
      refill_recvs(i);
    }
    return send_completions;
  }

  void cleanup() override {
    tcp_control_cleanup(rank_, world_size_);

    for (int i = 0; i < NUM_NICS_PER_GPU; i++) {
      auto& nic = nics_[i];
      for (int r = 0; r < world_size_; r++) {
        if (peers_[i][r].qp) ibv_destroy_qp(peers_[i][r].qp);
        if (peers_[i][r].ah) ibv_destroy_ah(peers_[i][r].ah);
      }
      if (nic.mr) ibv_dereg_mr(nic.mr);
      if (nic.cq) ibv_destroy_cq(nic.cq);
      if (nic.pd) ibv_dealloc_pd(nic.pd);
      if (nic.ctx) ibv_close_device(nic.ctx);
      if (nic.gpu_buf) cudaFree(nic.gpu_buf);
    }
  }

 private:
  void post_initial_recvs(int nic_idx) {
    auto& recv_qp = peers_[nic_idx][rank_].qp;
    for (int slot = 0; slot < 2048; slot++) {
      post_recv(nic_idx, recv_qp, slot);
    }
  }

  void refill_recvs(int nic_idx) {
    auto& recv_qp = peers_[nic_idx][rank_].qp;
    while (recv_completed_[nic_idx] > 0) {
      post_recv(nic_idx, recv_qp, recv_slot_[nic_idx]);
      recv_slot_[nic_idx]++;
      recv_completed_[nic_idx]--;
    }
  }

  void post_recv(int nic_idx, ibv_qp* recv_qp, int slot) {
    auto& nic = nics_[nic_idx];
    int msg_id = slot % NUM_MSGS;
    ibv_sge sge = {(uint64_t)nic.gpu_buf + msg_id * g_msg_size, g_msg_size,
                   nic.mr->lkey};
    ibv_recv_wr wr = {}, *bad_wr;
    wr.wr_id = slot;
    wr.num_sge = 1;
    wr.sg_list = &sge;

    if (ibv_post_recv(recv_qp, &wr, &bad_wr)) {
      perror("Failed to post/refill recv");
      exit(1);
    }
  }

  int rank_;
  int local_rank_;
  int world_size_;
  char const* master_ip_;
  std::vector<NicContext> nics_;
  std::vector<std::vector<RDMAConnectionInfo>> remote_info_;
  std::vector<std::vector<PeerEndpoint>> peers_;
  std::vector<int> recv_completed_;
  std::vector<int> recv_slot_;
};
#endif

void run_benchmark(int rank, int local_rank, int world_size,
                   AllToAllBackend& backend) {
  pin_to_numa(local_rank);
  CUDA_CHECK(cudaSetDevice(local_rank));

  int local_world_size = NUM_GPUS_PER_NODE;
  if (char const* env = std::getenv("LOCAL_WORLD_SIZE")) {
    local_world_size = std::max(1, std::atoi(env));
  }

  backend.setup();

  auto dispatch_table =
      generate_dispatch_table(rank, world_size, local_world_size,
                              g_rail_aligned);
  int node_rank = rank / local_world_size;
  CUDA_CHECK(cudaDeviceSynchronize());

  // Barrier to ensure all ranks have exchanged endpoints and posted receives.
  tcp_barrier(rank, world_size);

  constexpr int NUM_ROUNDS = 50;
  constexpr int WARMUP_ROUNDS = 20;
  std::vector<double> round_times;
  int total_ops = NUM_MSGS * TOPK;

  for (int round = 0; round < NUM_ROUNDS; round++) {
    tcp_barrier(rank, world_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    backend.begin_round();

    auto start = std::chrono::high_resolution_clock::now();
    int send_completed = 0;
    int send_inflight = 0;
    int network_ops = 0;

    for (int op_idx = 0; op_idx < total_ops; op_idx++) {
      int dst_rank = dispatch_table[op_idx];
      int dst_node = dst_rank / local_world_size;
      if (dst_node == node_rank) continue;
      network_ops++;

      int msg_id = op_idx / TOPK;
      backend.post_write(op_idx, dst_rank, msg_id);
      send_inflight++;

      while (send_inflight >= WINDOW_SIZE) {
        int ne = backend.poll_send_completions(send_inflight);
        send_completed += ne;
        send_inflight -= ne;
        if (ne == 0) sched_yield();
      }
    }

    while (send_completed < network_ops) {
      int ne = backend.poll_send_completions(network_ops - send_completed);
      send_completed += ne;
      if (ne == 0) sched_yield();
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_us =
        std::chrono::duration<double, std::micro>(end - start).count();

    round_times.push_back(elapsed_us);
    if (rank == 0) {
      printf("Round %d: %.2f us\n", round, elapsed_us);
    }

    tcp_barrier(rank, world_size);
  }

  double sum = 0.0;
  for (int i = WARMUP_ROUNDS; i < NUM_ROUNDS; i++) {
    sum += round_times[i];
  }
  double avg_us = sum / (NUM_ROUNDS - WARMUP_ROUNDS);

  printf("Rank %d: average of last %d rounds: %.2f us\n", rank,
         NUM_ROUNDS - WARMUP_ROUNDS, avg_us);

  if (rank == 0) {
    double total_data_gb = (NUM_MSGS * TOPK * g_msg_size) / 1e9;
    double elapsed_s = avg_us / 1e6;
    printf("Average time: %.2f us\n", avg_us);
    printf("Throughput: %.2f GB/s\n", total_data_gb / elapsed_s);
  }

  tcp_barrier(rank, world_size);
  backend.cleanup();
}

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr,
            "Usage: %s <rank> <world_size> <master_ip> [local_rank] "
            "[verbs|cxi] [msg_size_bytes] [cross|rail]\n",
            argv[0]);
    return 1;
  }

  int rank = atoi(argv[1]);
  int world_size = atoi(argv[2]);
  char const* master_ip = argv[3];

  int local_rank = (argc >= 5) ? atoi(argv[4]) : rank % NUM_GPUS_PER_NODE;
  char const* transport = (argc >= 6) ? argv[5] : "verbs";
  if (argc >= 7) {
    g_msg_size = std::max<size_t>(1, static_cast<size_t>(atoll(argv[6])));
  }
  char const* pattern = (argc >= 8) ? argv[7] : "cross";
  g_rail_aligned = strcmp(pattern, "rail") == 0;
  if (strcmp(pattern, "cross") != 0 && strcmp(pattern, "rail") != 0) {
    fprintf(stderr, "Unknown traffic pattern '%s'; expected cross or rail\n",
            pattern);
    return 1;
  }

#ifdef USE_LIBFABRIC_CXI
  if (strcmp(transport, "cxi") != 0) {
    fprintf(stderr, "This binary was built with USE_LIBFABRIC_CXI=1; pass cxi\n");
    return 1;
  }
  CxiAllToAllBackend backend(rank, local_rank, world_size, master_ip);
  run_benchmark(rank, local_rank, world_size, backend);
#else
  if (strcmp(transport, "verbs") != 0) {
    fprintf(stderr, "This binary was built for verbs/EFA; pass verbs\n");
    return 1;
  }
  VerbsAllToAllBackend backend(rank, local_rank, world_size, master_ip);
  run_benchmark(rank, local_rank, world_size, backend);
#endif

  return 0;
}
