/*
 * FIFO Performance Benchmark
 * Measures FIFO dispatch throughput and GPU-side latency
 * To run the benchmark:
 *    make -j
 *    torchrun --nproc_per_node=8 --standalone benchmark.py [-l] [-b] [-r]
 *    -l: Latency mode (RTT measurement)
 *    -b: Burst mode (no polling)
 *    -r: Random mode (each thread randomly selects a FIFO)
 *    -c: Control Mops mode, fifo evaluation for mops vs latency
 *    -x: CXI mode, proxy posts one small CXI write per FIFO command
 */

#include "../../include/fifo.hpp"
#include "../../include/fifo_util.hpp"
#include "launch_kernel_shim.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

#ifdef USE_LIBFABRIC_CXI
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_rma.h>
#endif

using namespace mscclpp;

// Configuration
struct BenchmarkConfig {
  uint32_t num_threads;        // Number of GPU threads pushing to FIFO
  uint32_t fifo_size;          // FIFO size
  uint32_t test_duration_ms;   // Test duration in milliseconds
  uint32_t warmup_iterations;  // Number of warmup iterations
  uint32_t batch_size;
  float gpu_clock_ghz;
  bool measure_latency;
  int mode;
  uint64_t sleep_cycles;
  float target_mops;
  uint32_t cxi_bytes;
  uint32_t cxi_max_outstanding;
};

// Metrics collected from GPU
struct ThreadMetrics {
  uint64_t push_count;          // Number of successful pushes
  uint64_t total_cycles;        // Total cycles spent (for latency calculation)
  uint64_t max_latency_cycles;  // Maximum latency observed
  uint64_t min_latency_cycles;  // Minimum latency observed
};

// Maximum number of latency samples per thread for percentile calculation
constexpr int MAX_LATENCY_SAMPLES = 10000;

class FifoBackend {
 public:
  virtual ~FifoBackend() = default;
  virtual uint64_t progress() = 0;
  virtual bool hasPending() const = 0;
  virtual bool canSubmit(size_t fifo_idx) const = 0;
  virtual uint64_t submit(size_t fifo_idx, Fifo& fifo,
                          ProxyTrigger trigger) = 0;
};

class PopOnlyBackend final : public FifoBackend {
 public:
  uint64_t progress() override { return 0; }
  bool hasPending() const override { return false; }
  bool canSubmit(size_t) const override { return true; }
  uint64_t submit(size_t, Fifo& fifo, ProxyTrigger trigger) override {
    // Flip back the MSB that was set by the device. The plain FIFO benchmark
    // does not consume the payload, but keep the same trigger handling shape
    // as the production proxy.
    trigger.snd ^= (uint64_t{1} << uint64_t{63});
    fifo.pop();
    return 1;
  }
};

// Host-side proxy that polls FIFOs and delegates work to a backend.
class FifoProxy {
 public:
  FifoProxy(std::vector<Fifo*> fifos, std::unique_ptr<FifoBackend> backend)
      : fifos_(std::move(fifos)),
        backend_(std::move(backend)),
        stop_(false),
        processed_count_(0) {}

  void start() { thread_ = std::thread(&FifoProxy::run, this); }

  void stop() {
    stop_ = true;
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  uint64_t getProcessedCount() const { return processed_count_; }

 private:
  void run() {
    while (!stop_ || backend_->hasPending()) {
      processed_count_ += backend_->progress();
      if (stop_) continue;

      for (size_t i = 0; i < fifos_.size(); ++i) {
        if (!backend_->canSubmit(i)) continue;
        ProxyTrigger trigger = fifos_[i]->poll();
        if (trigger.fst == 0) continue;
        processed_count_ += backend_->submit(i, *fifos_[i], trigger);
      }
    }
  }

  std::vector<Fifo*> fifos_;
  std::unique_ptr<FifoBackend> backend_;
  std::thread thread_;
  std::atomic<bool> stop_;
  std::atomic<uint64_t> processed_count_;
};

#ifdef USE_LIBFABRIC_CXI
namespace {

void checkFi(int rc, char const* what) {
  if (rc < 0) {
    throw std::runtime_error(std::string(what) + " failed: " +
                             fi_strerror(-rc));
  }
}

struct CxiEndpoint {
  fid_fabric* fabric = nullptr;
  fid_domain* domain = nullptr;
  fid_ep* ep = nullptr;
  fid_cq* cq = nullptr;
  fid_av* av = nullptr;
  fi_info* info = nullptr;
  std::vector<uint8_t> name;
  fi_addr_t peer = FI_ADDR_UNSPEC;
};

class CxiLoopback {
 public:
  CxiLoopback(size_t bytes, size_t max_outstanding)
      : bytes_(bytes),
        src_(std::max<size_t>(bytes, 8), 0xab),
        dst_(std::max<size_t>(bytes, 8), 0),
        contexts_(max_outstanding) {
    openEndpoint(a_);
    openEndpoint(b_);
    connectEndpoints(a_, b_);
    src_mr_ = registerMr(a_, src_.data(), src_.size());
    dst_mr_ = registerMr(b_, dst_.data(), dst_.size());
  }

  ~CxiLoopback() { close(); }

  CxiLoopback(CxiLoopback const&) = delete;
  CxiLoopback& operator=(CxiLoopback const&) = delete;

  bool canPost() const { return inflight_ < contexts_.size(); }

  bool post(size_t fifo_idx, Fifo* fifo) {
    if (inflight_ >= contexts_.size()) return false;
    WriteContext& ctx = contexts_[(next_context_++) % contexts_.size()];
    ctx.fifo = fifo;
    ctx.fifo_idx = fifo_idx;
    ctx.in_use = true;
    ssize_t rc = fi_write(a_.ep, src_.data(), bytes_, fi_mr_desc(src_mr_),
                          a_.peer, 0, fi_mr_key(dst_mr_), &ctx.context);
    checkFi(static_cast<int>(rc), "fi_write(CXI FIFO bench)");
    ++inflight_;
    return true;
  }

  uint64_t poll(std::vector<uint8_t>& pending) {
    uint64_t completed = 0;
    fi_cq_entry entries[64] = {};
    for (;;) {
      ssize_t rc = fi_cq_read(a_.cq, entries, 64);
      if (rc == -FI_EAGAIN) return completed;
      if (rc < 0) {
        fi_cq_err_entry err = {};
        ssize_t erc = fi_cq_readerr(a_.cq, &err, 0);
        if (erc >= 0) {
          throw std::runtime_error(
              std::string("CXI FIFO bench CQ error: ") +
              fi_cq_strerror(a_.cq, err.prov_errno, err.err_data, nullptr, 0));
        }
        checkFi(static_cast<int>(rc), "fi_cq_read(CXI FIFO bench)");
      }
      for (ssize_t i = 0; i < rc; ++i) {
        auto* ctx = static_cast<WriteContext*>(entries[i].op_context);
        if (!ctx || !ctx->in_use || !ctx->fifo) {
          throw std::runtime_error("invalid CXI FIFO bench completion context");
        }
        ctx->fifo->pop();
        if (ctx->fifo_idx < pending.size()) pending[ctx->fifo_idx] = 0;
        ctx->fifo = nullptr;
        ctx->fifo_idx = static_cast<size_t>(-1);
        ctx->in_use = false;
        --inflight_;
        ++completed;
      }
    }
  }

  size_t inflight() const { return inflight_; }

 private:
  struct WriteContext {
    fi_context context = {};
    Fifo* fifo = nullptr;
    size_t fifo_idx = static_cast<size_t>(-1);
    bool in_use = false;
  };

  static void openEndpoint(CxiEndpoint& e) {
    fi_info* hints = fi_allocinfo();
    if (!hints) throw std::runtime_error("fi_allocinfo failed");
    hints->caps = FI_RMA | FI_WRITE | FI_REMOTE_WRITE | FI_LOCAL_COMM |
                  FI_REMOTE_COMM;
    hints->mode = FI_CONTEXT;
    hints->ep_attr->type = FI_EP_RDM;
    hints->fabric_attr->prov_name = strdup("cxi");
    hints->domain_attr->mr_mode =
        FI_MR_ENDPOINT | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
    hints->domain_attr->threading = FI_THREAD_DOMAIN;

    int rc = fi_getinfo(FI_VERSION(1, 15), nullptr, nullptr, 0, hints,
                        &e.info);
    fi_freeinfo(hints);
    checkFi(rc, "fi_getinfo(cxi)");

    checkFi(fi_fabric(e.info->fabric_attr, &e.fabric, nullptr), "fi_fabric");
    checkFi(fi_domain(e.fabric, e.info, &e.domain, nullptr), "fi_domain");

    fi_cq_attr cq_attr = {};
    cq_attr.format = FI_CQ_FORMAT_CONTEXT;
    cq_attr.size = 131072;
    checkFi(fi_cq_open(e.domain, &cq_attr, &e.cq, nullptr), "fi_cq_open");

    fi_av_attr av_attr = {};
    av_attr.type = FI_AV_MAP;
    checkFi(fi_av_open(e.domain, &av_attr, &e.av, nullptr), "fi_av_open");

    checkFi(fi_endpoint(e.domain, e.info, &e.ep, nullptr), "fi_endpoint");
    checkFi(fi_ep_bind(e.ep, &e.cq->fid, FI_TRANSMIT | FI_RECV),
            "fi_ep_bind(cq)");
    checkFi(fi_ep_bind(e.ep, &e.av->fid, 0), "fi_ep_bind(av)");
    checkFi(fi_enable(e.ep), "fi_enable");

    size_t name_len = 0;
    rc = fi_getname(&e.ep->fid, nullptr, &name_len);
    if (rc != -FI_ETOOSMALL) checkFi(rc, "fi_getname(size)");
    e.name.resize(name_len);
    checkFi(fi_getname(&e.ep->fid, e.name.data(), &name_len), "fi_getname");
    e.name.resize(name_len);
  }

  static void connectEndpoints(CxiEndpoint& a, CxiEndpoint& b) {
    int rc = fi_av_insert(a.av, b.name.data(), 1, &a.peer, 0, nullptr);
    if (rc != 1) {
      if (rc < 0) checkFi(rc, "fi_av_insert(a)");
      throw std::runtime_error("fi_av_insert(a) inserted no address");
    }
    rc = fi_av_insert(b.av, a.name.data(), 1, &b.peer, 0, nullptr);
    if (rc != 1) {
      if (rc < 0) checkFi(rc, "fi_av_insert(b)");
      throw std::runtime_error("fi_av_insert(b) inserted no address");
    }
  }

  static fid_mr* registerMr(CxiEndpoint& e, void* ptr, size_t len) {
    iovec iov = {};
    iov.iov_base = ptr;
    iov.iov_len = len;
    fi_mr_attr attr = {};
    attr.mr_iov = &iov;
    attr.iov_count = 1;
    attr.access = FI_SEND | FI_RECV | FI_WRITE | FI_READ | FI_REMOTE_WRITE |
                  FI_REMOTE_READ;
    fid_mr* mr = nullptr;
    checkFi(fi_mr_regattr(e.domain, &attr, 0, &mr), "fi_mr_regattr");
    checkFi(fi_mr_bind(mr, &e.ep->fid, 0), "fi_mr_bind");
    checkFi(fi_control(&mr->fid, FI_ENABLE, nullptr),
            "fi_control(FI_ENABLE)");
    return mr;
  }

  static void closeEndpoint(CxiEndpoint& e) {
    if (e.ep) fi_close(&e.ep->fid);
    if (e.av) fi_close(&e.av->fid);
    if (e.cq) fi_close(&e.cq->fid);
    if (e.domain) fi_close(&e.domain->fid);
    if (e.fabric) fi_close(&e.fabric->fid);
    if (e.info) fi_freeinfo(e.info);
    e = {};
  }

  void close() {
    if (src_mr_) fi_close(&src_mr_->fid);
    if (dst_mr_) fi_close(&dst_mr_->fid);
    src_mr_ = nullptr;
    dst_mr_ = nullptr;
    closeEndpoint(a_);
    closeEndpoint(b_);
  }

  size_t bytes_ = 0;
  CxiEndpoint a_;
  CxiEndpoint b_;
  fid_mr* src_mr_ = nullptr;
  fid_mr* dst_mr_ = nullptr;
  std::vector<uint8_t> src_;
  std::vector<uint8_t> dst_;
  std::vector<WriteContext> contexts_;
  size_t next_context_ = 0;
  size_t inflight_ = 0;
};

class CxiFifoBackend final : public FifoBackend {
 public:
  CxiFifoBackend(size_t num_fifos, uint32_t bytes, uint32_t max_outstanding)
      : cxi_(bytes, max_outstanding), pending_(num_fifos, 0) {}

  uint64_t progress() override { return cxi_.poll(pending_); }

  bool hasPending() const override { return cxi_.inflight() > 0; }

  bool canSubmit(size_t fifo_idx) const override {
    return fifo_idx < pending_.size() && pending_[fifo_idx] == 0 &&
           cxi_.canPost();
  }

  uint64_t submit(size_t fifo_idx, Fifo& fifo, ProxyTrigger trigger) override {
    (void)trigger;
    pending_[fifo_idx] = 1;
    if (!cxi_.post(fifo_idx, &fifo)) {
      pending_[fifo_idx] = 0;
    }
    return 0;
  }

 private:
  CxiLoopback cxi_;
  std::vector<uint8_t> pending_;
};

}  // namespace
#endif

// Print throughput results
void printThroughputResults(
    std::vector<ThreadMetrics> const& metrics, uint64_t processed_count,
    double duration_sec, BenchmarkConfig const& config,
    std::vector<uint64_t> const* latency_samples = nullptr) {
  uint64_t total_pushes = 0;
  uint64_t total_cycles = 0;
  for (auto const& m : metrics) {
    total_pushes += m.push_count;
    total_cycles += m.total_cycles;
  }

  double throughput_mops = total_pushes / duration_sec / 1e6;
  double proxy_throughput_mops = processed_count / duration_sec / 1e6;

  if (config.mode == 4) {
    double avg_latency_ns =
        (total_pushes > 0)
            ? (double)total_cycles / total_pushes / config.gpu_clock_ghz / 2.0
            : 0;
    double p99_latency_ns = 0;
    if (latency_samples && !latency_samples->empty()) {
      std::vector<uint64_t> sorted_samples = *latency_samples;
      std::sort(sorted_samples.begin(), sorted_samples.end());
      size_t p99_idx = (sorted_samples.size() * 99) / 100;
      if (p99_idx >= sorted_samples.size()) p99_idx = sorted_samples.size() - 1;
      p99_latency_ns = sorted_samples[p99_idx] / config.gpu_clock_ghz / 2.0;
    }
    printf("%11.1f | %7u | %11.2f | %16.0f | %16.0f\n", config.target_mops,
           config.num_threads / 32, throughput_mops, avg_latency_ns,
           p99_latency_ns);
    return;
  }

  if (config.mode == 5) {
    printf("Threads: %4u | FIFO Size: %4u | CXI write bytes: %4u | ",
           config.num_threads, config.fifo_size, config.cxi_bytes);
    printf("GPU Pushes: %6.2f Mops/s | CXI Completed: %6.2f Mops/s",
           throughput_mops, proxy_throughput_mops);
    printf("\n");
    return;
  }

  printf("Threads: %4u | FIFO Size: %4u | ", config.num_threads,
         config.fifo_size);
  printf("GPU Pushes: %6.2f Mops/s | Proxy Processed: %6.2f Mops/s",
         throughput_mops, proxy_throughput_mops);

  if (config.measure_latency && total_pushes > 0) {
    uint64_t total_cycles = 0;
    uint64_t max_latency = 0;
    uint64_t min_latency = UINT64_MAX;

    for (auto const& m : metrics) {
      total_cycles += m.total_cycles;
      max_latency = std::max(max_latency, m.max_latency_cycles);
      if (m.min_latency_cycles > 0) {
        min_latency = std::min(min_latency, m.min_latency_cycles);
      }
    }

    double avg_cycles = (double)total_cycles / total_pushes;
    // Convert cycles to nanoseconds using actual GPU clock rate
    double avg_latency_ns = avg_cycles / config.gpu_clock_ghz;
    double max_latency_ns = max_latency / config.gpu_clock_ghz;
    double min_latency_ns =
        (min_latency == UINT64_MAX) ? 0 : min_latency / config.gpu_clock_ghz;

    // Calculate 99th percentile if we have samples
    double p99_latency_ns = 0;
    if (latency_samples && !latency_samples->empty()) {
      std::vector<uint64_t> sorted_samples = *latency_samples;
      std::sort(sorted_samples.begin(), sorted_samples.end());
      size_t p99_idx = (sorted_samples.size() * 99) / 100;
      if (p99_idx >= sorted_samples.size()) p99_idx = sorted_samples.size() - 1;
      p99_latency_ns = sorted_samples[p99_idx] / config.gpu_clock_ghz;
    }

    printf(" | RTT Latency (ns) - Avg: %.0f, Min: %.0f, P99: %.0f, Max: %.0f",
           avg_latency_ns, min_latency_ns, p99_latency_ns, max_latency_ns);
  }

  printf("\n");
}

constexpr int NUM_SMS = 128;
constexpr int NUM_FIFOS = 32;
constexpr int NUM_PROXIES = 4;
constexpr int FIFOS_PER_PROXY = NUM_FIFOS / NUM_PROXIES;

// Run single benchmark test
void runBenchmark(BenchmarkConfig const& config) {
  // Create 32 FIFOs
  std::vector<std::unique_ptr<Fifo>> fifos;
  std::vector<FifoDeviceHandle> deviceHandles;
  for (int i = 0; i < NUM_FIFOS; i++) {
    fifos.push_back(std::make_unique<Fifo>(config.fifo_size));
    deviceHandles.push_back(fifos[i]->deviceHandle());
  }

  // Copy device handles to GPU
  FifoDeviceHandle* d_fifo_handles;
  cudaMalloc(&d_fifo_handles, sizeof(FifoDeviceHandle) * NUM_FIFOS);
  cudaMemcpy(d_fifo_handles, deviceHandles.data(),
             sizeof(FifoDeviceHandle) * NUM_FIFOS, cudaMemcpyHostToDevice);

  // Allocate device metrics
  ThreadMetrics* d_metrics;
  cudaMalloc(&d_metrics, sizeof(ThreadMetrics) * config.num_threads);
  cudaMemset(d_metrics, 0, sizeof(ThreadMetrics) * config.num_threads);

  // Allocate latency samples buffer (for percentile calculation)
  uint64_t* d_latency_samples = nullptr;
  if (config.measure_latency) {
    cudaMalloc(&d_latency_samples,
               sizeof(uint64_t) * config.num_threads * MAX_LATENCY_SAMPLES);
    cudaMemset(d_latency_samples, 0,
               sizeof(uint64_t) * config.num_threads * MAX_LATENCY_SAMPLES);
  }

  // Stop flag
  bool* d_stop_flag;
  cudaMallocManaged(&d_stop_flag, sizeof(bool));
  *d_stop_flag = false;

  // Start 4 proxy threads, each managing 8 FIFOs.
  std::vector<std::unique_ptr<FifoProxy>> proxies;
  for (int i = 0; i < NUM_PROXIES; i++) {
    std::vector<Fifo*> proxy_fifos;
    for (int j = 0; j < FIFOS_PER_PROXY; j++) {
      proxy_fifos.push_back(fifos[i * FIFOS_PER_PROXY + j].get());
    }
    std::unique_ptr<FifoBackend> backend;
    if (config.mode == 5) {
#ifdef USE_LIBFABRIC_CXI
      backend = std::make_unique<CxiFifoBackend>(
          proxy_fifos.size(), config.cxi_bytes, config.cxi_max_outstanding);
#else
      throw std::runtime_error(
          "FIFO CXI mode requires building with USE_LIBFABRIC_CXI=1");
#endif
    } else {
      backend = std::make_unique<PopOnlyBackend>();
    }
    proxies.push_back(
        std::make_unique<FifoProxy>(std::move(proxy_fifos), std::move(backend)));
    proxies.back()->start();
  }

  // Launch GPU kernel
  dim3 grid(NUM_SMS);
  dim3 block((config.num_threads + grid.x - 1) / grid.x);

  auto start_time = std::chrono::high_resolution_clock::now();

  if (config.mode == 0 || config.mode == 5) {
    // Use throughput kernel with batching
    launchFifoKernel(grid, block, d_fifo_handles, d_metrics, config.num_threads,
                     config.test_duration_ms, config.warmup_iterations,
                     d_stop_flag, config.gpu_clock_ghz, config.batch_size,
                     NUM_FIFOS, d_latency_samples, MAX_LATENCY_SAMPLES);
  } else if (config.mode == 1) {
    // Use RTT latency kernel - polls after each push to measure round-trip time
    launchFifoLatencyKernel(grid, block, d_fifo_handles, d_metrics,
                            config.num_threads, config.test_duration_ms,
                            config.warmup_iterations, d_stop_flag,
                            config.gpu_clock_ghz, NUM_FIFOS, d_latency_samples,
                            MAX_LATENCY_SAMPLES);
  } else if (config.mode == 2) {
    // Use burst kernel - threads push as fast as possible without polling
    launchFifoBurstKernel(grid, block, d_fifo_handles, d_metrics,
                          config.num_threads, config.test_duration_ms,
                          config.warmup_iterations, d_stop_flag,
                          config.gpu_clock_ghz, NUM_FIFOS, d_latency_samples,
                          MAX_LATENCY_SAMPLES);
  } else if (config.mode == 3) {
    // Use random kernel - each thread randomly selects a FIFO
    launchFifoRandomKernel(grid, block, d_fifo_handles, d_metrics,
                           config.num_threads, config.test_duration_ms,
                           config.warmup_iterations, d_stop_flag,
                           config.gpu_clock_ghz, NUM_FIFOS, d_latency_samples,
                           MAX_LATENCY_SAMPLES);
  } else if (config.mode == 4) {
    // Use control MOPs mode, to get latency vs mops curve
    launchFifoControlledMopsKernel(
        grid, block, d_fifo_handles, d_metrics, config.num_threads,
        config.test_duration_ms, config.warmup_iterations, d_stop_flag,
        config.gpu_clock_ghz, NUM_FIFOS, d_latency_samples, MAX_LATENCY_SAMPLES,
        config.sleep_cycles);
  }

  // Wait for test duration (or kernel completion for latency mode)
  std::this_thread::sleep_for(
      std::chrono::milliseconds(config.test_duration_ms + 500));

  // Signal stop
  *d_stop_flag = true;
  cudaDeviceSynchronize();

  auto end_time = std::chrono::high_resolution_clock::now();
  double duration_sec =
      std::chrono::duration<double>(end_time - start_time).count();

  // Stop all proxies and collect total processed count
  uint64_t total_processed = 0;
  for (auto& proxy : proxies) {
    proxy->stop();
    total_processed += proxy->getProcessedCount();
  }

  // Copy metrics back
  std::vector<ThreadMetrics> h_metrics(config.num_threads);
  cudaMemcpy(h_metrics.data(), d_metrics,
             sizeof(ThreadMetrics) * config.num_threads,
             cudaMemcpyDeviceToHost);

  // Collect latency samples if measuring latency
  std::vector<uint64_t> latency_samples;
  if (config.measure_latency && d_latency_samples) {
    std::vector<uint64_t> all_samples(config.num_threads * MAX_LATENCY_SAMPLES);
    cudaMemcpy(all_samples.data(), d_latency_samples,
               sizeof(uint64_t) * config.num_threads * MAX_LATENCY_SAMPLES,
               cudaMemcpyDeviceToHost);

    // Filter out zero samples (unused slots)
    for (auto const& sample : all_samples) {
      if (sample > 0) {
        latency_samples.push_back(sample);
      }
    }
  }

  // Print results
  printThroughputResults(h_metrics, total_processed, duration_sec, config,
                         latency_samples.empty() ? nullptr : &latency_samples);

  // Cleanup
  cudaFree(d_metrics);
  cudaFree(d_stop_flag);
  cudaFree(d_fifo_handles);
  if (d_latency_samples) {
    cudaFree(d_latency_samples);
  }
}

int main(int argc, char** argv) {
  // Get LOCAL_RANK from environment (set by torchrun)
  int local_rank = 0;
  char const* local_rank_env = std::getenv("LOCAL_RANK");
  if (local_rank_env != nullptr) {
    local_rank = std::atoi(local_rank_env);
  }

  // Get WORLD_SIZE from environment (total number of processes)
  int world_size = 1;
  char const* world_size_env = std::getenv("WORLD_SIZE");
  if (world_size_env != nullptr) {
    world_size = std::atoi(world_size_env);
  }

  // Initialize CUDA with the appropriate device
  cudaSetDevice(local_rank);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, local_rank);

  // Get GPU clock rate (attribute is in kHz).
  int clock_rate_khz = 0;
  cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, local_rank);
  float gpu_clock_ghz = clock_rate_khz / 1000000.0f;

  printf("========================================\n");
  printf("FIFO Performance Benchmark\n");
  printf("========================================\n");
  printf("Rank: %d/%d\n", local_rank, world_size);
  printf("GPU %d: %s\n", local_rank, prop.name);
  printf("SM count: %d\n", prop.multiProcessorCount);
  printf("GPU Clock: %.2f GHz\n", gpu_clock_ghz);
  printf("Configuration: %d FIFOs, %d Proxy Threads (%d FIFOs/proxy)\n\n",
         NUM_FIFOS, NUM_PROXIES, NUM_FIFOS / NUM_PROXIES);

  BenchmarkConfig config = {.num_threads = 32,
                            .fifo_size = 2048,
                            .test_duration_ms = 3000,
                            .warmup_iterations = 100,
                            .batch_size = 32,
                            .gpu_clock_ghz = gpu_clock_ghz,
                            .measure_latency = true,
                            .mode = 0,
                            .sleep_cycles = 0,
                            .target_mops = 0.0f,
                            .cxi_bytes = 8,
                            .cxi_max_outstanding = 1024};

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "-l") {
      config.mode = 1;
    } else if (std::string(argv[i]) == "-b") {
      config.mode = 2;
    } else if (std::string(argv[i]) == "-r") {
      config.mode = 3;
    } else if (std::string(argv[i]) == "-c") {
      config.mode = 4;
    } else if (std::string(argv[i]) == "-x") {
      config.mode = 5;
    } else if (std::string(argv[i]) == "--cxi-bytes" && i + 1 < argc) {
      config.cxi_bytes = static_cast<uint32_t>(std::atoi(argv[++i]));
    } else if (std::string(argv[i]) == "--cxi-max-outstanding" &&
               i + 1 < argc) {
      config.cxi_max_outstanding =
          static_cast<uint32_t>(std::atoi(argv[++i]));
    }
  }

  // Test configurations
  std::vector<uint32_t> thread_counts = {1, 32, 64, 128, 256, 512, 1024};
  std::vector<uint32_t> fifo_sizes = {2048, 4096};

  if (config.mode == 0) {
    // Throughput tests
    printf("--- FIFO Dispatch Throughput Tests ---\n");
    printf(
        "(Testing different thread counts and FIFO sizes with batch size "
        "%u)\n\n",
        config.batch_size);
  } else if (config.mode == 1) {
    // RTT Latency tests
    printf("--- FIFO RTT Latency Tests ---\n");
    printf("(Measuring round-trip time: push + host processing + poll)\n\n");
  } else if (config.mode == 2) {
    // Burst tests
    printf("--- FIFO Burst Tests ---\n");
    printf("(Testing different thread counts and FIFO sizes)\n\n");
  } else if (config.mode == 3) {
    // Random FIFO selection tests
    printf("--- FIFO Random Selection Tests ---\n");
    printf("(Each thread randomly selects a FIFO for each push)\n\n");
  } else if (config.mode == 4) {
    printf("(Measuring latency vs. Mops with controlled mops)\n");
    printf(
        "Target Mops | Threads | Actual Mops | Avg Latency (ns) | P99 "
        "Latency (ns)\n");

    config.num_threads = 1024;
    config.fifo_size = 4096;
    config.test_duration_ms = 5000;

    for (float target_mops = 0.5f; target_mops <= 22.0f; target_mops += 0.5f) {
      uint32_t num_active_threads = config.num_threads / 32;
      float ops_per_thread_per_sec = (target_mops * 1e6f) / num_active_threads;
      float sleep_time_sec = 1.0f / ops_per_thread_per_sec;
      uint64_t sleep_ns = (uint64_t)(sleep_time_sec * 1e9f * 4.0f);

      config.sleep_cycles = sleep_ns;
      config.target_mops = target_mops;
      runBenchmark(config);
    }
    return 0;
  } else if (config.mode == 5) {
    printf("--- FIFO CXI Post/Completion Tests ---\n");
    printf(
        "(Proxy posts one %u-byte CXI write per FIFO command and pops the "
        "FIFO after CQ completion; max outstanding per proxy: %u)\n\n",
        config.cxi_bytes, config.cxi_max_outstanding);
  }

  for (auto fifo_size : fifo_sizes) {
    printf("FIFO Size: %u\n", fifo_size);
    printf("-----------------------------------\n");
    for (auto num_threads : thread_counts) {
      if (num_threads > fifo_size) continue;
      config.num_threads = num_threads;
      config.fifo_size = fifo_size;
      runBenchmark(config);
    }
    printf("\n");
  }

  printf("\n========================================\n");
  printf("Benchmark Complete\n");
  printf("========================================\n");

  return 0;
}
