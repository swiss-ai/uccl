#include <rdma/fabric.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>

#include <chrono>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static void check_fi(int rc, const char* what) {
  if (rc < 0) {
    throw std::runtime_error(std::string(what) + ": " + fi_strerror(-rc));
  }
}

struct Endpoint {
  fid_fabric* fabric = nullptr;
  fid_domain* domain = nullptr;
  fid_ep* ep = nullptr;
  fid_cq* cq = nullptr;
  fid_av* av = nullptr;
  fi_info* info = nullptr;
  std::vector<uint8_t> name;
  fi_addr_t peer = FI_ADDR_UNSPEC;
};

static Endpoint open_ep() {
  Endpoint e;
  fi_info* hints = fi_allocinfo();
  if (!hints) throw std::runtime_error("fi_allocinfo failed");
  hints->caps = FI_RMA | FI_WRITE | FI_ATOMIC | FI_FENCE | FI_REMOTE_WRITE |
                FI_REMOTE_READ | FI_LOCAL_COMM | FI_REMOTE_COMM;
  hints->mode = FI_CONTEXT;
  hints->ep_attr->type = FI_EP_RDM;
  hints->fabric_attr->prov_name = strdup("cxi");
  hints->domain_attr->mr_mode =
      FI_MR_ENDPOINT | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
  hints->domain_attr->threading = FI_THREAD_DOMAIN;

  check_fi(fi_getinfo(FI_VERSION(1, 15), nullptr, nullptr, 0, hints, &e.info),
           "fi_getinfo");
  fi_freeinfo(hints);

  check_fi(fi_fabric(e.info->fabric_attr, &e.fabric, nullptr), "fi_fabric");
  check_fi(fi_domain(e.fabric, e.info, &e.domain, nullptr), "fi_domain");

  fi_cq_attr cq_attr = {};
  cq_attr.format = FI_CQ_FORMAT_DATA;
  cq_attr.size = 1024;
  check_fi(fi_cq_open(e.domain, &cq_attr, &e.cq, nullptr), "fi_cq_open");

  fi_av_attr av_attr = {};
  av_attr.type = FI_AV_MAP;
  check_fi(fi_av_open(e.domain, &av_attr, &e.av, nullptr), "fi_av_open");

  check_fi(fi_endpoint(e.domain, e.info, &e.ep, nullptr), "fi_endpoint");
  check_fi(fi_ep_bind(e.ep, &e.cq->fid, FI_TRANSMIT | FI_RECV),
           "fi_ep_bind(cq)");
  check_fi(fi_ep_bind(e.ep, &e.av->fid, 0), "fi_ep_bind(av)");
  check_fi(fi_enable(e.ep), "fi_enable");

  size_t name_len = 0;
  int rc = fi_getname(&e.ep->fid, nullptr, &name_len);
  if (rc != -FI_ETOOSMALL) check_fi(rc, "fi_getname(size)");
  e.name.resize(name_len);
  check_fi(fi_getname(&e.ep->fid, e.name.data(), &name_len), "fi_getname");
  e.name.resize(name_len);
  return e;
}

static void connect_eps(Endpoint& a, Endpoint& b) {
  int rc = fi_av_insert(a.av, b.name.data(), 1, &a.peer, 0, nullptr);
  if (rc != 1) {
    if (rc < 0) check_fi(rc, "fi_av_insert(a)");
    throw std::runtime_error("fi_av_insert(a) inserted no address");
  }
  rc = fi_av_insert(b.av, a.name.data(), 1, &b.peer, 0, nullptr);
  if (rc != 1) {
    if (rc < 0) check_fi(rc, "fi_av_insert(b)");
    throw std::runtime_error("fi_av_insert(b) inserted no address");
  }
}

static fid_mr* reg_mr(fid_domain* domain, fid_ep* ep, void* ptr, size_t len,
                      uint64_t access) {
  iovec iov = {};
  iov.iov_base = ptr;
  iov.iov_len = len;
  fi_mr_attr attr = {};
  attr.mr_iov = &iov;
  attr.iov_count = 1;
  attr.access = access;
  fid_mr* mr = nullptr;
  check_fi(fi_mr_regattr(domain, &attr, 0, &mr), "fi_mr_regattr");
  check_fi(fi_mr_bind(mr, &ep->fid, 0), "fi_mr_bind");
  check_fi(fi_control(&mr->fid, FI_ENABLE, nullptr), "fi_control(FI_ENABLE)");
  return mr;
}

static void wait_cq(fid_cq* cq, const char* what) {
  fi_cq_data_entry entry = {};
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  for (;;) {
    ssize_t rc = fi_cq_read(cq, &entry, 1);
    if (rc == 1) return;
    if (rc == -FI_EAGAIN) {
      if (std::chrono::steady_clock::now() > deadline) {
        throw std::runtime_error(std::string("timeout waiting for ") + what);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    if (rc < 0) {
      fi_cq_err_entry err = {};
      ssize_t erc = fi_cq_readerr(cq, &err, 0);
      if (erc >= 0) {
        throw std::runtime_error(std::string(what) + " CQ error: " +
                                 fi_cq_strerror(cq, err.prov_errno,
                                                 err.err_data, nullptr, 0));
      }
      check_fi(static_cast<int>(rc), what);
    }
  }
}

int main() {
  Endpoint a = open_ep();
  Endpoint b = open_ep();
  connect_eps(a, b);

  alignas(64) uint64_t src[8] = {};
  alignas(64) uint64_t dst[8] = {};
  alignas(64) uint64_t add_value = 7;
  alignas(64) uint64_t counter = 3;
  for (int i = 0; i < 8; ++i) src[i] = 0xabc000 + i;

  uint64_t access = FI_SEND | FI_RECV | FI_WRITE | FI_READ | FI_REMOTE_WRITE |
                    FI_REMOTE_READ;
  fid_mr* src_mr = reg_mr(a.domain, a.ep, src, sizeof(src), access);
  fid_mr* add_mr =
      reg_mr(a.domain, a.ep, &add_value, sizeof(add_value), access);
  fid_mr* dst_mr = reg_mr(b.domain, b.ep, dst, sizeof(dst), access);
  fid_mr* ctr_mr = reg_mr(b.domain, b.ep, &counter, sizeof(counter), access);

  fi_context write_ctx = {};
  ssize_t rc = fi_write(a.ep, src, sizeof(src), fi_mr_desc(src_mr), a.peer, 0,
                        fi_mr_key(dst_mr), &write_ctx);
  check_fi(static_cast<int>(rc), "fi_write");

  fi_context atomic_ctx = {};
  fi_ioc msg_iov = {};
  msg_iov.addr = &add_value;
  msg_iov.count = 1;
  void* add_desc = fi_mr_desc(add_mr);
  fi_rma_ioc rma_iov = {};
  rma_iov.addr = 0;
  rma_iov.count = 1;
  rma_iov.key = fi_mr_key(ctr_mr);
  fi_msg_atomic msg = {};
  msg.msg_iov = &msg_iov;
  msg.desc = &add_desc;
  msg.iov_count = 1;
  msg.addr = a.peer;
  msg.rma_iov = &rma_iov;
  msg.rma_iov_count = 1;
  msg.datatype = FI_UINT64;
  msg.op = FI_SUM;
  msg.context = &atomic_ctx;
  size_t atomic_count = 0;
  check_fi(fi_atomicvalid(a.ep, FI_UINT64, FI_SUM, &atomic_count),
           "fi_atomicvalid(FI_UINT64/FI_SUM)");
  std::printf("atomicvalid count=%zu\n", atomic_count);
  rc = fi_atomicmsg(a.ep, &msg, 0);
  check_fi(static_cast<int>(rc), "fi_atomicmsg");

  wait_cq(a.cq, "write completion");
  wait_cq(a.cq, "atomic completion");

  for (int i = 0; i < 8; ++i) {
    if (dst[i] != src[i]) {
      std::fprintf(stderr, "payload mismatch at %d: got=%lx want=%lx\n", i,
                   dst[i], src[i]);
      return 2;
    }
  }
  if (counter != 10) {
    std::fprintf(stderr, "counter mismatch: got=%ld want=10\n", counter);
    return 3;
  }
  std::puts("cxi host write+atomic ok");
  return 0;
}
