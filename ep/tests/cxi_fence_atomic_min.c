#include <rdma/fabric.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(x) do { int _r = (int)(x); if (_r < 0) { \
  fprintf(stderr, "%s: %s (%d)\n", #x, fi_strerror(-_r), _r); exit(1); }} while (0)

struct ep {
  struct fid_fabric *fab; struct fid_domain *dom; struct fid_ep *ep;
  struct fid_cq *cq; struct fid_av *av; void *name; size_t name_len;
  fi_addr_t peer;
};

static struct ep open_ep(void) {
  struct ep e = {0};
  struct fi_info *h = fi_allocinfo(), *info = NULL;
  h->caps = FI_RMA | FI_WRITE | FI_ATOMIC | FI_FENCE | FI_REMOTE_WRITE |
            FI_REMOTE_READ | FI_LOCAL_COMM | FI_REMOTE_COMM;
  h->mode = FI_CONTEXT;
  h->ep_attr->type = FI_EP_RDM;
  h->fabric_attr->prov_name = strdup("cxi");
  h->domain_attr->mr_mode = FI_MR_ENDPOINT | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
  CHECK(fi_getinfo(FI_VERSION(1, 22), NULL, NULL, 0, h, &info));
  fi_freeinfo(h);
  CHECK(fi_fabric(info->fabric_attr, &e.fab, NULL));
  CHECK(fi_domain(e.fab, info, &e.dom, NULL));
  struct fi_cq_attr cq = { .format = FI_CQ_FORMAT_DATA, .size = 16 };
  CHECK(fi_cq_open(e.dom, &cq, &e.cq, NULL));
  struct fi_av_attr av = { .type = FI_AV_MAP };
  CHECK(fi_av_open(e.dom, &av, &e.av, NULL));
  CHECK(fi_endpoint(e.dom, info, &e.ep, NULL));
  CHECK(fi_ep_bind(e.ep, &e.cq->fid, FI_TRANSMIT | FI_RECV));
  CHECK(fi_ep_bind(e.ep, &e.av->fid, 0));
  CHECK(fi_enable(e.ep));
  int r = fi_getname(&e.ep->fid, NULL, &e.name_len);
  if (r != -FI_ETOOSMALL) CHECK(r);
  e.name = calloc(1, e.name_len);
  CHECK(fi_getname(&e.ep->fid, e.name, &e.name_len));
  fi_freeinfo(info);
  return e;
}

static struct fid_mr *mr(struct ep *e, void *p, size_t n, uint64_t acc) {
  struct iovec iov = { .iov_base = p, .iov_len = n };
  struct fi_mr_attr a = { .mr_iov = &iov, .iov_count = 1, .access = acc };
  struct fid_mr *m = NULL;
  CHECK(fi_mr_regattr(e->dom, &a, 0, &m));
  CHECK(fi_mr_bind(m, &e->ep->fid, 0));
  CHECK(fi_control(&m->fid, FI_ENABLE, NULL));
  return m;
}

static void cq1(struct ep *e) {
  struct fi_cq_data_entry c;
  for (int i = 0; i < 10000; ++i) {
    ssize_t r = fi_cq_read(e->cq, &c, 1);
    if (r == 1) return;
    if (r != -FI_EAGAIN) CHECK(r);
    usleep(1000);
  }
  fprintf(stderr, "CQ timeout\n"); exit(2);
}

int main(void) {
  struct ep a = open_ep(), b = open_ep();
  if (fi_av_insert(a.av, b.name, 1, &a.peer, 0, NULL) != 1) return 3;
  if (fi_av_insert(b.av, a.name, 1, &b.peer, 0, NULL) != 1) return 4;
  uint64_t src = 0x1234, dst = 0, add = 7, ctr = 3;
  uint64_t acc = FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;
  struct fid_mr *src_mr = mr(&a, &src, 8, acc), *add_mr = mr(&a, &add, 8, acc);
  struct fid_mr *dst_mr = mr(&b, &dst, 8, acc), *ctr_mr = mr(&b, &ctr, 8, acc);
  struct fi_context wctx = {0}, actx = {0};
  CHECK(fi_write(a.ep, &src, 8, fi_mr_desc(src_mr), a.peer, 0,
                 fi_mr_key(dst_mr), &wctx));
  struct fi_ioc iov = { .addr = &add, .count = 1 }; void *desc = fi_mr_desc(add_mr);
  struct fi_rma_ioc rma = { .addr = 0, .count = 1, .key = fi_mr_key(ctr_mr) };
  struct fi_msg_atomic msg = { .msg_iov = &iov, .desc = &desc, .iov_count = 1,
    .addr = a.peer, .rma_iov = &rma, .rma_iov_count = 1,
    .datatype = FI_UINT64, .op = FI_SUM, .context = &actx };
  CHECK(fi_atomicmsg(a.ep, &msg, FI_FENCE));
  cq1(&a); cq1(&a);
  printf("dst=0x%lx ctr=%lu\n", dst, ctr);
  return (dst == src && ctr == 10) ? 0 : 5;
}
