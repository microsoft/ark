// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <infiniband/verbs.h>
#include <malloc.h>
#include <map>
#include <unistd.h>

#include "file_io.h"
#include "logging.h"
#include "net/net_ib.h"

namespace ark {

// IB memory region
NetIbMr::NetIbMr(void *mr_, void *buffer_) : mr{mr_}, buffer{buffer_}
{
    this->info.addr = (uint64_t)this->get_buf();
    this->info.rkey = this->get_rkey();
}
NetIbMr::~NetIbMr()
{
    // IB resources should be freed by NetIbMgr.
}
void *NetIbMr::get_addr() const
{
    return ((struct ibv_mr *)this->mr)->addr;
}
uint32_t NetIbMr::get_lkey() const
{
    return ((struct ibv_mr *)this->mr)->lkey;
}
uint32_t NetIbMr::get_rkey() const
{
    return ((struct ibv_mr *)this->mr)->rkey;
}
size_t NetIbMr::get_length() const
{
    return ((struct ibv_mr *)this->mr)->length;
}

// IB queue pair
NetIbQp::NetIbQp(void *qp_, int port_)
    : qp{qp_}, wrs{new struct ibv_send_wr[ARK_NET_IB_MAX_SENDS]},
      sges{new struct ibv_sge[ARK_NET_IB_MAX_SENDS]}, wrn{0}
{
    assert(qp_ != nullptr);
    struct ibv_context *ctx = ((struct ibv_qp *)qp_)->context;
    struct ibv_port_attr port_attr;
    if (ibv_query_port(ctx, port_, &port_attr) != 0) {
        LOGERR("failed to query IB port: %d", port_);
    }
    this->info.lid = port_attr.lid;
    this->info.port = port_;
    this->info.link_layer = port_attr.link_layer;
    this->info.qpn = ((struct ibv_qp *)qp_)->qp_num;
    this->info.mtu = port_attr.active_mtu;
    if (port_attr.link_layer != IBV_LINK_LAYER_INFINIBAND) {
        union ibv_gid gid;
        if (ibv_query_gid(ctx, port_, 0, &gid) != 0) {
            LOGERR("failed to query GID");
        }
        this->info.spn = gid.global.subnet_prefix;
    }
    if (this->init() != 0) {
        LOGERR("failed to modify QP to INIT");
    }
    std::memset(this->wrs, 0,
                sizeof(struct ibv_send_wr) * ARK_NET_IB_MAX_SENDS);
    std::memset(this->sges, 0, sizeof(struct ibv_sge) * ARK_NET_IB_MAX_SENDS);
    LOG(DEBUG, "QP INIT: qpn=", this->info.qpn);
}

NetIbQp::~NetIbQp()
{
    // IB resources should be freed by NetIbMgr.
    delete reinterpret_cast<struct ibv_send_wr *>(this->wrs);
    delete reinterpret_cast<struct ibv_sge *>(this->sges);
}

int NetIbQp::init()
{
    struct ibv_qp_attr qp_attr;
    std::memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = this->info.port;
    qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
    if (ibv_modify_qp((struct ibv_qp *)this->qp, &qp_attr,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                          IBV_QP_ACCESS_FLAGS) != 0) {
        LOG(WARN, "ibv_modify_qp failed (", errno, ")");
        return -1;
    }
    return 0;
}

int NetIbQp::rtr(const NetIbQp::Info *info)
{
    struct ibv_qp_attr qp_attr;
    std::memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_RTR;
    qp_attr.path_mtu = IBV_MTU_1024;
    qp_attr.dest_qp_num = info->qpn;
    qp_attr.rq_psn = 0;
    qp_attr.max_dest_rd_atomic = 1;
    qp_attr.min_rnr_timer = 0x12;
    if (info->link_layer == IBV_LINK_LAYER_ETHERNET) {
        qp_attr.ah_attr.is_global = 1;
        qp_attr.ah_attr.grh.dgid.global.subnet_prefix = info->spn;
        qp_attr.ah_attr.grh.dgid.global.interface_id = info->lid;
        qp_attr.ah_attr.grh.flow_label = 0;
        qp_attr.ah_attr.grh.sgid_index = 0;
        qp_attr.ah_attr.grh.hop_limit = 255;
        qp_attr.ah_attr.grh.traffic_class = 0;
    } else {
        qp_attr.ah_attr.is_global = 0;
        qp_attr.ah_attr.dlid = info->lid;
    }
    qp_attr.ah_attr.sl = 0;
    qp_attr.ah_attr.src_path_bits = 0;
    qp_attr.ah_attr.port_num = info->port;
    LOG(DEBUG, "QP RTR: qpn=", this->info.qpn, " remote=", info->qpn);
    if (ibv_modify_qp((struct ibv_qp *)this->qp, &qp_attr,
                      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                          IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                          IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) !=
        0) {
        LOG(WARN, "ibv_modify_qp failed (", errno, ")");
        return -1;
    }
    return 0;
}

int NetIbQp::rts()
{
    struct ibv_qp_attr qp_attr;
    std::memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_RTS;
    qp_attr.timeout = 18;
    qp_attr.retry_cnt = 7;
    qp_attr.rnr_retry = 7;
    qp_attr.sq_psn = 0;
    qp_attr.max_rd_atomic = 1;
    if (ibv_modify_qp((struct ibv_qp *)this->qp, &qp_attr,
                      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                          IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                          IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
        LOG(WARN, "ibv_modify_qp failed (", errno, ")");
        return -1;
    }
    LOG(DEBUG, "QP RTS: qpn=", this->info.qpn);
    return 0;
}

int NetIbQp::stage_send(void *mr, const NetIbMr::Info *info, int size,
                        uint64_t wr_id, unsigned int imm_data, int offset)
{
    if (this->wrn >= ARK_NET_IB_MAX_SENDS) {
        LOG(WARN, "wrn=", this->wrn);
        return -1;
    }
    int wrn = this->wrn;
    struct ibv_send_wr *wr_ = &((struct ibv_send_wr *)this->wrs)[wrn];
    struct ibv_sge *sge_ = &((struct ibv_sge *)this->sges)[wrn];
    std::memset(wr_, 0, sizeof(struct ibv_send_wr));
    std::memset(sge_, 0, sizeof(struct ibv_sge));
    wr_->wr_id = wr_id;
    wr_->sg_list = sge_;
    wr_->num_sge = 1;
    wr_->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr_->imm_data = imm_data;
    wr_->send_flags = IBV_SEND_SIGNALED;
    wr_->wr.rdma.remote_addr = info->addr;
    wr_->wr.rdma.rkey = info->rkey;
    wr_->next = nullptr;
    sge_->addr = (uint64_t)(((NetIbMr *)mr)->get_buf()) + (uint64_t)offset;
    sge_->length = size;
    sge_->lkey = ((NetIbMr *)mr)->get_lkey();
    if (wrn > 0) {
        ((struct ibv_send_wr *)this->wrs)[wrn - 1].next = wr_;
    }
    // LOG(DEBUG, "stage_send addr=", (void *)sge_->addr,
    //     " remote_addr=", (void *)wr_->wr.rdma.remote_addr);
    this->wrn++;
    return this->wrn;
}

int NetIbQp::post_send()
{
    struct ibv_send_wr *bad_wr;
    int ret = ibv_post_send((struct ibv_qp *)this->qp,
                            (struct ibv_send_wr *)this->wrs, &bad_wr);
    if (ret != 0) {
        LOG(WARN, "ibv_post_send failed (", ret, ")");
        return -1;
    }
    std::memset(this->wrs, 0, sizeof(struct ibv_send_wr) * this->wrn);
    std::memset(this->sges, 0, sizeof(struct ibv_sge) * this->wrn);
    this->wrn = 0;
    return 0;
}

int NetIbQp::post_recv(uint64_t wr_id)
{
    struct ibv_recv_wr wr, *bad_wr;
    wr.wr_id = wr_id;
    wr.sg_list = nullptr;
    wr.num_sge = 0;
    wr.next = nullptr;
    if (ibv_post_recv((struct ibv_qp *)this->qp, &wr, &bad_wr) != 0) {
        LOG(WARN, "ibv_post_recv failed (", errno, ")");
        return -1;
    }
    return 0;
}

////////////////////////////////////////////////////////////////////////////////

// Holds resources of a single IB device.
NetIbMgr::NetIbMgr(int ib_dev_id, bool sep_sc_rc_)
    : wcs{new struct ibv_wc[ARK_NET_IB_CQ_POLL_NUM]}, sep_sc_rc{sep_sc_rc_}
{
    int num;
    struct ibv_device **devices = ibv_get_device_list(&num);
    if (ib_dev_id >= num) {
        LOGERR("ib_dev_id=", ib_dev_id, " num=", num);
    }
    this->device_name = ibv_get_device_name(devices[ib_dev_id]);
    struct ibv_context *ctx_ = ibv_open_device(devices[ib_dev_id]);
    std::string ibdev_path(devices[ib_dev_id]->ibdev_path);
    ibv_free_device_list(devices);
    if (ctx_ == nullptr) {
        LOGERR("failed to open IB device: ", this->device_name);
    }
    this->ctx = ctx_;
    // Get the NUMA node
    this->numa_node = -1;
    if (is_dir(ibdev_path)) {
        std::string numa_node_path = ibdev_path + "/device/numa_node";
        if (is_file(numa_node_path)) {
            std::string numa_node_str = read_file(numa_node_path);
            this->numa_node = std::stoi(numa_node_str);
        }
    }
    LOG(DEBUG, "opened IB device: ", this->device_name, " numa_node ",
        this->numa_node);
    // Check available ports
    struct ibv_device_attr devAttr;
    if (ibv_query_device(ctx_, &devAttr) != 0) {
        LOGERR("failed to query IB device: ", this->device_name);
    }
    for (int port = 1; port <= devAttr.phys_port_cnt; port++) {
        struct ibv_port_attr portAttr;
        if (ibv_query_port(ctx_, port, &portAttr) != 0) {
            LOG(WARN, "failed to query IB port: ", port);
            continue;
        }
        if (portAttr.state != IBV_PORT_ACTIVE) {
            continue;
        }
        if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND &&
            portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) {
            continue;
        }
        this->ports.emplace_back(port);
    }
    //
    if (sep_sc_rc_) {
        this->scq =
            ibv_create_cq(ctx_, ARK_NET_IB_CQ_SIZE, nullptr, nullptr, 0);
        this->rcq =
            ibv_create_cq(ctx_, ARK_NET_IB_CQ_SIZE, nullptr, nullptr, 0);
        this->cq = nullptr;
    } else {
        this->cq = ibv_create_cq(ctx_, ARK_NET_IB_CQ_SIZE, nullptr, nullptr, 0);
        this->scq = nullptr;
        this->rcq = nullptr;
    }
    this->pd = ibv_alloc_pd(ctx_);
}

NetIbMgr::~NetIbMgr()
{
    for (auto &mr : this->mrs) {
        ibv_dereg_mr((struct ibv_mr *)mr->get_mr());
    }
    for (auto &qp : this->qps) {
        ibv_destroy_qp((struct ibv_qp *)qp->get_qp());
    }
    ibv_dealloc_pd((struct ibv_pd *)this->pd);
    if (this->sep_sc_rc) {
        ibv_destroy_cq((struct ibv_cq *)this->scq);
        ibv_destroy_cq((struct ibv_cq *)this->rcq);
    } else {
        ibv_destroy_cq((struct ibv_cq *)this->cq);
    }
    ibv_close_device((struct ibv_context *)this->ctx);
    delete reinterpret_cast<struct ibv_wc *>(this->wcs);
}

NetIbQp *NetIbMgr::create_qp(int port)
{
    if (port < 0) {
        port = this->ports[0];
    } else {
        bool found = false;
        for (auto &p : this->ports) {
            if (p == port) {
                found = true;
                break;
            }
        }
        if (!found) {
            LOG(WARN, "invalid port: ", port);
            return nullptr;
        }
    }
    struct ibv_qp_init_attr qp_init_attr;
    std::memset(&qp_init_attr, 0, sizeof(struct ibv_qp_init_attr));
    qp_init_attr.sq_sig_all = 0;
    if (this->sep_sc_rc) {
        qp_init_attr.send_cq = (struct ibv_cq *)this->scq;
        qp_init_attr.recv_cq = (struct ibv_cq *)this->rcq;
    } else {
        qp_init_attr.send_cq = (struct ibv_cq *)this->cq;
        qp_init_attr.recv_cq = (struct ibv_cq *)this->cq;
    }
    qp_init_attr.qp_type = IBV_QPT_RC;
    qp_init_attr.cap.max_send_wr = 1024;
    qp_init_attr.cap.max_recv_wr = 128;
    qp_init_attr.cap.max_send_sge = 1;
    qp_init_attr.cap.max_recv_sge = 1;
    qp_init_attr.cap.max_inline_data = 0;
    struct ibv_qp *qp = ibv_create_qp((struct ibv_pd *)this->pd, &qp_init_attr);
    if (qp == nullptr) {
        LOG(WARN, "ibv_create_qp failed (", errno, ")");
        return nullptr;
    }
    this->qps.emplace_back(new NetIbQp{qp, port});
    return this->qps.back().get();
}

// Register a memory region for remote write.
NetIbMr *NetIbMgr::reg_mr(void *buffer, size_t size)
{
    if (size == 0) {
        return nullptr;
    }
    static __thread uintptr_t pageSize = 0;
    if (pageSize == 0) {
        pageSize = sysconf(_SC_PAGESIZE);
    }
    uintptr_t addr = (uintptr_t)buffer & -pageSize;
    size_t pages = ((uintptr_t)buffer + size - addr + pageSize - 1) / pageSize;
    struct ibv_mr *mr =
        ibv_reg_mr((struct ibv_pd *)this->pd, (void *)addr, pages * pageSize,
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                       IBV_ACCESS_REMOTE_READ | IBV_ACCESS_RELAXED_ORDERING);
    if (mr == nullptr) {
        LOG(WARN, "ibv_reg_mr failed (", errno, ")");
        return nullptr;
    }
    LOG(DEBUG, "MR addr ", (void *)addr, " buffer ", (void *)buffer, " size ",
        pages * pageSize);
    this->mrs.emplace_back(new NetIbMr{mr, buffer});
    return this->mrs.back().get();
}

int NetIbMgr::poll_cq()
{
    if (this->sep_sc_rc) {
        LOGERR("poll_cq not supported for separate send/recv CQs");
    }
    int ret = ibv_poll_cq((struct ibv_cq *)this->cq, ARK_NET_IB_CQ_POLL_NUM,
                          (struct ibv_wc *)this->wcs);
    if (ret < 0) {
        LOG(WARN, "ibv_poll_cq failed (", errno, ")");
        return -1;
    }
    this->wcn = ret;
    return ret;
}

int NetIbMgr::poll_scq()
{
    if (!this->sep_sc_rc) {
        LOGERR("poll_scq not supported for single send/recv CQ");
    }
    int ret = ibv_poll_cq((struct ibv_cq *)this->scq, ARK_NET_IB_CQ_POLL_NUM,
                          (struct ibv_wc *)this->wcs);
    if (ret < 0) {
        LOG(WARN, "ibv_poll_cq failed (", errno, ")");
        return -1;
    }
    this->wcn = ret;
    return ret;
}

int NetIbMgr::poll_rcq()
{
    if (!this->sep_sc_rc) {
        LOGERR("poll_rcq not supported for single send/recv CQ");
    }
    int ret = ibv_poll_cq((struct ibv_cq *)this->rcq, ARK_NET_IB_CQ_POLL_NUM,
                          (struct ibv_wc *)this->wcs);
    if (ret < 0) {
        LOG(WARN, "ibv_poll_cq failed (", errno, ")");
        return -1;
    }
    this->wcn = ret;
    return ret;
}

int NetIbMgr::get_wc_status(int i) const
{
    if (i < 0 || i >= this->wcn) {
        return -1;
    }
    struct ibv_wc *wc = (struct ibv_wc *)this->wcs + i;
    return wc->status;
}

const char *NetIbMgr::get_wc_status_str(int i) const
{
    if (i < 0 || i >= this->wcn) {
        return nullptr;
    }
    struct ibv_wc *wc = (struct ibv_wc *)this->wcs + i;
    return ibv_wc_status_str(wc->status);
}

uint64_t NetIbMgr::get_wc_wr_id(int i) const
{
    if (i < 0 || i >= this->wcn) {
        return -1;
    }
    struct ibv_wc *wc = (struct ibv_wc *)this->wcs + i;
    return wc->wr_id;
}

unsigned int NetIbMgr::get_wc_imm_data(int i) const
{
    if (i < 0 || i >= this->wcn) {
        return -1;
    }
    struct ibv_wc *wc = (struct ibv_wc *)this->wcs + i;
    return wc->imm_data;
}

////////////////////////////////////////////////////////////////////////////////

std::map<int, NetIbMgr *> ARK_NET_IB_MGR_GLOBAL;

// Get a NetIbMgr instance
NetIbMgr *get_net_ib_mgr(int ib_dev_id)
{
    auto it = ARK_NET_IB_MGR_GLOBAL.find(ib_dev_id);
    if (it == ARK_NET_IB_MGR_GLOBAL.end()) {
        ARK_NET_IB_MGR_GLOBAL[ib_dev_id] = new NetIbMgr{ib_dev_id};
        return ARK_NET_IB_MGR_GLOBAL[ib_dev_id];
    }
    return it->second;
}

// Get the number of IB devices
int get_net_ib_device_num()
{
    int num;
    struct ibv_device **devices = ibv_get_device_list(&num);
    ibv_free_device_list(devices);
    return num;
}

#define DIVUP(x, y) (((x) + (y)-1) / (y))
#define ROUNDUP(x, y) (DIVUP((x), (y)) * (y))

void *page_memalign(std::size_t size)
{
    std::size_t page_size = sysconf(_SC_PAGESIZE);
    void *p;
    int size_aligned = ROUNDUP(size, page_size);
    p = memalign(page_size, size_aligned);
    if (p == nullptr) {
        return nullptr;
    }
    memset(p, 0, size);
    return p;
}

} // namespace ark
