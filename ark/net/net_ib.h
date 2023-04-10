// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_NET_IB_H_
#define ARK_NET_IB_H_

#include <list>
#include <memory>
#include <string>
#include <vector>

#define ARK_NET_IB_CQ_SIZE 1024
#define ARK_NET_IB_CQ_POLL_NUM 4
#define ARK_NET_IB_MAX_SENDS 64

namespace ark {

// IB memory region
class NetIbMr
{
  public:
    // To be shared with the remote peer
    struct Info
    {
        uint64_t addr;
        uint32_t rkey;
    };

    NetIbMr(void *mr_, void *buffer_);
    ~NetIbMr();

    void *get_addr() const;
    uint32_t get_lkey() const;
    uint32_t get_rkey() const;
    size_t get_length() const;
    void *get_mr() const
    {
        return mr;
    }
    void *get_buf() const
    {
        return buffer;
    };
    const struct Info &get_info() const
    {
        return info;
    };

  private:
    void *mr;
    void *buffer;
    struct Info info;
};

// IB queue pair
class NetIbQp
{
  public:
    // To be shared with the remote peer
    struct Info
    {
        uint16_t lid;
        uint8_t port;
        uint8_t link_layer;
        uint32_t qpn;
        uint64_t spn;
        int mtu;
    };

    NetIbQp(void *qp_, int port_);
    ~NetIbQp();
    int init();
    int rtr(const NetIbQp::Info *info);
    int rts();
    int stage_send(void *mr, const NetIbMr::Info *info, int size,
                   uint64_t wr_id, unsigned int imm_data, int offset = 0);
    int post_send();
    int post_recv(uint64_t wr_id);

    //
    void *get_qp() const
    {
        return qp;
    }
    const struct Info &get_info() const
    {
        return info;
    }

  private:
    void *qp;
    struct Info info;
    void *wrs;
    void *sges;
    int wrn;
};

// Holds resources of a single IB device.
class NetIbMgr
{
  public:
    NetIbMgr(int ib_dev_id, bool sep_sc_rc_ = false);
    ~NetIbMgr();

    // Create a new queue pair.
    NetIbQp *create_qp(int port = -1);
    // Register a memory region.
    NetIbMr *reg_mr(void *buffer, size_t size);
    //
    int poll_cq();
    int poll_scq();
    int poll_rcq();

    int get_numa_node() const
    {
        return numa_node;
    };
    void *get_wcs() const
    {
        return wcs;
    };
    int get_wcn() const
    {
        return wcn;
    };
    int get_wc_status(int i) const;
    const char *get_wc_status_str(int i) const;
    uint64_t get_wc_wr_id(int i) const;
    unsigned int get_wc_imm_data(int i) const;

  private:
    std::string device_name;
    int numa_node;
    void *ctx;
    void *cq;
    void *scq;
    void *rcq;
    void *pd;
    void *wcs;
    const bool sep_sc_rc;
    int wcn;
    std::vector<int> ports;
    std::list<std::unique_ptr<NetIbQp>> qps;
    std::list<std::unique_ptr<NetIbMr>> mrs;
};

NetIbMgr *get_net_ib_mgr(int ib_dev_id);
int get_net_ib_device_num();
void *page_memalign(std::size_t size);

} // namespace ark

#endif // ARK_NET_IB_H_
