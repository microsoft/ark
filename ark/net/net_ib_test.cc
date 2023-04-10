// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#include "ark/gpu/gpu_logging.h"
#include "ark/gpu/gpu_mem.h"
#include "ark/init.h"
#include "ark/ipc/ipc_coll.h"
#include "ark/net/net_ib.h"
#include "ark/process.h"
#include "ark/unittest/unittest_utils.h"
#include <cstring>
#include <numa.h>
#include <string>

#define BW_TEST_BYTES 1073741824UL

static void numa_bind(int node)
{
    nodemask_t mask;
    nodemask_zero(&mask);
    nodemask_set_compat(&mask, node);
    numa_bind_compat(&mask);
}

static double sr_loop(ark::NetIbMgr *mgr, ark::NetIbQp *qp, ark::NetIbMr *mr1,
                      ark::NetIbMr *mr2, ark::NetIbMr::Info *rmi1,
                      ark::NetIbMr::Info *rmi2, int num_iter, int bytes,
                      bool is_recv, bool bidir)
{
    int ret;
    int max_num_wc = bidir ? 2 : 1;
    double start = ark::cpu_timer();
    for (int iter = 0; iter < num_iter; ++iter) {
        if (is_recv) {
            ret = qp->post_recv(0);
            UNITTEST_EQ(ret, 0);
            if (bidir) {
                ret = qp->stage_send(mr2, rmi2, bytes, 0, 0);
                UNITTEST_EQ(ret, 1);
                ret = qp->post_send();
                UNITTEST_EQ(ret, 0);
            }
        } else {
            if (bidir) {
                ret = qp->post_recv(0);
                UNITTEST_EQ(ret, 0);
            }
            ret = qp->stage_send(mr1, rmi1, bytes, 0, 0);
            UNITTEST_EQ(ret, 1);
            ret = qp->post_send();
            UNITTEST_EQ(ret, 0);
        }

        int num_wc = 0;
        do {
            ret = mgr->poll_cq();
            for (int i = 0; i < ret; ++i) {
                UNITTEST_EQ(mgr->get_wc_status(i), 0);
            }
            num_wc += ret;
        } while (num_wc < max_num_wc);
        UNITTEST_EQ(num_wc, max_num_wc);
    }
    return ark::cpu_timer() - start;
}

int test_net_ib_cpu_internal(std::size_t bytes, bool is_recv)
{
    int ret;
    int rank = is_recv ? 0 : 1;
    int dev = (ark::get_net_ib_device_num() >= 2) ? rank : 0;
    bytes = (bytes + 3) & -4;

    ark::NetIbMgr mgr{dev};

    void *buf = ark::page_memalign(bytes);
    UNITTEST_NE(buf, (void *)nullptr);
    ark::NetIbMr *mr = mgr.reg_mr(buf, bytes);
    UNITTEST_NE(mr, (ark::NetIbMr *)nullptr);

    ark::NetIbQp *qp = mgr.create_qp();
    UNITTEST_NE(qp, (ark::NetIbQp *)nullptr);

    const ark::NetIbMr::Info &lmi = mr->get_info();
    const ark::NetIbQp::Info &lqi = qp->get_info();

    ark::IpcAllGather iag_mr{"rdma_test_mr", rank, 2, &lmi, sizeof(lmi)};
    ark::IpcAllGather iag_qp{"rdma_test_qp", rank, 2, &lqi, sizeof(lqi)};
    iag_mr.sync();
    iag_qp.sync();

    ark::NetIbMr::Info rmi;
    ark::NetIbQp::Info rqi;
    std::memcpy(&rmi, iag_mr.get_data(rank == 0 ? 1 : 0), sizeof(rmi));
    std::memcpy(&rqi, iag_qp.get_data(rank == 0 ? 1 : 0), sizeof(rqi));

    ret = qp->rtr(&rqi);
    UNITTEST_EQ(ret, 0);

    ret = qp->rts();
    UNITTEST_EQ(ret, 0);

    // Leverage just for sync.
    iag_qp.sync();

    for (int iter = 0; iter < 1000; ++iter) {
        if (is_recv) {
            ret = qp->post_recv(0);
            UNITTEST_EQ(ret, 0);

            do {
                ret = mgr.poll_cq();
            } while (ret == 0);

            UNITTEST_EQ(ret, 1);
            if (iter == 9999) {
                UNITTEST_EQ(*(int *)buf, 9999);
                UNITTEST_EQ(*((int *)buf + 1), 0);
            }
        } else {
            // Should be sent
            *(int *)buf = iter;
            // Should not be sent
            *((int *)buf + 1) = iter;

            ret = qp->stage_send(mr, &rmi, sizeof(int), 0, 0);
            UNITTEST_EQ(ret, 1);

            ret = qp->post_send();
            UNITTEST_EQ(ret, 0);

            do {
                ret = mgr.poll_cq();
            } while (ret == 0);
            UNITTEST_EQ(ret, 1);
            UNITTEST_EQ(mgr.get_wc_status(0), 0);
        }
    }
    return 0;
}

int test_net_ib_cpu_bw_internal(std::size_t bytes, bool is_recv)
{
    int ret;
    int num_iter = 10;
    int rank = is_recv ? 0 : 1;
    int dev = rank + 1;
    bytes = (bytes + 3) & -4;

    ark::NetIbMgr mgr{(ark::get_net_ib_device_num() >= 2) ? dev : 0};

    numa_bind(mgr.get_numa_node());

    void *buf = ark::page_memalign(bytes);
    UNITTEST_NE(buf, (void *)nullptr);
    void *buf2 = ark::page_memalign(bytes);
    UNITTEST_NE(buf2, (void *)nullptr);
    ark::NetIbMr *mr = mgr.reg_mr(buf, bytes);
    UNITTEST_NE(mr, (ark::NetIbMr *)nullptr);
    ark::NetIbMr *mr2 = mgr.reg_mr(buf2, bytes);
    UNITTEST_NE(mr2, (ark::NetIbMr *)nullptr);

    ark::NetIbQp *qp = mgr.create_qp();
    UNITTEST_NE(qp, (ark::NetIbQp *)nullptr);

    const ark::NetIbMr::Info &lmi = mr->get_info();
    const ark::NetIbMr::Info &lmi2 = mr->get_info();
    const ark::NetIbQp::Info &lqi = qp->get_info();

    ark::IpcAllGather iag_mr{"rdma_test_mr", rank, 2, &lmi, sizeof(lmi)};
    ark::IpcAllGather iag_mr2{"rdma_test_mr2", rank, 2, &lmi2, sizeof(lmi2)};
    ark::IpcAllGather iag_qp{"rdma_test_qp", rank, 2, &lqi, sizeof(lqi)};
    iag_mr.sync();
    iag_mr2.sync();
    iag_qp.sync();

    ark::NetIbMr::Info rmi;
    ark::NetIbMr::Info rmi2;
    ark::NetIbQp::Info rqi;
    std::memcpy(&rmi, iag_mr.get_data(rank == 0 ? 1 : 0), sizeof(rmi));
    std::memcpy(&rmi2, iag_mr2.get_data(rank == 0 ? 1 : 0), sizeof(rmi2));
    std::memcpy(&rqi, iag_qp.get_data(rank == 0 ? 1 : 0), sizeof(rqi));

    ret = qp->rtr(&rqi);
    UNITTEST_EQ(ret, 0);

    ret = qp->rts();
    UNITTEST_EQ(ret, 0);

    // Leverage just for sync.
    iag_qp.sync();

    double elapsed;
    elapsed = sr_loop(&mgr, qp, mr, mr2, &rmi, &rmi2, num_iter, bytes, is_recv,
                      false);
    LOG(ark::INFO, "Uni-dir: ", bytes * num_iter / elapsed / 1e9, " GB/s");
    elapsed =
        sr_loop(&mgr, qp, mr, mr2, &rmi, &rmi2, num_iter, bytes, is_recv, true);
    LOG(ark::INFO, "Bi-dir: ", bytes * num_iter / elapsed / 1e9, " GB/s");
    return 0;
}

int test_net_ib_gpu_internal(std::size_t bytes, bool is_recv)
{
    int ret;
    int rank = is_recv ? 0 : 1;
    int dev = (ark::get_net_ib_device_num() >= 2) ? rank : 0;
    bytes = (bytes + 3) & -4;

    // Create a CUDA context
    CULOG(cuInit(0));
    CUdevice cudev;
    CUcontext cuctx;
    CULOG(cuDeviceGet(&cudev, rank));
    CULOG(cuCtxCreate(&cuctx, 0, cudev));
    CULOG(cuCtxSetCurrent(cuctx));

    ark::NetIbMgr mgr{dev};

    ark::GpuMem mem{"gpu_mem_" + std::to_string(rank), bytes, true};
    void *buf = mem.href();
    ark::NetIbMr *mr = mgr.reg_mr((void *)mem.ref(), bytes);
    UNITTEST_NE(mr, (ark::NetIbMr *)nullptr);

    ark::NetIbQp *qp = mgr.create_qp();
    UNITTEST_NE(qp, (ark::NetIbQp *)nullptr);

    const ark::NetIbMr::Info &lmi = mr->get_info();
    const ark::NetIbQp::Info &lqi = qp->get_info();

    ark::IpcAllGather iag_mr{"rdma_test_mr", rank, 2, &lmi, sizeof(lmi)};
    ark::IpcAllGather iag_qp{"rdma_test_qp", rank, 2, &lqi, sizeof(lqi)};
    iag_mr.sync();
    iag_qp.sync();

    ark::NetIbMr::Info rmi;
    ark::NetIbQp::Info rqi;
    std::memcpy(&rmi, iag_mr.get_data(rank == 0 ? 1 : 0), sizeof(rmi));
    std::memcpy(&rqi, iag_qp.get_data(rank == 0 ? 1 : 0), sizeof(rqi));

    ret = qp->rtr(&rqi);
    UNITTEST_EQ(ret, 0);

    ret = qp->rts();
    UNITTEST_EQ(ret, 0);

    // Leverage just for sync.
    iag_qp.sync();

    for (int iter = 0; iter < 1000; ++iter) {
        if (is_recv) {
            ret = qp->post_recv(0);
            UNITTEST_EQ(ret, 0);

            do {
                ret = mgr.poll_cq();
            } while (ret == 0);

            UNITTEST_EQ(ret, 1);
            if (iter == 9999) {
                UNITTEST_EQ(*(volatile int *)buf, 9999);
                UNITTEST_EQ(*((volatile int *)buf + 1), 0);
            }
        } else {
            // Should be sent
            *(volatile int *)buf = iter;
            // Should not be sent
            *((volatile int *)buf + 1) = iter;

            ret = qp->stage_send(mr, &rmi, sizeof(int), 0, 0);
            UNITTEST_EQ(ret, 1);

            ret = qp->post_send();
            UNITTEST_EQ(ret, 0);

            do {
                ret = mgr.poll_cq();
            } while (ret == 0);
            UNITTEST_EQ(ret, 1);
            UNITTEST_EQ(mgr.get_wc_status(0), 0);
        }
    }
    return 0;
}

int test_net_ib_gpu_bw_internal(std::size_t bytes, bool is_recv)
{
    int ret;
    int num_iter = 10;
    int rank = is_recv ? 0 : 1;
    int dev = rank + 1;
    bytes = (bytes + 3) & -4;

    // Create a CUDA context
    CULOG(cuInit(0));
    CUdevice cudev;
    CUcontext cuctx;
    CULOG(cuDeviceGet(&cudev, dev));
    CULOG(cuCtxCreate(&cuctx, 0, cudev));
    CULOG(cuCtxSetCurrent(cuctx));

    ark::NetIbMgr mgr{(ark::get_net_ib_device_num() >= 2) ? dev : 0};

    numa_bind(mgr.get_numa_node());

    ark::GpuMem mem{"gpu_mem_" + std::to_string(rank), bytes, true};
    ark::GpuMem mem2{"gpu_mem2_" + std::to_string(rank), bytes, true};
    ark::NetIbMr *mr = mgr.reg_mr((void *)mem.ref(), bytes);
    ark::NetIbMr *mr2 = mgr.reg_mr((void *)mem2.ref(), bytes);
    UNITTEST_NE(mr, (ark::NetIbMr *)nullptr);
    UNITTEST_NE(mr2, (ark::NetIbMr *)nullptr);

    ark::NetIbQp *qp = mgr.create_qp();
    UNITTEST_NE(qp, (ark::NetIbQp *)nullptr);

    const ark::NetIbMr::Info &lmi = mr->get_info();
    const ark::NetIbMr::Info &lmi2 = mr2->get_info();
    const ark::NetIbQp::Info &lqi = qp->get_info();

    ark::IpcAllGather iag_mr{"rdma_test_mr", rank, 2, &lmi, sizeof(lmi)};
    ark::IpcAllGather iag_mr2{"rdma_test_mr2", rank, 2, &lmi2, sizeof(lmi2)};
    ark::IpcAllGather iag_qp{"rdma_test_qp", rank, 2, &lqi, sizeof(lqi)};
    iag_mr.sync();
    iag_mr2.sync();
    iag_qp.sync();

    ark::NetIbMr::Info rmi;
    ark::NetIbMr::Info rmi2;
    ark::NetIbQp::Info rqi;
    std::memcpy(&rmi, iag_mr.get_data(rank == 0 ? 1 : 0), sizeof(rmi));
    std::memcpy(&rmi2, iag_mr2.get_data(rank == 0 ? 1 : 0), sizeof(rmi2));
    std::memcpy(&rqi, iag_qp.get_data(rank == 0 ? 1 : 0), sizeof(rqi));

    ret = qp->rtr(&rqi);
    UNITTEST_EQ(ret, 0);

    ret = qp->rts();
    UNITTEST_EQ(ret, 0);

    // Leverage just for sync.
    iag_qp.sync();

    double elapsed;
    elapsed = sr_loop(&mgr, qp, mr, mr2, &rmi, &rmi2, num_iter, bytes, is_recv,
                      false);
    LOG(ark::INFO, "Uni-dir: ", bytes * num_iter / elapsed / 1e9, " GB/s");
    elapsed =
        sr_loop(&mgr, qp, mr, mr2, &rmi, &rmi2, num_iter, bytes, is_recv, true);
    LOG(ark::INFO, "Bi-dir: ", bytes * num_iter / elapsed / 1e9, " GB/s");
    return 0;
}

//
ark::unittest::State test_net_ib_cpu()
{
    for (int i = 0; i < 100; ++i) {
        int pid0 = ark::proc_spawn([] {
            ark::unittest::Timeout timeout{30};
            std::size_t bytes = 1024 * 1024;
            return test_net_ib_cpu_internal(bytes, true);
        });
        UNITTEST_NE(pid0, -1);
        int pid1 = ark::proc_spawn([] {
            ark::unittest::Timeout timeout{30};
            std::size_t bytes = 1024 * 1024;
            return test_net_ib_cpu_internal(bytes, false);
        });
        UNITTEST_NE(pid1, -1);

        int ret = ark::proc_wait({pid0, pid1});
        UNITTEST_EQ(ret, 0);
    }
    return ark::unittest::SUCCESS;
}

//
ark::unittest::State test_net_ib_cpu_bw()
{
    int pid0 = ark::proc_spawn([] {
        ark::unittest::Timeout timeout{30};
        std::size_t bytes = BW_TEST_BYTES;
        return test_net_ib_cpu_bw_internal(bytes, true);
    });
    UNITTEST_NE(pid0, -1);
    int pid1 = ark::proc_spawn([] {
        ark::unittest::Timeout timeout{30};
        std::size_t bytes = BW_TEST_BYTES;
        return test_net_ib_cpu_bw_internal(bytes, false);
    });
    UNITTEST_NE(pid1, -1);

    int ret = ark::proc_wait({pid0, pid1});
    UNITTEST_EQ(ret, 0);

    return ark::unittest::SUCCESS;
}

//
ark::unittest::State test_net_ib_gpu()
{
    for (int i = 0; i < 10; ++i) {
        int pid0 = ark::proc_spawn([] {
            ark::unittest::Timeout timeout{30};
            std::size_t bytes = 1024 * 1024;
            return test_net_ib_gpu_internal(bytes, true);
        });
        UNITTEST_NE(pid0, -1);
        int pid1 = ark::proc_spawn([] {
            ark::unittest::Timeout timeout{30};
            std::size_t bytes = 1024 * 1024;
            return test_net_ib_gpu_internal(bytes, false);
        });
        UNITTEST_NE(pid1, -1);

        int ret = ark::proc_wait({pid0, pid1});
        UNITTEST_EQ(ret, 0);
    }
    return ark::unittest::SUCCESS;
}

//
ark::unittest::State test_net_ib_gpu_bw()
{
    int pid0 = ark::proc_spawn([] {
        ark::unittest::Timeout timeout{30};
        std::size_t bytes = BW_TEST_BYTES;
        return test_net_ib_gpu_bw_internal(bytes, true);
    });
    UNITTEST_NE(pid0, -1);
    int pid1 = ark::proc_spawn([] {
        ark::unittest::Timeout timeout{30};
        std::size_t bytes = BW_TEST_BYTES;
        return test_net_ib_gpu_bw_internal(bytes, false);
    });
    UNITTEST_NE(pid1, -1);

    int ret = ark::proc_wait({pid0, pid1});
    UNITTEST_EQ(ret, 0);

    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_net_ib_cpu);
    UNITTEST(test_net_ib_gpu);
    UNITTEST(test_net_ib_cpu_bw);
    UNITTEST(test_net_ib_gpu_bw);
    return 0;
}
