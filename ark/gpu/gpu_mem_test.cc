// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_mem.h"

#include "env.h"
#include "gpu/gpu_logging.h"
#include "include/ark.h"
#include "include/ark_utils.h"
#include "ipc/ipc_hosts.h"
#include "ipc/ipc_socket.h"
#include "unittest/unittest_utils.h"

ark::unittest::State test_gpu_mem() {
    int pid = ark::utils::proc_spawn([] {
        ark::unittest::Timeout timeout{5};

        // Create a context of GPU 0.
        GLOG(ark::gpuInit(0));
        ark::gpuDevice dev0;
        ark::gpuCtx ctx0;
        GLOG(ark::gpuDeviceGet(&dev0, 0));
        GLOG(ark::gpuCtxCreate(&ctx0, 0, dev0));
        GLOG(ark::gpuCtxSetCurrent(ctx0));

        // Local memory in GPU 0.
        ark::GpuMem mem0{4096};

        // Create a context of GPU 1.
        ark::gpuDevice dev1;
        ark::gpuCtx ctx1;
        GLOG(ark::gpuDeviceGet(&dev1, 1));
        GLOG(ark::gpuCtxCreate(&ctx1, 0, dev1));
        GLOG(ark::gpuCtxSetCurrent(ctx1));

        // Remote memory in GPU 1.
        ark::GpuMem mem1{4096};

        // Set data on GPU 0.
        GLOG(ark::gpuCtxSetCurrent(ctx0));
        GLOG(ark::gpuMemsetD32(mem0.ref(), 7, 1024));
        // Check data on GPU 0.
        volatile int *href0 = (volatile int *)mem0.href();
        for (int i = 0; i < 1024; ++i) {
            UNITTEST_EQ(href0[i], 7);
        }

        // Set data on GPU 1.
        GLOG(ark::gpuCtxSetCurrent(ctx1));
        GLOG(ark::gpuMemsetD32(mem1.ref(), 9, 1024));
        // Check data on GPU 1.
        volatile int *href1 = (volatile int *)mem1.href();
        for (int i = 0; i < 1024; ++i) {
            UNITTEST_EQ(href1[i], 9);
        }

        return 0;
    });
    UNITTEST_NE(pid, -1);

    int ret = ark::utils::proc_wait(pid);
    UNITTEST_EQ(ret, 0);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_gpu_mem_ipc() {
    int pid0 = ark::utils::proc_spawn([] {
        ark::unittest::Timeout timeout{5};

        int port_base = ark::get_env().ipc_listen_port_base;
        ark::IpcSocket is{ark::get_host(0), port_base};

        // Create a context of GPU 0.
        GLOG(ark::gpuInit(0));
        ark::gpuDevice dev;
        ark::gpuCtx ctx;
        GLOG(ark::gpuDeviceGet(&dev, 0));
        GLOG(ark::gpuCtxCreate(&ctx, 0, dev));
        GLOG(ark::gpuCtxSetCurrent(ctx));

        // Local memory in GPU 0.
        ark::GpuMem mem0{4096};

        // Write information of the local memory.
        const ark::GpuMem::Info &mem0_info = mem0.get_info();
        is.add_item("gpu_mem_0", &mem0_info, sizeof(mem0_info));

        // Get information of the remote memory.
        ark::GpuMem::Info mem1_info;
        auto s = is.query_item(ark::get_host(0), port_base + 1, "gpu_mem_1",
                               &mem1_info, sizeof(mem1_info), true);
        UNITTEST_TRUE(s == ark::IpcSocket::SUCCESS);

        // Remote memory in GPU 1.
        ark::GpuMem mem1{mem1_info};

        // Wait until another process writes data on the local GPU 0.
        volatile int *href = (volatile int *)mem0.href();
        for (int i = 0; i < 1024; ++i) {
            while (href[i] != 7) {
            }
        }

        // Set data on the remote GPU 1.
        GLOG(ark::gpuMemsetD32(mem1.ref(), 9, 1024));

        return 0;
    });
    UNITTEST_NE(pid0, -1);

    int pid1 = ark::utils::proc_spawn([] {
        ark::unittest::Timeout timeout{5};

        int port_base = ark::get_env().ipc_listen_port_base;
        ark::IpcSocket is{ark::get_host(0), port_base + 1};

        // Create a context of GPU 1.
        GLOG(ark::gpuInit(0));
        ark::gpuDevice dev;
        ark::gpuCtx ctx;
        GLOG(ark::gpuDeviceGet(&dev, 1));
        GLOG(ark::gpuCtxCreate(&ctx, 0, dev));
        GLOG(ark::gpuCtxSetCurrent(ctx));

        // Local memory in GPU 1.
        ark::GpuMem mem1{4096};

        // Write information of the local memory.
        const ark::GpuMem::Info &mem1_info = mem1.get_info();
        is.add_item("gpu_mem_1", &mem1_info, sizeof(mem1_info));

        // Get information of the remote memory.
        ark::GpuMem::Info mem0_info;
        auto s = is.query_item(ark::get_host(0), port_base, "gpu_mem_0",
                               &mem0_info, sizeof(mem0_info), true);
        UNITTEST_TRUE(s == ark::IpcSocket::SUCCESS);

        // Remote memory in GPU 0.
        ark::GpuMem mem0{mem0_info};

        // Set data on the remote GPU 0.
        GLOG(ark::gpuMemsetD32(mem0.ref(), 7, 1024));

        // Wait until another process writes data on the local GPU 1.
        volatile int *href = (volatile int *)mem1.href();
        for (int i = 0; i < 1024; ++i) {
            while (href[i] != 9) {
            }
        }

        return 0;
    });
    UNITTEST_NE(pid1, -1);

    int ret = ark::utils::proc_wait({pid0, pid1});
    UNITTEST_EQ(ret, 0);
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_gpu_mem);
    UNITTEST(test_gpu_mem_ipc);
    return 0;
}
