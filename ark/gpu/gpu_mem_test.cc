#include "ark/gpu/gpu_logging.h"
#include "ark/gpu/gpu_mem.h"
#include "ark/init.h"
#include "ark/process.h"
#include "ark/unittest/unittest_utils.h"

using namespace ark;
using namespace std;

unittest::State test_gpu_mem_no_ipc()
{
    int pid = proc_spawn([] {
        unittest::Timeout timeout{3};

        // Create a CUDA context of GPU 0.
        CULOG(cuInit(0));
        CUdevice dev0;
        CUcontext ctx0;
        CULOG(cuDeviceGet(&dev0, 0));
        CULOG(cuCtxCreate(&ctx0, 0, dev0));
        CULOG(cuCtxSetCurrent(ctx0));

        // Local memory in GPU 0.
        GpuMem mem0{"gpu_mem_0", 4096, true};

        // Create a CUDA context of GPU 1.
        CUdevice dev1;
        CUcontext ctx1;
        CULOG(cuDeviceGet(&dev1, 1));
        CULOG(cuCtxCreate(&ctx1, 0, dev1));
        CULOG(cuCtxSetCurrent(ctx1));

        // Remote memory in GPU 1.
        GpuMem mem1{"gpu_mem_1", 4096, true};

        // Set data on GPU 0.
        CULOG(cuCtxSetCurrent(ctx0));
        CULOG(cuMemsetD32(mem0.ref(), 7, 1024));
        // Check data on GPU 0.
        volatile int *href0 = (volatile int *)mem0.href();
        for (int i = 0; i < 1024; ++i) {
            UNITTEST_EQ(href0[i], 7);
        }

        // Set data on GPU 1.
        CULOG(cuCtxSetCurrent(ctx1));
        CULOG(cuMemsetD32(mem1.ref(), 9, 1024));
        // Check data on GPU 1.
        volatile int *href1 = (volatile int *)mem1.href();
        for (int i = 0; i < 1024; ++i) {
            UNITTEST_EQ(href1[i], 9);
        }

        return 0;
    });
    UNITTEST_NE(pid, -1);

    int ret = proc_wait(pid);
    UNITTEST_EQ(ret, 0);
    return unittest::SUCCESS;
}

unittest::State test_gpu_mem_ipc()
{
    int pid0 = proc_spawn([] {
        unittest::Timeout timeout{3};

        // Create a CUDA context of GPU 0.
        CULOG(cuInit(0));
        CUdevice dev;
        CUcontext ctx;
        CULOG(cuDeviceGet(&dev, 0));
        CULOG(cuCtxCreate(&ctx, 0, dev));
        CULOG(cuCtxSetCurrent(ctx));

        // Local memory in GPU 0.
        GpuMem mem0{"gpu_mem_0", 4096, true};
        // Remote memory in GPU 1.
        GpuMem mem1{"gpu_mem_1", 4096, false};

        // Wait until another process writes data on the local GPU 0.
        volatile int *href = (volatile int *)mem0.href();
        for (int i = 0; i < 1024; ++i) {
            while (href[i] != 7) {
            }
        }

        // Set data on the remote GPU 1.
        CULOG(cuMemsetD32(mem1.ref(), 9, 1024));

        return 0;
    });
    UNITTEST_NE(pid0, -1);

    int pid1 = proc_spawn([] {
        unittest::Timeout timeout{3};

        // Create a CUDA context of GPU 1.
        CULOG(cuInit(0));
        CUdevice dev;
        CUcontext ctx;
        CULOG(cuDeviceGet(&dev, 1));
        CULOG(cuCtxCreate(&ctx, 0, dev));
        CULOG(cuCtxSetCurrent(ctx));

        // Remote memory in GPU 0.
        GpuMem mem0{"gpu_mem_0", 4096, false};
        // Local memory in GPU 1.
        GpuMem mem1{"gpu_mem_1", 4096, true};

        // Set data on the remote GPU 0.
        CULOG(cuMemsetD32(mem0.ref(), 7, 1024));

        // Wait until another process writes data on the local GPU 1.
        volatile int *href = (volatile int *)mem1.href();
        for (int i = 0; i < 1024; ++i) {
            while (href[i] != 9) {
            }
        }

        return 0;
    });
    UNITTEST_NE(pid1, -1);

    int ret = proc_wait({pid0, pid1});
    UNITTEST_EQ(ret, 0);
    return unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_gpu_mem_no_ipc);
    UNITTEST(test_gpu_mem_ipc);
    return 0;
}
