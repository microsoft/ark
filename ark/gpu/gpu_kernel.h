// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_KERNEL_H_
#define ARK_GPU_KERNEL_H_

#include <string>
#include <thread>
#include <vector>

#include "gpu/gpu.h"
#include "gpu/gpu_mgr.h"

#define ARK_BUF_NAME "ARK_BUF"
#define ARK_SC_NAME "ARK_SC"
#define ARK_RC_NAME "ARK_RC"
#define ARK_LSS_NAME "ARK_LOOP_SYNC_STATE"
#define ARK_REQ_NAME "ARK_REQUEST"
#define ARK_CLKS_NAME "ARK_CLKS"

namespace ark {

class GpuKernel {
   public:
    GpuKernel(const std::string &name, const std::vector<std::string> &codes,
              const std::array<unsigned int, 3> &grid_dims,
              const std::array<unsigned int, 3> &block_dims,
              unsigned int smem_bytes, std::initializer_list<GpuBuf *> buf_args,
              std::initializer_list<size_t> buf_offs,
              std::initializer_list<std::pair<void *, size_t>> args,
              const std::string &gpubin);
    ~GpuKernel();

    void compile(const GpuInfo &gpu_info);
    GpuState launch(GpuStream stream);

    const std::string &get_name() { return name; }
    const std::vector<std::string> &get_codes() { return codes; }
    const std::string &get_gpubin() { return gpubin; }
    int get_function_attribute(gpuFunctionAttribute attr) const;
    bool is_compiled() const { return this->kernel != nullptr; }

   protected:
    const std::string name;
    std::vector<std::string> codes;
    std::array<unsigned int, 3> const gd;
    std::array<unsigned int, 3> const bd;
    unsigned int smem_bytes;
    // Input data buffers of this kernel.
    std::vector<GpuBuf *> buf_args;
    //
    std::vector<size_t> buf_offs;
    // Pointers to an entry of each buffer in `buf_args`.
    std::vector<GpuPtr> ptr_args;
    int num_params;
    void **params;
    std::string gpubin;

    gpuModule module;
    gpuFunction kernel = nullptr;
};

class GpuLoopKernel : public GpuKernel {
   public:
    GpuLoopKernel(const std::string &name,
                  const std::vector<std::string> &codes_body,
                  unsigned int num_sm, unsigned int num_warp,
                  unsigned int smem_bytes, const std::string &gpubin,
                  GpuMgrCtx *ctx);

    void compile(const GpuInfo &gpu_info);
    GpuState launch(gpuStream stream, bool disable_timing = true);
    void load();
    void run(int iter = 1);
    bool poll();
    void wait();
    void stop();

    float get_elapsed_msec() const { return elapsed_msec; }
    const long long int *get_clocks() const {
        return (const long long int *)clocks->href();
    }

   private:
    GpuMgrCtx *ctx;
    GpuEvent timer_begin;
    GpuEvent timer_end;

    std::unique_ptr<GpuMem> flag;
    std::unique_ptr<GpuMem> clocks;

    volatile int *flag_href;

    GpuStream stream = nullptr;
    bool is_recording = false;
    float elapsed_msec = -1;
};

}  // namespace ark

#endif  // ARK_GPU_KERNEL_H_
