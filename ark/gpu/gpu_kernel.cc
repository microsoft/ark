// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#include "cpu_timer.h"
#include "env.h"
#include "gpu/gpu_compile.h"
#include "gpu/gpu_kernel.h"
#include "gpu/gpu_logging.h"

using namespace std;

// This should be a power of 2.
#define CLKS_CNT 1048576
//
#define MAX_LOOP_COUNTER 10000000

namespace ark {

//
GpuKernel::GpuKernel(const string &name_, const vector<string> &codes_,
                     const array<unsigned int, 3> &grid_dims,
                     const array<unsigned int, 3> &block_dims,
                     unsigned int smem_bytes_,
                     initializer_list<GpuBuf *> buf_args_,
                     initializer_list<size_t> buf_offs_,
                     initializer_list<pair<void *, size_t>> args,
                     const string &cubin_)
    : name{name_}, codes{codes_}, gd{grid_dims}, bd{block_dims},
      smem_bytes{smem_bytes_}, buf_args{buf_args_}, buf_offs{buf_offs_},
      ptr_args(buf_args.size(), 0), num_params{(int)(buf_args.size() +
                                                     args.size())},
      params{new void *[num_params]}, cubin{cubin_}
{
    if (this->name.size() == 0) {
        LOG(ERROR, "Invalid kernel name: ", this->name);
    }
    assert(this->params != nullptr);
    // Default offset zero.
    if (this->buf_offs.size() == 0) {
        this->buf_offs.resize(buf_args.size(), 0);
    }
    int idx = buf_args.size();
    for (auto &pa : args) {
        void *p = malloc(pa.second);
        assert(p != nullptr);
        if (pa.first != nullptr) {
            ::memcpy(p, pa.first, pa.second);
        }
        this->params[idx++] = p;
    }
}

//
GpuKernel::~GpuKernel()
{
    if (this->params != nullptr) {
        for (int i = buf_args.size(); i < this->num_params; ++i) {
            if (this->params[i] != nullptr) {
                free(this->params[i]);
            }
        }
        delete this->params;
        this->params = nullptr;
    }
}

//
void GpuKernel::compile(const GpuInfo &gpu_info)
{
    if (this->is_compiled()) {
        return;
    }
    unsigned int max_reg_cnt = gpu_info.max_registers_per_block /
                               (this->bd[0] * this->bd[1] * this->bd[2]);
    if (max_reg_cnt >= gpu_info.max_registers_per_thread) {
        max_reg_cnt = gpu_info.max_registers_per_thread - 1;
    }
    //
    if (this->cubin.empty()) {
        this->cubin = gpu_compile(this->codes, gpu_info.arch, max_reg_cnt);
    }

    //
    size_t num_opts = 5;
    size_t buflen = 8192;
    std::unique_ptr<CUjit_option[]> opts(new CUjit_option[num_opts]);
    std::unique_ptr<void *[]> optvals(new void *[num_opts]);
    std::string infobuf;
    std::string errbuf;

    infobuf.resize(buflen, ' ');
    errbuf.resize(buflen, ' ');

    int enable = 1;

    opts[0] = CU_JIT_INFO_LOG_BUFFER;
    optvals[0] = (void *)infobuf.data();

    opts[1] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    optvals[1] = (void *)(long)buflen;

    opts[2] = CU_JIT_ERROR_LOG_BUFFER;
    optvals[2] = (void *)errbuf.data();

    opts[3] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    optvals[3] = (void *)(long)buflen;

    opts[4] = CU_JIT_GENERATE_DEBUG_INFO;
    optvals[4] = (void *)(long)enable;

    if (cuModuleLoadDataEx(&this->module, this->cubin.c_str(), num_opts,
                           opts.get(), optvals.get()) != CUDA_SUCCESS) {
        LOG(DEBUG, infobuf);
        LOG(ERROR, "cuModuleLoadDataEx() failed: ", errbuf);
    }
    CULOG(cuModuleGetFunction(&this->kernel, this->module, this->name.c_str()));
    //
    int static_smem_size_bytes;
    CULOG(cuFuncGetAttribute(&static_smem_size_bytes,
                             CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                             this->kernel));
    int dynamic_smem_size_bytes = smem_bytes - static_smem_size_bytes;
    CULOG(cuFuncSetAttribute(this->kernel,
                             CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                             dynamic_smem_size_bytes));
}

//
GpuState GpuKernel::launch(GpuStream stream)
{
    if (!this->is_compiled()) {
        LOG(ERROR, "Kernel is not compiled yet.");
    }
    auto it_ptr = this->ptr_args.begin();
    auto it_off = this->buf_offs.begin();
    for (GpuBuf *buf : this->buf_args) {
        *it_ptr++ = buf->ref(*it_off++);
    }
    void **params = this->params;
    for (size_t i = 0; i < this->ptr_args.size(); ++i) {
        *params++ = &this->ptr_args[i];
    }
    return cuLaunchKernel(this->kernel, this->gd[0], this->gd[1], this->gd[2],
                          this->bd[0], this->bd[1], this->bd[2],
                          this->smem_bytes, stream, this->params, 0);
}

int GpuKernel::get_function_attribute(CUfunction_attribute attr) const
{
    if (this->kernel == nullptr) {
        LOG(ERROR, "Kernel is not compiled yet.");
    }
    int ret;
    CULOG(cuFuncGetAttribute(&ret, attr, this->kernel));
    return ret;
}

////////////////////////////////////////////////////////////////////////////////

GpuLoopKernel::GpuLoopKernel(const string &name_,
                             const vector<string> &codes_body,
                             unsigned int num_sm, unsigned int num_warp,
                             unsigned int smem_bytes, const string &cubin_,
                             GpuMgrCtx *ctx_)
    : GpuKernel{name_,
                {},
                {num_sm, 1, 1},
                {num_warp * 32, 1, 1},
                (smem_bytes < 4) ? 4 : smem_bytes,
                {},
                {},
                {{0, sizeof(GpuPtr)}, {0, sizeof(GpuPtr)}},
                cubin_},
      ctx{ctx_}, timer_begin{ctx_->create_event(false)}, timer_end{
                                                             ctx_->create_event(
                                                                 false)}
{
    ctx_->set_current();
    this->flag = make_unique<GpuMem>(sizeof(int));
    this->clocks = make_unique<GpuMem>(CLKS_CNT * sizeof(long long int));
    this->flag_href = (volatile int *)this->flag->href(0);

    *(GpuPtr *)this->params[0] = this->flag->ref(0);
    std::memset(this->clocks->href(), 0, this->clocks->get_bytes());

    if (codes_body.size() > 0) {
        const string *ark_loop_body_code = nullptr;
        for (auto &code : codes_body) {
            if (code.find("ark_loop_body") == string::npos) {
                //
                // this->codes.emplace_back(body_prefix.str() + code);
            } else {
                ark_loop_body_code = &code;
            }
        }
        assert(ark_loop_body_code != nullptr);

        stringstream ss;
        // clang-format off
        ss <<
        "// THIS KERNEL IS MACHINE-GENERATED BY ARK.\n"
        "#define ARK_THREADS_PER_BLOCK " << num_warp * 32 << "\n"
        "#define ARK_KERNELS_SYNC_CLKS_CNT " << CLKS_CNT << "\n"
        "__device__ volatile unsigned long long int *" ARK_REQ_NAME ";\n"
        "__device__ volatile unsigned           int *_ARK_SC;\n"
        "__device__ volatile unsigned           int *_ARK_RC;\n"
        "__device__ long long int *_ARK_CLKS;\n"
        "__device__ int _ITER = 0;\n"
        "#include \"ark_kernels.h\"\n"
        "__device__ ark::sync::State " ARK_LSS_NAME ";\n"
        "__device__ char *" ARK_BUF_NAME ";\n"
        << *ark_loop_body_code <<
        "extern \"C\" __global__ __launch_bounds__(" << num_warp * 32 << ", 1)\n"
        "void " << name_ << "(volatile int *_it)\n"
        "{\n"
        "  for (;;) {\n"
        "    if (threadIdx.x == 0 && blockIdx.x == 0) {\n"
        "      int iter;\n"
        "      while ((iter = *_it) == 0) {}\n"
        "      _ITER = iter;\n"
        "    }\n"
        "    ark::sync_gpu<" << num_sm << ">(" ARK_LSS_NAME ");\n"
        "    if (_ITER < 0) {\n"
        "      return;\n"
        "    }\n"
        "    for (int _i = 0; _i < _ITER; ++_i) {\n"
        "      ark_loop_body(_i);\n"
        "      ark::sync_gpu<" << num_sm << ">(" ARK_LSS_NAME ");\n"
        "    }\n"
        "    if (threadIdx.x == 0 && blockIdx.x == 0) {\n"
        "      *_it = 0;\n"
        "    }\n"
        "  }\n"
        "}\n";
        // clang-format on
        this->codes.emplace_back(ss.str());
    }
}

void GpuLoopKernel::compile(const GpuInfo &gpu_info)
{
    this->ctx->set_current();
    if (this->is_compiled()) {
        return;
    }
    // Compile the code.
    GpuKernel::compile(gpu_info);
}

void GpuLoopKernel::load()
{
    this->ctx->set_current();
    //
    if (!this->is_compiled()) {
        LOG(ERROR, "Need to compile first before initialization.");
    }
    if (this->stream != nullptr) {
        // Wait until previous works finish.
        this->wait();
    } else {
        // Initialize global variables in the loop kernel.
        GpuPtr buf_ptr_val = this->ctx->get_data_ref();
        GpuPtr lss_ptr_addr;
        GpuPtr buf_ptr_addr;
        CULOG(cuModuleGetGlobal(&lss_ptr_addr, 0, this->module, ARK_LSS_NAME));
        CULOG(cuModuleGetGlobal(&buf_ptr_addr, 0, this->module, ARK_BUF_NAME));
        CULOG(cuMemsetD32(lss_ptr_addr, 0, 4));
        CULOG(cuMemcpyHtoD(buf_ptr_addr, &buf_ptr_val, sizeof(GpuPtr)));
        //
        GpuPtr sc_ptr_val = this->ctx->get_sc_ref(0);
        GpuPtr rc_ptr_val = this->ctx->get_rc_ref(0);
        GpuPtr sc_ptr_addr;
        GpuPtr rc_ptr_addr;
        CULOG(cuModuleGetGlobal(&sc_ptr_addr, 0, this->module, ARK_SC_NAME));
        CULOG(cuModuleGetGlobal(&rc_ptr_addr, 0, this->module, ARK_RC_NAME));
        CULOG(cuMemcpyHtoD(sc_ptr_addr, &sc_ptr_val, sizeof(GpuPtr)));
        CULOG(cuMemcpyHtoD(rc_ptr_addr, &rc_ptr_val, sizeof(GpuPtr)));
        //
        GpuPtr db_ptr_val = this->ctx->get_request_ref();
        GpuPtr db_ptr_addr;
        CULOG(cuModuleGetGlobal(&db_ptr_addr, 0, this->module, ARK_REQ_NAME));
        CULOG(cuMemcpyHtoD(db_ptr_addr, &db_ptr_val, sizeof(GpuPtr)));
        //
        GpuPtr clks_ptr_val = this->clocks->ref();
        GpuPtr clks_ptr_addr;
        CULOG(
            cuModuleGetGlobal(&clks_ptr_addr, 0, this->module, ARK_CLKS_NAME));
        CULOG(cuMemcpyHtoD(clks_ptr_addr, &clks_ptr_val, sizeof(GpuPtr)));
        // set the data buffer pointers of remote gpus

        int nrph = get_env().num_ranks_per_host;
        int nodes_id = this->ctx->get_gpu_id() / nrph;
        // only set the GPU remote data buf pointers of the GPUs on the same
        // node
        for (int i = nodes_id * nrph;
             i < (nodes_id + 1) * nrph && i < this->ctx->get_world_size();
             i++) {
            GpuPtr data_buf_value = this->ctx->get_data_ref(i);
            if (data_buf_value == 0) {
                continue;
            }
            GpuPtr data_buf_ptr;
            string data_buf_name = ARK_BUF_NAME + std::to_string(i);
            CUresult _e = cuModuleGetGlobal(&data_buf_ptr, 0, this->module,
                                            data_buf_name.c_str());
            // in some test code the symbol _ARK_BUF_0 is not defined
            if (_e == CUDA_ERROR_NOT_FOUND) {
                LOG(DEBUG, "global variable ", data_buf_name, " not found");
                continue;
            }
            // CULOG(_e);
            LOG(DEBUG, data_buf_name, " data_buf_ptr=", std::hex, data_buf_ptr,
                " data_buf_value=", data_buf_value);
            CULOG(cuMemcpyHtoD(data_buf_ptr, &data_buf_value, sizeof(GpuPtr)));
        }
    }
}

GpuState GpuLoopKernel::launch(CUstream stream, bool disable_timing)
{
    this->elapsed_msec = -1;
    if (!this->is_compiled()) {
        LOG(ERROR, "Need to compile first before initialization.");
    } else if (stream == nullptr) {
        LOG(ERROR, "Given an invalid stream.");
    } else if (this->stream != nullptr) {
        if (this->stream == stream) {
            LOG(WARN, "Ignore launching twice.");
            return CUDA_SUCCESS;
        } else {
            LOG(ERROR, "This loop kernel is already running.");
        }
    }
    if (!disable_timing) {
        CULOG(cuEventRecord(this->timer_begin, stream));
    }
    // Initialize loop flags.
    *(this->flag_href) = 0;
    GpuState res = GpuKernel::launch(stream);
    if (res == CUDA_SUCCESS) {
        this->stream = stream;
        if (!disable_timing) {
            CULOG(cuEventRecord(this->timer_end, stream));
            this->is_recording = true;
        }
    }
    return res;
}

void GpuLoopKernel::run(int iter)
{
    if (iter > 0) {
        volatile int *href = this->flag_href;
        while (*href > 0) {
        }
        *href = iter;
    }
}

bool GpuLoopKernel::poll()
{
    return *(this->flag_href) <= 0;
}

void GpuLoopKernel::wait()
{
    volatile int *href = this->flag_href;
    int cnt = MAX_LOOP_COUNTER;
    while (*href > 0) {
        if (--cnt > 0) {
            continue;
        }
        // Check if the kernel encountered an error.
        CUresult res = cuStreamQuery(this->stream);
        if (res == CUDA_SUCCESS) {
            if (*href > 0) {
                LOG(WARN, "Stream is finished but the loop flag is still set.");
                break;
            } else {
                LOG(WARN, "wait() is delayed by a stream query. Regarding "
                          "timing measurements may be inaccurate.");
                break;
            }
        } else if (res == CUDA_ERROR_NOT_READY) {
            cnt = MAX_LOOP_COUNTER;
        } else {
            CULOG(res);
        }
    }
}

void GpuLoopKernel::stop()
{
    this->wait();
    *(this->flag_href) = -1;
    CULOG(cuStreamSynchronize(this->stream));
    if (is_recording) {
        CULOG(cuEventElapsedTime(&(this->elapsed_msec), this->timer_begin,
                                 this->timer_end));
        this->is_recording = false;
    }
    this->stream = nullptr;
}

} // namespace ark
