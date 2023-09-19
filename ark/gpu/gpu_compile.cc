// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "cpu_timer.h"
#include "env.h"
#include "gpu/gpu_compile.h"
#include "gpu/gpu_logging.h"
#include "include/ark.h"
#include "random.h"
#include "threading.h"

#define ARK_USE_NVRTC 0
#define ARK_DEBUG_KERNEL 0

#if (ARK_USE_NVRTC)
#include <nvrtc.h>
#endif // (ARK_USE_NVRTC)

using namespace std;

namespace ark {

#if (ARK_USE_NVRTC)
const string nvrtc_compile(const string &ark_root, const string &arch,
                           const string &code, unsigned int max_reg_cnt)
{
    nvrtcProgram prog;
    NVRTCLOG(nvrtcCreateProgram(&prog, code.c_str(), nullptr, 0, 0, 0));
    string opt_arch = "-arch=compute_" + arch;
    string opt_arch_def = "--define-macro=ARK_TARGET_CUDA_ARCH=" + arch;
    string opt_reg = "-maxrregcount=" + to_string(max_reg_cnt);
    string opt_inc_0 = "-I" + ark_root + "/include";
    string opt_inc_1 = "-I" + ark_root + "/include/kernels";
    string opt_inc_2 = "-I" + ark_root + "/include/kernels/nvrtc";
    const char *opts[] = {
        opt_arch.c_str(),
        "-std=c++17",
        "-default-device",
#if (ARK_DEBUG_KERNEL)
        "--device-debug",
        "--generate-line-info",
#endif // (ARK_DEBUG_KERNEL)
        opt_reg.c_str(),
        opt_arch_def.c_str(),
        opt_inc_0.c_str(),
        opt_inc_1.c_str(),
        opt_inc_2.c_str(),
        "-I/usr/local/cuda/include",
    };
    // Print compile options for debugging.
    stringstream ss;
    for (size_t i = 0; i < sizeof(opts) / sizeof(opts[0]); ++i) {
        ss << opts[i] << " ";
    }
    LOG(DEBUG, ss.str());
    // Compile.
    nvrtcResult compileResult =
        nvrtcCompileProgram(prog, sizeof(opts) / sizeof(opts[0]), opts);
    // Obtain compilation log from the program.
    size_t log_size;
    NVRTCLOG(nvrtcGetProgramLogSize(prog, &log_size));
    if (log_size > 1) {
        char *log = new char[log_size];
        NVRTCLOG(nvrtcGetProgramLog(prog, log));
        // LOG(ERROR, endl, log, endl);
        LOG(DEBUG, endl, log, endl);
        delete[] log;
    }
    NVRTCLOG(compileResult);
    // Obtain PTX from the program.
    size_t ptx_size;
    NVRTCLOG(nvrtcGetPTXSize(prog, &ptx_size));
    char *ptx = new char[ptx_size];
    NVRTCLOG(nvrtcGetPTX(prog, ptx));
    NVRTCLOG(nvrtcDestroyProgram(&prog));
    // Write the result PTX file.
    return string(ptx);
}

const string link(const vector<string> &ptxs)
{
    unsigned int buflen = 8192;
    char *infobuf = new char[buflen];
    char *errbuf = new char[buflen];
    assert(infobuf != nullptr);
    assert(errbuf != nullptr);
    int enable = 1;
    int num_opts = 5;
    CUjit_option *opts = new CUjit_option[num_opts];
    void **optvals = new void *[num_opts];
    assert(opts != nullptr);
    assert(optvals != nullptr);

    opts[0] = CU_JIT_INFO_LOG_BUFFER;
    optvals[0] = (void *)infobuf;

    opts[1] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    optvals[1] = (void *)(long)buflen;

    opts[2] = CU_JIT_ERROR_LOG_BUFFER;
    optvals[2] = (void *)errbuf;

    opts[3] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    optvals[3] = (void *)(long)buflen;

    opts[4] = CU_JIT_GENERATE_DEBUG_INFO;
    optvals[4] = (void *)(long)enable;

    CUlinkState lstate;
    CULOG(cuLinkCreate(num_opts, opts, optvals, &lstate));
    for (const auto &ptx : ptxs) {
        CULOG(cuLinkAddData(lstate, CU_JIT_INPUT_PTX, (void *)ptx.c_str(),
                            ptx.size() + 1, 0, 0, 0, 0));
    }
    char **cubin = nullptr;
    size_t cubin_size;
    CUresult res = cuLinkComplete(lstate, (void **)cubin, &cubin_size);
    if (res != CUDA_SUCCESS) {
        LOG(DEBUG, errbuf);
        CULOG(res);
    }
    assert(cubin != nullptr);
    string ret{*cubin};
    CULOG(cuLinkDestroy(lstate));
    delete[] infobuf;
    delete[] errbuf;
    return ret;
}

#endif // (ARK_USE_NVRTC)

const string gpu_compile(const vector<string> &codes,
                         const GpuArchType &arch_type, unsigned int max_reg_cnt)
{
    const string &ark_root = get_env().path_root_dir;
    string arch;
    if (arch_type == GPU_ARCH_CUDA_60) {
        arch = "60";
    } else if (arch_type == GPU_ARCH_CUDA_70) {
        arch = "70";
    } else if (arch_type == GPU_ARCH_CUDA_80) {
        arch = "80";
    } else if (arch_type == GPU_ARCH_CUDA_90) {
        arch = "90";
    } else {
        arch = "";
    }

#if (ARK_USE_NVRTC)
    vector<string> ptxs;
    for (auto &code : codes) {
        ptxs.emplace_back(nvrtc_compile(ark_root, arch, code, max_reg_cnt));
    }
    // return link(ark_root, ptxs);
    return ptxs[0];
#else
    // assert(false);
    // return "";
    vector<pair<string, string>> items;
    items.reserve(codes.size());
    srand();
    for (auto &code : codes) {
        string rand_str;
        for (;;) {
            rand_str = rand_anum(16);
            bool retry = false;
            for (auto &p : items) {
                if (p.second == rand_str) {
                    retry = true;
                    break;
                }
            }
            if (!retry) {
                break;
            }
        }
        // TODO: retry if the file name already exists.
        items.emplace_back(code, "/tmp/ark_" + rand_str);
    }
    assert(items.size() == 1);
    para_exec<pair<string, string>>(
        items, 20, [&arch, &ark_root, max_reg_cnt](pair<string, string> &item) {
            string cu_file_path = item.second + ".cu";
            // Write CUDA code file.
            {
                ofstream cu_file(cu_file_path, ios::out | ios::trunc);
                cu_file << item.first;
            }
            // Compile command using NVCC.
            stringstream exec_cmd;
            exec_cmd << "/usr/local/cuda/bin/nvcc -cubin ";
#if (ARK_DEBUG_KERNEL)
            exec_cmd << "-G ";
#endif // (ARK_DEBUG_KERNEL)
            if (max_reg_cnt > 0) {
                exec_cmd << "-maxrregcount " << max_reg_cnt << " ";
            }
            stringstream define_args;
            stringstream include_args;
            // clang-format off
            define_args << "-DARK_TARGET_CUDA_ARCH=" << arch << " "
                        << "-DARK_COMM_SW=1" << " ";
            include_args << "-I" << ark_root << "/include "
                         << "-I" << ark_root << "/include/kernels ";
            if (get_env().use_mscclpp) {
                define_args << "-DARK_USE_MSCCLPP=1 ";
                include_args << "-I" << get_env().mscclpp_include_dir << " ";
            }
            exec_cmd << "-ccbin g++ -std c++17 -lcuda "
                << define_args.str() << include_args.str() <<
                "-gencode arch=compute_" << arch
                << ",code=sm_" << arch << " "
                "-o " << item.second << ".cubin "
                << cu_file_path << " 2>&1";
            // clang-format on
            double start = cpu_timer();
            LOG(INFO, "Compiling: ", cu_file_path);
            LOG(DEBUG, exec_cmd.str());
            // Run the command.
            array<char, 4096> buffer;
            stringstream exec_print;
            unique_ptr<FILE, decltype(&pclose)> pipe(
                popen(exec_cmd.str().c_str(), "r"), pclose);
            if (!pipe) {
                LOG(ERROR, "popen() failed");
            }
            while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
                exec_print << buffer.data();
            }
            string exec_print_str = exec_print.str();
            if (exec_print_str.size() > 0) {
                LOG(ERROR, endl, exec_cmd.str(), endl, exec_print_str, endl);
            }
            LOG(INFO, "Compile succeed: ", cu_file_path, " (",
                cpu_timer() - start, " seconds)");
        });
    string cu_file_path = items[0].second + ".cu";
    string cubin_file_path = items[0].second + ".cubin";
    ifstream cubin_file(cubin_file_path);
    stringstream ss;
    ss << cubin_file.rdbuf();
    // remove(cu_file_path.c_str());
    remove(cubin_file_path.c_str());
    return ss.str();
#endif // (ARK_USE_NVRTC)
}

} // namespace ark
