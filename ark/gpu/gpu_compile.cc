// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_compile.h"

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>

#include "cpu_timer.h"
#include "env.h"
#include "gpu/gpu_logging.h"
#include "include/ark.h"
#include "random.h"

#define ARK_DEBUG_KERNEL 0

using namespace std;

namespace ark {

template <typename ItemType>
static void para_exec(std::vector<ItemType> &items, int max_num_threads,
                      const std::function<void(ItemType &)> &func) {
    size_t nthread = (size_t)max_num_threads;
    if (nthread > items.size()) {
        nthread = items.size();
    }
    std::vector<std::thread> threads;
    threads.reserve(nthread);
    std::mutex mtx;
    size_t idx = 0;
    for (size_t i = 0; i < nthread; ++i) {
        threads.emplace_back([&items, &mtx, &idx, &func] {
            size_t local_idx = -1;
            for (;;) {
                {
                    const std::lock_guard<std::mutex> lock(mtx);
                    local_idx = idx++;
                }
                if (local_idx >= items.size()) break;
                func(items[local_idx]);
            }
        });
    }
    for (auto &t : threads) {
        t.join();
    }
}

const string gpu_compile(const vector<string> &codes,
                         const GpuArchType &arch_type,
                         unsigned int max_reg_cnt) {
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
    vector<pair<string, string> > items;
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
    para_exec<pair<string, string> >(
        items, 20, [&arch, &ark_root, max_reg_cnt](pair<string, string> &item) {
            string cu_file_path = item.second + ".cu";
            // Write GPU kernel code file.
            {
                ofstream cu_file(cu_file_path, ios::out | ios::trunc);
                cu_file << item.first;
            }
            // Compile command using NVCC.
            stringstream exec_cmd;
            exec_cmd << "/usr/local/cuda/bin/nvcc -cubin ";
#if (ARK_DEBUG_KERNEL)
            exec_cmd << "-G ";
#endif  // (ARK_DEBUG_KERNEL)
            if (max_reg_cnt > 0) {
                exec_cmd << "-maxrregcount " << max_reg_cnt << " ";
            }
            stringstream include_args;
            // clang-format off
            include_args << "-I" << ark_root << "/include "
                         << "-I" << ark_root << "/include/kernels ";
            exec_cmd << "-ccbin g++ -std c++17 -lcuda "
                "--define-macro=ARK_TARGET_CUDA_ARCH=" << arch << " "
                "--define-macro=ARK_COMM_SW=1 " <<
                include_args.str() <<
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
                LOG(ERROR, endl, exec_print_str, endl);
            }
            LOG(INFO, "Compile succeed: ", cu_file_path, " (",
                cpu_timer() - start, " seconds)");
        });
    string cu_file_path = items[0].second + ".cu";
    string gpubin_file_path = items[0].second + ".cubin";
    ifstream gpubin_file(gpubin_file_path);
    stringstream ss;
    ss << gpubin_file.rdbuf();
    // remove(cu_file_path.c_str());
    remove(gpubin_file_path.c_str());
    return ss.str();
#endif  // (ARK_USE_NVRTC)
}

}  // namespace ark
