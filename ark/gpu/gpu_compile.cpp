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
#include <iomanip>
#include <mutex>
#include <thread>

#include "ark/random.hpp"
#include "cpu_timer.h"
#include "env.h"
#include "file_io.h"
#include "gpu/gpu_logging.h"

#define ARK_DEBUG_KERNEL 0

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

// TODO: use a stronger hash function
static std::string fnv1a_hash(const std::string &str) {
    const uint64_t FNV_prime = 1099511628211u;
    const uint64_t FNV_offset_basis = 14695981039346656037u;
    uint64_t hash = FNV_offset_basis;
    for (const auto &c : str) {
        hash ^= static_cast<uint64_t>(c);
        hash *= FNV_prime;
    }
    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(16) << hash;
    return ss.str();
}

static const std::string gpu_compile_command(
    const std::string &code_file_path, const std::string &ark_root,
    const Arch &arch, [[maybe_unused]] unsigned int max_reg_cnt,
    const std::string &output_file_path) {
#if defined(ARK_CUDA)
    if (!arch.belongs_to(ARCH_CUDA)) {
        ERR(InvalidUsageError, "Invalid architecture: ", arch.name());
    }
    const std::string &cc = arch.category()[1];

    std::vector<std::string> args;

    // TODO: use the compiler found by cmake.
    args.emplace_back("/usr/local/cuda/bin/nvcc");
    args.emplace_back("-cubin");
#if (ARK_DEBUG_KERNEL)
    args.emplace_back("-G");
#endif  // (ARK_DEBUG_KERNEL)
    if (max_reg_cnt > 0) {
        args.emplace_back("-maxrregcount " + std::to_string(max_reg_cnt));
    }
    args.emplace_back("-ccbin g++");
    args.emplace_back("-std c++17");
    args.emplace_back("-lcuda");
    args.emplace_back("--define-macro=ARK_TARGET_CUDA_ARCH=" + cc);
    args.emplace_back("-I" + ark_root + "/include");
    args.emplace_back("-I" + ark_root + "/include/kernels");
    args.emplace_back("-gencode arch=compute_" + cc + ",code=sm_" + cc);
    args.emplace_back("-o " + output_file_path);
    args.emplace_back(code_file_path);
    args.emplace_back("2>&1");

#elif defined(ARK_ROCM)
    // Remove prepending "rocm_" from arch to get the compute capability
    if (!arch.belongs_to(ARCH_ROCM)) {
        ERR(InvalidUsageError, "Invalid architecture: ", arch.name());
    }
    const std::string &cc = to_lower(arch.category()[1]);

    std::vector<std::string> args;

    // TODO: use the compiler found by cmake.
    args.emplace_back("LANG=C /usr/bin/hipcc");
    args.emplace_back("--genco");
#if (ARK_DEBUG_KERNEL)
    args.emplace_back("-O0");
#endif  // (ARK_DEBUG_KERNEL)
    args.emplace_back("-std=c++17");
    args.emplace_back("--define-macro=ARK_TARGET_ROCM_ARCH=" + cc);
    args.emplace_back("-I" + ark_root + "/include");
    args.emplace_back("-I" + ark_root + "/include/kernels");
    args.emplace_back("--offload-arch=gfx" + cc);
    args.emplace_back("-o " + output_file_path);
    args.emplace_back(code_file_path);
    args.emplace_back("2>&1");
#endif

    // Compile command.
    std::stringstream compile_cmd;
    compile_cmd << args[0];
    for (size_t i = 1; i < args.size(); ++i) {
        compile_cmd << " " << args[i];
    }
    return compile_cmd.str();
}

const std::string gpu_compile(const std::vector<std::string> &codes,
                              const Arch &arch, unsigned int max_reg_cnt) {
    const std::string &ark_root = get_env().path_root_dir;

    // (code, file name prefix) pairs
    std::vector<std::pair<std::string, std::string> > items;
    items.reserve(codes.size());
    srand();
    for (auto &code : codes) {
        std::string hash_str = fnv1a_hash(code);
        items.emplace_back(code, "/tmp/ark_" + hash_str);
    }
    assert(items.size() == 1);
    para_exec<std::pair<std::string, std::string> >(
        items, 20,
        [&arch, &ark_root,
         max_reg_cnt](std::pair<std::string, std::string> &item) {
            std::string code_file_path = item.second + ".cu";
            std::string bin_file_path = item.second + ".cubin";
            if (is_exist(code_file_path) && is_exist(bin_file_path)) {
                if (!get_env().ignore_binary_cache) {
                    LOG(INFO, "Reusing cached binary for ", code_file_path);
                    return;
                }
                // Remove the cached binary.
                int err = remove_file(bin_file_path);
                if (err != 0) {
                    ERR(SystemError,
                        "Failed to remove a cache file: ", bin_file_path,
                        " (errno ", err, ")");
                }
            }
            // Write GPU kernel code file.
            {
                std::ofstream code_file(code_file_path,
                                        std::ios::out | std::ios::trunc);
                code_file << item.first;
            }
            const std::string compile_cmd =
                gpu_compile_command(code_file_path, ark_root, arch, max_reg_cnt,
                                    item.second + ".cubin");
            double start = cpu_timer();
            LOG(INFO, "Compiling: ", code_file_path);
            LOG(DEBUG, compile_cmd);
            // Run the command.
            std::array<char, 4096> buffer;
            std::stringstream exec_print;
            std::unique_ptr<FILE, decltype(&pclose)> pipe(
                popen(compile_cmd.c_str(), "r"), pclose);
            if (!pipe) {
                ERR(SystemError, "popen() failed");
            }
            while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
                exec_print << buffer.data();
            }
            std::string exec_print_str = exec_print.str();
            if (exec_print_str.size() > 0) {
                ERR(ExecutorError, "\n", compile_cmd, "\n", exec_print_str,
                    "\n");
            }
            LOG(INFO, "Compile succeed: ", code_file_path, " (",
                cpu_timer() - start, " seconds)");
        });
    std::string gpubin_file_path = items[0].second + ".cubin";
    return read_file(gpubin_file_path);
}

}  // namespace ark
