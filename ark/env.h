// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_ENV_H_
#define ARK_ENV_H_

#include <string>

namespace ark {

// Environment variables.
struct Env {
    Env();
    // Log level.
    std::string log_level;
    // Root directory where ARK is installed.
    std::string path_root_dir;
    // Temporary directory.
    std::string path_tmp_dir;
    // If true, we do not remove temporal files in `path_tmp_dir`.
    bool keep_tmp;
    // Hostfile.
    std::string hostfile;
    // Number of ranks per host.
    int num_ranks_per_host;
    // Disable IB
    bool disable_ib;
    // Ignore compiled binary cache.
    bool ignore_binary_cache;
    // Enforce to compile a specific plan file.
    std::string enforce_plan_path;
    // MSCCL++ bootstrap port.
    int mscclpp_port;
};

// Get the global Env.
const Env &get_env(bool reset = false);

}  // namespace ark

#endif  // ARK_ENV_H_
