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
    // Base value of listen socket ports.
    int ipc_listen_port_base;
    // Number of ranks per host.
    int num_ranks_per_host;
    // Disable IB.
    bool disable_ib;
    // Disable P2P GPU memcpy.
    bool disable_p2p_memcpy;
    // Disable the heuristic ARK graph optimization.
    bool disable_graph_opt;
    // Ignore compiled binary cache.
    bool ignore_binary_cache;
    // Prefix of shared memory file names.
    std::string shm_name_prefix;
};

// Get the global Env.
const Env &get_env();

}  // namespace ark

#endif  // ARK_ENV_H_
