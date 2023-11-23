// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "env.h"

#include <cstdlib>
#include <memory>
#include <type_traits>

#define DEFAULT_ARK_LOG_LEVEL "INFO"
#define DEFAULT_ARK_ROOT "/usr/local/ark"
#define DEFAULT_ARK_TMP "/tmp/ark"
#define DEFAULT_ARK_KEEP_TMP false
#define DEFAULT_ARK_HOSTFILE_NAME "hostfile"
#define DEFAULT_ARK_IPC_LISTEN_PORT_BASE 42000
#define DEFAULT_ARK_NUM_RANKS_PER_HOST 8
#define DEFAULT_ARK_DISABLE_IB false
#define DEFAULT_ARK_DISABLE_P2P_MEMCPY false
#define DEFAULT_ARK_DISABLE_GRAPH_OPT false
#define DEFAULT_ARK_IGNORE_BINARY_CACHE false
#define DEFAULT_ARK_SHM_NAME_PREFIX "ark."
#define DEFAULT_ARK_ENFORCE_KERNEL_CODE_PATH ""
#define DEFAULT_ARK_USE_MSCCLPP false
#define DEFAULT_ARK_MSCCLPP_INCLUDE_DIR "/usr/local/mscclpp/include"
#define DEFAULT_ARK_MSCCLPP_PORT 50051

template <typename T>
T env(const std::string &env_name, const T &default_val) {
    const char *env_ca = getenv(env_name.c_str());
    if (env_ca == nullptr) {
        return default_val;
    }
    if constexpr (std::is_same<T, int>::value) {
        return atoi(env_ca);
    } else if constexpr (std::is_same<T, bool>::value) {
        std::string env_str(env_ca);
        return (env_str == "1");
    } else {
        return std::string(env_ca);
    }
    // Should not reach here.
    return T{};
}

namespace ark {

Env::Env() {
    // Get log level.
    this->log_level = env<std::string>("ARK_LOG_LEVEL", DEFAULT_ARK_LOG_LEVEL);
    // Check if ARK_ROOT is set.
    this->path_root_dir = env<std::string>("ARK_ROOT", DEFAULT_ARK_ROOT);
    // Set temporal directory path.
    this->path_tmp_dir = env<std::string>("ARK_TMP", DEFAULT_ARK_TMP);
    // If `ARK_KEEP_TMP=1`, we do not remove temporal files in `ARK_TMP`.
    this->keep_tmp = env<bool>("ARK_KEEP_TMP", DEFAULT_ARK_KEEP_TMP);
    // Get the hostfile path.
    this->hostfile = env<std::string>(
        "ARK_HOSTFILE", this->path_root_dir + "/" + DEFAULT_ARK_HOSTFILE_NAME);
    // Get the listen socket port.
    this->ipc_listen_port_base =
        env<int>("ARK_IPC_LISTEN_PORT_BASE", DEFAULT_ARK_IPC_LISTEN_PORT_BASE);
    // Get the number of ranks per host.
    this->num_ranks_per_host =
        env<int>("ARK_NUM_RANKS_PER_HOST", DEFAULT_ARK_NUM_RANKS_PER_HOST);
    // If `ARK_DISABLE_IB=1`, we disable IB networking.
    this->disable_ib = env<bool>("ARK_DISABLE_IB", DEFAULT_ARK_DISABLE_IB);
    // If `ARK_DISABLE_P2P_MEMCPY=1`, we disable P2P memcpy between GPUs.
    this->disable_p2p_memcpy =
        env<bool>("ARK_DISABLE_P2P_MEMCPY", DEFAULT_ARK_DISABLE_P2P_MEMCPY);
    // If `ARK_DISABLE_GRAPH_OPT=1`, we disable graph optimization.
    this->disable_graph_opt =
        env<bool>("ARK_DISABLE_GRAPH_OPT", DEFAULT_ARK_DISABLE_GRAPH_OPT);
    // If `ARK_IGNORE_BINARY_CACHE=1`, we ignore compiled binary cache.
    this->ignore_binary_cache =
        env<bool>("ARK_IGNORE_BINARY_CACHE", DEFAULT_ARK_IGNORE_BINARY_CACHE);
    //
    this->shm_name_prefix =
        env<std::string>("ARK_SHM_NAME_PREFIX", DEFAULT_ARK_SHM_NAME_PREFIX);
    //
    this->enforce_kernel_code_path = env<std::string>(
        "ARK_ENFORCE_KERNEL_CODE_PATH", DEFAULT_ARK_ENFORCE_KERNEL_CODE_PATH);
    // If `ARK_USE_MSCCLPP=1`, we use MSCCL++.
    this->use_mscclpp = env<bool>("ARK_USE_MSCCLPP", DEFAULT_ARK_USE_MSCCLPP);
    // Get the MSCCL++ include directory path.
    this->mscclpp_include_dir = env<std::string>(
        "ARK_MSCCLPP_INCLUDE_DIR", DEFAULT_ARK_MSCCLPP_INCLUDE_DIR);
    // Get the port number of MSCCLPP.
    this->mscclpp_port = env<int>("ARK_MSCCLPP_PORT", DEFAULT_ARK_MSCCLPP_PORT);
}

// Global Env.
std::shared_ptr<Env> _ARK_ENV_GLOBAL = nullptr;

// Get the global Env.
const Env &get_env() {
    if (_ARK_ENV_GLOBAL.get() == nullptr) {
        _ARK_ENV_GLOBAL.reset(new Env);
    }
    return *_ARK_ENV_GLOBAL;
}

}  // namespace ark
