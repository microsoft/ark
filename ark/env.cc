// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "env.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

#define DEFAULT_ARK_ROOT "/usr/local/ark"
#define DEFAULT_ARK_TMP "/tmp/ark"
#define DEFAULT_ARK_HOSTFILE_NAME "hostfile"
#define DEFAULT_ARK_IPC_LISTEN_PORT_BASE 42000
#define DEFAULT_ARK_NUM_RANKS_PER_HOST 8
#define DEFAULT_ARK_DISABLE_GRAPH_OPT false
#define DEFAULT_ARK_SHM_NAME_PREFIX "ark."
#define DEFAULT_ARK_USE_MSCCLPP false
#define DEFAULT_ARK_MSCCLPP_INCLUDE_DIR "/usr/local/mscclpp/include"
#define DEFAULT_ARK_MSCCLPP_PORT 50051

namespace ark {

Env::Env() {
    // Get log level.
    this->log_level = getenv("ARK_LOG_LEVEL");
    // Check if ARK_ROOT is set.
    const char *root_ca = getenv("ARK_ROOT");
    if (root_ca == nullptr) {
        root_ca = DEFAULT_ARK_ROOT;
    }
    this->path_root_dir = root_ca;
    // Set temporal directory path.
    const char *tmp_ca = getenv("ARK_TMP");
    if (tmp_ca == nullptr) {
        this->path_tmp_dir = DEFAULT_ARK_TMP;
    } else {
        this->path_tmp_dir = tmp_ca;
    }
    // If `ARK_KEEP_TMP=1`, we do not remove temporal files in `ARK_TMP`.
    const char *keep_tmp_ca = getenv("ARK_KEEP_TMP");
    if (keep_tmp_ca != nullptr && strncmp(keep_tmp_ca, "1", 2) == 0) {
        this->keep_tmp = true;
    } else {
        this->keep_tmp = false;
    }
    // Get the PCIe name (domain:bus:slot.function) of the FPGA.
    const char *fpga_ca = getenv("ARK_FPGA_DBSF");
    if (fpga_ca == nullptr) {
        this->fpga_dbsf = "";
    } else {
        this->fpga_dbsf = fpga_ca;
    }
    // Get the hostfile path.
    const char *hostfile_ca = getenv("ARK_HOSTFILE");
    if (hostfile_ca == nullptr) {
        this->hostfile = this->path_root_dir + "/" + DEFAULT_ARK_HOSTFILE_NAME;
    } else {
        this->hostfile = hostfile_ca;
    }
    // Get the listen socket port.
    const char *ipc_ca = getenv("ARK_IPC_LISTEN_PORT_BASE");
    if (ipc_ca == nullptr) {
        this->ipc_listen_port_base = DEFAULT_ARK_IPC_LISTEN_PORT_BASE;
    } else {
        this->ipc_listen_port_base = atoi(ipc_ca);
    }
    // Get the number of ranks per host.
    const char *ranks_ca = getenv("ARK_NUM_RANKS_PER_HOST");
    if (ranks_ca == nullptr) {
        this->num_ranks_per_host = DEFAULT_ARK_NUM_RANKS_PER_HOST;
    } else {
        this->num_ranks_per_host = atoi(ranks_ca);
    }
    // If `ARK_DISABLE_IB=1`, we disable IB networking.
    const char *disable_ib_ca = getenv("ARK_DISABLE_IB");
    if ((disable_ib_ca != nullptr) && (strncmp(disable_ib_ca, "1", 2) == 0)) {
        this->disable_ib = true;
    } else {
        this->disable_ib = false;
    }
    // If `ARK_DISABLE_P2P_MEMCPY=1`, we disable P2P CUDA memcpy.
    const char *disable_p2p_memcpy_ca = getenv("ARK_DISABLE_P2P_MEMCPY");
    if ((disable_p2p_memcpy_ca != nullptr) &&
        (strncmp(disable_p2p_memcpy_ca, "1", 2) == 0)) {
        this->disable_p2p_memcpy = true;
    } else {
        this->disable_p2p_memcpy = false;
    }
    // If `ARK_DISABLE_GRAPH_OPT=1`, we disable graph optimization.
    const char *disable_graph_opt_ca = getenv("ARK_DISABLE_GRAPH_OPT");
    if (disable_graph_opt_ca == nullptr) {
        this->disable_graph_opt = DEFAULT_ARK_DISABLE_GRAPH_OPT;
    } else if (strncmp(disable_graph_opt_ca, "1", 2) == 0) {
        this->disable_graph_opt = true;
    } else {
        this->disable_graph_opt = false;
    }
    //
    const char *shm_name_prefix_ca = getenv("ARK_SHM_NAME_PREFIX");
    if (shm_name_prefix_ca == nullptr) {
        this->shm_name_prefix = DEFAULT_ARK_SHM_NAME_PREFIX;
    } else {
        this->shm_name_prefix = shm_name_prefix_ca;
    }
    // If `ARK_USE_MSCCLPP=1`, we use MSCCL++.
    const char *use_mscclpp_ca = getenv("ARK_USE_MSCCLPP");
    if (use_mscclpp_ca == nullptr) {
        this->use_mscclpp = DEFAULT_ARK_USE_MSCCLPP;
    } else if (strncmp(use_mscclpp_ca, "1", 2) == 0) {
        this->use_mscclpp = true;
    } else {
        this->use_mscclpp = false;
    }
    // Get the MSCCL++ include directory path.
    const char *mscclpp_include_dir_ca = getenv("ARK_MSCCLPP_INCLUDE_DIR");
    if (mscclpp_include_dir_ca == nullptr) {
        this->mscclpp_include_dir = DEFAULT_ARK_MSCCLPP_INCLUDE_DIR;
    } else {
        this->mscclpp_include_dir = mscclpp_include_dir_ca;
    }
    // Get the port number of MSCCL++.
    const char *mscclpp_port_ca = getenv("ARK_MSCCLPP_PORT");
    if (mscclpp_port_ca == nullptr) {
        this->mscclpp_port = DEFAULT_ARK_MSCCLPP_PORT;
    } else {
        this->mscclpp_port = atoi(mscclpp_port_ca);
    }
}

// Global Env.
Env *_ARK_ENV_GLOBAL = nullptr;

// Get the global Env.
const Env &get_env() {
    if (_ARK_ENV_GLOBAL == nullptr) {
        _ARK_ENV_GLOBAL = new Env;
        assert(_ARK_ENV_GLOBAL != nullptr);
    }
    return *_ARK_ENV_GLOBAL;
}

}  // namespace ark
