#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "ark/env.h"

using namespace std;

#define ARK_IPC_LISTEN_PORT_BASE_DEFAULT 42000
#define ARK_HOSTFILE_DEFAULT "hostfile"

namespace ark {

Env::Env()
{
    // Get log level.
    this->log_level = getenv("ARK_LOG_LEVEL");
    // Check if ARK_ROOT is set.
    const char *root_ca = getenv("ARK_ROOT");
    if (root_ca == nullptr) {
        cerr << "ARK_ROOT is not set.\n";
        exit(1);
    }
    this->path_root_dir = root_ca;
    // Set temporal directory path.
    const char *tmp_ca = getenv("ARK_TMP");
    if (tmp_ca == nullptr) {
        this->path_tmp_dir = this->path_root_dir + "/tmp";
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
        this->hostfile = this->path_root_dir + "/" + ARK_HOSTFILE_DEFAULT;
    } else {
        this->hostfile = hostfile_ca;
    }
    // Get the listen socket port.
    const char *ipc_ca = getenv("ARK_IPC_LISTEN_PORT_BASE");
    if (ipc_ca == nullptr) {
        this->ipc_listen_port_base = ARK_IPC_LISTEN_PORT_BASE_DEFAULT;
    } else {
        this->ipc_listen_port_base = atoi(ipc_ca);
    }
    // Get the number of ranks per host.
    const char *ranks_ca = getenv("ARK_NUM_RANKS_PER_HOST");
    if (ranks_ca == nullptr) {
        this->num_ranks_per_host = 8;
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
    const char *scheduler_ca = getenv("ARK_SCHEDULER");
    if (scheduler_ca == nullptr) {
        this->scheduler = "Default";
    } else {
        this->scheduler = scheduler_ca;
    }
}

// Global Env.
Env *_ARK_ENV_GLOBAL = nullptr;

// Get the global Env.
const Env &get_env()
{
    if (_ARK_ENV_GLOBAL == nullptr) {
        _ARK_ENV_GLOBAL = new Env;
        assert(_ARK_ENV_GLOBAL != nullptr);
    }
    return *_ARK_ENV_GLOBAL;
}

} // namespace ark
