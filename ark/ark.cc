// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <sstream>
#include <string>
#include <vector>

#include "env.h"
#include "file_io.h"
#include "include/ark.h"
#include "logging.h"

#define SHM_DIR "/dev/shm/"

namespace ark {

std::string version()
{
    std::stringstream ss;
    ss << ARK_MAJOR << "." << ARK_MINOR << "." << ARK_PATCH;
    return ss.str();
}

void init()
{
    LOG(DEBUG, "init ark");
    // Clean up the shared memory directory. This is useful when the previous
    // run crashed, as this forces to remove locks generated by previous runs.
    // This may crash other ARK processes running on the same machine, if there
    // are any.
    const std::string shm_dir = SHM_DIR;
    const size_t len = shm_dir.size();
    const std::string &prefix = get_env().shm_name_prefix;
    std::vector<std::string> paths = list_dir(shm_dir);
    for (auto &path : paths) {
        if (path.substr(len, prefix.size()) == prefix) {
            if (remove_file(path) != 0) {
                LOG(ERROR, "init failed: failed to remove ", path, " (errno ",
                    errno, ")");
            }
        }
    }
}

} // namespace ark
