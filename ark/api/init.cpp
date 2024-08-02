// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/init.hpp"

#include <sstream>
#include <string>
#include <vector>

#include "env.h"
#include "file_io.h"
#include "logging.hpp"

#define SHM_DIR "/dev/shm/"

namespace ark {

void init() {
    LOG(DEBUG, "init ark");

    // Get the environment variables.
    (void)get_env(true);

    // Create the temporary directory if it does not exist.
    const std::string &tmp_dir = get_env().path_tmp_dir;
    if (!is_exist(tmp_dir)) {
        if (create_dir(tmp_dir) != 0) {
            ERR(SystemError,
                "init failed: failed to create temporary directory ", tmp_dir,
                " (errno ", errno, ")");
        }
    } else if (!get_env().keep_tmp) {
        // Clear the temporary directory if it exists and keep_tmp is false.
        if (clear_dir(tmp_dir) != 0) {
            ERR(SystemError,
                "init failed: failed to clear temporary directory ", tmp_dir,
                " (errno ", errno, ")");
        }
    }
}

}  // namespace ark
