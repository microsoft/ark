// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ipc/ipc_hosts.h"

#include <arpa/inet.h>
#include <netdb.h>

#include <fstream>
#include <vector>

#include "env.h"
#include "file_io.h"
#include "include/ark.h"
#include "logging.h"

namespace ark {

static std::vector<std::string> hosts;

const std::string get_host(int idx, bool reset) {
    if (reset) {
        hosts.clear();
    }
    if (hosts.size() == 0) {
        const auto &hostfile = get_env().hostfile;
        if (!is_file(hostfile)) {
            LOG(WARN, "cannot open hostfile: ", hostfile, ", assume localhost");
            hosts.push_back("127.0.0.1");
        } else {
            std::ifstream ifs(hostfile);
            std::string line;
            int host_idx = 0;
            while (std::getline(ifs, line)) {
                // Hostname to IP
                struct hostent *ent = ::gethostbyname(line.c_str());
                if (ent == nullptr) {
                    ERR(InvalidUsageError, "cannot resolve hostname: ", line);
                }
                char *host = ::inet_ntoa(*(struct in_addr *)ent->h_addr);
                LOG(INFO, "HOST ", host_idx, ": ", host);
                hosts.emplace_back(host);
                host_idx++;
            }
        }
    }
    if ((idx < 0) || (idx >= (int)hosts.size())) {
        ERR(InvalidUsageError, "invalid host index: ", idx);
    }
    return hosts[idx];
}

}  // namespace ark
