// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <arpa/inet.h>
#include <cstring>
#include <netdb.h>
#include <vector>

#include "env.h"
#include "ipc/ipc_hosts.h"
#include "logging.h"

namespace ark {

static std::vector<std::string> hosts;

const std::string &get_host(int idx)
{
    if (hosts.size() == 0) {
        const char *hostfile = get_env().hostfile.c_str();
        FILE *fp = fopen(hostfile, "r");
        if (fp == nullptr) {
            LOG(WARN, "cannot open hostfile: ", hostfile, ", assume localhost");
            hosts.push_back("127.0.0.1");
        } else {
            char buf[1024];
            buf[1023] = 0;
            int host_idx = 0;
            while (fgets(buf, sizeof(buf), fp) != nullptr) {
                if (buf[1023] != 0) {
                    LOGERR("hostfile line too long: ", buf);
                }
                // Erase the newline character
                int l = strlen(buf);
                buf[l - 1] = 0;
                // Hostname to IP
                struct hostent *ent = gethostbyname(buf);
                if (ent == nullptr) {
                    LOGERR("cannot resolve hostname: ", buf);
                }
                char *host = inet_ntoa(*(struct in_addr *)ent->h_addr);
                LOG(INFO, "HOST ", host_idx, ": ", host);
                hosts.emplace_back(host);
                host_idx++;
            }
        }
    }
    if ((idx < 0) || (idx >= (int)hosts.size())) {
        LOGERR("invalid host index: ", idx);
    }
    return hosts[idx];
}

} // namespace ark
