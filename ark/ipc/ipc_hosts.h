// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_IPC_HOSTS_H_
#define ARK_IPC_HOSTS_H_

#include <string>

namespace ark {

/// Return a hostname from the hostfile.
/// @param idx Index of the hostname to return.
/// @param reset Whether to reread the hostfile.
/// @return The hostname.
const std::string get_host(int idx, bool reset = false);

}  // namespace ark

#endif  // ARK_IPC_HOSTS_H_
