// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_UTILS_NET_HPP_
#define ARK_UTILS_NET_HPP_

#include <string>

namespace ark {

/// Return a hostname from the hostfile.
/// @param idx Index of the hostname to return.
/// @param reset Whether to reread the hostfile.
/// @return The hostname.
const std::string get_host(int idx, bool reset = false);

}  // namespace ark

#endif  // ARK_UTILS_NET_HPP_
