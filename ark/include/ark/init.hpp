// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_INIT_HPP
#define ARK_INIT_HPP

namespace ark {

/// Initialize the ARK runtime.
///
/// This function should be called by the user before any other functions are
/// called. It is safe to call this function multiple times.
void init();

}  // namespace ark

#endif  // ARK_INIT_HPP
