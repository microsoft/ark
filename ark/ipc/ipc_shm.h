// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#ifndef ARK_IPC_SHM_H_
#define ARK_IPC_SHM_H_

namespace ark {

// Create a shm file if it does not exist, and truncate it to zero bytes.
// If the creation fails, return -1.
int ipc_shm_create(const char *name);
// Try opening a shm file.
// Return its file descriptor on success, otherwise return -1.
int ipc_shm_open(const char *name);
// Open a shm file and return its file descriptor.
// If opening fails due to non-existence, block until it can open.
// If opening fails due to any other reasons, return -1.
int ipc_shm_open_blocking(const char *name);
// Destroy a shm file.
int ipc_shm_destroy(const char *name);
// Return zero if we can open the shm file.
int ipc_shm_exist(const char *name);

} // namespace ark

#endif // ARK_IPC_SHM_H_
