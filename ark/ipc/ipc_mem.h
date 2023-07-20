// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_IPC_ENTRY_H_
#define ARK_IPC_ENTRY_H_

#include <string>

#include "ipc/ipc_lock.h"

namespace ark {

// A single IpcMem consists of a data file and a lock file
// to ensure atomic read/write on the data file.
class IpcMem
{
  public:
    // Constructor.
    IpcMem(const std::string &name, bool create, bool try_create = false);
    // Destructor.
    ~IpcMem();

    // Allocate/re-allocate the shared memory space of the data file.
    // Return the current mmapped address if the given `bytes`
    // is less than `total_bytes`.
    void *alloc(std::size_t bytes);
    // Return the lock file mmap address.
    IpcLock *get_lock() const
    {
        return this->lock;
    }
    // Return the current mmap address.
    void *get_addr() const
    {
        return this->addr;
    }
    // Get the bytes of the mmapped memory space of the data file.
    std::size_t get_bytes() const
    {
        return total_bytes;
    }
    // Return true if object is the data creator.
    bool is_create() const
    {
        return create;
    }

  private:
    const std::string name;
    // If true, this object will create shared object files and
    // may change their sizes or destroy them.
    // If false, this object will busy wait for the creator if those
    // files do not exist or when their sizes are not as expected.
    bool create;
    // Pointer to the mmapped memory space of the lock file.
    IpcLock *lock = nullptr;
    // Pointer to the mmapped memory space of the data file.
    void *addr = nullptr;
    // Size of the mmapped memory space of the data file.
    std::size_t total_bytes = 0;
};

} // namespace ark

#endif // ARK_IPC_ENTRY_H_
