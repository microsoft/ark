// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "cpu_timer.h"
#include "env.h"
#include "ipc/ipc_mem.h"
#include "ipc/ipc_shm.h"
#include "logging.h"

using namespace std;

#define NAME_LOCK_POSTFIX ".lock"
#define NAME_DATA_POSTFIX ".data"

namespace ark {

// Constructor.
//
// When the producer is statically determined:
//   Producer:  create_=true,   try_create=any
//   Consumer:  create_=false,  try_create=false
//
// To elect a producer:
//   Everybody: create_=false,  try_create=true
//
// NOTE: To elect the producer, we need to make sure that the lock file name
// does not already exist, otherwise it will cause a deadlock.
//
IpcMem::IpcMem(const string &name_, bool create_, bool try_create)
    : name{name_}, create{create_}
{
    assert(name_.size() > 0);
    string lock_name_str =
        get_env().shm_name_prefix + name_ + NAME_LOCK_POSTFIX;
    const char *lock_name = lock_name_str.c_str();
    int fd;
    if (create_ || try_create) {
        // Try creating a lock file.
        fd = ipc_shm_create(lock_name);
        if (fd != -1) {
            // Succeed.
            int r = ftruncate(fd, sizeof(IpcLock));
            assert(r == 0);
            create_ = true;
        } else if (create_) {
            fd = ipc_shm_open(lock_name);
            assert(fd != -1);
            int r = ftruncate(fd, sizeof(IpcLock));
            assert(r == 0);
        }
    }
    if (!create_) {
        // Wait until we can open the lock file.
        fd = ipc_shm_open_blocking(lock_name);
        assert(fd != -1);
        struct stat s;
        for (;;) {
            int r = fstat(fd, &s);
            assert(r == 0);
            if (s.st_size > 0) {
                assert(s.st_size == sizeof(IpcLock));
                break;
            }
            // Wait until the creator finishes `ftruncate()`.
            cpu_ntimer_sleep(1000);
        }
    }
    // Get mmap of the lock.
    this->lock = (IpcLock *)mmap(0, sizeof(IpcLock), PROT_READ | PROT_WRITE,
                                 MAP_SHARED, fd, 0);
    assert(this->lock != MAP_FAILED);
    close(fd);
    if (create_) {
        // Initialize and acquire the lock.
        int r = ipc_lock_init(this->lock);
        assert(r == 0);
        // Release the lock immediately.
        r = ipc_lock_release(this->lock);
        assert(r == 0);
    } else {
        // Wait until the lock is initialized.
        // This must finish shortly, so we just wait polling.
        while (!this->lock->is_init) {
            cpu_ntimer_sleep(1000);
        }
    }
    this->create = create_;
}

// Destructor.
IpcMem::~IpcMem()
{
    if (this->lock != nullptr) {
        if (this->create) {
            string lock_name =
                get_env().shm_name_prefix + this->name + NAME_LOCK_POSTFIX;
            shm_unlink(lock_name.c_str());
        }
        munmap(this->lock, sizeof(IpcLock));
    }
    if (this->addr != nullptr) {
        if (this->create) {
            string data_name =
                get_env().shm_name_prefix + this->name + NAME_DATA_POSTFIX;
            shm_unlink(data_name.c_str());
        }
        munmap(this->addr, this->total_bytes);
    }
}

// Allocate/re-allocate the shared memory space of the data file.
// Return the current mmapped address if the given `bytes`
// is equal to or less than `total_bytes`.
void *IpcMem::alloc(size_t bytes)
{
    assert((bytes != 0) || !this->create);
    if ((bytes != 0) && (bytes <= this->total_bytes)) {
        // If `total_bytes` is zero, nullptr is returned.
        return this->addr;
    }
    // Open the data file.
    string data_name_str =
        get_env().shm_name_prefix + this->name + NAME_DATA_POSTFIX;
    const char *data_name = data_name_str.c_str();
    int fd;
    if ((this->total_bytes == 0) && this->create) {
        // Create an empty data file.
        fd = ipc_shm_create(data_name);
        if (fd == -1) {
            fd = ipc_shm_open(data_name);
            if (fd == -1) {
                LOGERR("ipc_shm_open: ", strerror(errno), " (", errno, ")");
            }
        }
    } else if (this->create) {
        // Open the existing data file.
        fd = ipc_shm_open(data_name);
        if (fd == -1) {
            LOGERR("ipc_shm_open: ", strerror(errno), " (", errno, ")");
        }
    } else {
        // Wait until the data file appears.
        fd = ipc_shm_open_blocking(data_name);
        if (fd == -1) {
            LOGERR("ipc_shm_open_blocking: ", strerror(errno), " (", errno,
                   ")");
        }
    }
    if (this->create) {
        assert(bytes > 0);
        // Truncate the file size.
        int r = ftruncate(fd, bytes);
        assert(r == 0);
    } else {
        // Wait until the file size becomes equal to or larger than `bytes`.
        // We assume that in most cases the file size is already as large as
        // `bytes` or is going to be so very soon.
        // NOTE: this method cannot prevent race conditions if the data file
        // size may decrease.
        struct stat s;
        for (;;) {
            int r = fstat(fd, &s);
            assert(r == 0);
            if ((bytes == 0) && (s.st_size > 0)) {
                bytes = s.st_size;
                break;
            } else if ((bytes > 0) && (size_t)s.st_size >= bytes) {
                break;
            }
            // Wait until the creator finishes `ftruncate()`.
            cpu_ntimer_sleep(1000);
        }
    }
    // Create a new mmap.
    void *old = this->addr;
    this->addr = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    assert(this->addr != MAP_FAILED);
    // Remove the old mmap.
    if (old != nullptr) {
        munmap(old, this->total_bytes);
    }
    close(fd);
    this->total_bytes = bytes;
    return this->addr;
}

} // namespace ark
