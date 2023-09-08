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
IpcMem::IpcMem(const string &name, bool create, bool try_create)
    : name_{name}, create_{create}
{
    assert(name_.size() > 0);
    string lock_name_str =
        get_env().shm_name_prefix + name_ + NAME_LOCK_POSTFIX;
    const char *lock_name = lock_name_str.c_str();
    int fd;
    if (create || try_create) {
        // Try creating a lock file.
        fd = ipc_shm_create(lock_name);
        if (fd != -1) {
            // Succeed.
            int r = ftruncate(fd, sizeof(IpcLock));
            if (r != 0) {
                LOG(ERROR, "ftruncate failed (errno ", r, ")");
            }
            create = true;
        } else if (create) {
            fd = ipc_shm_open(lock_name);
            assert(fd != -1);
            int r = ftruncate(fd, sizeof(IpcLock));
            if (r != 0) {
                LOG(ERROR, "ftruncate failed (errno ", r, ")");
            }
        }
    }
    if (!create) {
        // Wait until we can open the lock file.
        fd = ipc_shm_open_blocking(lock_name);
        assert(fd != -1);
        struct stat s;
        for (;;) {
            int r = fstat(fd, &s);
            if (r != 0) {
                LOG(ERROR, "fstat failed (errno ", r, ")");
            }
            if (s.st_size > 0) {
                assert(s.st_size == sizeof(IpcLock));
                break;
            }
            // Wait until the creator finishes `ftruncate()`.
            cpu_ntimer_sleep(1000);
        }
    }
    // Get mmap of the lock.
    lock_ = (IpcLock *)mmap(0, sizeof(IpcLock), PROT_READ | PROT_WRITE,
                            MAP_SHARED, fd, 0);
    assert(lock_ != MAP_FAILED);
    close(fd);
    if (create) {
        // Initialize and acquire the lock.
        int r = ipc_lock_init(lock_);
        if (r != 0) {
            LOG(ERROR, "ipc_lock_init failed (errno ", r, ")");
        }
        // Release the lock immediately.
        r = ipc_lock_release(lock_);
        if (r != 0) {
            LOG(ERROR, "ipc_lock_release failed (errno ", r, ")");
        }
    } else {
        // Wait until the lock is initialized.
        // This must finish shortly, so we just wait polling.
        while (!lock_->is_init) {
            cpu_ntimer_sleep(1000);
        }
    }
    create_ = create;
    locked_ = false;
}

// Destructor.
IpcMem::~IpcMem()
{
    if (lock_ != nullptr) {
        if (locked_) {
            this->unlock();
        }
        if (create_) {
            string lock_name =
                get_env().shm_name_prefix + name_ + NAME_LOCK_POSTFIX;
            shm_unlink(lock_name.c_str());
        }
        munmap(lock_, sizeof(IpcLock));
    }
    if (addr_ != nullptr) {
        if (create_) {
            string data_name =
                get_env().shm_name_prefix + name_ + NAME_DATA_POSTFIX;
            shm_unlink(data_name.c_str());
        }
        munmap(addr_, total_bytes_);
    }
}

void IpcMem::lock()
{
    assert(lock_ != nullptr);
    assert(!locked_);
    int r = ipc_lock_acquire(lock_);
    if (r != 0) {
        LOG(ERROR, "ipc_lock_acquire failed (errno ", r, ")");
    }
    locked_ = true;
}

void IpcMem::unlock()
{
    assert(lock_ != nullptr);
    assert(locked_);
    int r = ipc_lock_release(lock_);
    if (r != 0) {
        LOG(ERROR, "ipc_lock_release failed (errno ", r, ")");
    }
    locked_ = false;
}

bool IpcMem::is_locked() const
{
    return locked_;
}

// Allocate/re-allocate the shared memory space of the data file.
// Return the current mmapped address if the given `bytes`
// is equal to or less than `total_bytes`.
void *IpcMem::alloc(size_t bytes)
{
    assert((bytes != 0) || !create_);
    if ((bytes != 0) && (bytes <= total_bytes_)) {
        // If `total_bytes` is zero, nullptr is returned.
        return addr_;
    }
    // Open the data file.
    string data_name_str =
        get_env().shm_name_prefix + name_ + NAME_DATA_POSTFIX;
    const char *data_name = data_name_str.c_str();
    int fd;
    if ((total_bytes_ == 0) && create_) {
        // Create an empty data file.
        fd = ipc_shm_create(data_name);
        if (fd == -1) {
            fd = ipc_shm_open(data_name);
            if (fd == -1) {
                LOG(ERROR, "ipc_shm_open: ", strerror(errno), " (", errno, ")");
            }
        }
    } else if (create_) {
        // Open the existing data file.
        fd = ipc_shm_open(data_name);
        if (fd == -1) {
            LOG(ERROR, "ipc_shm_open: ", strerror(errno), " (", errno, ")");
        }
    } else {
        // Wait until the data file appears.
        fd = ipc_shm_open_blocking(data_name);
        if (fd == -1) {
            LOG(ERROR, "ipc_shm_open_blocking: ", strerror(errno), " (", errno,
                ")");
        }
    }
    if (create_) {
        assert(bytes > 0);
        // Truncate the file size.
        int r = ftruncate(fd, bytes);
        if (r != 0) {
            LOG(ERROR, "ftruncate failed (errno ", r, ")");
        }
    } else {
        // Wait until the file size becomes equal to or larger than `bytes`.
        // We assume that in most cases the file size is already as large as
        // `bytes` or is going to be so very soon.
        // NOTE: this method cannot prevent race conditions if the data file
        // size may decrease.
        struct stat s;
        for (;;) {
            int r = fstat(fd, &s);
            if (r != 0) {
                LOG(ERROR, "fstat failed (errno ", r, ")");
            }
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
    void *old = addr_;
    addr_ = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    assert(addr_ != MAP_FAILED);
    // Remove the old mmap.
    if (old != nullptr) {
        munmap(old, total_bytes_);
    }
    close(fd);
    total_bytes_ = bytes;
    return addr_;
}

} // namespace ark
