// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ipc_lock.h"

#include <cassert>

#include "logging.h"

namespace ark {

// Initialize the lock.
int ipc_lock_init(IpcLock *lock) {
    assert(lock != nullptr);
    int ret;
    pthread_mutexattr_t attr;
    ret = pthread_mutexattr_init(&attr);
    if (ret != 0) {
        return ret;
    }
    ret = pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    if (ret != 0) {
        return ret;
    }
    ret = pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
    if (ret != 0) {
        return ret;
    }
    ret = pthread_mutex_init(&(lock->mtx), &attr);
    if (ret != 0) {
        return ret;
    }
    // Acquire this lock before setting `is_init`.
    ret = pthread_mutex_lock(&(lock->mtx));
    if (ret != 0) {
        return ret;
    }
    lock->is_init = true;
    return 0;
}

// Destroy the lock.
int ipc_lock_destroy(IpcLock *lock) {
    assert(lock != nullptr);
    // Regardless of whether the destruction successes or not,
    // do not reuse this lock.
    lock->is_init = false;
    return pthread_mutex_destroy(&(lock->mtx));
}

// Acquire the lock.
int ipc_lock_acquire(IpcLock *lock) {
    assert(lock != nullptr);
    return pthread_mutex_lock(&(lock->mtx));
}

// Release the lock.
int ipc_lock_release(IpcLock *lock) {
    assert(lock != nullptr);
    return pthread_mutex_unlock(&(lock->mtx));
}

////////////////////////////////////////////////////////////////////////////////

// Constructor.
IpcLockGuard::IpcLockGuard(IpcLock *lock_) : lock{lock_} {
    assert(lock_ != nullptr);
    assert(lock->is_init == true);
    int r = ipc_lock_acquire(lock_);
    if (r != 0) {
        ERR(SystemError, "ipc_lock_acquire failed (errno ", r, ")");
    }
}

// Desctructor.
IpcLockGuard::~IpcLockGuard() { ipc_lock_release(this->lock); }

}  // namespace ark
