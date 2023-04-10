// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#ifndef ARK_IPC_LOCK_H_
#define ARK_IPC_LOCK_H_

#include <pthread.h>

namespace ark {

// IPC lock.
struct IpcLock
{
    // Mutex lock.
    pthread_mutex_t mtx;
    // True if this lock is initialized.
    bool is_init = false;
};

// Initialize and acquire the lock immediately.
int ipc_lock_init(IpcLock *lock);
// Destroy the lock.
int ipc_lock_destroy(IpcLock *lock);
// Acquire the lock.
int ipc_lock_acquire(IpcLock *lock);
// Release the lock.
int ipc_lock_release(IpcLock *lock);

// IPC lock guard.
class IpcLockGuard
{
  public:
    // Constructor.
    IpcLockGuard(IpcLock *lock);
    // Desctructor.
    ~IpcLockGuard();

  private:
    // Lock to guard.
    IpcLock *lock;
};

} // namespace ark

#endif // ARK_IPC_LOCK_H_
