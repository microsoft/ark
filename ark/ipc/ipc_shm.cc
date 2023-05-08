// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/inotify.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ark/ipc/ipc_shm.h"
#include "ark/logging.h"

#define SHM_DIR "/dev/shm/"
#define SHM_MODE 0666

using namespace std;

namespace ark {

// Create a shm file if it does not exist, and truncate it to zero bytes.
// If the creation fails, return -1.
int ipc_shm_create(const char *name)
{
    return shm_open(name, O_RDWR | O_CREAT | O_EXCL, SHM_MODE);
}

// Try opening a shm file.
// Return its file descriptor on success, otherwise return -1.
int ipc_shm_open(const char *name)
{
    return shm_open(name, O_RDWR, SHM_MODE);
}

// Open a shm file and return its file descriptor.
// If opening fails due to non-existence, block until it can open.
// If opening fails due to any other reasons, return -1.
int ipc_shm_open_blocking(const char *name)
{
    // Monitor file creations in the shm directory.
    int ifd = inotify_init1(IN_NONBLOCK);
    if (ifd == -1) {
        LOGERR("inotify_init1: ", strerror(errno), " (", errno, ")");
    }
    if (inotify_add_watch(ifd, SHM_DIR, IN_CREATE) == -1) {
        close(ifd);
        LOGERR("inotify_add_watch: ", strerror(errno), " (", errno, ")");
    }
    // Check whether the file already exists.
    // NOTE: `inotify_add_watch()` should come before `shm_open()`
    // to avoid race conditions.
    int fd = shm_open(name, O_RDWR, SHM_MODE);
    if ((fd != -1) || (errno != ENOENT)) {
        close(ifd);
        return fd;
    }
    pollfd pfd;
    pfd.fd = ifd;
    pfd.events = POLLIN;
    char ibuf[1024] __attribute__((aligned(__alignof__(inotify_event))));
    for (;;) {
        int poll_num = poll(&pfd, 1, -1);
        if (poll_num == -1) {
            if (errno != EINTR) {
                close(ifd);
                LOGERR("poll: ", strerror(errno), " (", errno, ")");
            }
        } else if ((poll_num > 0) && (pfd.revents & POLLIN)) {
            // Read inotify events.
            for (;;) {
                ssize_t len = read(ifd, ibuf, sizeof(ibuf));
                if (len == -1) {
                    if (errno != EAGAIN) {
                        close(ifd);
                        LOGERR("read: ", strerror(errno), " (", errno, ")");
                    }
                    break;
                }
                char *p = ibuf;
                while (p < (ibuf + len)) {
                    const inotify_event *event = (const inotify_event *)p;
                    if (event->mask & IN_CREATE) {
                        if (strncmp(name, event->name, strlen(name)) == 0) {
                            close(ifd);
                            return shm_open(name, O_RDWR, SHM_MODE);
                        }
                    }
                    p += sizeof(inotify_event) + event->len;
                }
            }
        }
    }
    // Never reaches here.
    return -1;
}

// Destroy a shm file.
int ipc_shm_destroy(const char *name)
{
    return shm_unlink(name);
}

// Return zero if we can open the shm file.
int ipc_shm_exist(const char *name)
{
    int fd = shm_open(name, O_RDWR, SHM_MODE);
    if (fd != -1) {
        close(fd);
        return true;
    }
    return false;
}

} // namespace ark
