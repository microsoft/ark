// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ipc/ipc_socket.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>

#include "include/ark.h"
#include "logging.h"

#define MAX_LISTEN_LEN 4096
#define MAX_ITEM_NAME_LEN 256

namespace ark {

IpcSocket::IpcSocket(const std::string &ip_, int port_, bool create_)
    : ip{ip_}, port{port_}, create{create_} {
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port_);
    addr.sin_addr.s_addr = inet_addr(ip_.c_str());
    this->sock_listen = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    int ret;
    ret = setsockopt(this->sock_listen, SOL_SOCKET, SO_REUSEADDR, &opt,
                     sizeof(int));
    if (ret != 0) {
        ERR(SystemError, "setsockopt: ", strerror(errno), " (", errno, ")");
    }
    int flags = fcntl(this->sock_listen, F_GETFL, 0);
    if (flags == -1) {
        ERR(SystemError, "fcntl: ", strerror(errno), " (", errno, ")");
    }
    if (fcntl(this->sock_listen, F_SETFL, flags | O_NONBLOCK) == -1) {
        ERR(SystemError, "fcntl: ", strerror(errno), " (", errno, ")");
    }
    LOG(DEBUG, "listen ", ip_, ":", port_);
    ret = bind(this->sock_listen, (struct sockaddr *)&addr, sizeof(addr));
    if (ret != 0) {
        ERR(SystemError, "bind(", ip_, ":", port_, "): ", strerror(errno), " (",
            errno, ")");
    }
    ret = listen(this->sock_listen, MAX_LISTEN_LEN);
    if (ret != 0) {
        ERR(SystemError, "listen: ", strerror(errno), " (", errno, ")");
    }
    this->run_server = true;
    this->server = std::thread([&] {
        int epfd = epoll_create1(0);
        if (epfd == -1) {
            ERR(SystemError, "epoll_create1: ", strerror(errno), " (", errno,
                ")");
        }
        struct epoll_event ev;
        ev.events = EPOLLIN;
        ev.data.fd = this->sock_listen;
        if (epoll_ctl(epfd, EPOLL_CTL_ADD, this->sock_listen, &ev) == -1) {
            ERR(SystemError, "epoll_ctl: ", strerror(errno), " (", errno, ")");
        }
        struct epoll_event events[MAX_LISTEN_LEN];
        while (this->run_server) {
            int num = epoll_wait(epfd, events, MAX_LISTEN_LEN, 100);
            if (num == -1) {
                if (errno != EINTR) {
                    ERR(SystemError, "epoll_wait: ", strerror(errno), " (",
                        errno, ")");
                }
            } else if (num > 0) {
                this->serve_item();
            }
        }
    });
}

IpcSocket::~IpcSocket() {
    this->run_server = false;
    if (this->server.joinable()) {
        this->server.join();
    }
    close(this->sock_listen);
    for (auto &item : this->items) {
        if (item.second.data != nullptr) {
            free(item.second.data);
        }
    }
}

IpcSocket::State IpcSocket::add_item(const std::string &name, const void *data,
                                     int size) {
    if (name.size() > MAX_ITEM_NAME_LEN) {
        ERR(InvalidUsageError, "name too long");
    }
    void *copy;
    if ((data == nullptr) || (size == 0)) {
        copy = nullptr;
    } else {
        copy = malloc(size);
        memcpy(copy, data, size);
    }
    struct Item item;
    item.data = copy;
    item.size = size;
    item.cnt = 0;
    this->items.emplace(name, item);
    return SUCCESS;
}

IpcSocket::State IpcSocket::query_item_internal(const std::string &ip, int port,
                                                const std::string &name,
                                                void *data, int size,
                                                bool block) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = inet_addr(ip.c_str());
    int ret;
    for (;;) {
        ret = connect(sock, (struct sockaddr *)&addr, sizeof(addr));
        if (ret == 0) {
            break;
        } else if (block) {
            if ((errno != EINTR) && (errno != ECONNREFUSED)) {
                ERR(SystemError, "connect: ", strerror(errno), " (", errno,
                    ")");
            }
            sched_yield();
        } else {
            close(sock);
            return CONNECT_FAILED;
        }
    }
    int flags = fcntl(sock, F_GETFL, 0);
    if (flags == -1) {
        ERR(SystemError, "fcntl: ", strerror(errno), " (", errno, ")");
    }
    if (fcntl(sock, F_SETFL, flags | O_NONBLOCK) == -1) {
        ERR(SystemError, "fcntl: ", strerror(errno), " (", errno, ")");
    }
    ret = this->send_all(sock, name.c_str(), name.size());
    if (ret != 0) {
        close(sock);
        return SEND_FAILED;
    }
    ret = this->recv_all(sock, data, size);
    if (ret < 0) {
        close(sock);
        return RECV_FAILED;
    } else if (ret == 0) {
        close(sock);
        return ITEM_NOT_FOUND;
    }
    close(sock);
    return SUCCESS;
}

IpcSocket::State IpcSocket::query_item(const std::string &ip, int port,
                                       const std::string &name, void *data,
                                       int size, bool block) {
    State s;
    for (;;) {
        s = query_item_internal(ip, port, name, data, size, block);
        if (!block || (s != ITEM_NOT_FOUND)) {
            break;
        }
        sched_yield();
    }
    return s;
}

IpcSocket::State IpcSocket::serve_item() {
    int sock = accept(this->sock_listen, NULL, NULL);
    if (sock < 0) {
        return ACCEPT_FAILED;
    }
    int flags = fcntl(sock, F_GETFL, 0);
    if (flags == -1) {
        ERR(SystemError, "fcntl: ", strerror(errno), " (", errno, ")");
    }
    if (fcntl(sock, F_SETFL, flags | O_NONBLOCK) == -1) {
        ERR(SystemError, "fcntl: ", strerror(errno), " (", errno, ")");
    }
    // TODO: need a better way to make sure the whole name is received
    char name[MAX_ITEM_NAME_LEN + 1];
    int ret = this->recv_try(sock, name, MAX_ITEM_NAME_LEN);
    if (ret < 0) {
        close(sock);
        return RECV_FAILED;
    }
    name[ret] = '\0';
    auto it = this->items.find(name);
    if (it != this->items.end()) {
        ret = this->send_all(sock, it->second.data, it->second.size);
        if (ret != 0) {
            close(sock);
            return SEND_FAILED;
        }
        it->second.cnt++;
    }
    close(sock);
    return SUCCESS;
}

const IpcSocket::Item *IpcSocket::get_item(const std::string &name) const {
    auto it = this->items.find(name);
    if (it != this->items.end()) {
        return &it->second;
    }
    return nullptr;
}

////////////////////////////////////////////////////////////////////////////////

int IpcSocket::send_all(int sock, const void *buf, int size) {
    int sent = 0;
    const char *ptr = (const char *)buf;
    while (sent < size) {
        int ret = send(sock, ptr + sent, size - sent, 0);
        if (ret < 0) {
            if ((errno == EINTR) || (errno == EAGAIN)) {
                continue;
            }
            return ret;
        }
        sent += ret;
    }
    return 0;
}

int IpcSocket::recv_try(int sock, void *buf, int size) {
    int cnt = 0;
    int received = 0;
    char *ptr = (char *)buf;
    while (received < size) {
        int ret = recv(sock, ptr + received, size - received, 0);
        if (ret < 0) {
            if ((errno == EINTR) || (errno == EAGAIN)) {
                if (++cnt > 10) {
                    // Avoid deadlocks
                    sched_yield();
                }
                continue;
            }
            return ret;
        } else if (ret < size - received) {
            received += ret;
            break;
        }
        received += ret;
    }
    return received;
}

int IpcSocket::recv_all(int sock, void *buf, int size) {
    int cnt = 0;
    int received = 0;
    char *ptr = (char *)buf;
    while (received < size) {
        int ret = recv(sock, ptr + received, size - received, 0);
        if (ret < 0) {
            if ((errno == EINTR) || (errno == EAGAIN)) {
                if (++cnt > 10) {
                    // Avoid deadlocks
                    sched_yield();
                }
                continue;
            }
            return ret;
        } else if (ret == 0) {
            return 0;
        }
        received += ret;
    }
    return received;
}

}  // namespace ark
