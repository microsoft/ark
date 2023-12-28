// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_IPC_IPC_SOCKET_H_
#define ARK_IPC_IPC_SOCKET_H_

#include <map>
#include <string>
#include <thread>

namespace ark {

class IpcSocket {
   public:
    typedef enum {
        SUCCESS = 0,
        ACCEPT_FAILED,
        CONNECT_FAILED,
        SEND_FAILED,
        RECV_FAILED,
        ITEM_NOT_FOUND,
    } State;

    struct Item {
        void *data;
        int size;
        int cnt;
    };

    IpcSocket(const std::string &ip_, int port_, bool create_ = true);
    ~IpcSocket();

    State add_item(const std::string &name, const void *data, int size);
    State query_item(const std::string &ip, int port, const std::string &name,
                     void *data, int size, bool block = false);
    State serve_item();

    const Item *get_item(const std::string &name) const;

   private:
    State query_item_internal(const std::string &ip, int port,
                              const std::string &name, void *data, int size,
                              bool block);

    int send_all(int sock, const void *buf, int size);
    int recv_try(int sock, void *buf, int size);
    int recv_all(int sock, void *buf, int size);

    const std::string ip;
    const int port;
    const bool create;

    int sock_listen;
    bool run_server;
    std::thread server;

    std::map<std::string, struct Item> items;
};

}  // namespace ark

#endif  // ARK_IPC_IPC_SOCKET_H_
