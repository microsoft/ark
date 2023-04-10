// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_IPC_COLL_H_
#define ARK_IPC_COLL_H_

#include "ark/ipc/ipc_mem.h"

namespace ark {

//
class IpcAllGather
{
  public:
    // Constructor.
    IpcAllGather(const std::string &name, int rank, int size, const void *addr,
                 std::size_t bytes);
    // Desctructor.
    ~IpcAllGather();
    //
    void sync();
    // Get data of the given rank.
    void *get_data(int rank_) const;

  private:
    IpcMem *mem;
    int rank;
    int size;
    // Per-rank bytes.
    std::size_t bytes;
};

} // namespace ark

#endif // ARK_IPC_COLL_H_
