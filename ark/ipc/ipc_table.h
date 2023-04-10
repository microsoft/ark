#ifndef ARK_IPC_TABLE_H_
#define ARK_IPC_TABLE_H_

#include <list>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ark/ipc/ipc_mem.h"

namespace ark {

class IpcTable
{
  public:
    IpcTable(const std::string &name, int rank, size_t elem_bytes);
    void *add_entry(int rank, const std::string &key);
    void freeze();
    void *get_entry(int rank, const std::string &key);

  private:
    void import_rank_data(int rank);

    const std::string name;
    const int rank;
    const size_t elem_bytes;

    size_t total_val_bytes = 0;

    std::list<std::unique_ptr<IpcMem>> key_storage;
    std::list<std::unique_ptr<IpcMem>> val_storage;
    std::vector<IpcMem *> keys;
    std::vector<IpcMem *> vals;
    std::vector<std::map<std::string, void *>> maps;

    std::stringstream key_stream;
    bool freezed = false;
};

} // namespace ark

#endif // ARK_IPC_TABLE_H_
