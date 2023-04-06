#include <algorithm>
#include <cstring>

#include "ark/ipc/ipc_table.h"
#include "ark/logging.h"

using namespace std;

namespace ark {

IpcTable::IpcTable(const string &name_, int rank_, size_t elem_bytes_)
    : name{name_}, rank{rank_}, elem_bytes{elem_bytes_}
{
    IpcMem *key = new IpcMem{name_ + ".key" + to_string(rank_), true};
    IpcMem *val = new IpcMem{name_ + ".val" + to_string(rank_), true};
    this->key_storage.emplace_back(key);
    this->val_storage.emplace_back(val);
    this->keys.resize(rank_ + 1, nullptr);
    this->vals.resize(rank_ + 1, nullptr);
    this->maps.resize(rank_ + 1);
    this->keys[rank_] = key;
    this->vals[rank_] = val;
}

void *IpcTable::add_entry(int rank, const string &key)
{
    if (key.find(' ') != string::npos) {
        LOGERR("the key should not contain any blank character.");
    }
    if (rank == this->rank) {
        auto &rank_map = this->maps[rank];
        if (rank_map.find(key) != rank_map.end()) {
            return rank_map[key];
        }
        this->key_stream << key << ' ';
        size_t offset = this->total_val_bytes;
        this->total_val_bytes += this->elem_bytes;
        void *ptr = this->vals[rank]->alloc(this->total_val_bytes);
        void *ret = (char *)ptr + offset;
        this->maps[rank][key] = ret;
        return ret;
    } else {
        bool do_init = false;
        if (rank <= (int)this->keys.size()) {
            this->keys.resize(rank + 1, nullptr);
            this->vals.resize(rank + 1, nullptr);
            this->maps.resize(rank + 1);
            do_init = true;
        } else if (this->keys[rank] == nullptr) {
            do_init = true;
        }
        if (do_init) {
            IpcMem *key =
                new IpcMem{this->name + ".key" + to_string(rank), false};
            IpcMem *val =
                new IpcMem{this->name + ".val" + to_string(rank), false};
            this->key_storage.emplace_back(key);
            this->key_storage.emplace_back(val);
            this->keys[rank] = key;
            this->vals[rank] = val;
        }
        auto &rank_map = this->maps[rank];
        if (rank_map.find(key) != rank_map.end()) {
            return rank_map[key];
        }
        rank_map[key] = nullptr;
        return nullptr;
    }
}

//
void IpcTable::freeze()
{
    const string key_str = this->key_stream.str();
    const int len = (int)key_str.size();
    const char *key_c_str = key_str.c_str();
    char *pk = (char *)this->keys[this->rank]->alloc(len + 1);
    memcpy(pk, key_c_str, len);
    pk[len] = '\0';

    for (int i = 0; i < (int)this->keys.size(); ++i) {
        if ((i == this->rank) || (this->keys[i] == nullptr)) {
            continue;
        }
        this->import_rank_data(i);
    }
    this->freezed = true;
}

void *IpcTable::get_entry(int rank, const string &key)
{
    if (rank == this->rank) {
        if (!this->freezed) {
            return this->add_entry(rank, key);
        }
    } else {
        void *ret = this->add_entry(rank, key);
        if ((ret != nullptr) || !this->freezed) {
            return ret;
        }
        this->import_rank_data(rank);
    }
    auto &rank_map = this->maps[rank];
    if (rank_map.find(key) == rank_map.end()) {
        LOGERR("the entry does not exist: rank=", rank, ", key=", key);
    }
    return rank_map[key];
}

void IpcTable::import_rank_data(int rank)
{
    char *keys_ptr = (char *)this->keys[rank]->alloc(0);
    string keys_str = string{keys_ptr};
    size_t num_keys = count(keys_str.begin(), keys_str.end(), ' ');
    char *vals_ptr =
        (char *)this->vals[rank]->alloc(num_keys * this->elem_bytes);

    auto &rank_map = this->maps[rank];
    string::size_type pos_s = 0;
    string::size_type pos_e;
    for (size_t j = 0; j < num_keys; ++j) {
        pos_e = keys_str.find(' ', pos_s);
        rank_map[keys_str.substr(pos_s, pos_e - pos_s)] =
            (void *)(vals_ptr + j * this->elem_bytes);
        pos_s = pos_e + 1;
    }
}

} // namespace ark
