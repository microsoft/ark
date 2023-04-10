#include <cstring>

#include "ark/cpu_timer.h"
#include "ark/ipc/ipc_coll.h"
#include "ark/logging.h"
namespace ark {

// Constructor.
IpcAllGather::IpcAllGather(const std::string &name_, int rank_, int size_,
                           const void *addr_, std::size_t bytes_)
    : rank{rank_}, size{size_}, bytes{bytes_}
{
    this->mem = new IpcMem{name_, false, true};
    char *ptr = (char *)this->mem->alloc(bytes_ * size_ + 8);
    if (addr_ != nullptr) {
        void *data = ptr + rank_ * bytes_;
        std::memcpy(data, addr_, bytes_);
    }
}

// Desctructor.
IpcAllGather::~IpcAllGather()
{
    delete this->mem;
}

void IpcAllGather::sync()
{
    char *ptr = (char *)this->mem->get_addr();
    volatile int *cnt = (volatile int *)(ptr + this->bytes * this->size);
    volatile int *flag = cnt + 1;
    int is_dec;
    {
        IpcLockGuard lg{this->mem->get_lock()};
        int old = *cnt;
        is_dec = *flag;
        if (is_dec) {
            *cnt = old - 1;
            if (old == 1) {
                *flag = 0;
            }
        } else {
            *cnt = old + 1;
            if (old == this->size - 1) {
                *flag = 1;
            }
        }
    }
    if (is_dec) {
        while (*flag == 1) {
            cpu_ntimer_sleep(1e6);
        }
    } else {
        while (*flag == 0) {
            cpu_ntimer_sleep(1e6);
        }
    }
}

void *IpcAllGather::get_data(int rank_) const
{
    return (char *)this->mem->get_addr() + rank_ * this->bytes;
}

} // namespace ark
