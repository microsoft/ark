// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#ifndef ARK_THREADING_H_
#define ARK_THREADING_H_

#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace ark {

template <typename ItemType>
void para_exec(std::vector<ItemType> &items, int max_num_threads,
               const std::function<void(ItemType &)> &func)
{
    size_t nthread = (size_t)max_num_threads;
    if (nthread > items.size()) {
        nthread = items.size();
    }
    std::vector<std::thread> threads;
    threads.reserve(nthread);
    std::mutex mtx;
    size_t idx = 0;
    for (size_t i = 0; i < nthread; ++i) {
        threads.emplace_back([&items, &mtx, &idx, &func] {
            size_t local_idx = -1;
            for (;;) {
                {
                    const std::lock_guard<std::mutex> lock(mtx);
                    local_idx = idx++;
                }
                if (local_idx >= items.size())
                    break;
                func(items[local_idx]);
            }
        });
    }
    for (auto &t : threads) {
        t.join();
    }
}

} // namespace ark

#endif // ARK_THREADING_H_