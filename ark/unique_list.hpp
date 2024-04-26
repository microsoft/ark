// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_UNIQUE_LIST_HPP_
#define ARK_UNIQUE_LIST_HPP_

#include <cstddef>
#include <list>
#include <map>
#include <vector>

namespace ark {

template <typename T>
class UniqueList {
   private:
    std::list<T> list_;
    std::map<T, typename std::list<T>::iterator> index_;

   public:
    UniqueList() = default;

    explicit UniqueList(const std::vector<T> &vec) {
        for (const auto &value : vec) {
            push_back(value);
        }
    }

    UniqueList(const UniqueList &other) = default;

    UniqueList(UniqueList &&other) = default;

    UniqueList &operator=(const UniqueList &other) = default;

    UniqueList &operator=(UniqueList &&other) = default;

    const T &front() const { return list_.front(); }

    const T &back() const { return list_.back(); }

    const T &operator[](size_t idx) const {
        auto it = list_.begin();
        std::advance(it, idx);
        return *it;
    }

    void push_back(const T &value) {
        auto it = index_.find(value);
        if (it == index_.end()) {
            list_.push_back(value);
            index_[value] = --list_.end();
        }
    }

    void erase(const T &value) {
        auto it = index_.find(value);
        if (it != index_.end()) {
            list_.erase(it->second);
            index_.erase(it);
        }
    }

    void erase(typename std::list<T>::iterator it) {
        index_.erase(*it);
        list_.erase(it);
    }

    void clear() {
        list_.clear();
        index_.clear();
    }

    size_t index(const T &value) const {
        auto it = index_.find(value);
        return (it == index_.end())
                   ? -1
                   : std::distance(
                         list_.begin(),
                         static_cast<typename std::list<T>::const_iterator>(
                             it->second));
    }

    typename std::list<T>::iterator begin() { return list_.begin(); }

    typename std::list<T>::const_iterator begin() const {
        return list_.begin();
    }

    typename std::list<T>::iterator end() { return list_.end(); }

    typename std::list<T>::const_iterator end() const { return list_.end(); }

    typename std::list<T>::iterator find(const T &value) {
        auto it = index_.find(value);
        return (it == index_.end()) ? end() : it->second;
    }

    typename std::list<T>::const_iterator find(const T &value) const {
        auto it = index_.find(value);
        return (it == index_.end()) ? end() : it->second;
    }

    bool empty() const { return list_.empty(); }

    bool contains(const T &value) const {
        return index_.find(value) != index_.end();
    }

    size_t size() const { return index_.size(); }
};

}  // namespace ark

#endif  // ARK_UNIQUE_LIST_HPP_
