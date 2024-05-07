// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_TENSOR_HPP_
#define ARK_MODEL_TENSOR_HPP_

#include <set>
#include <vector>

#include "ark/dims.hpp"
#include "ark/model_ref.hpp"
#include "json.hpp"

namespace ark {

class ModelBuffer {
   public:
    // (remote_rank, tag)
    using TagInfo = std::pair<int, int>;

    ModelBuffer(int rank = -1);

    ModelBuffer(size_t id, int rank, const std::vector<TagInfo> &send_tags,
                const std::vector<TagInfo> &recv_tags);

    size_t id() const { return id_; }

    int rank() const { return rank_; }

    const std::set<TagInfo> &send_tags() const { return send_tags_; }

    const std::set<TagInfo> &recv_tags() const { return recv_tags_; }

    void tag_send(int remote_rank, int tag);

    void tag_recv(int remote_rank, int tag);

    nlohmann::ordered_json serialize() const;

    static std::shared_ptr<ModelBuffer> deserialize(const json &serialized);

   private:
    size_t id_;
    int rank_;
    std::set<TagInfo> send_tags_;
    std::set<TagInfo> recv_tags_;
};

class ModelDataT;
using ModelDataType = std::shared_ptr<ModelDataT>;

class ModelTensor {
   public:
    ModelTensor(ModelDataType data_type, ModelBufferRef buffer,
                const Dims &shape, const Dims &strides = {},
                const Dims &offsets = {}, const Dims &pads = {});

    ModelTensor(const ModelTensor &other);

    size_t id() const { return id_; }

    ModelDataType data_type() const { return data_type_; }

    ModelBufferRef buffer() const { return buffer_; }

    const Dims &shape() const { return shape_; }

    const Dims &strides() const { return strides_; }

    const Dims &offsets() const { return offsets_; }

    const Dims &pads() const { return pads_; }

    size_t shape_bytes() const;

    size_t strides_bytes() const;

    bool is_sequential() const;

    nlohmann::ordered_json serialize() const;

    static std::shared_ptr<ModelTensor> deserialize(const json &serialized);

   private:
    static size_t next_id();

    size_t id_;
    ModelDataType data_type_;
    ModelBufferRef buffer_;
    Dims shape_;
    Dims strides_;
    Dims offsets_;
    Dims pads_;
};

}  // namespace ark

#endif  // ARK_MODEL_TENSOR_HPP_
