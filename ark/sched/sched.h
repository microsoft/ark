// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_SCHED_H_
#define ARK_SCHED_H_

#include "include/ark.h"
#include "sched/sched_codegen.h"
#include "sched/sched_opgraph.h"
#include "sched/sched_stream.h"

namespace ark {

struct BufInfo {
    // all the information of a GPU data buffer
    BufInfo(int gpu_id_, size_t bytes_, TensorBuf *tbuf_, int sid_,
            size_t offset_)
        : gpu_id{gpu_id_},
          bytes{bytes_},
          tbuf{tbuf_},
          sid{sid_},
          offset{offset_} {}
    // gpu_id: the id of the GPU where the buffer is allocated. If the
    // gpu_id is the same as this rank's gpu_id, the buffer is allocated on
    // the GPU, otherwise it will be imported from another GPU.
    int gpu_id;
    size_t bytes;
    TensorBuf *tbuf;
    // sid: a unique id of the buffer, used to identify the buffer when
    // we need to export or import the buffer. If the TensorBuf is located
    // on this GPU and the sid is not -1, it will be exported.
    int sid;
    size_t offset;
};

class BaseScheduler {
   public:
    BaseScheduler(Model &model, int gpu_id, int rank_, int world_size_,
                  int num_warps_per_sm_ = 16);

    // create context on gpu for the model
    std::shared_ptr<GpuContext> create_context();

    const OpConfig *sched_op_config(const Op *op);

    virtual void schedule() = 0;

    //
    virtual std::vector<std::string> gen_code() = 0;

   protected:
    Model *model;
    std::shared_ptr<GpuManager> gpu_mgr;
    int rank;
    int world_size;
    std::shared_ptr<GpuContext> ctx;
    int num_warps_per_sm;
    std::unique_ptr<CodeGenerator> codegen;

    std::vector<std::unique_ptr<SchedOpSeq>> opseqs;

    // the information of the GPU buffers
    std::vector<BufInfo> buf_infos;
};

class DefaultScheduler : public BaseScheduler {
   public:
    DefaultScheduler(Model &model, int gpu_id, int rank_, int world_size_,
                     int num_warps_per_sm = 16);

    std::vector<std::string> gen_code();
    void schedule();

   protected:
    void configure_gpu_buf(const std::list<Tensor *> &model_tensors);
    void heuristic_optimize_model(Model &model, Model::Impl *model_impl,
                                  const GpuManager::Info &gpu_info, int num_sm);
    void heuristic_optimize_matmul(Model &model, Model::Impl *model_impl,
                                   Op &matmul_op,
                                   const GpuManager::Info &gpu_info,
                                   int num_sm);

   private:
    void recursive_schedule(std::list<OpNode *> &nodes,
                            std::set<OpNode *> &seen_nodes);

    std::unique_ptr<OpGraph> op_graph;
    std::vector<std::unique_ptr<SchedStream>> comp_stream;
    std::vector<std::unique_ptr<SchedStream>> comm_stream;
};

}  // namespace ark

#endif  // ARK_SCHED_H_
