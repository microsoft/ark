#ifndef ARK_EXECUTOR_H_
#define ARK_EXECUTOR_H_

#include <string>

#include "ark/gpu/gpu_kernel.h"
#include "ark/model.h"
#include "ark/sched/sched.h"

namespace ark {

// Convenience class for executing a model.
class Executor
{
  public:
    Executor(const int gpu_id_, int rank_, int world_size_, const Model &model,
             const std::string &name);
    ~Executor();

    void compile();
    void launch();
    void run(int iter);
    float stop();
    GpuBuf *get_gpu_buf(Tensor *tns) const;
    Tensor *get_tensor(Tensor *tns) const;
    void tensor_memcpy(Tensor *tns, const void *src, size_t bytes);
    void tensor_memcpy(void *dst, Tensor *src, size_t bytes);
    void tensor_clear(Tensor *tns);

  private:
    const int gpu_id;
    const int rank;
    const int world_size;
    GpuMgrCtx *ctx;
    SchedulerBase *sched;
    GpuLoopKernel *glk = nullptr;
    GpuStream stream = nullptr;
};

} // namespace ark

#endif // ARK_EXECUTOR_H_
