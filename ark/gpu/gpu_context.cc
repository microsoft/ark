#include "gpu/gpu_context.h"

#include "gpu/gpu_manager.h"

namespace ark {
class GpuContext::Impl {
   public:
    Impl() {}
    ~Impl() {}
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    void export_memory(void *ptr, std::size_t bytes, int eid) {}
    void import_memory(void *ptr, std::size_t bytes, int eid) {}
    void freeze(bool wait) {}

   private:
    std::shared_ptr<GpuManager> manager_;
};

}  // namespace ark
