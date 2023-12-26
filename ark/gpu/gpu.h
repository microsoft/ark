// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_H_
#define ARK_GPU_H_

#include <functional>

#if (!defined(ARK_CUDA) && !defined(ARK_ROCM))
#error "ARK_CUDA or ARK_ROCM must be defined"
#define ARK_CUDA  // dummy
#endif            // !defined(ARK_CUDA) && !defined(ARK_ROCM)

#if defined(ARK_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#elif defined(ARK_ROCM)
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#endif

#define ARK_GPU_DEFINE_TYPE_ALIAS(alias, type) typedef type alias;

#define ARK_GPU_DEFINE_CONSTANT_ALIAS(alias, constant) \
    constexpr auto alias = constant;

#define ARK_GPU_DEFINE_FUNC_ALIAS(alias, func)    \
    template <typename... Args>                   \
    inline auto alias(Args &&... args) {          \
        return func(std::forward<Args>(args)...); \
    }

namespace ark {

#if defined(ARK_CUDA)

ARK_GPU_DEFINE_TYPE_ALIAS(gpuError, cudaError);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuStream, cudaStream_t);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuEvent, cudaEvent_t);

ARK_GPU_DEFINE_TYPE_ALIAS(gpuDrvError, CUresult);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuModule, CUmodule);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuFunction, CUfunction);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuDeviceptr, CUdeviceptr);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuFunctionAttribute, CUfunction_attribute);

// runtime API
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuSuccess, cudaSuccess);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuErrorNotReady, cudaErrorNotReady);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeComputeCapabilityMajor,
                              cudaDevAttrComputeCapabilityMajor);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeComputeCapabilityMinor,
                              cudaDevAttrComputeCapabilityMinor);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeMaxThreadsPerBlock,
                              cudaDevAttrMaxThreadsPerBlock);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeMultiprocessorCount,
                              cudaDevAttrMultiProcessorCount);
ARK_GPU_DEFINE_CONSTANT_ALIAS(
    gpuDeviceAttributeMaxSharedMemoryPerMultiprocessor,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeSharedMemPerBlockOptin,
                              cudaDevAttrMaxSharedMemoryPerBlockOptin);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeClockRate,
                              cudaDevAttrClockRate);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeWarpSize, cudaDevAttrWarpSize);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeMaxRegistersPerBlock,
                              cudaDevAttrMaxRegistersPerBlock);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributePciDomainID,
                              cudaDevAttrPciDomainId);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributePciBusId, cudaDevAttrPciBusId);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributePciDeviceId,
                              cudaDevAttrPciDeviceId);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuEventDisableTiming, cudaEventDisableTiming);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuHostAllocMapped, cudaHostAllocMapped);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuHostAllocWriteCombined,
                              cudaHostAllocWriteCombined);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuHostRegisterMapped, cudaHostRegisterMapped);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuMemcpyDeviceToHost, cudaMemcpyDeviceToHost);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuMemcpyDeviceToDevice,
                              cudaMemcpyDeviceToDevice);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuMemcpyHostToDevice, cudaMemcpyHostToDevice);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuStreamNonBlocking, cudaStreamNonBlocking);

// driver API
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDrvSuccess, CUDA_SUCCESS);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuErrorNotFound, CUDA_ERROR_NOT_FOUND);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuFuncAttributeSharedSizeBytes,
                              CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuFuncAttributeMaxDynamicSharedSizeBytes,
                              CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuPointerAttributeSyncMemops,
                              CU_POINTER_ATTRIBUTE_SYNC_MEMOPS);

// runtime API
ARK_GPU_DEFINE_FUNC_ALIAS(gpuGetErrorString, cudaGetErrorString);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuDeviceSynchronize, cudaDeviceSynchronize);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuHostAlloc, cudaHostAlloc);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuHostFree, cudaFreeHost);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuHostRegister, cudaHostRegister);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuHostUnregister, cudaHostUnregister);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuHostGetDevicePointer, cuMemHostGetDevicePointer);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMalloc, cudaMalloc);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuFree, cudaFree);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemGetInfo, cudaMemGetInfo);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemcpy, cudaMemcpy);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemcpyAsync, cudaMemcpyAsync);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemsetAsync, cudaMemsetAsync);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuSetDevice, cudaSetDevice);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuStreamCreateWithFlags, cudaStreamCreateWithFlags);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuStreamDestroy, cudaStreamDestroy);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuStreamQuery, cudaStreamQuery);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuStreamSynchronize, cudaStreamSynchronize);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuEventCreate, cudaEventCreate);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuEventDestroy, cudaEventDestroy);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuEventRecord, cudaEventRecord);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuEventElapsedTime, cudaEventElapsedTime);

// driver API
ARK_GPU_DEFINE_FUNC_ALIAS(gpuDeviceGetAttribute, cudaDeviceGetAttribute);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuDrvGetErrorString, cuGetErrorString);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuModuleLoadData, cuModuleLoadData);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuModuleLoadDataEx, cuModuleLoadDataEx);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuModuleGetFunction, cuModuleGetFunction);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuModuleGetGlobal, cuModuleGetGlobal);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuModuleLaunchKernel, cuLaunchKernel);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuFuncGetAttribute, cuFuncGetAttribute);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuFuncSetAttribute, cuFuncSetAttribute);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemsetD32Async, cuMemsetD32Async);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemGetAddressRange, cuMemGetAddressRange);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuPointerSetAttribute, cuPointerSetAttribute);


#elif defined(ARK_ROCM)

ARK_GPU_DEFINE_TYPE_ALIAS(gpuError, hipError_t);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuCtx, hipCtx_t);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuModule, hipModule_t);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuFunction, hipFunction_t);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuDeviceptr, hipDeviceptr_t);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuStream, hipStream_t);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuEvent, hipEvent_t);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuFunctionAttribute, hipFunction_attribute);

ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuSuccess, hipSuccess);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuErrorNotFound, hipErrorNotFound);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuErrorNotReady, hipErrorNotReady);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeComputeCapabilityMajor,
                              hipDeviceAttributeComputeCapabilityMajor);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeComputeCapabilityMinor,
                              hipDeviceAttributeComputeCapabilityMinor);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeMaxThreadsPerBlock,
                              hipDeviceAttributeMaxThreadsPerBlock);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeMultiprocessorCount,
                              hipDeviceAttributeMultiprocessorCount);
ARK_GPU_DEFINE_CONSTANT_ALIAS(
    gpuDeviceAttributeMaxSharedMemoryPerMultiprocessor,
    hipDeviceAttributeMaxSharedMemoryPerMultiprocessor);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeSharedMemPerBlockOptin,
                              hipDeviceAttributeMaxSharedMemoryPerBlock);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeClockRate,
                              hipDeviceAttributeClockRate);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeWarpSize,
                              hipDeviceAttributeWarpSize);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeMaxRegistersPerBlock,
                              hipDeviceAttributeMaxRegistersPerBlock);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributePciDomainID,
                              hipDeviceAttributePciDomainID);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributePciBusId,
                              hipDeviceAttributePciBusId);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributePciDeviceId,
                              hipDeviceAttributePciDeviceId);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuHostRegisterMapped, hipHostRegisterMapped);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuHostAllocMapped, hipHostMallocMapped);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuHostAllocWriteCombined,
                              hipHostMallocWriteCombined);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuFuncAttributeSharedSizeBytes,
                              HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuFuncAttributeMaxDynamicSharedSizeBytes,
                              hipFuncAttributeMaxDynamicSharedMemorySize);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuPointerAttributeSyncMemops,
                              HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuStreamNonBlocking, hipStreamNonBlocking);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuEventDisableTiming, hipEventDisableTiming);

ARK_GPU_DEFINE_FUNC_ALIAS(gpuGetErrorString, hipDrvGetErrorString);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuDeviceGet, hipDeviceGet);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuDeviceGetCount, hipGetDeviceCount);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuDeviceGetAttribute, hipDeviceGetAttribute);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuDeviceSynchronize, hipDeviceSynchronize);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuModuleLoadData, hipModuleLoadData);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuModuleLoadDataEx, hipModuleLoadDataEx);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuModuleGetFunction, hipModuleGetFunction);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuModuleGetGlobal, hipModuleGetGlobal);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuModuleLaunchKernel, hipModuleLaunchKernel);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuFuncGetAttribute, hipFuncGetAttribute);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuFuncSetAttribute, hipFuncSetAttribute);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuHostAlloc, hipHostMalloc);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuHostFree, hipHostFree);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuHostRegister, hipHostRegister);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuHostUnregister, hipHostUnregister);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuHostGetDevicePointer, hipHostGetDevicePointer);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMalloc, hipMalloc);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuFree, hipFree);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemGetInfo, hipMemGetInfo);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemGetAddressRange, hipMemGetAddressRange);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemcpyDtoD, hipMemcpyDtoD);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemcpyDtoDAsync, hipMemcpyDtoDAsync);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemcpyHtoD, hipMemcpyHtoD);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemcpyHtoDAsync, hipMemcpyHtoDAsync);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemcpyDtoH, hipMemcpyDtoH);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemcpyDtoHAsync, hipMemcpyDtoHAsync);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemsetD32, hipMemsetD32);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemsetD32Async, hipMemsetD32Async);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemsetD8Async, hipMemsetD8Async);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuPointerSetAttribute, hipPointerSetAttribute);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuSetDevice, hipSetDevice);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuStreamCreateWithFlags, hipStreamCreateWithFlags);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuStreamDestroy, hipStreamDestroy);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuStreamQuery, hipStreamQuery);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuStreamSynchronize, hipStreamSynchronize);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuEventCreate, hipEventCreateWithFlags);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuEventDestroy, hipEventDestroy);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuEventRecord, hipEventRecord);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuEventElapsedTime, hipEventElapsedTime);

#endif

}  // namespace ark

#endif  // ARK_GPU_H_
