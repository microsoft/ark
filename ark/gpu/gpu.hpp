// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_HPP_
#define ARK_GPU_HPP_

#include <functional>

#if (!defined(ARK_CUDA) && !defined(ARK_ROCM))
#error "ARK_CUDA or ARK_ROCM must be defined"
#define ARK_CUDA  // dummy
#endif            // !defined(ARK_CUDA) && !defined(ARK_ROCM)

#if defined(ARK_CUDA)

#include <cuda.h>
#include <cuda_runtime.h>
#define ARK_GPU_DEFINE_TYPE_ALIAS(alias, cuda_type, rocm_type) \
    using alias = cuda_type;
#define ARK_GPU_DEFINE_CONSTANT_ALIAS(alias, cuda_const, rocm_const) \
    constexpr auto alias = cuda_const;
#define ARK_GPU_DEFINE_FUNC_ALIAS(alias, cuda_func, rocm_func) \
    template <typename... Args>                                \
    inline auto alias(Args &&... args) {                       \
        return cuda_func(std::forward<Args>(args)...);         \
    }

#elif defined(ARK_ROCM)

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#define ARK_GPU_DEFINE_TYPE_ALIAS(alias, cuda_type, rocm_type) \
    using alias = rocm_type;
#define ARK_GPU_DEFINE_CONSTANT_ALIAS(alias, cuda_const, rocm_const) \
    constexpr auto alias = rocm_const;
#define ARK_GPU_DEFINE_FUNC_ALIAS(alias, cuda_func, rocm_func) \
    template <typename... Args>                                \
    inline auto alias(Args &&... args) {                       \
        return rocm_func(std::forward<Args>(args)...);         \
    }

#endif  // defined(ARK_ROCM)

namespace ark {

ARK_GPU_DEFINE_TYPE_ALIAS(gpuError, cudaError_t, hipError_t);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuEvent, cudaEvent_t, hipEvent_t);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuStream, cudaStream_t, hipStream_t);

ARK_GPU_DEFINE_TYPE_ALIAS(gpuDeviceptr, CUdeviceptr, hipDeviceptr_t);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuDrvError, CUresult, hipError_t);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuModule, CUmodule, hipModule_t);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuFunction, CUfunction, hipFunction_t);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuFunctionAttribute, CUfunction_attribute,
                          hipFunction_attribute);

// runtime API
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuSuccess, cudaSuccess, hipSuccess);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuErrorNotReady, cudaErrorNotReady,
                              hipErrorNotReady);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeComputeCapabilityMajor,
                              cudaDevAttrComputeCapabilityMajor,
                              hipDeviceAttributeComputeCapabilityMajor);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeComputeCapabilityMinor,
                              cudaDevAttrComputeCapabilityMinor,
                              hipDeviceAttributeComputeCapabilityMinor);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeMaxThreadsPerBlock,
                              cudaDevAttrMaxThreadsPerBlock,
                              hipDeviceAttributeMaxThreadsPerBlock);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeMultiprocessorCount,
                              cudaDevAttrMultiProcessorCount,
                              hipDeviceAttributeMultiprocessorCount);
ARK_GPU_DEFINE_CONSTANT_ALIAS(
    gpuDeviceAttributeMaxSharedMemoryPerMultiprocessor,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor,
    hipDeviceAttributeMaxSharedMemoryPerMultiprocessor);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeSharedMemPerBlockOptin,
                              cudaDevAttrMaxSharedMemoryPerBlockOptin,
                              hipDeviceAttributeMaxSharedMemoryPerBlock);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeClockRate, cudaDevAttrClockRate,
                              hipDeviceAttributeClockRate);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeWarpSize, cudaDevAttrWarpSize,
                              hipDeviceAttributeWarpSize);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeMaxRegistersPerBlock,
                              cudaDevAttrMaxRegistersPerBlock,
                              hipDeviceAttributeMaxRegistersPerBlock);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributePciDomainID,
                              cudaDevAttrPciDomainId,
                              hipDeviceAttributePciDomainID);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributePciBusId, cudaDevAttrPciBusId,
                              hipDeviceAttributePciBusId);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributePciDeviceId,
                              cudaDevAttrPciDeviceId,
                              hipDeviceAttributePciDeviceId);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuEventDisableTiming, cudaEventDisableTiming,
                              hipEventDisableTiming);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuHostAllocMapped, cudaHostAllocMapped,
                              hipHostMallocMapped);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuHostAllocWriteCombined,
                              cudaHostAllocWriteCombined,
                              hipHostMallocWriteCombined);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuMemcpyDeviceToHost, cudaMemcpyDeviceToHost,
                              hipMemcpyDeviceToHost);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuMemcpyDeviceToDevice, cudaMemcpyDeviceToDevice,
                              hipMemcpyDeviceToDevice);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuMemcpyHostToDevice, cudaMemcpyHostToDevice,
                              hipMemcpyHostToDevice);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuStreamNonBlocking, cudaStreamNonBlocking,
                              hipStreamNonBlocking);

// driver API
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDrvSuccess, CUDA_SUCCESS, hipSuccess);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuErrorNotFound, CUDA_ERROR_NOT_FOUND,
                              hipErrorNotFound);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuFuncAttributeSharedSizeBytes,
                              CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                              HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuFuncAttributeMaxDynamicSharedSizeBytes,
                              CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                              hipFuncAttributeMaxDynamicSharedMemorySize);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuPointerAttributeSyncMemops,
                              CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                              HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS);

// runtime API
ARK_GPU_DEFINE_FUNC_ALIAS(gpuGetErrorString, cudaGetErrorString,
                          hipGetErrorString);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuGetLastError, cudaGetLastError, hipGetLastError);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuDeviceGetAttribute, cudaDeviceGetAttribute,
                          hipDeviceGetAttribute);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuDeviceSynchronize, cudaDeviceSynchronize,
                          hipDeviceSynchronize);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuHostAlloc, cudaHostAlloc, hipHostMalloc);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuHostFree, cudaFreeHost, hipHostFree);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuHostRegister, cudaHostRegister, hipHostRegister);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuHostUnregister, cudaHostUnregister,
                          hipHostUnregister);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuHostGetDevicePointer, cuMemHostGetDevicePointer,
                          hipHostGetDevicePointer);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMalloc, cudaMalloc, hipMalloc);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuFree, cudaFree, hipFree);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemGetInfo, cudaMemGetInfo, hipMemGetInfo);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemcpy, cudaMemcpy, hipMemcpy);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemcpyAsync, cudaMemcpyAsync, hipMemcpyAsync);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemsetAsync, cudaMemsetAsync, hipMemsetAsync);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuSetDevice, cudaSetDevice, hipSetDevice);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuStreamCreateWithFlags, cudaStreamCreateWithFlags,
                          hipStreamCreateWithFlags);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuStreamDestroy, cudaStreamDestroy,
                          hipStreamDestroy);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuStreamQuery, cudaStreamQuery, hipStreamQuery);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuStreamSynchronize, cudaStreamSynchronize,
                          hipStreamSynchronize);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuEventCreateWithFlags, cudaEventCreateWithFlags,
                          hipEventCreateWithFlags);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuEventDestroy, cudaEventDestroy, hipEventDestroy);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuEventRecord, cudaEventRecord, hipEventRecord);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuEventElapsedTime, cudaEventElapsedTime,
                          hipEventElapsedTime);

// driver API
ARK_GPU_DEFINE_FUNC_ALIAS(gpuDrvGetErrorString, cuGetErrorString,
                          hipDrvGetErrorString);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuModuleLoadData, cuModuleLoadData,
                          hipModuleLoadData);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuModuleLoadDataEx, cuModuleLoadDataEx,
                          hipModuleLoadDataEx);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuModuleGetFunction, cuModuleGetFunction,
                          hipModuleGetFunction);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuModuleGetGlobal, cuModuleGetGlobal,
                          hipModuleGetGlobal);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuModuleLaunchKernel, cuLaunchKernel,
                          hipModuleLaunchKernel);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuFuncGetAttribute, cuFuncGetAttribute,
                          hipFuncGetAttribute);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuFuncSetAttribute, cuFuncSetAttribute,
                          hipFuncSetAttribute);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemsetD32Async, cuMemsetD32Async,
                          hipMemsetD32Async);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemGetAddressRange, cuMemGetAddressRange,
                          hipMemGetAddressRange);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuPointerSetAttribute, cuPointerSetAttribute,
                          hipPointerSetAttribute);

}  // namespace ark

#endif  // ARK_GPU_HPP_
