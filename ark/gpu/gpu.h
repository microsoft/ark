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

ARK_GPU_DEFINE_TYPE_ALIAS(gpuError, CUresult);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuDevice, CUdevice);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuCtx, CUcontext);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuModule, CUmodule);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuFunction, CUfunction);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuDeviceptr, CUdeviceptr);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuStream, CUstream);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuEvent, CUevent);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuJitOption, CUjit_option);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuFunctionAttribute, CUfunction_attribute);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuIpcMemHandle, CUipcMemHandle);

ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuSuccess, CUDA_SUCCESS);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuErrorNotFound, CUDA_ERROR_NOT_FOUND);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuErrorNotReady, CUDA_ERROR_NOT_READY);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuErrorNotInitialized,
                              CUDA_ERROR_NOT_INITIALIZED);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuErrorDeinitialized, CUDA_ERROR_DEINITIALIZED);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuErrorPeerAccessUnsupported,
                              CUDA_ERROR_PEER_ACCESS_UNSUPPORTED);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeComputeCapabilityMajor,
                              CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeComputeCapabilityMinor,
                              CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeMaxThreadsPerBlock,
                              CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeMultiprocessorCount,
                              CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
ARK_GPU_DEFINE_CONSTANT_ALIAS(
    gpuDeviceAttributeMaxSharedMemoryPerMultiprocessor,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR);
ARK_GPU_DEFINE_CONSTANT_ALIAS(
    gpuDeviceAttributeSharedMemPerBlockOptin,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeClockRate,
                              CU_DEVICE_ATTRIBUTE_CLOCK_RATE);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeWarpSize,
                              CU_DEVICE_ATTRIBUTE_WARP_SIZE);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributeMaxRegistersPerBlock,
                              CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributePciDomainID,
                              CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributePciBusId,
                              CU_DEVICE_ATTRIBUTE_PCI_BUS_ID);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuDeviceAttributePciDeviceId,
                              CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuCtxMapHost, CU_CTX_MAP_HOST);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuHostAllocMapped, CU_MEMHOSTALLOC_DEVICEMAP);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuHostAllocWriteCombined,
                              CU_MEMHOSTALLOC_WRITECOMBINED);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuHostRegisterMapped,
                              CU_MEMHOSTREGISTER_DEVICEMAP);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuJitInfoLogBuffer, CU_JIT_INFO_LOG_BUFFER);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuJitInfoLogBufferSizeBytes,
                              CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuJitErrorLogBuffer, CU_JIT_ERROR_LOG_BUFFER);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuJitErrorLogBufferSizeBytes,
                              CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuJitGenerateDebugInfo,
                              CU_JIT_GENERATE_DEBUG_INFO);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuFuncAttributeSharedSizeBytes,
                              CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuFuncAttributeMaxDynamicSharedSizeBytes,
                              CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuPointerAttributeSyncMemops,
                              CU_POINTER_ATTRIBUTE_SYNC_MEMOPS);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuStreamNonBlocking, CU_STREAM_NON_BLOCKING);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuEventDisableTiming, CU_EVENT_DISABLE_TIMING);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuIpcMemLazyEnablePeerAccess,
                              CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);

ARK_GPU_DEFINE_FUNC_ALIAS(gpuInit, cuInit);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuGetErrorString, cuGetErrorString);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuDeviceGet, cuDeviceGet);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuDeviceGetCount, cuDeviceGetCount);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuDeviceGetAttribute, cuDeviceGetAttribute);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuDevicePrimaryCtxRetain, cuDevicePrimaryCtxRetain);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuDevicePrimaryCtxRelease,
                          cuDevicePrimaryCtxRelease);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuDeviceSynchronize, cuCtxSynchronize);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuCtxCreate, cuCtxCreate);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuCtxDestroy, cuCtxDestroy);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuCtxSetCurrent, cuCtxSetCurrent);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuModuleLoadData, cuModuleLoadData);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuModuleLoadDataEx, cuModuleLoadDataEx);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuModuleGetFunction, cuModuleGetFunction);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuModuleGetGlobal, cuModuleGetGlobal);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuModuleLaunchKernel, cuLaunchKernel);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuFuncGetAttribute, cuFuncGetAttribute);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuFuncSetAttribute, cuFuncSetAttribute);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuHostAlloc, cuMemHostAlloc);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuHostFree, cuMemFreeHost);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuHostRegister, cuMemHostRegister);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuHostUnregister, cuMemHostUnregister);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuHostGetDevicePointer, cuMemHostGetDevicePointer);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemAlloc, cuMemAlloc);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemFree, cuMemFree);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemGetInfo, cuMemGetInfo);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemGetAddressRange, cuMemGetAddressRange);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemcpyDtoD, cuMemcpyDtoD);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemcpyDtoDAsync, cuMemcpyDtoDAsync);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemcpyHtoD, cuMemcpyHtoD);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemcpyHtoDAsync, cuMemcpyHtoDAsync);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemcpyDtoH, cuMemcpyDtoH);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemcpyDtoHAsync, cuMemcpyDtoHAsync);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemsetD32, cuMemsetD32);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemsetD32Async, cuMemsetD32Async);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemsetD8Async, cuMemsetD8Async);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuPointerSetAttribute, cuPointerSetAttribute);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuStreamCreate, cuStreamCreate);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuStreamDestroy, cuStreamDestroy);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuStreamQuery, cuStreamQuery);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuStreamSynchronize, cuStreamSynchronize);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuEventCreate, cuEventCreate);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuEventDestroy, cuEventDestroy);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuEventRecord, cuEventRecord);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuEventElapsedTime, cuEventElapsedTime);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuIpcGetMemHandle, cuIpcGetMemHandle);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuIpcOpenMemHandle, cuIpcOpenMemHandle);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuIpcCloseMemHandle, cuIpcCloseMemHandle);

#elif defined(ARK_ROCM)

ARK_GPU_DEFINE_TYPE_ALIAS(gpuError, hipError_t);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuDevice, hipDevice_t);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuCtx, hipCtx_t);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuModule, hipModule_t);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuFunction, hipFunction_t);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuDeviceptr, hipDeviceptr_t);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuStream, hipStream_t);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuEvent, hipEvent_t);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuJitOption, hipJitOption);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuFunctionAttribute, hipFunction_attribute);
ARK_GPU_DEFINE_TYPE_ALIAS(gpuIpcMemHandle, hipIpcMemHandle_t);

ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuSuccess, hipSuccess);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuErrorNotFound, hipErrorNotFound);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuErrorNotReady, hipErrorNotReady);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuErrorDeinitialized, hipErrorDeinitialized);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuErrorNotInitialized, hipErrorNotInitialized);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuErrorPeerAccessUnsupported,
                              hipErrorPeerAccessUnsupported);
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
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuCtxMapHost, hipDeviceMapHost);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuHostRegisterMapped, hipHostRegisterMapped);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuHostAllocMapped, hipHostMallocMapped);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuHostAllocWriteCombined,
                              hipHostMallocWriteCombined);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuJitInfoLogBuffer, hipJitOptionInfoLogBuffer);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuJitInfoLogBufferSizeBytes,
                              hipJitOptionInfoLogBufferSizeBytes);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuJitErrorLogBuffer, hipJitOptionErrorLogBuffer);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuJitErrorLogBufferSizeBytes,
                              hipJitOptionErrorLogBufferSizeBytes);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuJitGenerateDebugInfo,
                              hipJitOptionGenerateDebugInfo);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuFuncAttributeSharedSizeBytes,
                              HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuFuncAttributeMaxDynamicSharedSizeBytes,
                              hipFuncAttributeMaxDynamicSharedMemorySize);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuPointerAttributeSyncMemops,
                              HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuStreamNonBlocking, hipStreamNonBlocking);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuEventDisableTiming, hipEventDisableTiming);
ARK_GPU_DEFINE_CONSTANT_ALIAS(gpuIpcMemLazyEnablePeerAccess,
                              hipIpcMemLazyEnablePeerAccess);

ARK_GPU_DEFINE_FUNC_ALIAS(gpuInit, hipInit);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuGetErrorString, hipDrvGetErrorString);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuDeviceGet, hipDeviceGet);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuDeviceGetCount, hipGetDeviceCount);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuDeviceGetAttribute, hipDeviceGetAttribute);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuDevicePrimaryCtxRetain, hipDevicePrimaryCtxRetain);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuDevicePrimaryCtxRelease,
                          hipDevicePrimaryCtxRelease);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuDeviceSynchronize, hipDeviceSynchronize);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuCtxCreate, hipCtxCreate);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuCtxDestroy, hipCtxDestroy);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuCtxSetCurrent, hipCtxSetCurrent);
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
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemAlloc, hipMalloc);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuMemFree, hipFree);
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
ARK_GPU_DEFINE_FUNC_ALIAS(gpuStreamCreate, hipStreamCreateWithFlags);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuStreamDestroy, hipStreamDestroy);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuStreamQuery, hipStreamQuery);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuStreamSynchronize, hipStreamSynchronize);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuEventCreate, hipEventCreateWithFlags);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuEventDestroy, hipEventDestroy);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuEventRecord, hipEventRecord);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuEventElapsedTime, hipEventElapsedTime);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuIpcGetMemHandle, hipIpcGetMemHandle);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuIpcOpenMemHandle, hipIpcOpenMemHandle);
ARK_GPU_DEFINE_FUNC_ALIAS(gpuIpcCloseMemHandle, hipIpcCloseMemHandle);

#endif

}  // namespace ark

#endif  // ARK_GPU_H_
