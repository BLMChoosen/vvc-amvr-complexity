/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2026, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "CudaRuntime.h"

#include <cuda_runtime.h>

#include <memory>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace vtm::cuda_backend
{

struct RuntimeContext
{
  int                         device = -1;
  std::array<cudaStream_t, 3> streams{};
  std::array<cudaEvent_t, 3>  fences{};
  cudaMemPool_t               memoryPool = nullptr;
  std::uint64_t              *distortionResultsDevice = nullptr;
  std::uint64_t              *distortionResultsHost = nullptr;
  std::size_t                 distortionCapacity = 0;
  std::uint64_t               distortionDispatches = 0;
#if VTM_CUDA_TESTING
  unsigned                    asyncReleaseFailures = 0;
  unsigned                    immediateReleaseFailures = 0;
  unsigned                    distortionAllocationFailureStep = 0;
  unsigned                    distortionExecutionFailures = 0;
#endif
};

namespace
{

void checkCuda(const cudaError_t result, const char *operation)
{
  if (result != cudaSuccess)
  {
    throw std::runtime_error(std::string("CUDA ") + operation + " failed: " + cudaGetErrorString(result));
  }
}

std::size_t queueIndex(const CudaQueue queue)
{
  switch (queue)
  {
  case CudaQueue::Upload: return 0;
  case CudaQueue::Compute: return 1;
  case CudaQueue::Download: return 2;
  }
  throw std::runtime_error("Invalid CUDA queue");
}

std::size_t fenceIndex(const CudaFence fence)
{
  switch (fence)
  {
  case CudaFence::UploadComplete: return 0;
  case CudaFence::ComputeComplete: return 1;
  case CudaFence::DownloadComplete: return 2;
  }
  throw std::runtime_error("Invalid CUDA fence");
}

void rememberCudaError(cudaError_t &firstError, const char *&firstOperation, const cudaError_t result,
                       const char *operation) noexcept
{
  if (result != cudaSuccess && firstError == cudaSuccess)
  {
    firstError = result;
    firstOperation = operation;
  }
}

void releaseRuntimeContext(RuntimeContext *context, const bool checked, const bool synchronize)
{
  if (context == nullptr)
  {
    return;
  }

  cudaError_t firstError = cudaSuccess;
  const char *firstOperation = nullptr;
  rememberCudaError(firstError, firstOperation, cudaSetDevice(context->device), "device selection during shutdown");
  if (context->distortionResultsDevice != nullptr)
  {
    rememberCudaError(firstError, firstOperation,
                      cudaFreeAsync(context->distortionResultsDevice, context->streams[queueIndex(CudaQueue::Compute)]),
                      "distortion result release");
    context->distortionResultsDevice = nullptr;
  }
  for (cudaStream_t stream : context->streams)
  {
    if (synchronize && stream != nullptr)
    {
      rememberCudaError(firstError, firstOperation, cudaStreamSynchronize(stream),
                        "stream synchronization during shutdown");
    }
  }
  if (context->distortionResultsHost != nullptr)
  {
    rememberCudaError(firstError, firstOperation, cudaFreeHost(context->distortionResultsHost),
                      "pinned distortion result release");
    context->distortionResultsHost = nullptr;
  }
  for (cudaEvent_t &fence : context->fences)
  {
    if (fence != nullptr)
    {
      rememberCudaError(firstError, firstOperation, cudaEventDestroy(fence), "fence destruction");
      fence = nullptr;
    }
  }
  for (cudaStream_t &stream : context->streams)
  {
    if (stream != nullptr)
    {
      rememberCudaError(firstError, firstOperation, cudaStreamDestroy(stream), "stream destruction");
      stream = nullptr;
    }
  }
  if (context->memoryPool != nullptr)
  {
    rememberCudaError(firstError, firstOperation, cudaMemPoolDestroy(context->memoryPool), "memory pool destruction");
    context->memoryPool = nullptr;
  }
  delete context;

  if (checked && firstError != cudaSuccess)
  {
    throw std::runtime_error(std::string("CUDA ") + firstOperation + " failed: " + cudaGetErrorString(firstError));
  }
}

void ensureDistortionCapacity(RuntimeContext *context, const std::size_t required)
{
  if (required <= context->distortionCapacity)
  {
    return;
  }

  if (required > CUDA_MAX_DISTORTION_CANDIDATES)
  {
    throw std::runtime_error("CUDA distortion batch exceeds the fixed scratch limit");
  }
  // Allocate the bounded maximum on first use. This avoids replacement of live scratch and makes every
  // partial-allocation rollback transactional: either both resources become owned by the context or neither does.
  constexpr std::size_t capacity = CUDA_MAX_DISTORTION_CANDIDATES;

  cudaStream_t stream = context->streams[queueIndex(CudaQueue::Compute)];
  checkCuda(cudaStreamSynchronize(stream), "distortion scratch synchronization");

  std::uint64_t *newResultsDevice = nullptr;
  std::uint64_t *newResultsHost = nullptr;
  try
  {
#if VTM_CUDA_TESTING
    if (context->distortionAllocationFailureStep == 1)
    {
      context->distortionAllocationFailureStep = 0;
      throw std::runtime_error("Injected CUDA distortion device allocation failure");
    }
#endif
    checkCuda(cudaMallocFromPoolAsync(reinterpret_cast<void **>(&newResultsDevice),
                                      capacity * sizeof(*newResultsDevice), context->memoryPool, stream),
              "distortion result allocation");
#if VTM_CUDA_TESTING
    if (context->distortionAllocationFailureStep == 2)
    {
      context->distortionAllocationFailureStep = 0;
      throw std::runtime_error("Injected CUDA distortion pinned allocation failure");
    }
#endif
    checkCuda(cudaHostAlloc(reinterpret_cast<void **>(&newResultsHost), capacity * sizeof(*newResultsHost),
                            cudaHostAllocPortable),
              "pinned distortion result allocation");
  }
  catch (...)
  {
    if (newResultsDevice != nullptr)
    {
      cudaFreeAsync(newResultsDevice, stream);
    }
    cudaStreamSynchronize(stream);
    if (newResultsHost != nullptr)
    {
      cudaFreeHost(newResultsHost);
    }
    throw;
  }

  if (context->distortionResultsDevice != nullptr || context->distortionResultsHost != nullptr)
  {
    if (newResultsDevice != nullptr)
    {
      (void) cudaFreeAsync(newResultsDevice, stream);
      (void) cudaStreamSynchronize(stream);
    }
    if (newResultsHost != nullptr)
    {
      (void) cudaFreeHost(newResultsHost);
    }
    throw std::runtime_error("CUDA distortion scratch ownership is inconsistent");
  }
  context->distortionResultsDevice = newResultsDevice;
  context->distortionResultsHost = newResultsHost;
  context->distortionCapacity = capacity;
}

template<typename Sample>
__global__ void sadBatchKernel(const Sample *source, const std::size_t sourcePitchBytes,
                               const Sample *referenceBase, const std::size_t referencePitchBytes,
                               const std::uint32_t width, const std::uint32_t height,
                               const std::uint32_t candidateColumns, const std::uint16_t candidateStepX,
                               const std::uint16_t candidateStepY,
                               const std::uint8_t subShift, const std::uint8_t distortionShift,
                               std::uint64_t *results)
{
  const std::uint32_t candidate = blockIdx.x;
  const std::uint32_t candidateX = candidate % candidateColumns;
  const std::uint32_t candidateY = candidate / candidateColumns;
  const std::size_t sourceStride = sourcePitchBytes / sizeof(Sample);
  const std::size_t referenceStride = referencePitchBytes / sizeof(Sample);
  const Sample *reference = referenceBase
                            + static_cast<std::size_t>(candidateY) * candidateStepY * referenceStride
                            + static_cast<std::size_t>(candidateX) * candidateStepX;
  const std::uint32_t sampledRows = height >> subShift;
  const std::uint64_t sampleCount = static_cast<std::uint64_t>(width) * sampledRows;

  std::uint64_t sum = 0;
  for (std::uint64_t sample = threadIdx.x; sample < sampleCount; sample += blockDim.x)
  {
    const std::uint32_t sampledY = static_cast<std::uint32_t>(sample / width);
    const std::uint32_t x = static_cast<std::uint32_t>(sample - static_cast<std::uint64_t>(sampledY) * width);
    const std::uint32_t y = sampledY << subShift;
    const std::int64_t difference = static_cast<std::int64_t>(source[static_cast<std::size_t>(y) * sourceStride + x])
                                    - static_cast<std::int64_t>(reference[static_cast<std::size_t>(y) * referenceStride + x]);
    sum += static_cast<std::uint64_t>(difference < 0 ? -difference : difference);
  }

  __shared__ std::uint64_t partial[256];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned offset = blockDim.x / 2; offset != 0; offset >>= 1)
  {
    if (threadIdx.x < offset)
    {
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
  {
    results[candidate] = (partial[0] << subShift) >> distortionShift;
  }
}

}   // namespace

RuntimeContext *createRuntimeContext(const int device)
{
  int deviceCount = 0;
  checkCuda(cudaGetDeviceCount(&deviceCount), "device discovery");
  if (device < 0 || device >= deviceCount)
  {
    throw std::runtime_error("GPUDevice " + std::to_string(device) + " is invalid; available CUDA devices: "
                             + std::to_string(deviceCount));
  }

  std::unique_ptr<RuntimeContext> context(new RuntimeContext);
  context->device = device;

  try
  {
    checkCuda(cudaSetDevice(device), "device selection");

    int memoryPoolsSupported = 0;
    checkCuda(cudaDeviceGetAttribute(&memoryPoolsSupported, cudaDevAttrMemoryPoolsSupported, device),
              "memory pool capability query");
    if (!memoryPoolsSupported)
    {
      throw std::runtime_error("Selected CUDA device does not support stream-ordered memory pools");
    }

    for (cudaStream_t &stream : context->streams)
    {
      checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream creation");
    }
    for (cudaEvent_t &fence : context->fences)
    {
      checkCuda(cudaEventCreateWithFlags(&fence, cudaEventDisableTiming), "fence creation");
    }

    cudaMemPoolProps poolProperties{};
    poolProperties.allocType     = cudaMemAllocationTypePinned;
    poolProperties.handleTypes   = cudaMemHandleTypeNone;
    poolProperties.location.type = cudaMemLocationTypeDevice;
    poolProperties.location.id   = device;
    checkCuda(cudaMemPoolCreate(&context->memoryPool, &poolProperties), "memory pool creation");
  }
  catch (...)
  {
    destroyRuntimeContext(context.release());
    throw;
  }

  return context.release();
}

void synchronizeRuntimeContext(RuntimeContext *context)
{
  if (context == nullptr)
  {
    return;
  }
  checkCuda(cudaSetDevice(context->device), "device selection");
  for (cudaStream_t stream : context->streams)
  {
    checkCuda(cudaStreamSynchronize(stream), "stream synchronization");
  }
}

void shutdownRuntimeContext(RuntimeContext *context, const bool synchronize)
{
  releaseRuntimeContext(context, true, synchronize);
}

void destroyRuntimeContext(RuntimeContext *context) noexcept
{
  try
  {
    releaseRuntimeContext(context, false, true);
  }
  catch (...)
  {
    // Last-resort destructor cleanup must never throw.
  }
}

void recordFence(RuntimeContext *context, const CudaQueue queue, const CudaFence fence)
{
  checkCuda(cudaSetDevice(context->device), "device selection");
  checkCuda(cudaEventRecord(context->fences[fenceIndex(fence)], context->streams[queueIndex(queue)]), "fence record");
}

void waitFence(RuntimeContext *context, const CudaQueue queue, const CudaFence fence)
{
  checkCuda(cudaSetDevice(context->device), "device selection");
  checkCuda(cudaStreamWaitEvent(context->streams[queueIndex(queue)], context->fences[fenceIndex(fence)], 0),
            "fence wait");
}

void *allocateDevice(RuntimeContext *context, const std::size_t bytes, const CudaQueue queue)
{
  if (bytes == 0)
  {
    throw std::runtime_error("CUDA allocation size must be greater than zero");
  }
  checkCuda(cudaSetDevice(context->device), "device selection");
  void *allocation = nullptr;
  checkCuda(cudaMallocFromPoolAsync(&allocation, bytes, context->memoryPool, context->streams[queueIndex(queue)]),
            "memory pool allocation");
  return allocation;
}

void releaseDevice(RuntimeContext *context, void *allocation, const CudaQueue queue)
{
  if (allocation == nullptr)
  {
    return;
  }
#if VTM_CUDA_TESTING
  if (context->asyncReleaseFailures > 0)
  {
    --context->asyncReleaseFailures;
    throw std::runtime_error("Injected CUDA asynchronous release failure");
  }
#endif
  checkCuda(cudaSetDevice(context->device), "device selection");
  checkCuda(cudaFreeAsync(allocation, context->streams[queueIndex(queue)]), "memory pool release");
}

void releaseDeviceImmediate(RuntimeContext *context, void *allocation)
{
  if (allocation == nullptr)
  {
    return;
  }
#if VTM_CUDA_TESTING
  if (context->immediateReleaseFailures > 0)
  {
    --context->immediateReleaseFailures;
    throw std::runtime_error("Injected CUDA synchronous release failure");
  }
#endif
  checkCuda(cudaSetDevice(context->device), "device selection");
  checkCuda(cudaFree(allocation), "synchronous memory release fallback");
}

void *allocatePinnedHost(const std::size_t bytes)
{
  if (bytes == 0)
  {
    throw std::runtime_error("CUDA pinned allocation size must be greater than zero");
  }
  void *allocation = nullptr;
  checkCuda(cudaMallocHost(&allocation, bytes), "pinned host allocation");
  return allocation;
}

void releasePinnedHost(void *allocation) noexcept
{
  if (allocation != nullptr)
  {
    (void) cudaFreeHost(allocation);
  }
}

void copy2DToDeviceAsync(RuntimeContext *context, void *destination, const std::size_t destinationPitch,
                         const void *source, const std::size_t sourcePitch, const std::size_t widthBytes,
                         const std::size_t height, const CudaQueue queue)
{
  checkCuda(cudaSetDevice(context->device), "device selection");
  checkCuda(cudaMemcpy2DAsync(destination, destinationPitch, source, sourcePitch, widthBytes, height,
                              cudaMemcpyHostToDevice, context->streams[queueIndex(queue)]),
            "two-dimensional upload");
}

void copy2DToHostAsync(RuntimeContext *context, void *destination, const std::size_t destinationPitch,
                       const void *source, const std::size_t sourcePitch, const std::size_t widthBytes,
                       const std::size_t height, const CudaQueue queue)
{
  checkCuda(cudaSetDevice(context->device), "device selection");
  checkCuda(cudaMemcpy2DAsync(destination, destinationPitch, source, sourcePitch, widthBytes, height,
                              cudaMemcpyDeviceToHost, context->streams[queueIndex(queue)]),
            "two-dimensional download");
}

void synchronizeQueue(RuntimeContext *context, const CudaQueue queue)
{
  checkCuda(cudaSetDevice(context->device), "device selection");
  checkCuda(cudaStreamSynchronize(context->streams[queueIndex(queue)]), "queue synchronization");
}

void computeDistortionBatch(RuntimeContext *context, const CudaDistortionBatchDesc &batch,
                            const void *sourceDevice, const std::size_t sourcePitchBytes,
                            const void *referenceDevice, const std::size_t referencePitchBytes,
                            std::uint64_t *results)
{
  checkCuda(cudaSetDevice(context->device), "device selection");
  const std::uint32_t candidateCount = batch.candidateGrid.columns * batch.candidateGrid.rows;
  ensureDistortionCapacity(context, candidateCount);

  cudaStream_t stream = context->streams[queueIndex(CudaQueue::Compute)];
#if VTM_CUDA_TESTING
  if (context->distortionExecutionFailures > 0)
  {
    --context->distortionExecutionFailures;
    throw std::runtime_error("Injected CUDA distortion execution failure");
  }
#endif

  constexpr unsigned threads = 256;
  // VTM 24 builds use FULL_NBIT=1 for both Pel configurations, so SAD retains all source precision.
  constexpr std::uint8_t distortionShift = 0;
  if (batch.elementSize == 2)
  {
    sadBatchKernel<std::int16_t><<<candidateCount, threads, 0, stream>>>(
      static_cast<const std::int16_t *>(sourceDevice), sourcePitchBytes,
      static_cast<const std::int16_t *>(referenceDevice), referencePitchBytes, batch.width, batch.height,
      batch.candidateGrid.columns, batch.candidateGrid.stepX, batch.candidateGrid.stepY,
      batch.subShift, distortionShift,
      context->distortionResultsDevice);
  }
  else
  {
    sadBatchKernel<std::int32_t><<<candidateCount, threads, 0, stream>>>(
      static_cast<const std::int32_t *>(sourceDevice), sourcePitchBytes,
      static_cast<const std::int32_t *>(referenceDevice), referencePitchBytes, batch.width, batch.height,
      batch.candidateGrid.columns, batch.candidateGrid.stepX, batch.candidateGrid.stepY,
      batch.subShift, distortionShift,
      context->distortionResultsDevice);
  }
  checkCuda(cudaGetLastError(), "SAD batch launch");
  checkCuda(cudaMemcpyAsync(context->distortionResultsHost, context->distortionResultsDevice,
                            static_cast<std::size_t>(candidateCount) * sizeof(*results),
                            cudaMemcpyDeviceToHost, stream),
            "distortion result download");
  checkCuda(cudaStreamSynchronize(stream), "distortion batch completion");
  std::memcpy(results, context->distortionResultsHost,
              static_cast<std::size_t>(candidateCount) * sizeof(*results));
  ++context->distortionDispatches;
}

std::uint64_t distortionBatchDispatchCount(const RuntimeContext *context)
{
  return context->distortionDispatches;
}

void recoverDistortionRuntime(RuntimeContext *context) noexcept
{
  if (context == nullptr)
  {
    return;
  }
  (void) cudaSetDevice(context->device);
  const std::size_t computeIndex = queueIndex(CudaQueue::Compute);
  if (context->streams[computeIndex] != nullptr)
  {
    (void) cudaStreamSynchronize(context->streams[computeIndex]);
    if (context->distortionResultsDevice != nullptr)
    {
      if (cudaFreeAsync(context->distortionResultsDevice, context->streams[computeIndex]) == cudaSuccess)
      {
        context->distortionResultsDevice = nullptr; // ownership transferred even if later synchronization fails
        (void) cudaStreamSynchronize(context->streams[computeIndex]);
      }
      else
      {
        (void) cudaGetLastError(); // retain ownership so normal context teardown can retry
      }
    }
    (void) cudaStreamDestroy(context->streams[computeIndex]);
    context->streams[computeIndex] = nullptr;
  }
  if (context->distortionResultsHost != nullptr)
  {
    if (cudaFreeHost(context->distortionResultsHost) == cudaSuccess)
    {
      context->distortionResultsHost = nullptr;
    }
  }
  context->distortionCapacity = 0;
  (void) cudaGetLastError();
  (void) cudaStreamCreateWithFlags(&context->streams[computeIndex], cudaStreamNonBlocking);
}

#if VTM_CUDA_TESTING
void injectReleaseFailures(RuntimeContext *context, const unsigned asyncFailures, const unsigned immediateFailures)
{
  context->asyncReleaseFailures = asyncFailures;
  context->immediateReleaseFailures = immediateFailures;
}


void injectDistortionFailures(RuntimeContext *context, const unsigned allocationFailureStep,
                              const unsigned executionFailures)
{
  context->distortionAllocationFailureStep = allocationFailureStep;
  context->distortionExecutionFailures = executionFailures;
}
#endif

}   // namespace vtm::cuda_backend
