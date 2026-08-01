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

#include <algorithm>
#include <memory>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <atomic>
#include <chrono>
#include <vector>

namespace vtm::cuda_backend
{

struct AlfScratch
{
  CudaAlfCtuParam   *ctusDevice = nullptr;
  CudaAlfCtuParam   *ctusHost = nullptr;
  CudaAlfClassifier *classifiersDevice = nullptr;
  void              *outputDevice = nullptr;
  std::size_t         ctuCapacity = 0;
  std::size_t         classifierCapacity = 0;
  std::size_t         outputCapacity = 0;
};

struct DbfScratch
{
  CudaDbfLumaTask *tasksDevice = nullptr;
  CudaDbfLumaTask *tasksHost = nullptr;
  std::uint32_t *laneOffsetsDevice = nullptr;
  std::uint32_t *laneOffsetsHost = nullptr;
  void *outputDevice = nullptr;
  std::size_t taskCapacity = 0;
  std::size_t laneCapacity = 0;
  std::size_t outputCapacity = 0;
};

struct RuntimeContext
{
  int                         device = -1;
  std::array<cudaStream_t, 6> streams{};
  std::array<cudaEvent_t, 3>  fences{};
  cudaMemPool_t               memoryPool = nullptr;
  std::uint64_t              *distortionResultsDevice = nullptr;
  std::uint64_t              *distortionResultsHost = nullptr;
  std::size_t                 distortionCapacity = 0;
  std::uint64_t               distortionDispatches = 0;
  CudaQpaTask                *qpaTasksDevice = nullptr;
  CudaQpaTask                *qpaTasksHost = nullptr;
  CudaQpaResult              *qpaResultsDevice = nullptr;
  CudaQpaResult              *qpaResultsHost = nullptr;
  std::size_t                 qpaCapacity = 0;
  std::uint64_t               qpaDispatches = 0;
  std::uint64_t               qpaTasks = 0;
  AlfScratch                  alfScratch{};
  // A failed old-buffer release must retain ownership until recovery/teardown.
  AlfScratch                  alfRetiredScratch{};
  std::uint64_t               alfDispatches = 0;
  std::uint64_t               alfCtus = 0;
  std::uint64_t               alfPixels = 0;
  std::uint64_t               alfParameterUploadBytes = 0;
  std::uint64_t               alfDiagnosticDownloadBytes = 0;
  std::uint64_t               alfCommitBytes = 0;
  std::uint64_t               alfSynchronizations = 0;
  std::uint64_t               alfElapsedNanoseconds = 0;
  std::uint64_t               alfPeakScratchBytes = 0;
  DbfScratch                  dbfScratch{};
  std::uint64_t              dbfDispatches = 0;
  std::uint64_t              dbfTasks = 0;
  std::uint64_t              dbfPixels = 0;
  std::uint64_t              dbfParameterUploadBytes = 0;
  std::uint64_t              dbfCommitBytes = 0;
  std::uint64_t              dbfSynchronizations = 0;
  std::uint64_t              dbfElapsedNanoseconds = 0;
  std::uint64_t              dbfPeakScratchBytes = 0;
  CudaBatchTestFailurePoint   distortionFailurePoint = CudaBatchTestFailurePoint::None;
  CudaBatchTestFailurePoint   qpaFailurePoint = CudaBatchTestFailurePoint::None;
#if VTM_CUDA_TESTING
  unsigned                    asyncReleaseFailures = 0;
  unsigned                    immediateReleaseFailures = 0;
  unsigned                    distortionAllocationFailureStep = 0;
  unsigned                    distortionExecutionFailures = 0;
  unsigned                    qpaAllocationFailureStep = 0;
  unsigned                    qpaExecutionFailures = 0;
  CudaAlfTestFailurePoint     alfFailurePoint = CudaAlfTestFailurePoint::None;
  CudaDbfTestFailurePoint     dbfFailurePoint = CudaDbfTestFailurePoint::None;
#endif
};

namespace
{

#if VTM_CUDA_TESTING
std::atomic<unsigned> pinnedReleaseFailures{ 0 };
#endif

void checkCuda(const cudaError_t result, const char *operation)
{
  if (result != cudaSuccess)
  {
    throw std::runtime_error(std::string("CUDA ") + operation + " failed: " + cudaGetErrorString(result));
  }
}

std::uint64_t alfScratchBytes(const AlfScratch &scratch) noexcept
{
  std::uint64_t bytes = 0;
  if (scratch.ctusDevice != nullptr)
  {
    bytes += static_cast<std::uint64_t>(scratch.ctuCapacity) * sizeof(CudaAlfCtuParam);
  }
  if (scratch.ctusHost != nullptr)
  {
    bytes += static_cast<std::uint64_t>(scratch.ctuCapacity) * sizeof(CudaAlfCtuParam);
  }
  if (scratch.classifiersDevice != nullptr)
  {
    bytes += static_cast<std::uint64_t>(scratch.classifierCapacity) * sizeof(CudaAlfClassifier);
  }
  if (scratch.outputDevice != nullptr)
  {
    bytes += scratch.outputCapacity;
  }
  return bytes;
}

std::uint64_t dbfScratchBytes(const DbfScratch &scratch) noexcept
{
  std::uint64_t bytes = 0;
  if (scratch.tasksDevice != nullptr) bytes += scratch.taskCapacity * sizeof(CudaDbfLumaTask);
  if (scratch.tasksHost != nullptr) bytes += scratch.taskCapacity * sizeof(CudaDbfLumaTask);
  if (scratch.laneOffsetsDevice != nullptr) bytes += (scratch.laneCapacity + 1) * sizeof(std::uint32_t);
  if (scratch.laneOffsetsHost != nullptr) bytes += (scratch.laneCapacity + 1) * sizeof(std::uint32_t);
  if (scratch.outputDevice != nullptr) bytes += scratch.outputCapacity;
  return bytes;
}

bool isEmpty(const AlfScratch &scratch) noexcept
{
  return scratch.ctusDevice == nullptr && scratch.ctusHost == nullptr
         && scratch.classifiersDevice == nullptr && scratch.outputDevice == nullptr;
}

#if VTM_CUDA_TESTING
bool consumeAlfFailure(RuntimeContext *context, const CudaAlfTestFailurePoint point) noexcept
{
  if (context->alfFailurePoint != point)
  {
    return false;
  }
  context->alfFailurePoint = CudaAlfTestFailurePoint::None;
  return true;
}

bool consumeBatchFailure(CudaBatchTestFailurePoint &configured,
                         const CudaBatchTestFailurePoint point) noexcept
{
  if (configured != point)
  {
    return false;
  }
  configured = CudaBatchTestFailurePoint::None;
  return true;
}
#else
bool consumeAlfFailure(RuntimeContext *, CudaAlfTestFailurePoint) noexcept { return false; }
bool consumeBatchFailure(CudaBatchTestFailurePoint &, CudaBatchTestFailurePoint) noexcept { return false; }
#endif

#if VTM_CUDA_TESTING
bool consumeDbfFailure(RuntimeContext *context, const CudaDbfTestFailurePoint point) noexcept
{
  if (context->dbfFailurePoint != point) return false;
  context->dbfFailurePoint = CudaDbfTestFailurePoint::None;
  return true;
}
#else
bool consumeDbfFailure(RuntimeContext *, CudaDbfTestFailurePoint) noexcept { return false; }
#endif

std::size_t queueIndex(const CudaQueue queue)
{
  switch (queue)
  {
  case CudaQueue::Upload: return 0;
  case CudaQueue::Compute: return 1;
  case CudaQueue::Download: return 2;
  case CudaQueue::Qpa: return 3;
  case CudaQueue::Alf: return 4;
  case CudaQueue::Dbf: return 5;
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
  for (void **allocation : { reinterpret_cast<void **>(&context->qpaTasksDevice),
                             reinterpret_cast<void **>(&context->qpaResultsDevice) })
  {
    if (*allocation != nullptr)
    {
      rememberCudaError(firstError, firstOperation,
                        cudaFreeAsync(*allocation, context->streams[queueIndex(CudaQueue::Qpa)]),
                        "QPA scratch release");
      *allocation = nullptr;
    }
  }
  for (AlfScratch *scratch : { &context->alfScratch, &context->alfRetiredScratch })
  {
    for (void **allocation : { reinterpret_cast<void **>(&scratch->ctusDevice),
                               reinterpret_cast<void **>(&scratch->classifiersDevice),
                               &scratch->outputDevice })
    {
      if (*allocation != nullptr)
      {
        const cudaError_t result = cudaFreeAsync(*allocation, context->streams[queueIndex(CudaQueue::Alf)]);
        rememberCudaError(firstError, firstOperation, result, "ALF scratch release");
        if (result == cudaSuccess)
        {
          *allocation = nullptr;
        }
      }
    }
  }
  for (void **allocation : { reinterpret_cast<void **>(&context->dbfScratch.tasksDevice),
                             reinterpret_cast<void **>(&context->dbfScratch.laneOffsetsDevice),
                             &context->dbfScratch.outputDevice })
  {
    if (*allocation != nullptr)
    {
      const cudaError_t result = cudaFreeAsync(*allocation, context->streams[queueIndex(CudaQueue::Dbf)]);
      rememberCudaError(firstError, firstOperation, result, "DBF scratch release");
      if (result == cudaSuccess) *allocation = nullptr;
    }
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
  for (void **allocation : { reinterpret_cast<void **>(&context->qpaTasksHost),
                             reinterpret_cast<void **>(&context->qpaResultsHost) })
  {
    if (*allocation != nullptr)
    {
      rememberCudaError(firstError, firstOperation, cudaFreeHost(*allocation), "pinned QPA scratch release");
      *allocation = nullptr;
    }
  }
  for (AlfScratch *scratch : { &context->alfScratch, &context->alfRetiredScratch })
  {
    for (void **allocation : { reinterpret_cast<void **>(&scratch->ctusDevice),
                               reinterpret_cast<void **>(&scratch->classifiersDevice),
                               &scratch->outputDevice })
    {
      if (*allocation != nullptr)
      {
        const cudaError_t result = cudaFree(*allocation);
        rememberCudaError(firstError, firstOperation, result, "ALF scratch immediate release");
        if (result == cudaSuccess)
        {
          *allocation = nullptr;
        }
      }
    }
    if (scratch->ctusHost != nullptr)
    {
      const cudaError_t result = cudaFreeHost(scratch->ctusHost);
      rememberCudaError(firstError, firstOperation, result, "pinned ALF parameter release");
      if (result == cudaSuccess)
      {
        scratch->ctusHost = nullptr;
      }
    }
  }
  for (void **allocation : { reinterpret_cast<void **>(&context->dbfScratch.tasksDevice),
                             reinterpret_cast<void **>(&context->dbfScratch.laneOffsetsDevice),
                             &context->dbfScratch.outputDevice })
  {
    if (*allocation != nullptr)
    {
      const cudaError_t result = cudaFree(*allocation);
      rememberCudaError(firstError, firstOperation, result, "DBF scratch immediate release");
      if (result == cudaSuccess) *allocation = nullptr;
    }
  }
  for (void **allocation : { reinterpret_cast<void **>(&context->dbfScratch.tasksHost),
                             reinterpret_cast<void **>(&context->dbfScratch.laneOffsetsHost) })
  {
    if (*allocation != nullptr)
    {
      const cudaError_t result = cudaFreeHost(*allocation);
      rememberCudaError(firstError, firstOperation, result, "pinned DBF scratch release");
      if (result == cudaSuccess) *allocation = nullptr;
    }
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

void ensureQpaCapacity(RuntimeContext *context, const std::size_t required)
{
  if (required <= context->qpaCapacity)
  {
    return;
  }
  if (required > CUDA_MAX_QPA_TASKS)
  {
    throw std::runtime_error("CUDA QPA batch exceeds the fixed queue limit");
  }

  constexpr std::size_t capacity = CUDA_MAX_QPA_TASKS;
  cudaStream_t stream = context->streams[queueIndex(CudaQueue::Qpa)];
  checkCuda(cudaStreamSynchronize(stream), "QPA scratch synchronization");

  CudaQpaTask *newTasksDevice = nullptr;
  CudaQpaTask *newTasksHost = nullptr;
  CudaQpaResult *newResultsDevice = nullptr;
  CudaQpaResult *newResultsHost = nullptr;
  try
  {
#if VTM_CUDA_TESTING
    if (context->qpaAllocationFailureStep == 1)
    {
      context->qpaAllocationFailureStep = 0;
      throw std::runtime_error("Injected CUDA QPA task device allocation failure");
    }
#endif
    checkCuda(cudaMallocFromPoolAsync(reinterpret_cast<void **>(&newTasksDevice),
                                      capacity * sizeof(*newTasksDevice), context->memoryPool, stream),
              "QPA task allocation");
#if VTM_CUDA_TESTING
    if (context->qpaAllocationFailureStep == 2)
    {
      context->qpaAllocationFailureStep = 0;
      throw std::runtime_error("Injected CUDA QPA result device allocation failure");
    }
#endif
    checkCuda(cudaMallocFromPoolAsync(reinterpret_cast<void **>(&newResultsDevice),
                                      capacity * sizeof(*newResultsDevice), context->memoryPool, stream),
              "QPA result allocation");
#if VTM_CUDA_TESTING
    if (context->qpaAllocationFailureStep == 3)
    {
      context->qpaAllocationFailureStep = 0;
      throw std::runtime_error("Injected CUDA QPA task pinned allocation failure");
    }
#endif
    checkCuda(cudaHostAlloc(reinterpret_cast<void **>(&newTasksHost), capacity * sizeof(*newTasksHost),
                            cudaHostAllocPortable),
              "pinned QPA task allocation");
#if VTM_CUDA_TESTING
    if (context->qpaAllocationFailureStep == 4)
    {
      context->qpaAllocationFailureStep = 0;
      throw std::runtime_error("Injected CUDA QPA result pinned allocation failure");
    }
#endif
    checkCuda(cudaHostAlloc(reinterpret_cast<void **>(&newResultsHost), capacity * sizeof(*newResultsHost),
                            cudaHostAllocPortable),
              "pinned QPA result allocation");
  }
  catch (...)
  {
    if (newTasksDevice != nullptr) (void) cudaFreeAsync(newTasksDevice, stream);
    if (newResultsDevice != nullptr) (void) cudaFreeAsync(newResultsDevice, stream);
    (void) cudaStreamSynchronize(stream);
    if (newTasksHost != nullptr) (void) cudaFreeHost(newTasksHost);
    if (newResultsHost != nullptr) (void) cudaFreeHost(newResultsHost);
    throw;
  }

  if (context->qpaTasksDevice != nullptr || context->qpaResultsDevice != nullptr
      || context->qpaTasksHost != nullptr || context->qpaResultsHost != nullptr)
  {
    (void) cudaFreeAsync(newTasksDevice, stream);
    (void) cudaFreeAsync(newResultsDevice, stream);
    (void) cudaStreamSynchronize(stream);
    (void) cudaFreeHost(newTasksHost);
    (void) cudaFreeHost(newResultsHost);
    throw std::runtime_error("CUDA QPA scratch ownership is inconsistent");
  }

  context->qpaTasksDevice = newTasksDevice;
  context->qpaTasksHost = newTasksHost;
  context->qpaResultsDevice = newResultsDevice;
  context->qpaResultsHost = newResultsHost;
  context->qpaCapacity = capacity;
}

void releaseAlfScratchChecked(RuntimeContext *context, AlfScratch &scratch, cudaStream_t stream)
{
  if (isEmpty(scratch))
  {
    scratch = AlfScratch{};
    return;
  }
  if (consumeAlfFailure(context, CudaAlfTestFailurePoint::OldDeviceRelease))
  {
    throw std::runtime_error("Injected CUDA ALF old device scratch release failure");
  }
  for (void **allocation : { reinterpret_cast<void **>(&scratch.ctusDevice),
                             reinterpret_cast<void **>(&scratch.classifiersDevice),
                             &scratch.outputDevice })
  {
    if (*allocation != nullptr)
    {
      checkCuda(cudaFreeAsync(*allocation, stream), "old ALF device scratch release");
      *allocation = nullptr;
    }
  }
  checkCuda(cudaStreamSynchronize(stream), "old ALF scratch release completion");
  ++context->alfSynchronizations;
  if (scratch.ctusHost != nullptr)
  {
    if (consumeAlfFailure(context, CudaAlfTestFailurePoint::OldPinnedRelease))
    {
      throw std::runtime_error("Injected CUDA ALF old pinned scratch release failure");
    }
    checkCuda(cudaFreeHost(scratch.ctusHost), "old pinned ALF parameter release");
    scratch.ctusHost = nullptr;
  }
  scratch = AlfScratch{};
}

void ensureAlfCapacity(RuntimeContext *context, const std::size_t ctuCount,
                       const std::size_t classifierCount, const std::size_t outputBytes)
{
  AlfScratch &current = context->alfScratch;
  if (ctuCount <= current.ctuCapacity && classifierCount <= current.classifierCapacity
      && outputBytes <= current.outputCapacity)
  {
    return;
  }
  if (!isEmpty(context->alfRetiredScratch))
  {
    throw std::runtime_error("CUDA ALF retired scratch was not recovered before growth");
  }
  constexpr std::size_t scratchBudget = std::size_t{ 256 } * 1024 * 1024;
  const std::size_t ctuBytes = ctuCount * sizeof(CudaAlfCtuParam);
  const std::size_t classifierBytes = classifierCount * sizeof(CudaAlfClassifier);
  const std::size_t requestedBytes = 2 * ctuBytes + classifierBytes + outputBytes;
  if (ctuCount > CUDA_ALF_MAX_CTUS || ctuBytes > scratchBudget / 2
      || classifierBytes > scratchBudget - 2 * ctuBytes
      || outputBytes > scratchBudget - 2 * ctuBytes - classifierBytes
      || alfScratchBytes(current) > scratchBudget - requestedBytes)
  {
    throw std::runtime_error("CUDA ALF scratch budget exceeded");
  }

  cudaStream_t stream = context->streams[queueIndex(CudaQueue::Alf)];
  checkCuda(cudaStreamSynchronize(stream), "ALF scratch synchronization");
  ++context->alfSynchronizations;
  AlfScratch candidate{};
  candidate.ctuCapacity = ctuCount;
  candidate.classifierCapacity = classifierCount;
  candidate.outputCapacity = outputBytes;
  try
  {
    if (consumeAlfFailure(context, CudaAlfTestFailurePoint::GrowCtuDevice))
      throw std::runtime_error("Injected CUDA ALF CTU allocation failure");
    checkCuda(cudaMallocFromPoolAsync(reinterpret_cast<void **>(&candidate.ctusDevice), ctuBytes,
                                      context->memoryPool, stream), "ALF CTU parameter allocation");
    if (consumeAlfFailure(context, CudaAlfTestFailurePoint::GrowPinnedCtu))
      throw std::runtime_error("Injected CUDA ALF pinned allocation failure");
    checkCuda(cudaHostAlloc(reinterpret_cast<void **>(&candidate.ctusHost), ctuBytes, cudaHostAllocPortable),
              "pinned ALF CTU parameter allocation");
    if (consumeAlfFailure(context, CudaAlfTestFailurePoint::GrowClassifiers))
      throw std::runtime_error("Injected CUDA ALF classifier allocation failure");
    checkCuda(cudaMallocFromPoolAsync(reinterpret_cast<void **>(&candidate.classifiersDevice), classifierBytes,
                                      context->memoryPool, stream), "ALF classifier allocation");
    if (consumeAlfFailure(context, CudaAlfTestFailurePoint::GrowOutput))
      throw std::runtime_error("Injected CUDA ALF output allocation failure");
    checkCuda(cudaMallocFromPoolAsync(&candidate.outputDevice, outputBytes, context->memoryPool, stream),
              "ALF output allocation");
    checkCuda(cudaStreamSynchronize(stream), "ALF scratch allocation completion");
    ++context->alfSynchronizations;
  }
  catch (...)
  {
    // Preserve every successfully allocated pointer for the context recovery path.
    context->alfRetiredScratch = candidate;
    context->alfPeakScratchBytes = std::max(context->alfPeakScratchBytes,
      alfScratchBytes(current) + alfScratchBytes(context->alfRetiredScratch));
    throw;
  }

  std::swap(current, candidate);
  context->alfRetiredScratch = candidate;
  context->alfPeakScratchBytes = std::max(context->alfPeakScratchBytes,
    alfScratchBytes(current) + alfScratchBytes(context->alfRetiredScratch));
  releaseAlfScratchChecked(context, context->alfRetiredScratch, stream);
}

void ensureDbfCapacity(RuntimeContext *context, const std::size_t tasks, const std::size_t lanes,
                       const std::size_t outputBytes)
{
  DbfScratch &s = context->dbfScratch;
  if (tasks <= s.taskCapacity && lanes <= s.laneCapacity && outputBytes <= s.outputCapacity) return;
  if (tasks == 0 || tasks > CUDA_DBF_MAX_TASKS || lanes == 0)
    throw std::runtime_error("CUDA DBF scratch request is outside fixed limits");
  constexpr std::size_t budget = std::size_t{ 512 } * 1024 * 1024;
  const std::size_t taskBytes = tasks * sizeof(CudaDbfLumaTask);
  const std::size_t laneBytes = (lanes + 1) * sizeof(std::uint32_t);
  if (taskBytes > budget / 2 || laneBytes > budget - 2 * taskBytes
      || outputBytes > budget - 2 * taskBytes - 2 * laneBytes)
    throw std::runtime_error("CUDA DBF scratch budget exceeded");

  cudaStream_t stream = context->streams[queueIndex(CudaQueue::Dbf)];
  checkCuda(cudaStreamSynchronize(stream), "DBF scratch synchronization");
  ++context->dbfSynchronizations;
  DbfScratch candidate{};
  candidate.taskCapacity = tasks;
  candidate.laneCapacity = lanes;
  candidate.outputCapacity = outputBytes;
  try
  {
    if (consumeDbfFailure(context, CudaDbfTestFailurePoint::Allocation))
      throw std::runtime_error("Injected CUDA DBF allocation failure");
    checkCuda(cudaMallocFromPoolAsync(reinterpret_cast<void **>(&candidate.tasksDevice), taskBytes,
                                      context->memoryPool, stream), "DBF task allocation");
    checkCuda(cudaHostAlloc(reinterpret_cast<void **>(&candidate.tasksHost), taskBytes,
                            cudaHostAllocPortable), "pinned DBF task allocation");
    checkCuda(cudaMallocFromPoolAsync(reinterpret_cast<void **>(&candidate.laneOffsetsDevice), laneBytes,
                                      context->memoryPool, stream), "DBF lane offset allocation");
    checkCuda(cudaHostAlloc(reinterpret_cast<void **>(&candidate.laneOffsetsHost), laneBytes,
                            cudaHostAllocPortable), "pinned DBF lane offset allocation");
    checkCuda(cudaMallocFromPoolAsync(&candidate.outputDevice, outputBytes, context->memoryPool, stream),
              "DBF transactional output allocation");
    checkCuda(cudaStreamSynchronize(stream), "DBF scratch allocation completion");
    ++context->dbfSynchronizations;
  }
  catch (...)
  {
    for (void *p : { static_cast<void *>(candidate.tasksDevice),
                     static_cast<void *>(candidate.laneOffsetsDevice), candidate.outputDevice })
      if (p != nullptr) (void) cudaFreeAsync(p, stream);
    (void) cudaStreamSynchronize(stream);
    if (candidate.tasksHost != nullptr) (void) cudaFreeHost(candidate.tasksHost);
    if (candidate.laneOffsetsHost != nullptr) (void) cudaFreeHost(candidate.laneOffsetsHost);
    throw;
  }
  DbfScratch old = s;
  s = candidate;
  context->dbfPeakScratchBytes = std::max(context->dbfPeakScratchBytes,
                                          dbfScratchBytes(s) + dbfScratchBytes(old));
  for (void *p : { static_cast<void *>(old.tasksDevice),
                   static_cast<void *>(old.laneOffsetsDevice), old.outputDevice })
    if (p != nullptr) checkCuda(cudaFreeAsync(p, stream), "old DBF device scratch release");
  checkCuda(cudaStreamSynchronize(stream), "old DBF scratch release completion");
  ++context->dbfSynchronizations;
  if (old.tasksHost != nullptr) checkCuda(cudaFreeHost(old.tasksHost), "old pinned DBF task release");
  if (old.laneOffsetsHost != nullptr) checkCuda(cudaFreeHost(old.laneOffsetsHost), "old pinned DBF lane release");
}

__device__ __forceinline__ int dbfAbs(const int x) { return x < 0 ? -x : x; }
__device__ __forceinline__ int dbfClip(const int lo, const int hi, const int x)
{
  return x < lo ? lo : (x > hi ? hi : x);
}

template<typename Sample>
__device__ int dbfCalcDP(const Sample *src, const std::ptrdiff_t offset)
{
  return dbfAbs(int(src[-3 * offset]) - 2 * int(src[-2 * offset]) + int(src[-offset]));
}

template<typename Sample>
__device__ int dbfCalcDQ(const Sample *src, const std::ptrdiff_t offset)
{
  return dbfAbs(int(src[2 * offset]) - 2 * int(src[offset]) + int(src[0]));
}

template<typename Sample>
__device__ bool dbfStrong(const Sample *src, const std::ptrdiff_t offset, const int d,
                          const int beta, const int tc, const bool largeP, const bool largeQ,
                          const int lenP, const int lenQ)
{
  const int m4 = int(src[0]), m3 = int(src[-offset]), m7 = int(src[3 * offset]);
  const int m0 = int(src[-4 * offset]), m2 = int(src[-2 * offset]);
  int sp3 = dbfAbs(m0 - m3), sq3 = dbfAbs(m7 - m4);
  if (largeP || largeQ)
  {
    if (largeP)
    {
      int mP4;
      if (lenP == 7)
      {
        const int mP5 = int(src[-5 * offset]), mP6 = int(src[-6 * offset]);
        const int mP7 = int(src[-7 * offset]);
        mP4 = int(src[-8 * offset]);
        sp3 += dbfAbs(mP5 - mP6 - mP7 + mP4);
      }
      else mP4 = int(src[-6 * offset]);
      sp3 = (sp3 + dbfAbs(m0 - mP4) + 1) >> 1;
    }
    if (largeQ)
    {
      int m11;
      if (lenQ == 7)
      {
        const int m8 = int(src[4 * offset]), m9 = int(src[5 * offset]), m10 = int(src[6 * offset]);
        m11 = int(src[7 * offset]);
        sq3 += dbfAbs(m8 - m9 - m10 + m11);
      }
      else m11 = int(src[5 * offset]);
      sq3 = (sq3 + dbfAbs(m11 - m7) + 1) >> 1;
    }
    return sp3 + sq3 < (beta * 3 >> 5) && d < (beta >> 4)
           && dbfAbs(m3 - m4) < ((tc * 5 + 1) >> 1);
  }
  return sp3 + sq3 < (beta >> 3) && d < (beta >> 2)
         && dbfAbs(m3 - m4) < ((tc * 5 + 1) >> 1);
}

__device__ int dbfCoeff(const int len, const int pos)
{
  if (len == 3) { const int a[3] = {53,32,11}; return a[pos]; }
  if (len == 5) { const int a[5] = {58,45,32,19,6}; return a[pos]; }
  const int a[7] = {59,50,41,32,23,14,5}; return a[pos];
}
__device__ int dbfTcWeight(const int len, const int pos)
{
  if (len == 3) { const int a[3] = {6,4,2}; return a[pos]; }
  if (len == 5) { const int a[5] = {6,5,4,3,2}; return a[pos]; }
  const int a[7] = {6,5,4,3,2,1,1}; return a[pos];
}

template<typename Sample>
__device__ void dbfLongFilter(Sample *src, const std::ptrdiff_t offset, const int lenP,
                              const int lenQ, const int tc)
{
  Sample *p = src - offset;
  Sample *q = src;
  const int refP = (int(p[-(lenP * offset - offset)]) + int(p[-lenP * offset]) + 1) >> 1;
  const int refQ = (int(q[lenQ * offset - offset]) + int(q[lenQ * offset]) + 1) >> 1;
  int middle;
  if (lenP == lenQ)
  {
    if (lenP == 5)
      middle = (2 * (int(p[0])+int(q[0])+int(p[-offset])+int(q[offset])+int(p[-2*offset])+int(q[2*offset]))
                + int(p[-3*offset])+int(q[3*offset])+int(p[-4*offset])+int(q[4*offset])+8) >> 4;
    else
      middle = (2*(int(p[0])+int(q[0]))+int(p[-offset])+int(q[offset])+int(p[-2*offset])+int(q[2*offset])
                +int(p[-3*offset])+int(q[3*offset])+int(p[-4*offset])+int(q[4*offset])
                +int(p[-5*offset])+int(q[5*offset])+int(p[-6*offset])+int(q[6*offset])+8) >> 4;
  }
  else if ((lenP == 7 && lenQ == 5) || (lenP == 5 && lenQ == 7))
    middle = (2*(int(p[0])+int(q[0])+int(p[-offset])+int(q[offset]))+int(p[-2*offset])+int(q[2*offset])
              +int(p[-3*offset])+int(q[3*offset])+int(p[-4*offset])+int(q[4*offset])
              +int(p[-5*offset])+int(q[5*offset])+8) >> 4;
  else if (lenP == 7 || lenQ == 7)
  {
    Sample *pt = lenP == 7 ? p : q;
    Sample *qt = lenP == 7 ? q : p;
    const std::ptrdiff_t op = lenP == 7 ? -offset : offset;
    const std::ptrdiff_t oq = -op;
    middle = (2*(int(pt[0])+int(qt[0]))+int(qt[0])+2*(int(qt[oq])+int(qt[2*oq]))
              +int(pt[op])+int(qt[oq])+int(pt[2*op])+int(pt[3*op])+int(pt[4*op])
              +int(pt[5*op])+int(pt[6*op])+8) >> 4;
  }
  else
    middle = (int(p[0])+int(q[0])+int(p[-offset])+int(q[offset])+int(p[-2*offset])+int(q[2*offset])
              +int(p[-3*offset])+int(q[3*offset])+4) >> 3;
  for (int pos=0; pos<lenP; ++pos)
  {
    const int old = int(p[-offset*pos]), cv = tc * dbfTcWeight(lenP,pos) >> 1;
    p[-offset*pos] = Sample(dbfClip(old-cv, old+cv,
      (middle*dbfCoeff(lenP,pos)+refP*(64-dbfCoeff(lenP,pos))+32)>>6));
  }
  for (int pos=0; pos<lenQ; ++pos)
  {
    const int old = int(q[offset*pos]), cv = tc * dbfTcWeight(lenQ,pos) >> 1;
    q[offset*pos] = Sample(dbfClip(old-cv, old+cv,
      (middle*dbfCoeff(lenQ,pos)+refQ*(64-dbfCoeff(lenQ,pos))+32)>>6));
  }
}

template<typename Sample>
__device__ void dbfPelFilter(Sample *src, const std::ptrdiff_t offset, const CudaDbfLumaTask &t,
                             const bool strong, const bool filterP, const bool filterQ)
{
  const int m4=src[0],m3=src[-offset],m5=src[offset],m2=src[-2*offset],m6=src[2*offset];
  const int m1=src[-3*offset],m7=src[3*offset],m0=src[-4*offset];
  const bool largeP=(t.flags&CUDA_DBF_SIDE_P_LARGE)!=0, largeQ=(t.flags&CUDA_DBF_SIDE_Q_LARGE)!=0;
  int mp1=0,mp2=0,mp3=0,m8=0,m9=0,m10=0;
  if ((t.flags&CUDA_DBF_PART_P_NO_FILTER) && largeP)
  { mp1=src[-5*offset];mp2=src[-6*offset];mp3=src[-7*offset]; }
  if ((t.flags&CUDA_DBF_PART_Q_NO_FILTER) && largeQ)
  { m8=src[4*offset];m9=src[5*offset];m10=src[6*offset]; }
  if (strong)
  {
    if (largeP || largeQ)
      dbfLongFilter(src,offset,largeP?t.maxFilterLenP:3,largeQ?t.maxFilterLenQ:3,t.tc);
    else
    {
      src[-offset]=Sample(dbfClip(m3-3*t.tc,m3+3*t.tc,(m1+2*m2+2*m3+2*m4+m5+4)>>3));
      src[0]=Sample(dbfClip(m4-3*t.tc,m4+3*t.tc,(m2+2*m3+2*m4+2*m5+m6+4)>>3));
      src[-2*offset]=Sample(dbfClip(m2-2*t.tc,m2+2*t.tc,(m1+m2+m3+m4+2)>>2));
      src[offset]=Sample(dbfClip(m5-2*t.tc,m5+2*t.tc,(m3+m4+m5+m6+2)>>2));
      src[-3*offset]=Sample(dbfClip(m1-t.tc,m1+t.tc,(2*m0+3*m1+m2+m3+m4+4)>>3));
      src[2*offset]=Sample(dbfClip(m6-t.tc,m6+t.tc,(m3+m4+m5+3*m6+2*m7+4)>>3));
    }
  }
  else
  {
    int delta=(9*(m4-m3)-3*(m5-m2)+8)>>4;
    if (dbfAbs(delta)<t.tc*10)
    {
      delta=dbfClip(-t.tc,t.tc,delta);
      src[-offset]=Sample(dbfClip(t.minSample,t.maxSample,m3+delta));
      src[0]=Sample(dbfClip(t.minSample,t.maxSample,m4-delta));
      const int tc2=t.tc>>1;
      if(filterP) src[-2*offset]=Sample(dbfClip(t.minSample,t.maxSample,m2+dbfClip(-tc2,tc2,((((m1+m3+1)>>1)-m2+delta)>>1))));
      if(filterQ) src[offset]=Sample(dbfClip(t.minSample,t.maxSample,m5+dbfClip(-tc2,tc2,((((m6+m4+1)>>1)-m5-delta)>>1))));
    }
  }
  if(t.flags&CUDA_DBF_PART_P_NO_FILTER)
  {
    src[-offset]=Sample(m3);src[-2*offset]=Sample(m2);src[-3*offset]=Sample(m1);
    if(largeP){src[-4*offset]=Sample(m0);src[-5*offset]=Sample(mp1);src[-6*offset]=Sample(mp2);src[-7*offset]=Sample(mp3);}
  }
  if(t.flags&CUDA_DBF_PART_Q_NO_FILTER)
  {
    src[0]=Sample(m4);src[offset]=Sample(m5);src[2*offset]=Sample(m6);
    if(largeQ){src[3*offset]=Sample(m7);src[4*offset]=Sample(m8);src[5*offset]=Sample(m9);src[6*offset]=Sample(m10);}
  }
}

template<typename Sample>
__device__ void dbfApplyTask(Sample *base, const std::size_t stride, const CudaDbfLumaTask &t)
{
  Sample *src0=base+std::size_t(t.y)*stride+t.x;
  const std::ptrdiff_t offset=t.direction==0?1:std::ptrdiff_t(stride);
  const std::ptrdiff_t step=t.direction==0?std::ptrdiff_t(stride):1;
  const int dp0=dbfCalcDP(src0,offset),dq0=dbfCalcDQ(src0,offset);
  Sample *src3=src0+3*step;
  const int dp3=dbfCalcDP(src3,offset),dq3=dbfCalcDQ(src3,offset);
  const bool largeP=(t.flags&CUDA_DBF_SIDE_P_LARGE)!=0,largeQ=(t.flags&CUDA_DBF_SIDE_Q_LARGE)!=0;
  bool usedLong=false, filterP=false, filterQ=false;
  if(largeP||largeQ)
  {
    const int dp0l=largeP?(dp0+dbfCalcDP(src0-3*offset,offset)+1)>>1:dp0;
    const int dp3l=largeP?(dp3+dbfCalcDP(src3-3*offset,offset)+1)>>1:dp3;
    const int dq0l=largeQ?(dq0+dbfCalcDQ(src0+3*offset,offset)+1)>>1:dq0;
    const int dq3l=largeQ?(dq3+dbfCalcDQ(src3+3*offset,offset)+1)>>1:dq3;
    if(dp0l+dq0l+dp3l+dq3l<t.beta)
    {
      const int side=(t.beta+(t.beta>>1))>>3;
      filterP=dp0l+dp3l<side;filterQ=dq0l+dq3l<side;
      usedLong=dbfStrong(src0,offset,2*(dp0l+dq0l),t.beta,t.tc,largeP,largeQ,t.maxFilterLenP,t.maxFilterLenQ)
        &&dbfStrong(src3,offset,2*(dp3l+dq3l),t.beta,t.tc,largeP,largeQ,t.maxFilterLenP,t.maxFilterLenQ);
      if(usedLong) for(int i=0;i<4;++i) dbfPelFilter(src0+i*step,offset,t,true,filterP,filterQ);
    }
  }
  if(!usedLong)
  {
    const int d0=dp0+dq0,d3=dp3+dq3;
    if(d0+d3<t.beta)
    {
      const int side=(t.beta+(t.beta>>1))>>3;
      filterP=t.maxFilterLenP>1&&t.maxFilterLenQ>1&&dp0+dp3<side;
      filterQ=t.maxFilterLenP>1&&t.maxFilterLenQ>1&&dq0+dq3<side;
      const bool strong=t.maxFilterLenP>2&&t.maxFilterLenQ>2
        &&dbfStrong(src0,offset,2*d0,t.beta,t.tc,false,false,3,3)
        &&dbfStrong(src3,offset,2*d3,t.beta,t.tc,false,false,3,3);
      for(int i=0;i<4;++i) dbfPelFilter(src0+i*step,offset,t,strong,filterP,filterQ);
    }
  }
}

template<typename Sample>
__global__ void dbfLaneKernel(Sample *base, const std::size_t pitchBytes,
                              const CudaDbfLumaTask *tasks, const std::uint32_t *offsets,
                              const std::uint32_t lanes)
{
  const std::uint32_t lane=blockIdx.x*blockDim.x+threadIdx.x;
  if(lane>=lanes)return;
  const std::size_t stride=pitchBytes/sizeof(Sample);
  for(std::uint32_t i=offsets[lane];i<offsets[lane+1];++i) dbfApplyTask(base,stride,tasks[i]);
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

template<typename Sample>
__global__ void qpaBatchKernel(const Sample *luma, const std::size_t pitchBytes,
                               const CudaQpaTask *tasks, CudaQpaResult *results)
{
  const CudaQpaTask task = tasks[blockIdx.x];
  const std::size_t stride = pitchBytes / sizeof(Sample);
  const CudaQpaRect flt = task.filterArea;
  const Sample *filterBase = luma + static_cast<std::size_t>(flt.y) * stride + flt.x;
  const std::uint64_t innerWidth = flt.width - 2;
  const std::uint64_t filterSamples = innerWidth * (flt.height - 2);

  std::uint64_t highpassSum = 0;
  for (std::uint64_t sample = threadIdx.x; sample < filterSamples; sample += blockDim.x)
  {
    const std::uint32_t y = static_cast<std::uint32_t>(sample / innerWidth) + 1;
    const std::uint32_t x = static_cast<std::uint32_t>(sample % innerWidth) + 1;
    const Sample *row = filterBase + static_cast<std::size_t>(y) * stride;
    const Sample *previousRow = row - stride;
    const Sample *nextRow = row + stride;
    // The host preflight admits only 8/10-bit samples. This exactly mirrors the original signed-int expression.
    const int f = 12 * int(row[x])
                  - 2 * (int(row[x - 1]) + int(row[x + 1]) + int(previousRow[x]) + int(nextRow[x]))
                  - int(previousRow[x - 1]) - int(previousRow[x + 1])
                  - int(nextRow[x - 1]) - int(nextRow[x + 1]);
    highpassSum += static_cast<std::uint64_t>(f < 0 ? -f : f);
  }

  const CudaQpaRect area = task.lumaArea;
  const Sample *areaBase = luma + static_cast<std::size_t>(area.y) * stride + area.x;
  const std::uint64_t areaSamples = static_cast<std::uint64_t>(area.width) * area.height;
  std::int64_t lumaSum = 0;
  for (std::uint64_t sample = threadIdx.x; sample < areaSamples; sample += blockDim.x)
  {
    const std::uint32_t y = static_cast<std::uint32_t>(sample / area.width);
    const std::uint32_t x = static_cast<std::uint32_t>(sample % area.width);
    lumaSum += areaBase[static_cast<std::size_t>(y) * stride + x];
  }

  __shared__ std::uint64_t highpassPartial[256];
  __shared__ std::int64_t  lumaPartial[256];
  highpassPartial[threadIdx.x] = highpassSum;
  lumaPartial[threadIdx.x] = lumaSum;
  __syncthreads();
  for (unsigned offset = blockDim.x / 2; offset != 0; offset >>= 1)
  {
    if (threadIdx.x < offset)
    {
      highpassPartial[threadIdx.x] += highpassPartial[threadIdx.x + offset];
      lumaPartial[threadIdx.x] += lumaPartial[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
  {
    CudaQpaResult result{};
    result.ticket = task.ticket;
    result.ctuAddr = task.ctuAddr;
    result.highpassSum = highpassPartial[0];
    result.lumaSum = lumaPartial[0];
    results[blockIdx.x] = result;
  }
}

template<typename Sample>
__device__ int alfSample(const Sample *source, const std::size_t stride,
                         const int width, const int height, int x, int y)
{
  x = x < 0 ? 0 : (x >= width ? width - 1 : x);
  y = y < 0 ? 0 : (y >= height ? height - 1 : y);
  return int(source[static_cast<std::size_t>(y) * stride + x]);
}

template<typename Sample>
__global__ void alfClassifyKernel(const Sample *source, const std::size_t pitchBytes,
                                  const CudaAlfLumaFrame frame, const std::uint32_t ctuCount,
                                  CudaAlfClassifier *classifiers)
{
  const std::uint32_t classesWide = (frame.width + 3) >> 2;
  const std::uint32_t classesHigh = (frame.height + 3) >> 2;
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= classesWide * classesHigh)
  {
    return;
  }
  const int x0 = int(index % classesWide) * 4;
  const int y0 = int(index / classesWide) * 4;
  const std::uint32_t ctuX = std::uint32_t(x0) / frame.ctuWidth;
  const std::uint32_t ctuY = std::uint32_t(y0) / frame.ctuHeight;
  const std::uint32_t ctu = ctuY * frame.ctusInWidth + ctuX;
  // Disabled CTUs are never filtered. Initializing their classifier keeps optional diagnostics deterministic.
  if (ctu >= ctuCount)
  {
    classifiers[index] = CudaAlfClassifier{ 0, 0 };
    return;
  }

  const std::size_t stride = pitchBytes / sizeof(Sample);
  int sumV = 0, sumH = 0, sumD0 = 0, sumD1 = 0;
  const int yMod = y0 & (frame.vbCtuHeight - 1);
  const int firstRow = yMod == frame.vbPos ? 2 : 0;
  const int lastRow = yMod == frame.vbPos - 4 ? 4 : 6;
  for (int rr = firstRow; rr <= lastRow; rr += 2)
  {
    const int baseY = y0 + rr - 2;
    int rowMinus = baseY - 1;
    int row = baseY;
    int rowPlus = baseY + 1;
    int rowPlus2 = baseY + 2;
    if (baseY > 0 && (baseY & (frame.vbCtuHeight - 1)) == frame.vbPos - 2)
    {
      rowPlus2 = rowPlus;
    }
    else if (baseY > 0 && (baseY & (frame.vbCtuHeight - 1)) == frame.vbPos)
    {
      rowMinus = row;
    }
    for (int cc = 0; cc <= 6; cc += 2)
    {
      // The scalar VTM base pointer is shifted left by flP1 (3) before pixY=j+1 is applied.
      const int x = x0 + cc - 2;
      const int y = alfSample(source, stride, int(frame.width), int(frame.height), x, row);
      const int yp1 = alfSample(source, stride, int(frame.width), int(frame.height), x + 1, rowPlus);
      const int ver0 = 2 * y
                       - alfSample(source, stride, int(frame.width), int(frame.height), x, rowMinus)
                       - alfSample(source, stride, int(frame.width), int(frame.height), x, rowPlus);
      const int ver1 = 2 * yp1
                       - alfSample(source, stride, int(frame.width), int(frame.height), x + 1, row)
                       - alfSample(source, stride, int(frame.width), int(frame.height), x + 1, rowPlus2);
      const int hor0 = 2 * y
                       - alfSample(source, stride, int(frame.width), int(frame.height), x + 1, row)
                       - alfSample(source, stride, int(frame.width), int(frame.height), x - 1, row);
      const int hor1 = 2 * yp1
                       - alfSample(source, stride, int(frame.width), int(frame.height), x + 2, rowPlus)
                       - alfSample(source, stride, int(frame.width), int(frame.height), x, rowPlus);
      const int d00 = 2 * y
                      - alfSample(source, stride, int(frame.width), int(frame.height), x - 1, rowMinus)
                      - alfSample(source, stride, int(frame.width), int(frame.height), x + 1, rowPlus);
      const int d01 = 2 * yp1
                      - alfSample(source, stride, int(frame.width), int(frame.height), x, row)
                      - alfSample(source, stride, int(frame.width), int(frame.height), x + 2, rowPlus2);
      const int d10 = 2 * y
                      - alfSample(source, stride, int(frame.width), int(frame.height), x - 1, rowPlus)
                      - alfSample(source, stride, int(frame.width), int(frame.height), x + 1, rowMinus);
      const int d11 = 2 * yp1
                      - alfSample(source, stride, int(frame.width), int(frame.height), x, rowPlus2)
                      - alfSample(source, stride, int(frame.width), int(frame.height), x + 2, row);
      sumV += abs(ver0) + abs(ver1);
      sumH += abs(hor0) + abs(hor1);
      sumD0 += abs(d00) + abs(d01);
      sumD1 += abs(d10) + abs(d11);
    }
  }

  const int tempAct = sumV + sumH;
  const int scale = (yMod == frame.vbPos - 4 || yMod == frame.vbPos) ? 96 : 64;
  int activity = (tempAct * scale) >> (frame.bitDepth + 4);
  activity = activity < 0 ? 0 : (activity > 15 ? 15 : activity);
  constexpr int thresholds[16] = { 0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4 };
  int classIdx = thresholds[activity];

  const int hv1 = sumV > sumH ? sumV : sumH;
  const int hv0 = sumV > sumH ? sumH : sumV;
  const int dirHV = sumV > sumH ? 1 : 3;
  const int d1 = sumD0 > sumD1 ? sumD0 : sumD1;
  const int d0 = sumD0 > sumD1 ? sumD1 : sumD0;
  const int dirD = sumD0 > sumD1 ? 0 : 2;
  const bool diagonalMain = std::uint32_t(d1) * std::uint32_t(hv0)
                            > std::uint32_t(hv1) * std::uint32_t(d0);
  const int mainDirection = diagonalMain ? dirD : dirHV;
  const int secondaryDirection = diagonalMain ? dirHV : dirD;
  const int strongest = diagonalMain ? d1 : hv1;
  const int weakest = diagonalMain ? d0 : hv0;
  int strength = strongest > 2 * weakest ? 1 : 0;
  if (strongest * 2 > 9 * weakest) strength = 2;
  if (strength) classIdx += (((mainDirection & 1) << 1) + strength) * 5;
  constexpr int transposeTable[8] = { 0, 1, 0, 2, 2, 3, 1, 3 };
  classifiers[index] = CudaAlfClassifier{ std::uint8_t(classIdx),
                                          std::uint8_t(transposeTable[mainDirection * 2
                                                                      + (secondaryDirection >> 1)]) };
}

__device__ int alfClipPair(const int clip, const int current, const int a, const int b)
{
  int da = a - current;
  int db = b - current;
  da = da < -clip ? -clip : (da > clip ? clip : da);
  db = db < -clip ? -clip : (db > clip ? clip : db);
  return da + db;
}

template<typename Sample>
__global__ void alfFilterKernel(const Sample *source, const std::size_t sourcePitchBytes,
                                Sample *output, const std::size_t outputStride,
                                const CudaAlfLumaFrame frame, const CudaAlfCtuParam *ctus,
                                const std::uint32_t ctuCount, const CudaAlfClassifier *classifiers)
{
  const int x = int(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = int(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= int(frame.width) || y >= int(frame.height)) return;
  const std::size_t sourceStride = sourcePitchBytes / sizeof(Sample);
  const std::uint32_t ctuIndex = std::uint32_t(y) / frame.ctuHeight * frame.ctusInWidth
                                 + std::uint32_t(x) / frame.ctuWidth;
  const int current = alfSample(source, sourceStride, int(frame.width), int(frame.height), x, y);
  if (ctuIndex >= ctuCount)
  {
    output[static_cast<std::size_t>(y) * outputStride + x] = Sample(current);
    return;
  }
  if (!ctus[ctuIndex].enabled)
  {
    output[static_cast<std::size_t>(y) * outputStride + x] = Sample(current);
    return;
  }

  const std::uint32_t classesWide = (frame.width + 3) >> 2;
  const CudaAlfClassifier classifier = classifiers[std::uint32_t(y >> 2) * classesWide + std::uint32_t(x >> 2)];
  constexpr int map[4][13] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 },
    { 9, 4, 10, 8, 1, 5, 11, 7, 3, 0, 2, 6, 12 },
    { 0, 3, 2, 1, 8, 7, 6, 5, 4, 9, 10, 11, 12 },
    { 9, 8, 10, 4, 3, 7, 11, 5, 1, 0, 2, 6, 12 }
  };
  const CudaAlfCtuParam &ctu = ctus[ctuIndex];
  const std::uint32_t classBase = std::uint32_t(classifier.classIdx) * CUDA_ALF_COEFFICIENTS;
  int rows[7] = { y, y + 1, y - 1, y + 2, y - 2, y + 3, y - 3 };
  const int yVb = y & (frame.vbCtuHeight - 1);
  if (yVb < frame.vbPos && yVb >= frame.vbPos - 4)
  {
    rows[1] = yVb == frame.vbPos - 1 ? rows[0] : rows[1];
    rows[3] = yVb >= frame.vbPos - 2 ? rows[1] : rows[3];
    rows[5] = yVb >= frame.vbPos - 3 ? rows[3] : rows[5];
    rows[2] = yVb == frame.vbPos - 1 ? rows[0] : rows[2];
    rows[4] = yVb >= frame.vbPos - 2 ? rows[2] : rows[4];
    rows[6] = yVb >= frame.vbPos - 3 ? rows[4] : rows[6];
  }
  else if (yVb >= frame.vbPos && yVb <= frame.vbPos + 3)
  {
    rows[2] = yVb == frame.vbPos ? rows[0] : rows[2];
    rows[4] = yVb <= frame.vbPos + 1 ? rows[2] : rows[4];
    rows[6] = yVb <= frame.vbPos + 2 ? rows[4] : rows[6];
    rows[1] = yVb == frame.vbPos ? rows[0] : rows[1];
    rows[3] = yVb <= frame.vbPos + 1 ? rows[1] : rows[3];
    rows[5] = yVb <= frame.vbPos + 2 ? rows[3] : rows[5];
  }
  const int dx0[12] = { 0, 1, 0, -1, 2, 1, 0, -1, -2, 3, 2, 1 };
  const int dx1[12] = { 0, -1, 0, 1, -2, -1, 0, 1, 2, -3, -2, -1 };
  const int row0[12] = { 5, 3, 3, 3, 1, 1, 1, 1, 1, 0, 0, 0 };
  const int row1[12] = { 6, 4, 4, 4, 2, 2, 2, 2, 2, 0, 0, 0 };
  int sum = 0;
  for (int tap = 0; tap < 12; ++tap)
  {
    const int sourceTap = map[classifier.transposeIdx][tap];
    const int coeff = int(ctu.coefficients[classBase + sourceTap]);
    const int clip = ctu.clipValues[classBase + sourceTap];
    const int a = alfSample(source, sourceStride, int(frame.width), int(frame.height), x + dx0[tap], rows[row0[tap]]);
    const int b = alfSample(source, sourceStride, int(frame.width), int(frame.height), x + dx1[tap], rows[row1[tap]]);
    sum += coeff * alfClipPair(clip, current, a, b);
  }
  const bool nearVb = (yVb == frame.vbPos - 1 || yVb == frame.vbPos);
  sum = nearVb ? ((sum + (1 << 9)) >> 10) : ((sum + 64) >> 7);
  int filtered = current + sum;
  filtered = filtered < frame.minSample ? frame.minSample : (filtered > frame.maxSample ? frame.maxSample : filtered);
  output[static_cast<std::size_t>(y) * outputStride + x] = Sample(filtered);
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

bool releasePinnedHost(void *allocation) noexcept
{
  if (allocation == nullptr)
  {
    return true;
  }
#if VTM_CUDA_TESTING
  unsigned failures = pinnedReleaseFailures.load(std::memory_order_relaxed);
  while (failures != 0)
  {
    if (pinnedReleaseFailures.compare_exchange_weak(failures, failures - 1, std::memory_order_relaxed))
    {
      return false;
    }
  }
#endif
  return cudaFreeHost(allocation) == cudaSuccess;
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

  if (consumeBatchFailure(context->distortionFailurePoint, CudaBatchTestFailurePoint::KernelLaunch))
  {
    throw std::runtime_error("Injected CUDA SAD kernel launch failure");
  }

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
  if (consumeBatchFailure(context->distortionFailurePoint, CudaBatchTestFailurePoint::ResultDownload))
  {
    throw std::runtime_error("Injected CUDA SAD result download failure");
  }
  checkCuda(cudaMemcpyAsync(context->distortionResultsHost, context->distortionResultsDevice,
                            static_cast<std::size_t>(candidateCount) * sizeof(*results),
                            cudaMemcpyDeviceToHost, stream),
            "distortion result download");
  if (consumeBatchFailure(context->distortionFailurePoint, CudaBatchTestFailurePoint::Completion))
  {
    throw std::runtime_error("Injected CUDA SAD completion failure");
  }
  checkCuda(cudaStreamSynchronize(stream), "distortion batch completion");
  if (consumeBatchFailure(context->distortionFailurePoint, CudaBatchTestFailurePoint::Publication))
  {
    throw std::runtime_error("Injected CUDA SAD publication failure");
  }
  std::memcpy(results, context->distortionResultsHost,
              static_cast<std::size_t>(candidateCount) * sizeof(*results));
  ++context->distortionDispatches;
}

void computeQpaBatch(RuntimeContext *context, const CudaDevicePlaneDesc &source,
                     const CudaQpaTask *tasks, const std::uint32_t taskCount, CudaQpaResult *results)
{
  checkCuda(cudaSetDevice(context->device), "device selection");
  ensureQpaCapacity(context, taskCount);
  cudaStream_t stream = context->streams[queueIndex(CudaQueue::Qpa)];
#if VTM_CUDA_TESTING
  if (context->qpaExecutionFailures > 0)
  {
    --context->qpaExecutionFailures;
    throw std::runtime_error("Injected CUDA QPA execution failure");
  }
#endif

  const std::size_t taskBytes = static_cast<std::size_t>(taskCount) * sizeof(*tasks);
  const std::size_t resultBytes = static_cast<std::size_t>(taskCount) * sizeof(*results);
  std::memcpy(context->qpaTasksHost, tasks, taskBytes);
  if (consumeBatchFailure(context->qpaFailurePoint, CudaBatchTestFailurePoint::Upload))
  {
    throw std::runtime_error("Injected CUDA QPA task upload failure");
  }
  checkCuda(cudaMemcpyAsync(context->qpaTasksDevice, context->qpaTasksHost, taskBytes,
                            cudaMemcpyHostToDevice, stream),
            "QPA task upload");

  if (consumeBatchFailure(context->qpaFailurePoint, CudaBatchTestFailurePoint::KernelLaunch))
  {
    throw std::runtime_error("Injected CUDA QPA kernel launch failure");
  }

  constexpr unsigned threads = 256;
  if (source.elementSize == 2)
  {
    qpaBatchKernel<std::int16_t><<<taskCount, threads, 0, stream>>>(
      static_cast<const std::int16_t *>(source.data), source.pitchBytes,
      context->qpaTasksDevice, context->qpaResultsDevice);
  }
  else
  {
    qpaBatchKernel<std::int32_t><<<taskCount, threads, 0, stream>>>(
      static_cast<const std::int32_t *>(source.data), source.pitchBytes,
      context->qpaTasksDevice, context->qpaResultsDevice);
  }
  checkCuda(cudaGetLastError(), "QPA batch launch");
  if (consumeBatchFailure(context->qpaFailurePoint, CudaBatchTestFailurePoint::ResultDownload))
  {
    throw std::runtime_error("Injected CUDA QPA result download failure");
  }
  checkCuda(cudaMemcpyAsync(context->qpaResultsHost, context->qpaResultsDevice, resultBytes,
                            cudaMemcpyDeviceToHost, stream),
            "QPA result download");
  if (consumeBatchFailure(context->qpaFailurePoint, CudaBatchTestFailurePoint::Completion))
  {
    throw std::runtime_error("Injected CUDA QPA completion failure");
  }
  checkCuda(cudaStreamSynchronize(stream), "QPA batch completion");
  if (consumeBatchFailure(context->qpaFailurePoint, CudaBatchTestFailurePoint::ResultCorruption))
  {
    context->qpaResultsHost[0].ticket ^= 1;
  }
  for (std::uint32_t index = 0; index < taskCount; ++index)
  {
    if (context->qpaResultsHost[index].ticket != context->qpaTasksHost[index].ticket
        || context->qpaResultsHost[index].ctuAddr != context->qpaTasksHost[index].ctuAddr)
    {
      throw std::runtime_error("CUDA QPA staged result ticket/order mismatch");
    }
  }
  if (consumeBatchFailure(context->qpaFailurePoint, CudaBatchTestFailurePoint::Publication))
  {
    throw std::runtime_error("Injected CUDA QPA publication failure");
  }
  std::memcpy(results, context->qpaResultsHost, resultBytes);
  ++context->qpaDispatches;
  context->qpaTasks += taskCount;
}

void filterAlfLumaFrame(RuntimeContext *context, const CudaDevicePlaneDesc &plane,
                        const CudaAlfLumaFrame &frame, const CudaAlfCtuParam *ctus,
                        const std::uint32_t ctuCount, CudaAlfClassifier *classifiers)
{
  const auto start = std::chrono::steady_clock::now();
  checkCuda(cudaSetDevice(context->device), "device selection");
  const std::size_t classesWide = (frame.width + 3) >> 2;
  const std::size_t classesHigh = (frame.height + 3) >> 2;
  const std::size_t classifierCount = classesWide * classesHigh;
  const std::size_t outputStrideBytes = static_cast<std::size_t>(frame.width) * frame.elementSize;
  const std::size_t outputBytes = outputStrideBytes * frame.height;
  ensureAlfCapacity(context, ctuCount, classifierCount, outputBytes);
  cudaStream_t stream = context->streams[queueIndex(CudaQueue::Alf)];
  AlfScratch &scratch = context->alfScratch;

  const std::size_t ctuBytes = static_cast<std::size_t>(ctuCount) * sizeof(*ctus);
  std::memcpy(scratch.ctusHost, ctus, ctuBytes);
  if (consumeAlfFailure(context, CudaAlfTestFailurePoint::ParameterUpload))
    throw std::runtime_error("Injected CUDA ALF parameter upload failure");
  checkCuda(cudaMemcpyAsync(scratch.ctusDevice, scratch.ctusHost, ctuBytes,
                            cudaMemcpyHostToDevice, stream), "ALF CTU parameter upload");

  constexpr unsigned classifierThreads = 256;
  const unsigned classifierBlocks = unsigned((classifierCount + classifierThreads - 1) / classifierThreads);
  const dim3 filterThreads(16, 16);
  const dim3 filterBlocks((frame.width + filterThreads.x - 1) / filterThreads.x,
                          (frame.height + filterThreads.y - 1) / filterThreads.y);
  if (consumeAlfFailure(context, CudaAlfTestFailurePoint::KernelLaunch))
    throw std::runtime_error("Injected CUDA ALF kernel launch failure");
  if (frame.elementSize == 2)
  {
    alfClassifyKernel<std::int16_t><<<classifierBlocks, classifierThreads, 0, stream>>>(
      static_cast<const std::int16_t *>(plane.data), plane.pitchBytes, frame, ctuCount,
      scratch.classifiersDevice);
    alfFilterKernel<std::int16_t><<<filterBlocks, filterThreads, 0, stream>>>(
      static_cast<const std::int16_t *>(plane.data), plane.pitchBytes,
      static_cast<std::int16_t *>(scratch.outputDevice), outputStrideBytes / sizeof(std::int16_t),
      frame, scratch.ctusDevice, ctuCount, scratch.classifiersDevice);
  }
  else
  {
    alfClassifyKernel<std::int32_t><<<classifierBlocks, classifierThreads, 0, stream>>>(
      static_cast<const std::int32_t *>(plane.data), plane.pitchBytes, frame, ctuCount,
      scratch.classifiersDevice);
    alfFilterKernel<std::int32_t><<<filterBlocks, filterThreads, 0, stream>>>(
      static_cast<const std::int32_t *>(plane.data), plane.pitchBytes,
      static_cast<std::int32_t *>(scratch.outputDevice), outputStrideBytes / sizeof(std::int32_t),
      frame, scratch.ctusDevice, ctuCount, scratch.classifiersDevice);
  }
  checkCuda(cudaGetLastError(), "ALF kernel launch");
  if (consumeAlfFailure(context, CudaAlfTestFailurePoint::KernelCompletion))
    throw std::runtime_error("Injected CUDA ALF kernel completion failure");
  checkCuda(cudaStreamSynchronize(stream), "ALF transactional output completion");
  ++context->alfSynchronizations;
  if (classifiers != nullptr)
  {
    if (consumeAlfFailure(context, CudaAlfTestFailurePoint::DiagnosticDownload))
      throw std::runtime_error("Injected CUDA ALF diagnostic download failure");
    checkCuda(cudaMemcpyAsync(classifiers, scratch.classifiersDevice,
                              classifierCount * sizeof(*classifiers), cudaMemcpyDeviceToHost, stream),
              "ALF classifier diagnostic download");
    checkCuda(cudaStreamSynchronize(stream), "ALF classifier diagnostic completion");
    ++context->alfSynchronizations;
  }
  if (consumeAlfFailure(context, CudaAlfTestFailurePoint::CommitCopy))
    throw std::runtime_error("Injected CUDA ALF reconstruction commit copy failure");
  checkCuda(cudaMemcpy2DAsync(plane.data, plane.pitchBytes, scratch.outputDevice, outputStrideBytes,
                              outputStrideBytes, frame.height, cudaMemcpyDeviceToDevice, stream),
            "ALF reconstruction commit");
  if (consumeAlfFailure(context, CudaAlfTestFailurePoint::CommitCompletion))
    throw std::runtime_error("Injected CUDA ALF reconstruction commit completion failure");
  checkCuda(cudaStreamSynchronize(stream), "ALF reconstruction commit completion");
  ++context->alfSynchronizations;
  ++context->alfDispatches;
  context->alfCtus += ctuCount;
  context->alfPixels += static_cast<std::uint64_t>(frame.width) * frame.height;
  context->alfParameterUploadBytes += ctuBytes;
  if (classifiers != nullptr)
    context->alfDiagnosticDownloadBytes += classifierCount * sizeof(*classifiers);
  context->alfCommitBytes += outputBytes;
  context->alfElapsedNanoseconds += static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

void filterDbfLumaFrame(RuntimeContext *context, const CudaDevicePlaneDesc &plane,
                        const CudaDbfFrame &frame, const CudaDbfLumaTask *tasks,
                        const std::uint32_t taskCount)
{
  const auto start = std::chrono::steady_clock::now();
  checkCuda(cudaSetDevice(context->device), "device selection");
  const std::size_t rowBytes = std::size_t(frame.width) * frame.elementSize;
  const std::size_t fullWidth = std::size_t(plane.marginLeft) + frame.width + plane.marginRight;
  const std::size_t fullHeight = std::size_t(plane.marginTop) + frame.height + plane.marginBottom;
  const std::size_t scratchPitch = fullWidth * frame.elementSize;
  const std::size_t outputBytes = scratchPitch * fullHeight;
  const std::uint32_t verticalLanes = (frame.height + 3) >> 2;
  const std::uint32_t horizontalLanes = (frame.width + 3) >> 2;
  std::uint32_t directionCounts[2] = { 0, 0 };
  for (std::uint32_t i = 0; i < taskCount; ++i) ++directionCounts[tasks[i].direction];
  ensureDbfCapacity(context, std::max(directionCounts[0], directionCounts[1]),
                    std::max(verticalLanes, horizontalLanes), outputBytes);
  cudaStream_t stream = context->streams[queueIndex(CudaQueue::Dbf)];
  DbfScratch &s = context->dbfScratch;

  if (consumeDbfFailure(context, CudaDbfTestFailurePoint::SnapshotCopy))
    throw std::runtime_error("Injected CUDA DBF snapshot copy failure");
  const auto *planeBase = static_cast<const std::uint8_t *>(plane.data)
                          - std::size_t(plane.marginTop) * plane.pitchBytes
                          - std::size_t(plane.marginLeft) * frame.elementSize;
  checkCuda(cudaMemcpy2DAsync(s.outputDevice, scratchPitch, planeBase, plane.pitchBytes, scratchPitch,
                              fullHeight, cudaMemcpyDeviceToDevice, stream), "DBF snapshot copy");
  auto *outputActive = static_cast<std::uint8_t *>(s.outputDevice)
                       + std::size_t(plane.marginTop) * scratchPitch
                       + std::size_t(plane.marginLeft) * frame.elementSize;

  std::uint64_t uploaded = 0;
  for (std::uint8_t direction = 0; direction < 2; ++direction)
  {
    const std::uint32_t lanes = direction == 0 ? verticalLanes : horizontalLanes;
    const std::uint32_t count = directionCounts[direction];
    std::fill_n(s.laneOffsetsHost, std::size_t(lanes) + 1, 0u);
    for (std::uint32_t i = 0; i < taskCount; ++i)
      if (tasks[i].direction == direction)
      {
        const std::uint32_t lane = direction == 0 ? tasks[i].y >> 2 : tasks[i].x >> 2;
        ++s.laneOffsetsHost[lane + 1];
      }
    for (std::uint32_t lane = 0; lane < lanes; ++lane)
      s.laneOffsetsHost[lane + 1] += s.laneOffsetsHost[lane];
    std::vector<std::uint32_t> cursor(s.laneOffsetsHost, s.laneOffsetsHost + lanes);
    for (std::uint32_t i = 0; i < taskCount; ++i)
      if (tasks[i].direction == direction)
      {
        const std::uint32_t lane = direction == 0 ? tasks[i].y >> 2 : tasks[i].x >> 2;
        s.tasksHost[cursor[lane]++] = tasks[i];
      }
    if (consumeDbfFailure(context, CudaDbfTestFailurePoint::ParameterUpload))
      throw std::runtime_error("Injected CUDA DBF parameter upload failure");
    const std::size_t taskBytes = std::size_t(count) * sizeof(CudaDbfLumaTask);
    const std::size_t laneBytes = (std::size_t(lanes) + 1) * sizeof(std::uint32_t);
    checkCuda(cudaMemcpyAsync(s.tasksDevice, s.tasksHost, taskBytes, cudaMemcpyHostToDevice, stream),
              "DBF task upload");
    checkCuda(cudaMemcpyAsync(s.laneOffsetsDevice, s.laneOffsetsHost, laneBytes,
                              cudaMemcpyHostToDevice, stream), "DBF lane upload");
    uploaded += taskBytes + laneBytes;
    const CudaDbfTestFailurePoint launchPoint = direction == 0
      ? CudaDbfTestFailurePoint::VerticalLaunch : CudaDbfTestFailurePoint::HorizontalLaunch;
    if (consumeDbfFailure(context, launchPoint))
      throw std::runtime_error(direction == 0 ? "Injected CUDA DBF vertical launch failure"
                                              : "Injected CUDA DBF horizontal launch failure");
    constexpr unsigned threads = 256;
    const unsigned blocks = (lanes + threads - 1) / threads;
    if (frame.elementSize == 2)
      dbfLaneKernel<std::int16_t><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<std::int16_t *>(outputActive), scratchPitch, s.tasksDevice, s.laneOffsetsDevice, lanes);
    else
      dbfLaneKernel<std::int32_t><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<std::int32_t *>(outputActive), scratchPitch, s.tasksDevice, s.laneOffsetsDevice, lanes);
    checkCuda(cudaGetLastError(), direction == 0 ? "DBF vertical kernel launch" : "DBF horizontal kernel launch");
    const CudaDbfTestFailurePoint completionPoint = direction == 0
      ? CudaDbfTestFailurePoint::VerticalCompletion : CudaDbfTestFailurePoint::HorizontalCompletion;
    if (consumeDbfFailure(context, completionPoint))
      throw std::runtime_error(direction == 0 ? "Injected CUDA DBF vertical completion failure"
                                              : "Injected CUDA DBF horizontal completion failure");
    checkCuda(cudaStreamSynchronize(stream), direction == 0 ? "DBF vertical completion" : "DBF horizontal completion");
    ++context->dbfSynchronizations;
  }
  if (consumeDbfFailure(context, CudaDbfTestFailurePoint::CommitCopy))
    throw std::runtime_error("Injected CUDA DBF commit copy failure");
  checkCuda(cudaMemcpy2DAsync(plane.data, plane.pitchBytes, outputActive, scratchPitch, rowBytes,
                              frame.height, cudaMemcpyDeviceToDevice, stream), "DBF reconstruction commit");
  if (consumeDbfFailure(context, CudaDbfTestFailurePoint::CommitCompletion))
    throw std::runtime_error("Injected CUDA DBF commit completion failure");
  checkCuda(cudaStreamSynchronize(stream), "DBF reconstruction commit completion");
  ++context->dbfSynchronizations;
  ++context->dbfDispatches;
  context->dbfTasks += taskCount;
  context->dbfPixels += std::uint64_t(frame.width) * frame.height;
  context->dbfParameterUploadBytes += uploaded;
  context->dbfCommitBytes += rowBytes * frame.height;
  context->dbfElapsedNanoseconds += static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

std::uint64_t distortionBatchDispatchCount(const RuntimeContext *context)
{
  return context->distortionDispatches;
}

std::uint64_t qpaBatchDispatchCount(const RuntimeContext *context)
{
  return context->qpaDispatches;
}

std::uint64_t qpaTaskCount(const RuntimeContext *context)
{
  return context->qpaTasks;
}

AlfAccelerationStats alfStats(const RuntimeContext *context) noexcept
{
  AlfAccelerationStats stats{};
  if (context == nullptr) return stats;
  stats.dispatches = context->alfDispatches;
  stats.ctus = context->alfCtus;
  stats.pixels = context->alfPixels;
  stats.parameterUploadBytes = context->alfParameterUploadBytes;
  stats.diagnosticDownloadBytes = context->alfDiagnosticDownloadBytes;
  stats.commitBytes = context->alfCommitBytes;
  stats.runtimeSynchronizations = context->alfSynchronizations;
  stats.runtimeNanoseconds = context->alfElapsedNanoseconds;
  stats.scratchBytes = alfScratchBytes(context->alfScratch) + alfScratchBytes(context->alfRetiredScratch);
  stats.retiredScratchBytes = alfScratchBytes(context->alfRetiredScratch);
  stats.peakScratchBytes = context->alfPeakScratchBytes;
  stats.enabled = true;
  return stats;
}

DbfAccelerationStats dbfStats(const RuntimeContext *context) noexcept
{
  DbfAccelerationStats stats{};
  if (context == nullptr) return stats;
  stats.dispatches = context->dbfDispatches;
  stats.tasks = context->dbfTasks;
  stats.pixels = context->dbfPixels;
  stats.parameterUploadBytes = context->dbfParameterUploadBytes;
  stats.commitBytes = context->dbfCommitBytes;
  stats.runtimeSynchronizations = context->dbfSynchronizations;
  stats.runtimeNanoseconds = context->dbfElapsedNanoseconds;
  stats.scratchBytes = dbfScratchBytes(context->dbfScratch);
  stats.peakScratchBytes = context->dbfPeakScratchBytes;
  stats.enabled = true;
  return stats;
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
        context->distortionResultsDevice = nullptr;
        (void) cudaStreamSynchronize(context->streams[computeIndex]);
      }
      else
      {
        (void) cudaGetLastError();
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
#if VTM_CUDA_TESTING
  context->distortionFailurePoint = CudaBatchTestFailurePoint::None;
#endif
  (void) cudaGetLastError();
  (void) cudaStreamCreateWithFlags(&context->streams[computeIndex], cudaStreamNonBlocking);
}

void recoverQpaRuntime(RuntimeContext *context) noexcept
{
  if (context == nullptr)
  {
    return;
  }
  (void) cudaSetDevice(context->device);
  const std::size_t qpaIndex = queueIndex(CudaQueue::Qpa);
  if (context->streams[qpaIndex] != nullptr)
  {
    (void) cudaStreamSynchronize(context->streams[qpaIndex]);
    for (void **allocation : { reinterpret_cast<void **>(&context->qpaTasksDevice),
                               reinterpret_cast<void **>(&context->qpaResultsDevice) })
    {
      if (*allocation != nullptr && cudaFreeAsync(*allocation, context->streams[qpaIndex]) == cudaSuccess)
      {
        *allocation = nullptr;
        (void) cudaStreamSynchronize(context->streams[qpaIndex]);
      }
      else if (*allocation != nullptr)
      {
        (void) cudaGetLastError();
      }
    }
    (void) cudaStreamDestroy(context->streams[qpaIndex]);
    context->streams[qpaIndex] = nullptr;
  }
  for (void **allocation : { reinterpret_cast<void **>(&context->qpaTasksHost),
                             reinterpret_cast<void **>(&context->qpaResultsHost) })
  {
    if (*allocation != nullptr && cudaFreeHost(*allocation) == cudaSuccess)
    {
      *allocation = nullptr;
    }
  }
  context->qpaCapacity = 0;
#if VTM_CUDA_TESTING
  context->qpaFailurePoint = CudaBatchTestFailurePoint::None;
#endif
  (void) cudaGetLastError();
  (void) cudaStreamCreateWithFlags(&context->streams[qpaIndex], cudaStreamNonBlocking);
}

void recoverAlfRuntime(RuntimeContext *context) noexcept
{
  if (context == nullptr) return;
  (void) cudaSetDevice(context->device);
  const std::size_t index = queueIndex(CudaQueue::Alf);
  if (context->streams[index] != nullptr)
  {
    (void) cudaStreamSynchronize(context->streams[index]);
    for (AlfScratch *scratch : { &context->alfScratch, &context->alfRetiredScratch })
    {
      for (void **allocation : { reinterpret_cast<void **>(&scratch->ctusDevice),
                                 reinterpret_cast<void **>(&scratch->classifiersDevice),
                                 &scratch->outputDevice })
      {
        if (*allocation != nullptr && cudaFreeAsync(*allocation, context->streams[index]) == cudaSuccess)
        {
          *allocation = nullptr;
        }
        else if (*allocation != nullptr)
        {
          (void) cudaGetLastError();
        }
      }
    }
    (void) cudaStreamSynchronize(context->streams[index]);
    (void) cudaStreamDestroy(context->streams[index]);
    context->streams[index] = nullptr;
  }
  for (AlfScratch *scratch : { &context->alfScratch, &context->alfRetiredScratch })
  {
    for (void **allocation : { reinterpret_cast<void **>(&scratch->ctusDevice),
                               reinterpret_cast<void **>(&scratch->classifiersDevice),
                               &scratch->outputDevice })
    {
      if (*allocation != nullptr && cudaFree(*allocation) == cudaSuccess)
      {
        *allocation = nullptr;
      }
      else if (*allocation != nullptr)
      {
        (void) cudaGetLastError();
      }
    }
    if (scratch->ctusHost != nullptr && cudaFreeHost(scratch->ctusHost) == cudaSuccess)
    {
      scratch->ctusHost = nullptr;
    }
    else if (scratch->ctusHost != nullptr)
    {
      (void) cudaGetLastError();
    }
    if (isEmpty(*scratch))
    {
      *scratch = AlfScratch{};
    }
  }
#if VTM_CUDA_TESTING
  context->alfFailurePoint = CudaAlfTestFailurePoint::None;
#endif
  (void) cudaGetLastError();
  (void) cudaStreamCreateWithFlags(&context->streams[index], cudaStreamNonBlocking);
}

void recoverDbfRuntime(RuntimeContext *context) noexcept
{
  if (context == nullptr) return;
  (void) cudaSetDevice(context->device);
  const std::size_t index = queueIndex(CudaQueue::Dbf);
  if (context->streams[index] != nullptr)
  {
    (void) cudaStreamSynchronize(context->streams[index]);
    for (void **allocation : { reinterpret_cast<void **>(&context->dbfScratch.tasksDevice),
                               reinterpret_cast<void **>(&context->dbfScratch.laneOffsetsDevice),
                               &context->dbfScratch.outputDevice })
    {
      if (*allocation != nullptr && cudaFreeAsync(*allocation, context->streams[index]) == cudaSuccess)
        *allocation = nullptr;
      else if (*allocation != nullptr) (void) cudaGetLastError();
    }
    (void) cudaStreamSynchronize(context->streams[index]);
    (void) cudaStreamDestroy(context->streams[index]);
    context->streams[index] = nullptr;
  }
  for (void **allocation : { reinterpret_cast<void **>(&context->dbfScratch.tasksDevice),
                             reinterpret_cast<void **>(&context->dbfScratch.laneOffsetsDevice),
                             &context->dbfScratch.outputDevice })
  {
    if (*allocation != nullptr && cudaFree(*allocation) == cudaSuccess) *allocation = nullptr;
    else if (*allocation != nullptr) (void) cudaGetLastError();
  }
  for (void **allocation : { reinterpret_cast<void **>(&context->dbfScratch.tasksHost),
                             reinterpret_cast<void **>(&context->dbfScratch.laneOffsetsHost) })
  {
    if (*allocation != nullptr && cudaFreeHost(*allocation) == cudaSuccess) *allocation = nullptr;
    else if (*allocation != nullptr) (void) cudaGetLastError();
  }
  context->dbfScratch = DbfScratch{};
#if VTM_CUDA_TESTING
  context->dbfFailurePoint = CudaDbfTestFailurePoint::None;
#endif
  (void) cudaGetLastError();
  (void) cudaStreamCreateWithFlags(&context->streams[index], cudaStreamNonBlocking);
}

void injectReleaseFailures(RuntimeContext *context, const unsigned asyncFailures, const unsigned immediateFailures)
{
#if VTM_CUDA_TESTING
  context->asyncReleaseFailures = asyncFailures;
  context->immediateReleaseFailures = immediateFailures;
#else
  (void) context;
  (void) asyncFailures;
  (void) immediateFailures;
#endif
}

void injectDistortionFailures(RuntimeContext *context, const unsigned allocationFailureStep,
                              const unsigned executionFailures)
{
#if VTM_CUDA_TESTING
  context->distortionAllocationFailureStep = allocationFailureStep;
  context->distortionExecutionFailures = executionFailures;
#else
  (void) context;
  (void) allocationFailureStep;
  (void) executionFailures;
#endif
}

void injectQpaFailures(RuntimeContext *context, const unsigned allocationFailureStep,
                       const unsigned executionFailures)
{
#if VTM_CUDA_TESTING
  context->qpaAllocationFailureStep = allocationFailureStep;
  context->qpaExecutionFailures = executionFailures;
#else
  (void) context;
  (void) allocationFailureStep;
  (void) executionFailures;
#endif
}

void injectDistortionFailurePoint(RuntimeContext *context, const CudaBatchTestFailurePoint failurePoint)
{
#if VTM_CUDA_TESTING
  context->distortionFailurePoint = failurePoint;
#else
  (void) context;
  (void) failurePoint;
#endif
}

void injectQpaFailurePoint(RuntimeContext *context, const CudaBatchTestFailurePoint failurePoint)
{
#if VTM_CUDA_TESTING
  context->qpaFailurePoint = failurePoint;
#else
  (void) context;
  (void) failurePoint;
#endif
}

void injectAlfFailure(RuntimeContext *context, const CudaAlfTestFailurePoint failurePoint)
{
#if VTM_CUDA_TESTING
  context->alfFailurePoint = failurePoint;
#else
  (void) context;
  (void) failurePoint;
#endif
}

void injectDbfFailure(RuntimeContext *context, const CudaDbfTestFailurePoint failurePoint)
{
#if VTM_CUDA_TESTING
  context->dbfFailurePoint = failurePoint;
#else
  (void) context;
  (void) failurePoint;
#endif
}

void injectPinnedReleaseFailures(const unsigned failures)
{
#if VTM_CUDA_TESTING
  pinnedReleaseFailures.store(failures, std::memory_order_relaxed);
#else
  (void) failures;
#endif
}

}   // namespace vtm::cuda_backend
