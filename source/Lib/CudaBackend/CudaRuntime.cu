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
#include <stdexcept>
#include <string>

namespace vtm::cuda_backend
{

struct RuntimeContext
{
  int           device = -1;
  cudaStream_t  stream = nullptr;
  cudaEvent_t   readyEvent = nullptr;
  cudaEvent_t   completeEvent = nullptr;
  cudaMemPool_t memoryPool = nullptr;
  bool          main10Supported = false;
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

    cudaDeviceProp properties{};
    checkCuda(cudaGetDeviceProperties(&properties, device), "device property query");
    context->main10Supported = properties.major >= 5;

    checkCuda(cudaStreamCreateWithFlags(&context->stream, cudaStreamNonBlocking), "stream creation");
    checkCuda(cudaEventCreateWithFlags(&context->readyEvent, cudaEventDisableTiming), "ready event creation");
    checkCuda(cudaEventCreateWithFlags(&context->completeEvent, cudaEventDisableTiming), "completion event creation");

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
  checkCuda(cudaStreamSynchronize(context->stream), "stream synchronization");
}

void destroyRuntimeContext(RuntimeContext *context) noexcept
{
  if (context == nullptr)
  {
    return;
  }

  cudaSetDevice(context->device);
  if (context->stream != nullptr)
  {
    cudaStreamSynchronize(context->stream);
  }
  if (context->completeEvent != nullptr)
  {
    cudaEventDestroy(context->completeEvent);
  }
  if (context->readyEvent != nullptr)
  {
    cudaEventDestroy(context->readyEvent);
  }
  if (context->stream != nullptr)
  {
    cudaStreamDestroy(context->stream);
  }
  if (context->memoryPool != nullptr)
  {
    cudaMemPoolDestroy(context->memoryPool);
  }
  delete context;
}

bool supportsMain10(const RuntimeContext *context) noexcept
{
  return context != nullptr && context->main10Supported;
}

}   // namespace vtm::cuda_backend
