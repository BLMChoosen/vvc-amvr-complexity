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

#include "CudaContext.h"

#include <stdexcept>
#include <thread>
#include <exception>

#if VTM_ENABLE_CUDA
#include "CudaRuntime.h"
#endif

namespace vtm
{

struct CudaContext::Impl
{
#if VTM_ENABLE_CUDA
  cuda_backend::RuntimeContext *runtime = nullptr;
#endif
  std::thread::id ownerThread;
};

namespace
{

void requireOwnerThread(const std::thread::id &ownerThread)
{
  if (ownerThread != std::this_thread::get_id())
  {
    throw std::runtime_error("CUDA context APIs must be called from the thread that created the context");
  }
}

}   // namespace

CudaContext::CudaContext() : m_impl(new Impl)
{
}

CudaContext::~CudaContext() noexcept
{
#if VTM_ENABLE_CUDA
  if (m_impl->runtime != nullptr)
  {
    if (m_impl->ownerThread != std::this_thread::get_id())
    {
      std::terminate();
    }
    cuda_backend::destroyRuntimeContext(m_impl->runtime);
    m_impl->runtime = nullptr;
    m_impl->ownerThread = std::thread::id{};
  }
#endif
}

void CudaContext::create(const int device)
{
  if (isCreated())
  {
    throw std::runtime_error("CUDA context is already initialized");
  }
  if (device < 0)
  {
    throw std::runtime_error("GPUDevice must be zero or greater");
  }

#if VTM_ENABLE_CUDA
  m_impl->runtime = cuda_backend::createRuntimeContext(device);
  m_impl->ownerThread = std::this_thread::get_id();
#else
  (void) device;
  throw std::runtime_error("CUDA backend requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

void CudaContext::synchronize()
{
#if VTM_ENABLE_CUDA
  if (m_impl->runtime != nullptr)
  {
    requireOwnerThread(m_impl->ownerThread);
    cuda_backend::synchronizeRuntimeContext(m_impl->runtime);
  }
#endif
}

void CudaContext::shutdown(const bool synchronize)
{
#if VTM_ENABLE_CUDA
  if (m_impl->runtime != nullptr)
  {
    requireOwnerThread(m_impl->ownerThread);
    cuda_backend::RuntimeContext *runtime = m_impl->runtime;
    m_impl->runtime = nullptr;
    m_impl->ownerThread = std::thread::id{};
    cuda_backend::shutdownRuntimeContext(runtime, synchronize);
  }
#endif
}

void CudaContext::destroy()
{
  shutdown();
}

void CudaContext::recordFence(const CudaQueue queue, const CudaFence fence)
{
#if VTM_ENABLE_CUDA
  if (m_impl->runtime == nullptr)
  {
    throw std::runtime_error("CUDA context is not initialized");
  }
  requireOwnerThread(m_impl->ownerThread);
  cuda_backend::recordFence(m_impl->runtime, queue, fence);
#else
  (void) queue;
  (void) fence;
  throw std::runtime_error("CUDA backend requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

void CudaContext::waitFence(const CudaQueue queue, const CudaFence fence)
{
#if VTM_ENABLE_CUDA
  if (m_impl->runtime == nullptr)
  {
    throw std::runtime_error("CUDA context is not initialized");
  }
  requireOwnerThread(m_impl->ownerThread);
  cuda_backend::waitFence(m_impl->runtime, queue, fence);
#else
  (void) queue;
  (void) fence;
  throw std::runtime_error("CUDA backend requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

void *CudaContext::allocateDevice(const std::size_t bytes, const CudaQueue queue)
{
#if VTM_ENABLE_CUDA
  if (m_impl->runtime == nullptr)
  {
    throw std::runtime_error("CUDA context is not initialized");
  }
  requireOwnerThread(m_impl->ownerThread);
  return cuda_backend::allocateDevice(m_impl->runtime, bytes, queue);
#else
  (void) bytes;
  (void) queue;
  throw std::runtime_error("CUDA backend requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

void CudaContext::releaseDevice(void *allocation, const CudaQueue queue)
{
#if VTM_ENABLE_CUDA
  if (m_impl->runtime == nullptr)
  {
    throw std::runtime_error("CUDA context is not initialized");
  }
  requireOwnerThread(m_impl->ownerThread);
  cuda_backend::releaseDevice(m_impl->runtime, allocation, queue);
#else
  (void) allocation;
  (void) queue;
  throw std::runtime_error("CUDA backend requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

bool CudaContext::isCreated() const noexcept
{
#if VTM_ENABLE_CUDA
  return m_impl->runtime != nullptr;
#else
  return false;
#endif
}

bool CudaContext::isCompiled() noexcept
{
#if VTM_ENABLE_CUDA
  return true;
#else
  return false;
#endif
}

}   // namespace vtm
