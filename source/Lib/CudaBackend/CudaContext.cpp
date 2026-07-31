/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2026, ITU/ISO/IEC
 * All rights reserved.
 */

#include "CudaContext.h"

#include <stdexcept>

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
};

CudaContext::CudaContext() : m_impl(new Impl)
{
}

CudaContext::~CudaContext()
{
  destroy();
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
    cuda_backend::synchronizeRuntimeContext(m_impl->runtime);
  }
#endif
}

void CudaContext::destroy() noexcept
{
#if VTM_ENABLE_CUDA
  if (m_impl->runtime != nullptr)
  {
    cuda_backend::destroyRuntimeContext(m_impl->runtime);
    m_impl->runtime = nullptr;
  }
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

bool CudaContext::supportsMain10() const noexcept
{
#if VTM_ENABLE_CUDA
  return m_impl->runtime != nullptr && cuda_backend::supportsMain10(m_impl->runtime);
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
