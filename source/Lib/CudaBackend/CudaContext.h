/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2026, ITU/ISO/IEC
 * All rights reserved.
 */

#ifndef VTM_CUDA_CONTEXT_H
#define VTM_CUDA_CONTEXT_H

#include <memory>

namespace vtm
{

class CudaContext
{
public:
  CudaContext();
  ~CudaContext();

  CudaContext(const CudaContext &) = delete;
  CudaContext &operator=(const CudaContext &) = delete;

  void create(int device);
  void synchronize();
  void destroy() noexcept;

  bool isCreated() const noexcept;
  bool supportsMain10() const noexcept;
  static bool isCompiled() noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

}   // namespace vtm

#endif   // VTM_CUDA_CONTEXT_H
