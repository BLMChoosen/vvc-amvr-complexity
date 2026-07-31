/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2026, ITU/ISO/IEC
 * All rights reserved.
 */

#ifndef VTM_COMPUTE_BACKEND_H
#define VTM_COMPUTE_BACKEND_H

#include <string>

namespace vtm
{

enum class ComputeBackend
{
  CPU,
  CUDA
};

struct ComputeConfig
{
  ComputeBackend backend = ComputeBackend::CPU;
  int            device  = 0;
};

const char *computeBackendName(ComputeBackend backend) noexcept;
bool parseComputeBackend(const std::string &value, ComputeBackend &backend) noexcept;

}   // namespace vtm

#endif   // VTM_COMPUTE_BACKEND_H
