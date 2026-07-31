/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2026, ITU/ISO/IEC
 * All rights reserved.
 */

#include "ComputeBackend.h"

namespace vtm
{

const char *computeBackendName(const ComputeBackend backend) noexcept
{
  return backend == ComputeBackend::CUDA ? "cuda" : "cpu";
}

bool parseComputeBackend(const std::string &value, ComputeBackend &backend) noexcept
{
  if (value == "cpu")
  {
    backend = ComputeBackend::CPU;
    return true;
  }
  if (value == "cuda")
  {
    backend = ComputeBackend::CUDA;
    return true;
  }
  return false;
}

}   // namespace vtm
