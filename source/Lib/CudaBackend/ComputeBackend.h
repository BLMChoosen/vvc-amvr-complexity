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

#ifndef VTM_COMPUTE_BACKEND_H
#define VTM_COMPUTE_BACKEND_H

#include <string>
#include <cstdint>

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
  bool           enableExperimentalSad = false;
  bool           enableExperimentalQpa = false;
  bool           enableExperimentalAlf = false;
  bool           enableExperimentalDbf = false;
  bool           enableExperimentalLoopFilterChain = false;
};

struct CudaSadStats
{
  std::uint64_t dispatches = 0;
  std::uint64_t failures = 0;
  bool disabled = false;
};

// Backend-neutral decoder telemetry.  Keeping this type here prevents DecoderLib's
// public API from depending on CUDA ALF implementation headers.
struct AlfAccelerationStats
{
  std::uint64_t dispatches = 0;
  std::uint64_t ctus = 0;
  std::uint64_t pixels = 0;
  std::uint64_t parameterUploadBytes = 0;
  std::uint64_t diagnosticDownloadBytes = 0;
  std::uint64_t commitBytes = 0;
  std::uint64_t mirrorUploadBytes = 0;
  std::uint64_t mirrorDownloadBytes = 0;
  std::uint64_t runtimeSynchronizations = 0;
  std::uint64_t integrationSynchronizations = 0;
  std::uint64_t uploadSubmissionNanoseconds = 0;
  std::uint64_t runtimeNanoseconds = 0;
  std::uint64_t downloadNanoseconds = 0;
  std::uint64_t integrationNanoseconds = 0;
  std::uint64_t scratchBytes = 0;
  std::uint64_t retiredScratchBytes = 0;
  std::uint64_t peakScratchBytes = 0;
  std::uint64_t failures = 0;
  bool enabled = false;
  bool poisoned = false;
};

struct DbfAccelerationStats
{
  std::uint64_t dispatches = 0;
  std::uint64_t noOpFrames = 0;
  std::uint64_t tasks = 0;
  std::uint64_t pixels = 0;
  std::uint64_t parameterUploadBytes = 0;
  std::uint64_t commitBytes = 0;
  std::uint64_t mirrorUploadBytes = 0;
  std::uint64_t mirrorDownloadBytes = 0;
  std::uint64_t runtimeSynchronizations = 0;
  std::uint64_t mirrorSynchronizations = 0;
  std::uint64_t integrationSynchronizations = 0;
  std::uint64_t runtimeNanoseconds = 0;
  std::uint64_t descriptorCollectionNanoseconds = 0;
  std::uint64_t integrationNanoseconds = 0;
  std::uint64_t scratchBytes = 0;
  std::uint64_t retiredScratchBytes = 0;
  std::uint64_t peakScratchBytes = 0;
  std::uint64_t failures = 0;
  bool enabled = false;
  bool poisoned = false;
};

const char *computeBackendName(ComputeBackend backend) noexcept;
bool parseComputeBackend(const std::string &value, ComputeBackend &backend) noexcept;

}   // namespace vtm

#endif   // VTM_COMPUTE_BACKEND_H
