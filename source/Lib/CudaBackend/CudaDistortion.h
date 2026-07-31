/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2026, ITU/ISO/IEC
 * All rights reserved.
 */

#ifndef VTM_CUDA_DISTORTION_H
#define VTM_CUDA_DISTORTION_H

#include "CudaPictureMirror.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace vtm
{

enum class CudaDistortionMetric : std::uint8_t
{
  Sad
};

// The host pointers identify locations in registered picture mirrors. CudaContext translates them to the
// corresponding pitched device addresses before submission, so codec code never handles a CUDA type or header.
struct CudaDistortionCandidateDesc
{
  const void *reference;
};

struct CudaDistortionBatchDesc
{
  CudaMirrorHandle                    sourceMirror;
  CudaMirrorHandle                    referenceMirror;
  const void                         *source;
  const CudaDistortionCandidateDesc  *candidates;
  std::uint32_t                       candidateCount;
  std::uint32_t                       width;
  std::uint32_t                       height;
  std::uint8_t                        sourcePlane;
  std::uint8_t                        referencePlane;
  std::uint8_t                        elementSize;
  std::uint8_t                        bitDepth;
  std::uint8_t                        subShift;
  CudaDistortionMetric                metric;
};

static_assert(std::is_standard_layout<CudaDistortionCandidateDesc>::value
                && std::is_trivial<CudaDistortionCandidateDesc>::value,
              "CUDA distortion candidate descriptor must remain POD");
static_assert(std::is_standard_layout<CudaDistortionBatchDesc>::value
                && std::is_trivial<CudaDistortionBatchDesc>::value,
              "CUDA distortion batch descriptor must remain POD");

}   // namespace vtm

#endif   // VTM_CUDA_DISTORTION_H
