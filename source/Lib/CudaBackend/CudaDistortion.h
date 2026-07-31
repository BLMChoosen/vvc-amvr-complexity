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
constexpr std::uint32_t CUDA_MAX_DISTORTION_CANDIDATES = 1u << 20;
constexpr std::uint32_t CUDA_MAX_INTRA_CANDIDATES = 64;

struct CudaDistortionCandidateGridDesc
{
  const void *reference;
  std::uint32_t columns;
  std::uint32_t rows;
  std::uint16_t stepX;
  std::uint16_t stepY;
};

struct CudaDistortionBatchDesc
{
  CudaMirrorHandle                    sourceMirror;
  CudaMirrorHandle                    referenceMirror;
  const void                         *source;
  CudaDistortionCandidateGridDesc     candidateGrid;
  std::uint32_t                       width;
  std::uint32_t                       height;
  std::uint8_t                        sourcePlane;
  std::uint8_t                        referencePlane;
  std::uint8_t                        elementSize;
  std::uint8_t                        bitDepth;
  std::uint8_t                        subShift;
  CudaDistortionMetric                metric;
};

// Intra shortlist candidates are packed as candidateCount tightly-strided rectangles. The original block remains
// in its persistent picture mirror; only the transient predictions are uploaded for the batch.
struct CudaIntraSatdBatchDesc
{
  CudaMirrorHandle sourceMirror;
  const void      *source;
  const void      *candidates;
  std::uint32_t    candidateCount;
  std::uint32_t    width;
  std::uint32_t    height;
  std::uint8_t     sourcePlane;
  std::uint8_t     elementSize;
  std::uint8_t     bitDepth;
};

struct CudaSadHadResult
{
  std::uint64_t sad;
  std::uint64_t had;
};

static_assert(std::is_standard_layout<CudaDistortionCandidateGridDesc>::value
                && std::is_trivial<CudaDistortionCandidateGridDesc>::value,
              "CUDA distortion candidate grid descriptor must remain POD");
static_assert(std::is_standard_layout<CudaDistortionBatchDesc>::value
                && std::is_trivial<CudaDistortionBatchDesc>::value,
              "CUDA distortion batch descriptor must remain POD");
static_assert(std::is_standard_layout<CudaIntraSatdBatchDesc>::value
                && std::is_trivial<CudaIntraSatdBatchDesc>::value,
              "CUDA intra SATD batch descriptor must remain POD");
static_assert(std::is_standard_layout<CudaSadHadResult>::value && std::is_trivial<CudaSadHadResult>::value,
              "CUDA SAD/HAD result must remain POD");

}   // namespace vtm

#endif   // VTM_CUDA_DISTORTION_H
