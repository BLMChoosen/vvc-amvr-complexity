/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2026, ITU/ISO/IEC
 * All rights reserved.
 */

#ifndef VTM_CUDA_ALF_H
#define VTM_CUDA_ALF_H

#include "ComputeBackend.h"

#include <cstdint>
#include <type_traits>

namespace vtm
{

constexpr std::uint32_t CUDA_ALF_CLASSES = 25;
constexpr std::uint32_t CUDA_ALF_COEFFICIENTS = 13;
constexpr std::uint32_t CUDA_ALF_MAX_CTUS = 4096;
// Below this size launch/copy/synchronization overhead dominates on the initial RTX 5060 target.
// The feature remains experimental and off by default until end-to-end decoder wall time improves.
constexpr std::uint64_t CUDA_ALF_MIN_FRAME_PIXELS = UINT64_C(1920) * 1080;

enum class CudaAlfDispatchResult : std::uint8_t
{
  NotEligible,
  Executed
};

enum class CudaAlfTestFailurePoint : std::uint8_t
{
  None,
  GrowCtuDevice,
  GrowPinnedCtu,
  GrowClassifiers,
  GrowOutput,
  OldDeviceRelease,
  OldPinnedRelease,
  ParameterUpload,
  KernelLaunch,
  KernelCompletion,
  DiagnosticDownload,
  CommitCopy,
  CommitCompletion
};

struct CudaAlfClassifier
{
  std::uint8_t classIdx;
  std::uint8_t transposeIdx;
};

struct CudaAlfCtuParam
{
  std::uint32_t x;
  std::uint32_t y;
  std::uint32_t width;
  std::uint32_t height;
  std::uint8_t enabled;
  std::uint8_t reserved[3];
  std::int16_t coefficients[CUDA_ALF_CLASSES * CUDA_ALF_COEFFICIENTS];
  std::int32_t clipValues[CUDA_ALF_CLASSES * CUDA_ALF_COEFFICIENTS];
};

struct CudaAlfLumaFrame
{
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t ctuWidth;
  std::uint32_t ctuHeight;
  std::uint32_t ctusInWidth;
  std::uint32_t ctusInHeight;
  std::int32_t minSample;
  std::int32_t maxSample;
  std::int32_t vbCtuHeight;
  std::int32_t vbPos;
  std::uint8_t bitDepth;
  std::uint8_t elementSize;
  std::uint8_t reserved[2];
};

static_assert(std::is_standard_layout<CudaAlfClassifier>::value && std::is_trivial<CudaAlfClassifier>::value,
              "CUDA ALF classifier must remain POD");
static_assert(std::is_standard_layout<CudaAlfCtuParam>::value && std::is_trivial<CudaAlfCtuParam>::value,
              "CUDA ALF CTU parameters must remain POD");
static_assert(std::is_standard_layout<CudaAlfLumaFrame>::value && std::is_trivial<CudaAlfLumaFrame>::value,
              "CUDA ALF frame parameters must remain POD");
} // namespace vtm

#endif // VTM_CUDA_ALF_H
