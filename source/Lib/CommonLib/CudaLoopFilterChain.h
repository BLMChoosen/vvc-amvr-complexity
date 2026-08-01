/* The copyright in this software is being made available under the BSD
 * License. See COPYING for details. */

#ifndef VTM_CUDA_LOOP_FILTER_CHAIN_H
#define VTM_CUDA_LOOP_FILTER_CHAIN_H

#include "CudaDeblocking.h"

#include <cstdint>
#include <type_traits>

namespace vtm
{

constexpr std::uint64_t CUDA_LOOP_FILTER_CHAIN_MIN_FRAME_PIXELS = UINT64_C(1920) * 1080;
constexpr std::uint32_t CUDA_LOOP_FILTER_CHAIN_MAX_CTUS = 4096;
constexpr std::uint32_t CUDA_LOOP_FILTER_CHAIN_MAX_LMCS_LUT = 1u << 10;
constexpr std::uint32_t CUDA_SAO_NUM_OFFSETS = 32;

enum CudaLoopFilterStageFlag : std::uint8_t
{
  CUDA_LOOP_FILTER_LMCS = 1 << 0,
  CUDA_LOOP_FILTER_DBF  = 1 << 1,
  CUDA_LOOP_FILTER_SAO  = 1 << 2,
  CUDA_LOOP_FILTER_ALF  = 1 << 3
};

enum CudaLoopFilterUnsupportedFeature : std::uint8_t
{
  CUDA_LOOP_FILTER_UNSUPPORTED_CCALF = 1 << 0
};

enum class CudaLoopFilterChainDispatchResult : std::uint8_t
{
  NotEligible,
  NoOp,
  Executed
};

enum class CudaLoopFilterChainTestFailurePoint : std::uint8_t
{
  None,
  GrowLutDevice,
  GrowLutPinned,
  GrowSaoDevice,
  GrowSaoPinned,
  GrowSaoOutput,
  OldDeviceRelease,
  OldPinnedRelease,
  Upload,
  LmcsLaunch,
  LmcsCompletion,
  DbfStage,
  SaoSnapshot,
  SaoLaunch,
  SaoCompletion,
  AlfStage,
  Download,
  Commit,
  RecoveryDeviceRelease,
  RecoveryPinnedRelease
};

struct CudaSaoLumaCtuParam
{
  std::uint32_t x;
  std::uint32_t y;
  std::uint32_t width;
  std::uint32_t height;
  std::int32_t  offsets[CUDA_SAO_NUM_OFFSETS];
  std::int8_t   type;
  std::uint8_t  enabled;
  std::uint8_t  reserved[2];
};

struct CudaLoopFilterChainFrame
{
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t ctuWidth;
  std::uint32_t ctuHeight;
  std::uint32_t ctusInWidth;
  std::uint32_t ctusInHeight;
  std::uint32_t ctuCount;
  std::uint32_t lmcsLutSize;
  std::int32_t  minSample;
  std::int32_t  maxSample;
  std::uint8_t  bitDepth;
  std::uint8_t  elementSize;
  std::uint8_t  stages;
  std::uint8_t  unsupportedFeatures;
};

struct LoopFilterChainAccelerationStats
{
  std::uint64_t dispatches = 0;
  std::uint64_t pixels = 0;
  std::uint64_t dbfTasks = 0;
  std::uint64_t saoCtus = 0;
  std::uint64_t alfCtus = 0;
  std::uint64_t parameterUploadBytes = 0;
  std::uint64_t internalCopyBytes = 0;
  std::uint64_t mirrorUploadBytes = 0;
  std::uint64_t mirrorDownloadBytes = 0;
  std::uint64_t runtimeSynchronizations = 0;
  std::uint64_t integrationSynchronizations = 0;
  std::uint64_t collectionNanoseconds = 0;
  std::uint64_t lmcsNanoseconds = 0;
  std::uint64_t dbfNanoseconds = 0;
  std::uint64_t saoNanoseconds = 0;
  std::uint64_t alfNanoseconds = 0;
  std::uint64_t runtimeNanoseconds = 0;
  std::uint64_t integrationNanoseconds = 0;
  std::uint64_t scratchBytes = 0;
  std::uint64_t retiredScratchBytes = 0;
  std::uint64_t peakScratchBytes = 0;
  std::uint64_t failures = 0;
  std::uint64_t notEligible = 0;
  bool enabled = false;
  bool poisoned = false;
};

static_assert(std::is_standard_layout<CudaSaoLumaCtuParam>::value
                && std::is_trivial<CudaSaoLumaCtuParam>::value,
              "CUDA SAO CTU parameters must remain POD");
static_assert(std::is_standard_layout<CudaLoopFilterChainFrame>::value
                && std::is_trivial<CudaLoopFilterChainFrame>::value,
              "CUDA loop-filter chain frame must remain POD");

} // namespace vtm

#endif // VTM_CUDA_LOOP_FILTER_CHAIN_H
