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

/** \file     DecCu.cpp
    \brief    CU decoder class
*/

#include "DecCu.h"

#include "CommonLib/InterPrediction.h"
#include "CommonLib/IntraPrediction.h"
#include "CommonLib/Picture.h"
#include "CommonLib/UnitTools.h"

#include "CommonLib/dtrace_buffer.h"

#if VTM_ENABLE_DECODER_BATCH_PROFILING
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>
#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif
#endif

#if RExt__DECODER_DEBUG_TOOL_STATISTICS
#include "CommonLib/CodingStatistics.h"
#endif
#if K0149_BLOCK_STATISTICS
#include "CommonLib/ChromaFormat.h"
#include "CommonLib/dtrace_blockstatistics.h"
#endif

//! \ingroup DecoderLib
//! \{

// ====================================================================================================================
// Constructor / destructor / create / destroy
// ====================================================================================================================

#if VTM_ENABLE_DECODER_BATCH_PROFILING
namespace
{
enum class McProfilePath : uint8_t
{
  UNI = 0,
  UNI_RPR,
  IDENTICAL_UNI,
  IDENTICAL_UNI_RPR,
  UNI_WEIGHTED,
  UNI_WEIGHTED_RPR,
  BI_AVG,
  BI_AVG_RPR,
  BI_WEIGHTED,
  BI_WEIGHTED_RPR,
  BI_BCW,
  BI_BCW_RPR,
  NUM
};

enum class McProfileReason : uint8_t
{
  INTRA_OR_PLT = 0,
  IBC,
  GPM,
  AFFINE_OR_PROF,
  CIIP,
  SUB_PU,
  DMVR,
  BDOF,
  INVALID_DPB_REFERENCE,
  MIXED_EFFECTIVE_PATH,
  LMCS_CHROMA_ADJ,
  IBC_BUFFER_RESET,
  IBC_VPDU_RESET,
  IBC_PRE_MV_CONSUMER,
  PATH_CHANGE,
  PICTURE_BOUNDARY,
  STREAM_END,
  NUM
};

enum class FusedCore : uint8_t
{
  CORE_A = 0,
  CORE_B_RPR,
  NUM
};

enum class FusedRefMode : uint8_t
{
  UNI = 0,
  IDENTICAL_UNI,
  UNI_WEIGHTED,
  BI_AVG,
  BI_WEIGHTED,
  BI_BCW,
  NUM
};

enum class FusedReason : uint8_t
{
  INTRA_OR_PLT = 0,
  IBC,
  GPM,
  AFFINE_OR_PROF,
  CIIP,
  SUB_PU,
  DMVR,
  BDOF,
  INVALID_DPB_REFERENCE,
  RPR,
  MTS_OR_OTHER_TRANSFORM,
  SCALING_LIST,
  LFNST,
  SBT,
  JOINT_CBCR,
  ACT,
  LMCS_CHROMA_RESIDUAL,
  LMCS_LUMA,
  LMCS_CACHE_DEPENDENCY,
  IBC_BUFFER_RESET,
  IBC_VPDU_RESET,
  IBC_PRE_MV_CONSUMER,
  PICTURE_BOUNDARY,
  STREAM_END,
  NUM
};

const char *const fusedCoreNames[] = { "core_a", "core_b_rpr" };
const char *const fusedRefModeNames[] = {
  "uni", "identical_uni", "uni_weighted", "bi_avg", "bi_weighted", "bi_bcw"
};
const char *const fusedReasonNames[] = {
  "intra_or_plt", "ibc", "gpm", "affine_or_prof", "ciip", "sub_pu", "dmvr", "bdof",
  "invalid_dpb", "rpr", "mts_or_other_transform", "scaling_list", "lfnst", "sbt", "joint_cbcr",
  "act", "lmcs_chroma_residual", "lmcs_luma", "lmcs_cache_dependency", "ibc_buffer_reset", "ibc_vpdu_reset",
  "ibc_pre_mv_consumer", "picture_boundary", "stream_end"
};
static_assert(sizeof(fusedCoreNames) / sizeof(fusedCoreNames[0]) == static_cast<size_t>(FusedCore::NUM),
              "fused core names are incomplete");
static_assert(sizeof(fusedRefModeNames) / sizeof(fusedRefModeNames[0]) == static_cast<size_t>(FusedRefMode::NUM),
              "fused reference-mode names are incomplete");
static_assert(sizeof(fusedReasonNames) / sizeof(fusedReasonNames[0]) == static_cast<size_t>(FusedReason::NUM),
              "fused reason names are incomplete");

const char *const mcProfilePathNames[] = {
  "uni", "uni_rpr", "identical_uni", "identical_uni_rpr", "uni_weighted", "uni_weighted_rpr",
  "bi_avg", "bi_avg_rpr", "bi_weighted", "bi_weighted_rpr", "bi_bcw", "bi_bcw_rpr"
};
const char *const mcProfileReasonNames[] = {
  "intra_or_plt", "ibc", "gpm", "affine_or_prof", "ciip", "sub_pu", "dmvr", "bdof",
  "invalid_dpb", "mixed_effective_path", "lmcs_chroma_adj", "ibc_buffer_reset", "ibc_vpdu_reset",
  "ibc_pre_mv_consumer", "path_change", "picture_boundary", "stream_end"
};
static_assert(sizeof(mcProfilePathNames) / sizeof(mcProfilePathNames[0]) == static_cast<size_t>(McProfilePath::NUM),
              "decoder batching path names are incomplete");
static_assert(sizeof(mcProfileReasonNames) / sizeof(mcProfileReasonNames[0]) == static_cast<size_t>(McProfileReason::NUM),
              "decoder batching reason names are incomplete");

constexpr McProfileReason mcProfileModeReason(const bool isIbc, const bool isInter)
{
  return isIbc ? McProfileReason::IBC : (isInter ? McProfileReason::NUM : McProfileReason::INTRA_OR_PLT);
}
static_assert(mcProfileModeReason(true, false) == McProfileReason::IBC,
              "IBC must be classified before the generic non-inter rejection");
static_assert(mcProfileModeReason(false, false) == McProfileReason::INTRA_OR_PLT,
              "non-inter CUs must retain their dedicated rejection");

constexpr bool fusedLmcsLumaActive(const bool sliceLmcsEnabled, const bool ctuLmcsEnabled)
{
  return sliceLmcsEnabled && ctuLmcsEnabled;
}
static_assert(!fusedLmcsLumaActive(false, false) && !fusedLmcsLumaActive(true, false)
                && fusedLmcsLumaActive(true, true),
              "fused LMCS-luma eligibility must require both the slice and CTU flags");

constexpr bool mcProfileIbcFillObservable(const bool spsIbcEnabled)
{
  return spsIbcEnabled;
}
static_assert(!mcProfileIbcFillObservable(false),
              "xFillIBCBuffer writes are semantically unobservable when SPS IBC is disabled");

struct IbcFillModel
{
  uint64_t pending = 0;
  uint64_t maximumPending = 0;
  uint64_t totalQueued = 0;
  uint64_t totalApplied = 0;
  uint64_t disabledSpsNoops = 0;
  uint64_t lastQueuedSequence = 0;
  uint64_t lastAppliedSequence = 0;

  constexpr void enqueue(const bool observable)
  {
    if (!observable)
    {
      disabledSpsNoops++;
      return;
    }
    pending++;
    totalQueued++;
    lastQueuedSequence = totalQueued;
    if (pending > maximumPending) maximumPending = pending;
  }

  constexpr void completeInOrder()
  {
    totalApplied += pending;
    pending = 0;
    lastAppliedSequence = lastQueuedSequence;
  }

  constexpr bool boundaryIsValid() const
  {
    return pending == 0 && lastAppliedSequence == lastQueuedSequence;
  }
};

constexpr bool mcProfileIbcFillModelAssertions()
{
  IbcFillModel model;
  model.enqueue(true);
  model.enqueue(true);
  if (model.pending != 2 || model.maximumPending != 2 || model.totalApplied != 0) return false;
  model.completeInOrder();
  if (!model.boundaryIsValid() || model.totalApplied != 2 || model.lastAppliedSequence != 2) return false;
  model.enqueue(false);
  return model.boundaryIsValid() && model.disabledSpsNoops == 1;
}
static_assert(mcProfileIbcFillModelAssertions(),
              "pending IBC fills must batch, complete in order, and never cross a consumer/reset boundary");

bool mcProfileEnabledFromEnvironment()
{
  const char *value = std::getenv("VTM_DECODER_BATCH_PROFILE");
  return value != nullptr && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
}

bool mcProfileIdenticalMotion(const PredictionUnit &pu)
{
  const Slice &slice = *pu.cs->slice;
  if (slice.isInterB() && !pu.cs->pps->getWPBiPred() && pu.refIdx[0] >= 0 && pu.refIdx[1] >= 0)
  {
    const Picture *refPicL0 = slice.getRefPic(REF_PIC_LIST_0, pu.refIdx[0]);
    const Picture *refPicL1 = slice.getRefPic(REF_PIC_LIST_1, pu.refIdx[1]);
    return refPicL0 == refPicL1 && pu.mv[0] == pu.mv[1];
  }
  return false;
}

McProfilePath mcProfileRprVariant(const McProfilePath base, const bool rpr)
{
  return rpr ? static_cast<McProfilePath>(static_cast<unsigned>(base) + 1) : base;
}

FusedRefMode fusedRefMode(const McProfilePath path)
{
  return static_cast<FusedRefMode>(static_cast<unsigned>(path) / 2);
}

struct McPredictionClassification
{
  McProfilePath path = McProfilePath::UNI;
  std::array<bool, NUM_REF_PIC_LIST_01> effectiveLists{};

  uint64_t operationCount() const
  {
    return static_cast<uint64_t>(effectiveLists[REF_PIC_LIST_0])
         + static_cast<uint64_t>(effectiveLists[REF_PIC_LIST_1]);
  }

  uint64_t weightedMetadataCount() const
  {
    const FusedRefMode mode = fusedRefMode(path);
    return mode == FusedRefMode::UNI_WEIGHTED || mode == FusedRefMode::BI_WEIGHTED ? operationCount() : 0;
  }
};

McPredictionClassification mcProfileClassifyPrediction(const SliceType sliceType, const bool useWp,
                                                        const bool useWpBi, const uint8_t bcwIdx,
                                                        const bool directList0, const bool identicalMotion,
                                                        const uint8_t interDir, const int refIdxList0,
                                                        const int refIdxList1, const bool scaledReference)
{
  McPredictionClassification result;
  const bool list0Used = (interDir & 1) != 0 && refIdxList0 >= 0;
  const bool list1Used = (interDir & 2) != 0 && refIdxList1 >= 0;
  if (directList0)
  {
    result.effectiveLists[REF_PIC_LIST_0] = list0Used;
    const bool weighted = (sliceType == P_SLICE && useWp) || (sliceType == B_SLICE && useWpBi);
    result.path = weighted ? McProfilePath::UNI_WEIGHTED : McProfilePath::UNI;
  }
  else if (identicalMotion)
  {
    result.effectiveLists[REF_PIC_LIST_0] = list0Used;
    result.path = McProfilePath::IDENTICAL_UNI;
  }
  else
  {
    result.effectiveLists[REF_PIC_LIST_0] = list0Used;
    result.effectiveLists[REF_PIC_LIST_1] = list1Used;
    const bool bothLists = list0Used && list1Used;
    if (bothLists && sliceType == B_SLICE && useWpBi && bcwIdx == BCW_DEFAULT)
      result.path = McProfilePath::BI_WEIGHTED;
    else if (!bothLists && ((sliceType == P_SLICE && useWp) || (sliceType == B_SLICE && useWpBi)))
      result.path = McProfilePath::UNI_WEIGHTED;
    else if (bothLists && bcwIdx != BCW_DEFAULT)
      result.path = McProfilePath::BI_BCW;
    else
      result.path = bothLists ? McProfilePath::BI_AVG : McProfilePath::UNI;
  }
  result.path = mcProfileRprVariant(result.path, scaledReference);
  return result;
}

FusedReason fusedReasonFromMc(const McProfileReason reason)
{
  switch (reason)
  {
  case McProfileReason::INTRA_OR_PLT: return FusedReason::INTRA_OR_PLT;
  case McProfileReason::IBC: return FusedReason::IBC;
  case McProfileReason::GPM: return FusedReason::GPM;
  case McProfileReason::AFFINE_OR_PROF: return FusedReason::AFFINE_OR_PROF;
  case McProfileReason::CIIP: return FusedReason::CIIP;
  case McProfileReason::SUB_PU: return FusedReason::SUB_PU;
  case McProfileReason::DMVR: return FusedReason::DMVR;
  case McProfileReason::BDOF: return FusedReason::BDOF;
  case McProfileReason::INVALID_DPB_REFERENCE: return FusedReason::INVALID_DPB_REFERENCE;
  default: return FusedReason::NUM;
  }
}

struct LogHistogram
{
  std::array<uint64_t, 65> buckets{};
  uint64_t count = 0;
  uint64_t maximum = 0;

  void add(const uint64_t value)
  {
    unsigned bucket = 0;
    if (value != 0)
    {
      uint64_t remaining = value;
      while (remaining >>= 1) ++bucket;
      ++bucket;
    }
    buckets[bucket]++;
    count++;
    if (value > maximum) maximum = value;
  }

  uint64_t quantileUpperBound(const unsigned percentile) const
  {
    if (count == 0) return 0;
    const uint64_t rank = (count * percentile + 99) / 100;
    uint64_t cumulative = 0;
    for (unsigned bucket = 0; bucket < buckets.size(); ++bucket)
    {
      cumulative += buckets[bucket];
      if (cumulative >= rank)
      {
        if (bucket == 0) return 0;
        if (bucket == 64) return UINT64_MAX;
        return (uint64_t{ 1 } << bucket) - 1;
      }
    }
    return maximum;
  }
};

struct RunDistribution
{
  LogHistogram tasks;
  LogHistogram pixels;
  uint64_t totalTasks = 0;
  uint64_t totalPixels = 0;

  void add(const uint64_t taskCount, const uint64_t pixelCount)
  {
    tasks.add(taskCount);
    pixels.add(pixelCount);
    totalTasks += taskCount;
    totalPixels += pixelCount;
  }
};

constexpr unsigned transformSizeClasses = 7;
constexpr int transformQpMinimum = -64;
constexpr int transformQpMaximum = 127;

enum class TransformFeature : uint8_t
{
  DCT2 = 0,
  MTS,
  TRANSFORM_SKIP,
  LFNST,
  SBT,
  JOINT_CBCR,
  ACT,
  LMCS,
  REGULAR_DCT2_CANDIDATE,
  NUM
};

enum class DequantPath : uint8_t
{
  DEPENDENT_FLAT = 0,
  DEPENDENT_SCALING_LIST,
  SCALAR_FLAT,
  SCALAR_SCALING_LIST,
  NUM
};

const char *const transformFeatureNames[] = {
  "dct2", "mts", "transform_skip", "lfnst", "sbt", "joint_cbcr", "act", "lmcs",
  "regular_dct2_candidate"
};
const char *const dequantPathNames[] = {
  "dependent_flat", "dependent_scaling_list", "scalar_flat", "scalar_scaling_list"
};
const char *const transformComponentNames[] = { "Y", "Cb", "Cr" };

static_assert(sizeof(transformFeatureNames) / sizeof(transformFeatureNames[0]) ==
                static_cast<size_t>(TransformFeature::NUM),
              "transform feature names are incomplete");
static_assert(sizeof(dequantPathNames) / sizeof(dequantPathNames[0]) == static_cast<size_t>(DequantPath::NUM),
              "dequant path names are incomplete");

struct TransformAggregate
{
  uint64_t tasks = 0;
  uint64_t pixels = 0;
  uint64_t coefficients = 0;
  uint64_t nonzeroCoefficients = 0;

  void add(const uint64_t pixelCount, const uint64_t coefficientCount, const uint64_t nonzeroCount)
  {
    tasks++;
    pixels += pixelCount;
    coefficients += coefficientCount;
    nonzeroCoefficients += nonzeroCount;
  }
};

struct TransformBlockAggregate
{
  uint64_t blocks = 0;
  uint64_t cbfBlocks = 0;
  uint64_t zeroCbfBlocks = 0;
  uint64_t pixels = 0;
  uint64_t cbfPixels = 0;
  uint64_t zeroCbfPixels = 0;

  void add(const uint64_t pixelCount, const bool cbf)
  {
    blocks++;
    pixels += pixelCount;
    if (cbf)
    {
      cbfBlocks++;
      cbfPixels += pixelCount;
    }
    else
    {
      zeroCbfBlocks++;
      zeroCbfPixels += pixelCount;
    }
  }
};

struct TransformProfileSample
{
  ComponentID component = COMPONENT_Y;
  unsigned widthLog2 = 0;
  unsigned heightLog2 = 0;
  uint64_t pixels = 0;
  uint64_t coefficients = 0;
  uint64_t nonzeroCoefficients = 0;
  int qp = 0;
  DequantPath dequantPath = DequantPath::SCALAR_FLAT;
  std::array<bool, static_cast<size_t>(TransformFeature::NUM)> features{};
};

// These are measurement-only POD layouts, not a promised kernel ABI. They make the descriptor-byte model explicit
// and reproducible while the profiler evaluates whether a fused dispatch is worth implementing.
struct FusedCuDescriptorModel
{
  uint64_t outputPictureMetadataId;
  uint64_t sliceMetadataId;
  uint32_t destinationPlaneOffset[MAX_NUM_COMPONENT];
  uint32_t destinationStride[MAX_NUM_COMPONENT];
  int32_t  x;
  int32_t  y;
  uint16_t width;
  uint16_t height;
  uint16_t predictionUnitCount;
  uint16_t transformTaskCount;
  uint32_t flags;
  int32_t  poc;
};

struct FusedPredictionOperationDescriptorModel
{
  uint64_t referenceMetadataId;
  uint32_t destinationPlaneOffset[MAX_NUM_COMPONENT];
  uint32_t destinationStride[MAX_NUM_COMPONENT];
  uint32_t referencePlaneOffset[MAX_NUM_COMPONENT];
  uint32_t referenceStride[MAX_NUM_COMPONENT];
  int32_t  destinationX;
  int32_t  destinationY;
  int32_t  mvHor;
  int32_t  mvVer;
  int16_t  refIdx;
  uint16_t width;
  uint16_t height;
  uint8_t  refList;
  uint8_t  componentMask;
  uint8_t  combineMode;
  uint8_t  bcwIdx;
  uint8_t  interpolationFlags;
  uint8_t  reserved[3];
  uint32_t flags;
};

struct FusedPictureMetadataModel
{
  uint64_t pictureIdentity;
  uint64_t residentOutputHandle;
  uint32_t planeOffset[MAX_NUM_COMPONENT];
  uint32_t stride[MAX_NUM_COMPONENT];
  uint32_t width[MAX_NUM_COMPONENT];
  uint32_t height[MAX_NUM_COMPONENT];
  int32_t  poc;
  uint8_t  chromaFormat;
  uint8_t  bitDepthLuma;
  uint8_t  bitDepthChroma;
  uint8_t  flags;
};

struct FusedSliceMetadataModel
{
  uint64_t sliceIdentity;
  uint64_t pictureMetadataId;
  uint16_t numReferences[NUM_REF_PIC_LIST_01];
  int16_t  clipMinimum[MAX_NUM_COMPONENT];
  int16_t  clipMaximum[MAX_NUM_COMPONENT];
  uint8_t  sliceType;
  uint8_t  flags;
  uint8_t  chromaFormat;
  uint8_t  reserved;
};

struct FusedReferenceMetadataModel
{
  uint64_t referenceIdentity;
  uint64_t residentReferenceHandle;
  uint64_t sliceMetadataId;
  uint32_t planeOffset[MAX_NUM_COMPONENT];
  uint32_t stride[MAX_NUM_COMPONENT];
  uint32_t width[MAX_NUM_COMPONENT];
  uint32_t height[MAX_NUM_COMPONENT];
  int32_t  referencePoc;
  int16_t  refIdx;
  uint8_t  refList;
  uint8_t  flags;
};

struct FusedWeightedPredictionMetadataModel
{
  int32_t codedWeight[MAX_NUM_COMPONENT];
  int32_t codedOffset[MAX_NUM_COMPONENT];
  int32_t weight[MAX_NUM_COMPONENT];
  int32_t offset[MAX_NUM_COMPONENT];
  int32_t shift[MAX_NUM_COMPONENT];
  int32_t round[MAX_NUM_COMPONENT];
  uint32_t log2WeightDenom[MAX_NUM_COMPONENT];
  uint8_t presentMask;
  uint8_t reserved[3];
};

struct FusedRprMetadataModel
{
  int32_t scaleX;
  int32_t scaleY;
  int32_t currentWindowLeft;
  int32_t currentWindowRight;
  int32_t currentWindowTop;
  int32_t currentWindowBottom;
  int32_t referenceWindowLeft;
  int32_t referenceWindowRight;
  int32_t referenceWindowTop;
  int32_t referenceWindowBottom;
  uint8_t currentChromaFormat;
  uint8_t referenceChromaFormat;
  uint8_t flags;
  uint8_t reserved;
};

struct FusedBcwMetadataModel
{
  int8_t  weightList0;
  int8_t  weightList1;
  uint8_t bcwIdx;
  uint8_t log2WeightBase;
};

struct FusedTransformDescriptorModel
{
  uint32_t coefficientOffset;
  uint32_t residualOffset;
  uint16_t width;
  uint16_t height;
  int16_t  qp;
  uint8_t  component;
  uint8_t  horizontalTransform;
  uint8_t  verticalTransform;
  uint8_t  flags;
};

static_assert(std::is_trivial<FusedCuDescriptorModel>::value &&
                std::is_standard_layout<FusedCuDescriptorModel>::value,
              "fused CU descriptor model must remain POD");
static_assert(std::is_trivial<FusedPredictionOperationDescriptorModel>::value &&
                std::is_standard_layout<FusedPredictionOperationDescriptorModel>::value,
              "fused prediction descriptor model must remain POD");
static_assert(std::is_trivial<FusedTransformDescriptorModel>::value &&
                std::is_standard_layout<FusedTransformDescriptorModel>::value,
              "fused transform descriptor model must remain POD");
static_assert(std::is_trivial<FusedPictureMetadataModel>::value &&
                std::is_standard_layout<FusedPictureMetadataModel>::value
                && std::is_trivial<FusedSliceMetadataModel>::value
                && std::is_standard_layout<FusedSliceMetadataModel>::value
                && std::is_trivial<FusedReferenceMetadataModel>::value
                && std::is_standard_layout<FusedReferenceMetadataModel>::value
                && std::is_trivial<FusedWeightedPredictionMetadataModel>::value
                && std::is_standard_layout<FusedWeightedPredictionMetadataModel>::value
                && std::is_trivial<FusedRprMetadataModel>::value
                && std::is_standard_layout<FusedRprMetadataModel>::value
                && std::is_trivial<FusedBcwMetadataModel>::value
                && std::is_standard_layout<FusedBcwMetadataModel>::value,
              "fused shared metadata models must remain POD");

uint64_t fusedCheckedAdd(const uint64_t left, const uint64_t right)
{
  if (right > std::numeric_limits<uint64_t>::max() - left)
    THROW("decoder fused-batch profiler byte counter overflow");
  return left + right;
}

uint64_t fusedCheckedMultiply(const uint64_t left, const uint64_t right)
{
  if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left)
    THROW("decoder fused-batch profiler byte counter overflow");
  return left * right;
}

uint64_t fusedCeilDivide(const uint64_t dividend, const uint64_t divisor)
{
  if (divisor == 0) THROW("decoder fused-batch profiler division by zero");
  return fusedCheckedAdd(dividend / divisor, dividend % divisor != 0 ? 1 : 0);
}

struct FusedPictureMetadataKey
{
  uintptr_t identity = 0;
  int poc = 0;
  unsigned width = 0;
  unsigned height = 0;
  unsigned chromaFormat = 0;
  unsigned bitDepthLuma = 0;
  unsigned bitDepthChroma = 0;

  bool operator==(const FusedPictureMetadataKey &other) const
  {
    return identity == other.identity && poc == other.poc && width == other.width && height == other.height
        && chromaFormat == other.chromaFormat && bitDepthLuma == other.bitDepthLuma
        && bitDepthChroma == other.bitDepthChroma;
  }
};

struct FusedSliceMetadataKey
{
  uintptr_t identity = 0;
  uintptr_t pictureIdentity = 0;
  int poc = 0;
  int sliceType = 0;
  int numRefList0 = 0;
  int numRefList1 = 0;
  bool useWp = false;
  bool useWpBi = false;

  bool operator==(const FusedSliceMetadataKey &other) const
  {
    return identity == other.identity && pictureIdentity == other.pictureIdentity && poc == other.poc
        && sliceType == other.sliceType && numRefList0 == other.numRefList0 && numRefList1 == other.numRefList1
        && useWp == other.useWp && useWpBi == other.useWpBi;
  }
};

struct FusedReferenceMetadataKey
{
  uintptr_t sliceIdentity = 0;
  uintptr_t referenceIdentity = 0;
  int referencePoc = 0;
  int refIdx = -1;
  int refList = 0;
  int scaleX = 0;
  int scaleY = 0;
  std::array<int, 4> currentWindow{};
  std::array<int, 4> referenceWindow{};
  std::array<int, MAX_NUM_COMPONENT> codedWeight{};
  std::array<int, MAX_NUM_COMPONENT> codedOffset{};
  std::array<int, MAX_NUM_COMPONENT> weight{};
  std::array<int, MAX_NUM_COMPONENT> offset{};
  std::array<int, MAX_NUM_COMPONENT> shift{};
  std::array<int, MAX_NUM_COMPONENT> round{};
  std::array<uint32_t, MAX_NUM_COMPONENT> log2WeightDenom{};
  uint8_t bcwIdx = BCW_DEFAULT;
  bool weighted = false;
  bool rpr = false;
  bool bcw = false;

  bool operator==(const FusedReferenceMetadataKey &other) const
  {
    return sliceIdentity == other.sliceIdentity && referenceIdentity == other.referenceIdentity
        && referencePoc == other.referencePoc && refIdx == other.refIdx && refList == other.refList
        && scaleX == other.scaleX && scaleY == other.scaleY && currentWindow == other.currentWindow
        && referenceWindow == other.referenceWindow && codedWeight == other.codedWeight
        && codedOffset == other.codedOffset && weight == other.weight && offset == other.offset
        && shift == other.shift && round == other.round && log2WeightDenom == other.log2WeightDenom
        && bcwIdx == other.bcwIdx && weighted == other.weighted && rpr == other.rpr && bcw == other.bcw;
  }
};

struct FusedBcwMetadataKey
{
  uint8_t bcwIdx = BCW_DEFAULT;
  int8_t weightList0 = 0;
  int8_t weightList1 = 0;

  bool operator==(const FusedBcwMetadataKey &other) const
  {
    return bcwIdx == other.bcwIdx && weightList0 == other.weightList0 && weightList1 == other.weightList1;
  }

  bool matchesGlobalTable() const
  {
    return bcwIdx < BCW_NUM
        && weightList0 == getBcwWeight(bcwIdx, REF_PIC_LIST_0)
        && weightList1 == getBcwWeight(bcwIdx, REF_PIC_LIST_1);
  }
};

struct FusedCandidate
{
  bool active = false;
  uint64_t cus = 0;
  uint64_t tus = 0;
  uint64_t lumaPixels = 0;
  uint64_t componentPixels = 0;
  uint64_t predictionUnits = 0;
  uint64_t predictionOperations = 0;
  uint64_t transformTasks = 0;
  uint64_t descriptorBytes = 0;
  uint64_t qcoeffBytes = 0;
  bool rpr = false;
  bool weighted = false;
  bool bcw = false;
  uint64_t rprPredictionOperations = 0;
  FusedPictureMetadataKey pictureMetadata;
  FusedSliceMetadataKey sliceMetadata;
  std::vector<FusedReferenceMetadataKey> referenceMetadata;
  std::vector<FusedBcwMetadataKey> bcwMetadata;
  std::array<uint64_t, static_cast<size_t>(FusedRefMode::NUM)> refModes{};
  std::array<bool, transformSizeClasses * transformSizeClasses> scanShapes{};
  std::array<bool, transformSizeClasses> dct2Sizes{};
  std::array<FusedReason, static_cast<size_t>(FusedCore::NUM)> rejections = {
    FusedReason::NUM, FusedReason::NUM
  };
};

struct FusedWindow
{
  uint64_t cus = 0;
  uint64_t tus = 0;
  uint64_t lumaPixels = 0;
  uint64_t componentPixels = 0;
  uint64_t predictionUnits = 0;
  uint64_t predictionOperations = 0;
  uint64_t transformTasks = 0;
  uint64_t descriptorBytes = 0;
  uint64_t qcoeffBytes = 0;
  uint64_t dirtyDownloadBytes = 0;
  std::array<uint64_t, static_cast<size_t>(FusedRefMode::NUM)> refModes{};

  bool empty() const { return cus == 0; }

  void add(const FusedCandidate &candidate)
  {
    cus += candidate.cus;
    tus += candidate.tus;
    lumaPixels += candidate.lumaPixels;
    componentPixels += candidate.componentPixels;
    predictionUnits += candidate.predictionUnits;
    predictionOperations += candidate.predictionOperations;
    transformTasks += candidate.transformTasks;
    descriptorBytes = fusedCheckedAdd(descriptorBytes, candidate.descriptorBytes);
    qcoeffBytes = fusedCheckedAdd(qcoeffBytes, candidate.qcoeffBytes);
    dirtyDownloadBytes = fusedCheckedAdd(dirtyDownloadBytes,
                                          fusedCheckedMultiply(candidate.componentPixels, sizeof(Pel)));
    for (size_t mode = 0; mode < refModes.size(); ++mode) refModes[mode] += candidate.refModes[mode];
  }
};

enum class FusedMetric : uint8_t
{
  CUS = 0,
  TUS,
  LUMA_PIXELS,
  COMPONENT_PIXELS,
  PREDICTION_OPERATIONS,
  TRANSFORM_TASKS,
  DESCRIPTOR_BYTES,
  QCOEFF_BYTES,
  DIRTY_DOWNLOAD_BYTES,
  TOTAL_TRANSFER_BYTES,
  NUM
};

const char *const fusedMetricNames[] = {
  "cus", "tus", "luma_pixels", "component_pixels", "prediction_operations", "transform_tasks",
  "descriptor_bytes", "qcoeff_bytes", "dirty_download_bytes", "total_transfer_bytes"
};
static_assert(sizeof(fusedMetricNames) / sizeof(fusedMetricNames[0]) == static_cast<size_t>(FusedMetric::NUM),
              "fused metric names are incomplete");

struct FusedDistribution
{
  std::array<LogHistogram, static_cast<size_t>(FusedMetric::NUM)> metrics;
  std::array<uint64_t, static_cast<size_t>(FusedMetric::NUM)> totals{};
  std::array<uint64_t, static_cast<size_t>(FusedRefMode::NUM)> refModePredictionUnits{};
  std::array<uint64_t, static_cast<size_t>(FusedRefMode::NUM)> refModeWindows{};
  uint64_t windows = 0;

  void add(const FusedWindow &window)
  {
    const std::array<uint64_t, static_cast<size_t>(FusedMetric::NUM)> values = {
      window.cus, window.tus, window.lumaPixels, window.componentPixels, window.predictionOperations,
      window.transformTasks, window.descriptorBytes, window.qcoeffBytes, window.dirtyDownloadBytes,
      fusedCheckedAdd(fusedCheckedAdd(window.descriptorBytes, window.qcoeffBytes), window.dirtyDownloadBytes)
    };
    windows++;
    for (size_t metric = 0; metric < values.size(); ++metric)
    {
      metrics[metric].add(values[metric]);
      totals[metric] = fusedCheckedAdd(totals[metric], values[metric]);
    }
    for (size_t mode = 0; mode < window.refModes.size(); ++mode)
    {
      refModePredictionUnits[mode] += window.refModes[mode];
      if (window.refModes[mode] != 0) refModeWindows[mode]++;
    }
  }
};

struct FusedCoreProfile
{
  FusedWindow active;
  FusedDistribution distribution;
  std::array<uint64_t, static_cast<size_t>(FusedReason::NUM)> flushReasons{};
  std::array<uint64_t, static_cast<size_t>(FusedReason::NUM)> rejectedCus{};
  std::array<uint64_t, static_cast<size_t>(FusedReason::NUM)> rejectedPixels{};
  std::array<bool, transformSizeClasses * transformSizeClasses> scanShapes{};
  std::array<bool, transformSizeClasses> dct2Sizes{};
  uint64_t consideredCus = 0;
  uint64_t consideredPixels = 0;
  uint64_t eligibleCus = 0;
  uint64_t eligiblePixels = 0;
  uint64_t sharedScanBytesOnce = 0;
  uint64_t sharedMatrixBytesOnce = 0;
  uint64_t sharedDequantBytesOnce = 0;
  uint64_t sharedPictureBytesOnce = 0;
  uint64_t sharedSliceBytesOnce = 0;
  uint64_t sharedReferenceBytesOnce = 0;
  uint64_t sharedWeightedBytesOnce = 0;
  uint64_t sharedRprBytesOnce = 0;
  uint64_t sharedBcwBytesOnce = 0;
  uint64_t rprCus = 0;
  uint64_t rprPredictionOperations = 0;
  uint64_t weightedPredictionUnits = 0;
  uint64_t bcwPredictionUnits = 0;
  IbcFillModel ibcFills;
  uint64_t ibcImmediateFills = 0;
  uint64_t ibcBoundaryChecks = 0;
  uint64_t ibcBoundaryViolations = 0;
  std::vector<FusedPictureMetadataKey> seenPictures;
  std::vector<FusedSliceMetadataKey> seenSlices;
  std::vector<FusedReferenceMetadataKey> seenReferences;
  std::vector<FusedReferenceMetadataKey> seenWeightedReferences;
  std::vector<FusedReferenceMetadataKey> seenRprReferences;
  std::vector<FusedBcwMetadataKey> seenBcwEntries;

  template<typename T, typename Predicate>
  static bool addUnique(std::vector<T> &values, const T &value, Predicate same)
  {
    for (const T &existing : values)
      if (same(existing, value)) return false;
    values.push_back(value);
    return true;
  }

  void addSharedMetadata(const FusedCandidate &candidate)
  {
    if (addUnique(seenPictures, candidate.pictureMetadata,
                  [](const FusedPictureMetadataKey &a, const FusedPictureMetadataKey &b) { return a == b; }))
      sharedPictureBytesOnce = fusedCheckedAdd(sharedPictureBytesOnce, sizeof(FusedPictureMetadataModel));
    if (addUnique(seenSlices, candidate.sliceMetadata,
                  [](const FusedSliceMetadataKey &a, const FusedSliceMetadataKey &b) { return a == b; }))
      sharedSliceBytesOnce = fusedCheckedAdd(sharedSliceBytesOnce, sizeof(FusedSliceMetadataModel));

    for (const FusedReferenceMetadataKey &reference : candidate.referenceMetadata)
    {
      const auto sameBase = [](const FusedReferenceMetadataKey &a, const FusedReferenceMetadataKey &b)
      {
        return a.sliceIdentity == b.sliceIdentity && a.referenceIdentity == b.referenceIdentity
            && a.referencePoc == b.referencePoc && a.refIdx == b.refIdx && a.refList == b.refList;
      };
      if (addUnique(seenReferences, reference, sameBase))
        sharedReferenceBytesOnce = fusedCheckedAdd(sharedReferenceBytesOnce, sizeof(FusedReferenceMetadataModel));
      if (reference.weighted
          && addUnique(seenWeightedReferences, reference,
                       [](const FusedReferenceMetadataKey &a, const FusedReferenceMetadataKey &b)
                       {
                         return a.sliceIdentity == b.sliceIdentity && a.refIdx == b.refIdx && a.refList == b.refList
                             && a.codedWeight == b.codedWeight && a.codedOffset == b.codedOffset
                             && a.weight == b.weight && a.offset == b.offset && a.shift == b.shift
                             && a.round == b.round && a.log2WeightDenom == b.log2WeightDenom;
                       }))
        sharedWeightedBytesOnce = fusedCheckedAdd(sharedWeightedBytesOnce,
                                                  sizeof(FusedWeightedPredictionMetadataModel));
      if (reference.rpr
          && addUnique(seenRprReferences, reference,
                       [](const FusedReferenceMetadataKey &a, const FusedReferenceMetadataKey &b)
                       {
                         return a.sliceIdentity == b.sliceIdentity && a.referenceIdentity == b.referenceIdentity
                             && a.refIdx == b.refIdx && a.refList == b.refList && a.scaleX == b.scaleX
                             && a.scaleY == b.scaleY && a.currentWindow == b.currentWindow
                             && a.referenceWindow == b.referenceWindow;
                       }))
        sharedRprBytesOnce = fusedCheckedAdd(sharedRprBytesOnce, sizeof(FusedRprMetadataModel));
    }
    for (const FusedBcwMetadataKey &bcw : candidate.bcwMetadata)
    {
      if (!bcw.matchesGlobalTable()) THROW("decoder fused-batch profiler BCW metadata disagrees with global table");
      if (addUnique(seenBcwEntries, bcw,
                    [](const FusedBcwMetadataKey &a, const FusedBcwMetadataKey &b) { return a == b; }))
        sharedBcwBytesOnce = fusedCheckedAdd(sharedBcwBytesOnce, sizeof(FusedBcwMetadataModel));
    }
  }

  void addSharedTables(const FusedCandidate &candidate)
  {
    addSharedMetadata(candidate);
    if (candidate.transformTasks != 0 && sharedDequantBytesOnce == 0)
      sharedDequantBytesOnce = sizeof(g_invQuantScales);
    for (size_t shape = 0; shape < scanShapes.size(); ++shape)
    {
      if (!candidate.scanShapes[shape] || scanShapes[shape]) continue;
      scanShapes[shape] = true;
      const unsigned widthLog2 = static_cast<unsigned>(shape / transformSizeClasses);
      const unsigned heightLog2 = static_cast<unsigned>(shape % transformSizeClasses);
      const uint64_t samples = fusedCheckedMultiply(uint64_t{ 1 } << widthLog2,
                                                     uint64_t{ 1 } << heightLog2);
      sharedScanBytesOnce = fusedCheckedAdd(sharedScanBytesOnce,
                                             fusedCheckedMultiply(samples, sizeof(ScanElement)));
    }
    for (size_t sizeLog2 = 0; sizeLog2 < dct2Sizes.size(); ++sizeLog2)
    {
      if (!candidate.dct2Sizes[sizeLog2] || dct2Sizes[sizeLog2]) continue;
      dct2Sizes[sizeLog2] = true;
      const uint64_t size = uint64_t{ 1 } << sizeLog2;
      sharedMatrixBytesOnce = fusedCheckedAdd(sharedMatrixBytesOnce,
                                               fusedCheckedMultiply(fusedCheckedMultiply(size, size),
                                                                    sizeof(TMatrixCoeff)));
    }
  }
};

unsigned transformSizeClass(const unsigned size)
{
  return std::min<unsigned>(floorLog2(size), transformSizeClasses - 1);
}

constexpr size_t transformShapeIndex(const ComponentID component, const unsigned widthLog2,
                                     const unsigned heightLog2)
{
  return (static_cast<size_t>(component) * transformSizeClasses + widthLog2) * transformSizeClasses + heightLog2;
}

constexpr size_t transformGroupIndex(const ComponentID component, const unsigned widthLog2,
                                     const unsigned heightLog2, const DequantPath dequantPath)
{
  return transformShapeIndex(component, widthLog2, heightLog2) * static_cast<size_t>(DequantPath::NUM)
       + static_cast<size_t>(dequantPath);
}

uint64_t mcProfileProcessId()
{
#if defined(_WIN32)
  return static_cast<uint64_t>(_getpid());
#else
  return static_cast<uint64_t>(getpid());
#endif
}

std::atomic<uint64_t> mcProfileDecoderInstances{ 0 };
std::mutex mcProfileOutputMutex;
} // namespace

struct DecCu::McProfile
{
  std::array<RunDistribution, static_cast<size_t>(McProfilePath::NUM)> paths;
  RunDistribution inverseTransformRuns;
  RunDistribution reconstructionRuns;
  std::array<TransformBlockAggregate, MAX_NUM_COMPONENT * transformSizeClasses * transformSizeClasses> transformBlocks;
  std::array<TransformAggregate, MAX_NUM_COMPONENT * transformSizeClasses * transformSizeClasses> transformShapes;
  std::array<TransformAggregate, static_cast<size_t>(TransformFeature::NUM)> transformFeatures;
  std::array<TransformAggregate, static_cast<size_t>(DequantPath::NUM)> dequantPaths;
  std::array<RunDistribution,
             MAX_NUM_COMPONENT * transformSizeClasses * transformSizeClasses * static_cast<size_t>(DequantPath::NUM)>
    transformGroups;
  std::array<uint64_t,
             MAX_NUM_COMPONENT * transformSizeClasses * transformSizeClasses * static_cast<size_t>(DequantPath::NUM)>
    pendingTransformGroupTasks{};
  std::array<uint64_t,
             MAX_NUM_COMPONENT * transformSizeClasses * transformSizeClasses * static_cast<size_t>(DequantPath::NUM)>
    pendingTransformGroupPixels{};
  std::array<uint64_t, transformQpMaximum - transformQpMinimum + 1> transformQps{};
  TransformAggregate transformTotals;
  std::array<uint64_t, static_cast<size_t>(McProfileReason::NUM)> flushReasons{};
  std::array<uint64_t, static_cast<size_t>(McProfileReason::NUM)> rejectedCus{};
  std::array<FusedCoreProfile, static_cast<size_t>(FusedCore::NUM)> fusedCores;
  FusedCandidate fusedCandidate;
  std::array<bool, static_cast<size_t>(FusedCore::NUM)> lastFusedHadCandidate{};
  std::array<bool, static_cast<size_t>(FusedCore::NUM)> lastFusedDeferredFill{};
  IbcFillModel ibcFills;
  std::mutex stateMutex;
  uint64_t processId = mcProfileProcessId();
  uint64_t decoderInstance = 0;
  bool schedulerSelfTestPassed = false;
  std::thread::id ownerThread;
  uint64_t ownerThreadHash = 0;
  uint64_t hookCalls = 0;
  uint64_t threadMismatchHooks = 0;
  bool haveOwnerThread = false;
  uint64_t totalPixels = 0;
  uint64_t eligiblePixels = 0;
  uint64_t eligibleCus = 0;
  uint64_t pictures = 0;
  int firstPoc = 0;
  int lastPoc = 0;
  bool havePoc = false;
  int activePath = -1;
  uint64_t activeTasks = 0;
  uint64_t activePixels = 0;
  uint64_t inverseTransformTasks = 0;
  uint64_t inverseTransformPixels = 0;
  uint64_t reconstructionTasks = 0;
  uint64_t reconstructionPixels = 0;
  uint64_t transformQpUnderflow = 0;
  uint64_t transformQpOverflow = 0;
  uint64_t ibcBoundaryChecks = 0;
  uint64_t ibcBoundaryViolations = 0;
  uint64_t ibcPrepareChecks = 0;
  uint64_t ibcPrepareViolations = 0;

  void flushFusedCore(const size_t core, const FusedReason reason)
  {
    FusedCoreProfile &profile = fusedCores[core];
    if (!profile.active.empty())
    {
      profile.distribution.add(profile.active);
      profile.flushReasons[static_cast<size_t>(reason)]++;
      profile.active = {};
    }
    profile.ibcBoundaryChecks++;
    profile.ibcFills.completeInOrder();
    if (!profile.ibcFills.boundaryIsValid()) profile.ibcBoundaryViolations++;
  }

  void flushFused(const FusedReason reason)
  {
    for (size_t core = 0; core < fusedCores.size(); ++core) flushFusedCore(core, reason);
  }

  void rejectFusedCandidate(const FusedReason reason, const bool coreAOnly = false)
  {
    if (!fusedCandidate.active) return;
    const size_t lastCore = coreAOnly ? 1 : fusedCores.size();
    for (size_t core = 0; core < lastCore; ++core)
    {
      if (fusedCandidate.rejections[core] != FusedReason::NUM) continue;
      flushFusedCore(core, reason);
      fusedCandidate.rejections[core] = reason;
    }
  }

  void beginFusedCandidate(const FusedCandidate &candidate, const FusedReason rejection)
  {
    if (fusedCandidate.active)
    {
      // A prior candidate reaching this point would mean a missing post-CU hook. Keep the measurement conservative.
      rejectFusedCandidate(FusedReason::STREAM_END);
      finishFusedCandidate();
    }
    fusedCandidate = candidate;
    fusedCandidate.active = true;
    fusedCandidate.cus = 1;
    fusedCandidate.descriptorBytes = fusedCheckedAdd(sizeof(FusedCuDescriptorModel),
                                                      fusedCheckedMultiply(fusedCandidate.predictionOperations,
                                                        sizeof(FusedPredictionOperationDescriptorModel)));
    for (FusedCoreProfile &core : fusedCores)
    {
      core.consideredCus++;
      core.consideredPixels += fusedCandidate.lumaPixels;
    }
    if (rejection != FusedReason::NUM) rejectFusedCandidate(rejection);
    else if (fusedCandidate.rpr) rejectFusedCandidate(FusedReason::RPR, true);
  }

  void finishFusedCandidate()
  {
    if (!fusedCandidate.active) return;
    lastFusedHadCandidate.fill(false);
    lastFusedDeferredFill.fill(false);
    for (size_t core = 0; core < fusedCores.size(); ++core)
    {
      FusedCoreProfile &profile = fusedCores[core];
      lastFusedHadCandidate[core] = true;
      const FusedReason rejection = fusedCandidate.rejections[core];
      if (rejection != FusedReason::NUM)
      {
        profile.rejectedCus[static_cast<size_t>(rejection)]++;
        profile.rejectedPixels[static_cast<size_t>(rejection)] += fusedCandidate.lumaPixels;
        continue;
      }
      profile.active.add(fusedCandidate);
      profile.addSharedTables(fusedCandidate);
      profile.eligibleCus++;
      profile.eligiblePixels += fusedCandidate.lumaPixels;
      lastFusedDeferredFill[core] = true;
      if (fusedCandidate.rpr)
      {
        profile.rprCus++;
        profile.rprPredictionOperations += fusedCandidate.rprPredictionOperations;
      }
      profile.weightedPredictionUnits +=
        fusedCandidate.refModes[static_cast<size_t>(FusedRefMode::UNI_WEIGHTED)]
        + fusedCandidate.refModes[static_cast<size_t>(FusedRefMode::BI_WEIGHTED)];
      profile.bcwPredictionUnits += fusedCandidate.refModes[static_cast<size_t>(FusedRefMode::BI_BCW)];
    }
    fusedCandidate = {};
  }

  void recordFusedTransformBlock()
  {
    if (!fusedCandidate.active) return;
    fusedCandidate.tus++;
  }

  static FusedReason fusedTransformRejection(const TransformProfileSample &sample)
  {
    if (sample.dequantPath == DequantPath::DEPENDENT_SCALING_LIST
        || sample.dequantPath == DequantPath::SCALAR_SCALING_LIST)
      return FusedReason::SCALING_LIST;
    if (sample.features[static_cast<size_t>(TransformFeature::LFNST)]) return FusedReason::LFNST;
    if (sample.features[static_cast<size_t>(TransformFeature::SBT)]) return FusedReason::SBT;
    if (sample.features[static_cast<size_t>(TransformFeature::JOINT_CBCR)]) return FusedReason::JOINT_CBCR;
    if (sample.features[static_cast<size_t>(TransformFeature::ACT)]) return FusedReason::ACT;
    if (sample.features[static_cast<size_t>(TransformFeature::LMCS)]) return FusedReason::LMCS_CHROMA_RESIDUAL;
    if (sample.features[static_cast<size_t>(TransformFeature::MTS)]
        || (!sample.features[static_cast<size_t>(TransformFeature::DCT2)]
            && !sample.features[static_cast<size_t>(TransformFeature::TRANSFORM_SKIP)]))
      return FusedReason::MTS_OR_OTHER_TRANSFORM;
    return FusedReason::NUM;
  }

  void recordFusedInverseTransform(const TransformProfileSample &sample)
  {
    if (!fusedCandidate.active) return;
    const FusedReason rejection = fusedTransformRejection(sample);
    if (rejection != FusedReason::NUM)
    {
      rejectFusedCandidate(rejection);
      return;
    }
    fusedCandidate.transformTasks++;
    fusedCandidate.qcoeffBytes = fusedCheckedAdd(fusedCandidate.qcoeffBytes,
                                                 fusedCheckedMultiply(sample.coefficients, sizeof(TCoeff)));
    fusedCandidate.descriptorBytes = fusedCheckedAdd(fusedCandidate.descriptorBytes,
                                                      sizeof(FusedTransformDescriptorModel));
    const size_t shape = static_cast<size_t>(sample.widthLog2) * transformSizeClasses + sample.heightLog2;
    fusedCandidate.scanShapes[shape] = true;
    if (sample.features[static_cast<size_t>(TransformFeature::DCT2)])
    {
      fusedCandidate.dct2Sizes[sample.widthLog2] = true;
      fusedCandidate.dct2Sizes[sample.heightLog2] = true;
    }
  }

  void fusedDependency(const FusedReason reason, const bool rejectCurrent)
  {
    flushFused(reason);
    if (rejectCurrent) rejectFusedCandidate(reason);
  }

  void prepareFusedCu(const FusedCandidate &candidate, const FusedReason rejection)
  {
    std::lock_guard<std::mutex> lock(stateMutex);
    validateHookThreadLocked();
    beginFusedCandidate(candidate, rejection);
  }

  void finishFusedCu()
  {
    std::lock_guard<std::mutex> lock(stateMutex);
    validateHookThreadLocked();
    finishFusedCandidate();
  }

  void validateHookThreadLocked()
  {
    const std::thread::id current = std::this_thread::get_id();
    hookCalls++;
    if (!haveOwnerThread)
    {
      ownerThread = current;
      ownerThreadHash = static_cast<uint64_t>(std::hash<std::thread::id>{}(current));
      haveOwnerThread = true;
    }
    else if (current != ownerThread)
    {
      threadMismatchHooks++;
    }
  }

  void flushMc(const McProfileReason reason)
  {
    if (activePath < 0)
    {
      if (ibcFills.pending != 0)
      {
        ibcBoundaryViolations++;
        ibcFills.completeInOrder();
      }
      return;
    }
    paths[static_cast<size_t>(activePath)].add(activeTasks, activePixels);
    flushReasons[static_cast<size_t>(reason)]++;
    ibcFills.completeInOrder();
    activePath = -1;
    activeTasks = 0;
    activePixels = 0;
  }

  void flushPipeline()
  {
    if (inverseTransformTasks != 0)
      inverseTransformRuns.add(inverseTransformTasks, inverseTransformPixels);
    if (reconstructionTasks != 0)
      reconstructionRuns.add(reconstructionTasks, reconstructionPixels);
    inverseTransformTasks = 0;
    inverseTransformPixels = 0;
    reconstructionTasks = 0;
    reconstructionPixels = 0;
    flushTransformGroup();
  }

  void flushTransformGroup()
  {
    for (size_t group = 0; group < transformGroups.size(); ++group)
    {
      if (pendingTransformGroupTasks[group] == 0) continue;
      transformGroups[group].add(pendingTransformGroupTasks[group], pendingTransformGroupPixels[group]);
      pendingTransformGroupTasks[group] = 0;
      pendingTransformGroupPixels[group] = 0;
    }
  }

  void beginPocLocked(const int poc)
  {
    if (!havePoc)
    {
      firstPoc = lastPoc = poc;
      pictures = 1;
      havePoc = true;
      return;
    }
    if (poc != lastPoc)
    {
      flushMc(McProfileReason::PICTURE_BOUNDARY);
      flushPipeline();
      flushFused(FusedReason::PICTURE_BOUNDARY);
      lastPoc = poc;
      pictures++;
    }
  }

  void rejectLocked(const McProfileReason reason, const bool reconstructionDependency)
  {
    rejectedCus[static_cast<size_t>(reason)]++;
    flushMc(reason);
    if (reconstructionDependency) flushPipeline();
  }

  void prepareCu(const int poc, const uint64_t pixels, const McProfileReason rejection,
                 const bool reconstructionDependency)
  {
    std::lock_guard<std::mutex> lock(stateMutex);
    validateHookThreadLocked();
    beginPocLocked(poc);
    totalPixels += pixels;
    if (rejection != McProfileReason::NUM)
    {
      if (rejection == McProfileReason::IBC)
      {
        ibcPrepareChecks++;
        if (!ibcFills.boundaryIsValid()) ibcPrepareViolations++;
      }
      rejectLocked(rejection, reconstructionDependency);
    }
  }

  void queueMc(const McProfilePath path, const uint64_t pixels)
  {
    std::lock_guard<std::mutex> lock(stateMutex);
    validateHookThreadLocked();
    const int pathIndex = static_cast<int>(path);
    if (activePath >= 0 && activePath != pathIndex) flushMc(McProfileReason::PATH_CHANGE);
    if (activePath < 0) activePath = pathIndex;
    activeTasks++;
    activePixels += pixels;
    eligibleCus++;
    eligiblePixels += pixels;
  }

  void recordTransformBlock(const ComponentID component, const unsigned widthLog2, const unsigned heightLog2,
                            const uint64_t pixels, const bool cbf)
  {
    std::lock_guard<std::mutex> lock(stateMutex);
    validateHookThreadLocked();
    transformBlocks[transformShapeIndex(component, widthLog2, heightLog2)].add(pixels, cbf);
    if (component == COMPONENT_Y) recordFusedTransformBlock();
  }

  void recordInverseTransform(const TransformProfileSample &sample)
  {
    std::lock_guard<std::mutex> lock(stateMutex);
    validateHookThreadLocked();
    recordFusedInverseTransform(sample);
    inverseTransformTasks++;
    inverseTransformPixels += sample.pixels;
    transformTotals.add(sample.pixels, sample.coefficients, sample.nonzeroCoefficients);
    transformShapes[transformShapeIndex(sample.component, sample.widthLog2, sample.heightLog2)]
      .add(sample.pixels, sample.coefficients, sample.nonzeroCoefficients);
    dequantPaths[static_cast<size_t>(sample.dequantPath)].add(sample.pixels, sample.coefficients,
                                                              sample.nonzeroCoefficients);
    for (size_t feature = 0; feature < sample.features.size(); ++feature)
    {
      if (sample.features[feature])
        transformFeatures[feature].add(sample.pixels, sample.coefficients, sample.nonzeroCoefficients);
    }
    if (sample.qp < transformQpMinimum)
      transformQpUnderflow++;
    else if (sample.qp > transformQpMaximum)
      transformQpOverflow++;
    else
      transformQps[static_cast<size_t>(sample.qp - transformQpMinimum)]++;

    if (!sample.features[static_cast<size_t>(TransformFeature::REGULAR_DCT2_CANDIDATE)])
    {
      flushTransformGroup();
      return;
    }
    const size_t group = transformGroupIndex(sample.component, sample.widthLog2, sample.heightLog2, sample.dequantPath);
    pendingTransformGroupTasks[group]++;
    pendingTransformGroupPixels[group] += sample.pixels;
  }

  void recordReconstruction(const uint64_t pixels)
  {
    std::lock_guard<std::mutex> lock(stateMutex);
    validateHookThreadLocked();
    reconstructionTasks++;
    reconstructionPixels += pixels;
  }

  void lmcsChromaAdjDependency()
  {
    std::lock_guard<std::mutex> lock(stateMutex);
    validateHookThreadLocked();
    flushMc(McProfileReason::LMCS_CHROMA_ADJ);
    flushPipeline();
    fusedDependency(FusedReason::LMCS_CACHE_DEPENDENCY, true);
  }

  void ibcBufferBoundary(const McProfileReason reason)
  {
    std::lock_guard<std::mutex> lock(stateMutex);
    validateHookThreadLocked();
    flushMc(reason);
    flushPipeline();
    const FusedReason fusedReason = reason == McProfileReason::IBC_BUFFER_RESET ? FusedReason::IBC_BUFFER_RESET
                                    : reason == McProfileReason::IBC_VPDU_RESET ? FusedReason::IBC_VPDU_RESET
                                                                               : FusedReason::IBC_PRE_MV_CONSUMER;
    fusedDependency(fusedReason, false);
    ibcBoundaryChecks++;
    if (!ibcFills.boundaryIsValid()) ibcBoundaryViolations++;
  }

  void recordPendingIbcFill(const bool observable, const bool legacyQueued)
  {
    std::lock_guard<std::mutex> lock(stateMutex);
    validateHookThreadLocked();
    if (legacyQueued) ibcFills.enqueue(observable);
    for (size_t core = 0; core < fusedCores.size(); ++core)
    {
      if (!lastFusedHadCandidate[core]) continue;
      FusedCoreProfile &profile = fusedCores[core];
      if (lastFusedDeferredFill[core])
      {
        profile.ibcFills.enqueue(observable);
      }
      else if (observable)
      {
        profile.ibcImmediateFills++;
      }
      else
      {
        profile.ibcFills.enqueue(false);
      }
      lastFusedHadCandidate[core] = false;
      lastFusedDeferredFill[core] = false;
    }
  }

  static bool runFusedSchedulerSelfTests()
  {
    if (fusedLmcsLumaActive(false, false) || fusedLmcsLumaActive(true, false)
        || !fusedLmcsLumaActive(true, true))
      return false;

    const uint64_t maximum = std::numeric_limits<uint64_t>::max();
    if (fusedCeilDivide(maximum, 1) != maximum
        || fusedCeilDivide(maximum, 2) != maximum / 2 + 1
        || fusedCeilDivide(0, maximum) != 0)
      return false;

    const auto checkWeightedClassification = [](const SliceType sliceType, const bool useWp, const bool useWpBi,
                                                const uint8_t interDir, const int refIdxList0,
                                                const int refIdxList1, const McProfilePath expectedPath,
                                                const bool expectList0, const bool expectList1,
                                                const uint64_t expectedOperations)
    {
      const McPredictionClassification result = mcProfileClassifyPrediction(
        sliceType, useWp, useWpBi, BCW_DEFAULT, false, false, interDir, refIdxList0, refIdxList1, false);
      const std::array<int, NUM_REF_PIC_LIST_01> refIdx = { refIdxList0, refIdxList1 };
      uint64_t metadataEntries = 0;
      for (size_t list = 0; list < result.effectiveLists.size(); ++list)
      {
        if (!result.effectiveLists[list]) continue;
        // The production metadata loop calls getRefPic only for these entries.
        if (refIdx[list] < 0) return false;
        metadataEntries++;
      }
      return result.path == expectedPath
          && result.effectiveLists[REF_PIC_LIST_0] == expectList0
          && result.effectiveLists[REF_PIC_LIST_1] == expectList1
          && result.operationCount() == expectedOperations
          && result.weightedMetadataCount() == metadataEntries;
    };
    if (!checkWeightedClassification(B_SLICE, false, true, 1, 0, -1, McProfilePath::UNI_WEIGHTED,
                                      true, false, 1)
        || !checkWeightedClassification(B_SLICE, false, true, 2, -1, 0, McProfilePath::UNI_WEIGHTED,
                                         false, true, 1)
        || !checkWeightedClassification(B_SLICE, false, true, 3, 0, 1, McProfilePath::BI_WEIGHTED,
                                         true, true, 2)
        || !checkWeightedClassification(P_SLICE, true, false, 1, 0, -1, McProfilePath::UNI_WEIGHTED,
                                         true, false, 1))
      return false;

    const auto candidate = [](const FusedRefMode mode, const uint64_t predictionUnits,
                              const uint64_t predictionOperations)
    {
      FusedCandidate value;
      value.lumaPixels = 64;
      value.componentPixels = 96;
      value.predictionUnits = predictionUnits;
      value.predictionOperations = predictionOperations;
      value.refModes[static_cast<size_t>(mode)] = predictionUnits;
      value.pictureMetadata.identity = 1;
      value.sliceMetadata.identity = 2;
      value.sliceMetadata.pictureIdentity = 1;
      FusedReferenceMetadataKey reference;
      reference.sliceIdentity = 2;
      reference.referenceIdentity = 3;
      reference.refIdx = 0;
      value.referenceMetadata.push_back(reference);
      return value;
    };

    auto lmcsEligibility = std::make_unique<McProfile>(true);
    const std::array<std::array<bool, 2>, 3> lmcsFlags = { {
      { false, false }, { true, false }, { true, true }
    } };
    for (size_t index = 0; index < lmcsFlags.size(); ++index)
    {
      lmcsEligibility->prepareCu(index == 2 ? 1 : 0, 64, McProfileReason::NUM, false);
      const bool active = fusedLmcsLumaActive(lmcsFlags[index][0], lmcsFlags[index][1]);
      lmcsEligibility->beginFusedCandidate(candidate(FusedRefMode::UNI, 1, 1),
                                            active ? FusedReason::LMCS_LUMA : FusedReason::NUM);
      lmcsEligibility->finishFusedCandidate();
    }
    for (const FusedCoreProfile &core : lmcsEligibility->fusedCores)
    {
      if (core.consideredCus != 3 || core.eligibleCus != 2 || core.distribution.windows != 1
          || core.distribution.totals[static_cast<size_t>(FusedMetric::CUS)] != 2
          || core.flushReasons[static_cast<size_t>(FusedReason::PICTURE_BOUNDARY)] != 1
          || core.rejectedCus[static_cast<size_t>(FusedReason::LMCS_LUMA)] != 1)
        return false;
    }

    auto bcwMetadata = std::make_unique<McProfile>(true);
    for (uintptr_t sliceIdentity : { uintptr_t{ 10 }, uintptr_t{ 20 } })
    {
      FusedCandidate bcwCandidate = candidate(FusedRefMode::BI_BCW, 1, 2);
      bcwCandidate.sliceMetadata.identity = sliceIdentity;
      bcwCandidate.referenceMetadata[0].sliceIdentity = sliceIdentity;
      FusedBcwMetadataKey key;
      key.bcwIdx = 0;
      key.weightList0 = getBcwWeight(key.bcwIdx, REF_PIC_LIST_0);
      key.weightList1 = getBcwWeight(key.bcwIdx, REF_PIC_LIST_1);
      bcwCandidate.bcwMetadata.push_back(key);
      bcwMetadata->beginFusedCandidate(bcwCandidate, FusedReason::NUM);
      bcwMetadata->finishFusedCandidate();
    }
    for (const FusedCoreProfile &core : bcwMetadata->fusedCores)
      if (core.sharedBcwBytesOnce != sizeof(FusedBcwMetadataModel) || core.seenBcwEntries.size() != 1) return false;

    auto scheduler = std::make_unique<McProfile>(true);
    scheduler->beginFusedCandidate(candidate(FusedRefMode::UNI, 1, 1), FusedReason::NUM);
    scheduler->finishFusedCandidate();
    scheduler->recordPendingIbcFill(true, false);
    FusedCandidate bi = candidate(FusedRefMode::BI_AVG, 1, 2);
    scheduler->beginFusedCandidate(bi, FusedReason::NUM);
    scheduler->finishFusedCandidate();
    scheduler->recordPendingIbcFill(true, false);
    scheduler->fusedDependency(FusedReason::IBC_PRE_MV_CONSUMER, false);
    for (const FusedCoreProfile &core : scheduler->fusedCores)
    {
      if (core.distribution.windows != 1
          || core.distribution.totals[static_cast<size_t>(FusedMetric::CUS)] != 2
          || core.ibcFills.totalQueued != 2 || core.ibcFills.totalApplied != 2
          || !core.ibcFills.boundaryIsValid()) return false;
    }
    if (scheduler->ibcFills.totalQueued != 0) return false;

    FusedCandidate mixed = candidate(FusedRefMode::UNI, 2, 3);
    mixed.refModes[static_cast<size_t>(FusedRefMode::UNI)] = 1;
    mixed.refModes[static_cast<size_t>(FusedRefMode::BI_AVG)] = 1;
    scheduler->beginFusedCandidate(mixed, FusedReason::NUM);
    scheduler->finishFusedCandidate();
    scheduler->recordPendingIbcFill(true, false);
    scheduler->fusedDependency(FusedReason::IBC_BUFFER_RESET, false);
    for (const FusedCoreProfile &core : scheduler->fusedCores)
      if (core.ibcFills.totalQueued != 3 || core.ibcFills.totalApplied != 3) return false;

    auto metadata = std::make_unique<McProfile>(true);
    FusedCandidate plain = candidate(FusedRefMode::UNI, 1, 1);
    metadata->beginFusedCandidate(plain, FusedReason::NUM);
    metadata->finishFusedCandidate();
    metadata->flushFused(FusedReason::PICTURE_BOUNDARY);
    const FusedCoreProfile &plainCore = metadata->fusedCores[static_cast<size_t>(FusedCore::CORE_B_RPR)];
    const uint64_t plainShared = fusedCheckedAdd(fusedCheckedAdd(plainCore.sharedPictureBytesOnce,
                                                                 plainCore.sharedSliceBytesOnce),
                                                 plainCore.sharedReferenceBytesOnce);
    FusedCandidate rpr = candidate(FusedRefMode::UNI_WEIGHTED, 1, 1);
    rpr.rpr = true;
    rpr.weighted = true;
    rpr.rprPredictionOperations = 1;
    rpr.referenceMetadata[0].weighted = true;
    rpr.referenceMetadata[0].rpr = true;
    rpr.referenceMetadata[0].scaleX = 1;
    rpr.referenceMetadata[0].scaleY = 1;
    rpr.referenceMetadata[0].weight[0] = 1;
    metadata->beginFusedCandidate(rpr, FusedReason::NUM);
    metadata->finishFusedCandidate();
    metadata->recordPendingIbcFill(true, false);
    metadata->flushFused(FusedReason::STREAM_END);
    const FusedCoreProfile &rprCore = metadata->fusedCores[static_cast<size_t>(FusedCore::CORE_B_RPR)];
    if (metadata->fusedCores[static_cast<size_t>(FusedCore::CORE_A)].ibcImmediateFills != 1
        || rprCore.ibcFills.totalQueued != 1 || rprCore.ibcFills.totalApplied != 1
        || rprCore.sharedWeightedBytesOnce != sizeof(FusedWeightedPredictionMetadataModel)
        || rprCore.sharedRprBytesOnce != sizeof(FusedRprMetadataModel)
        || fusedCheckedAdd(plainShared, fusedCheckedAdd(rprCore.sharedWeightedBytesOnce,
                                                       rprCore.sharedRprBytesOnce)) <= plainShared)
      return false;

    auto pocBoundary = std::make_unique<McProfile>(true);
    pocBoundary->prepareCu(0, 64, McProfileReason::NUM, false);
    pocBoundary->beginFusedCandidate(candidate(FusedRefMode::UNI, 1, 1), FusedReason::NUM);
    pocBoundary->finishFusedCandidate();
    pocBoundary->recordPendingIbcFill(true, false);
    pocBoundary->prepareCu(1, 64, McProfileReason::NUM, false);
    const FusedCoreProfile &beforeRpr = pocBoundary->fusedCores[static_cast<size_t>(FusedCore::CORE_B_RPR)];
    if (beforeRpr.distribution.windows != 1 || beforeRpr.ibcFills.totalApplied != 1) return false;
    pocBoundary->beginFusedCandidate(rpr, FusedReason::NUM);
    if (pocBoundary->fusedCandidate.rejections[static_cast<size_t>(FusedCore::CORE_A)] != FusedReason::RPR
        || pocBoundary->fusedCandidate.rejections[static_cast<size_t>(FusedCore::CORE_B_RPR)] != FusedReason::NUM)
      return false;

    auto discard = std::make_unique<McProfile>(true);
    discard->beginFusedCandidate(candidate(FusedRefMode::UNI, 1, 1), FusedReason::NUM);
    TransformProfileSample supported;
    supported.widthLog2 = 2;
    supported.heightLog2 = 2;
    supported.pixels = 16;
    supported.coefficients = 16;
    supported.features[static_cast<size_t>(TransformFeature::DCT2)] = true;
    discard->recordFusedInverseTransform(supported);
    TransformProfileSample unsupported = supported;
    unsupported.features[static_cast<size_t>(TransformFeature::DCT2)] = false;
    unsupported.features[static_cast<size_t>(TransformFeature::MTS)] = true;
    discard->recordFusedInverseTransform(unsupported);
    discard->finishFusedCandidate();
    for (const FusedCoreProfile &core : discard->fusedCores)
    {
      if (core.eligibleCus != 0 || core.distribution.windows != 0
          || core.rejectedCus[static_cast<size_t>(FusedReason::MTS_OR_OTHER_TRANSFORM)] != 1)
        return false;
    }
    return true;
  }

  explicit McProfile(const bool isolatedSelfTest = false)
  {
    if (isolatedSelfTest) return;
    decoderInstance = mcProfileDecoderInstances.fetch_add(1, std::memory_order_relaxed) + 1;
    schedulerSelfTestPassed = runFusedSchedulerSelfTests();
    if (!schedulerSelfTestPassed) THROW("decoder fused-batch profiler scheduler/model self-test failed");
  }

  static void appendDistribution(std::ostream &output, const RunDistribution &distribution)
  {
    output << "{\"runs\":" << distribution.tasks.count
           << ",\"tasks\":" << distribution.totalTasks
           << ",\"pixels\":" << distribution.totalPixels
           << ",\"tasks_q50_upper\":" << distribution.tasks.quantileUpperBound(50)
           << ",\"tasks_q90_upper\":" << distribution.tasks.quantileUpperBound(90)
           << ",\"tasks_q99_upper\":" << distribution.tasks.quantileUpperBound(99)
           << ",\"tasks_max\":" << distribution.tasks.maximum
           << ",\"pixels_q50_upper\":" << distribution.pixels.quantileUpperBound(50)
           << ",\"pixels_q90_upper\":" << distribution.pixels.quantileUpperBound(90)
           << ",\"pixels_q99_upper\":" << distribution.pixels.quantileUpperBound(99)
           << ",\"pixels_max\":" << distribution.pixels.maximum << '}';
  }

  static void appendTransformAggregate(std::ostream &output, const TransformAggregate &aggregate)
  {
    output << "{\"tasks\":" << aggregate.tasks
           << ",\"pixels\":" << aggregate.pixels
           << ",\"coefficients\":" << aggregate.coefficients
           << ",\"nonzero_coefficients\":" << aggregate.nonzeroCoefficients << '}';
  }

  static void appendTransformBlockAggregate(std::ostream &output, const TransformBlockAggregate &aggregate)
  {
    output << "{\"blocks\":" << aggregate.blocks
           << ",\"cbf_blocks\":" << aggregate.cbfBlocks
           << ",\"zero_cbf_blocks\":" << aggregate.zeroCbfBlocks
           << ",\"pixels\":" << aggregate.pixels
           << ",\"cbf_pixels\":" << aggregate.cbfPixels
           << ",\"zero_cbf_pixels\":" << aggregate.zeroCbfPixels << '}';
  }

  static void appendFusedMetric(std::ostream &output, const LogHistogram &histogram, const uint64_t total)
  {
    output << "{\"total\":" << total
           << ",\"p50_upper\":" << histogram.quantileUpperBound(50)
           << ",\"p90_upper\":" << histogram.quantileUpperBound(90)
           << ",\"p99_upper\":" << histogram.quantileUpperBound(99)
           << ",\"max\":" << histogram.maximum << ",\"log2_buckets\":[";
    for (size_t bucket = 0; bucket < histogram.buckets.size(); ++bucket)
    {
      if (bucket != 0) output << ',';
      output << histogram.buckets[bucket];
    }
    output << "]}";
  }

  static void appendFusedCore(std::ostream &output, const FusedCoreProfile &profile)
  {
    uint64_t sharedBytes = fusedCheckedAdd(profile.sharedScanBytesOnce, profile.sharedMatrixBytesOnce);
    sharedBytes = fusedCheckedAdd(sharedBytes, profile.sharedDequantBytesOnce);
    sharedBytes = fusedCheckedAdd(sharedBytes, profile.sharedPictureBytesOnce);
    sharedBytes = fusedCheckedAdd(sharedBytes, profile.sharedSliceBytesOnce);
    sharedBytes = fusedCheckedAdd(sharedBytes, profile.sharedReferenceBytesOnce);
    sharedBytes = fusedCheckedAdd(sharedBytes, profile.sharedWeightedBytesOnce);
    sharedBytes = fusedCheckedAdd(sharedBytes, profile.sharedRprBytesOnce);
    sharedBytes = fusedCheckedAdd(sharedBytes, profile.sharedBcwBytesOnce);
    output << "{\"coverage\":{\"considered_cus\":" << profile.consideredCus
           << ",\"considered_luma_pixels\":" << profile.consideredPixels
           << ",\"eligible_cus\":" << profile.eligibleCus
           << ",\"eligible_luma_pixels\":" << profile.eligiblePixels
           << ",\"rejected_cus\":" << (profile.consideredCus - profile.eligibleCus)
           << ",\"rejected_luma_pixels\":" << (profile.consideredPixels - profile.eligiblePixels) << "}"
           << ",\"windows\":" << profile.distribution.windows << ",\"metrics\":{";
    for (size_t metric = 0; metric < profile.distribution.metrics.size(); ++metric)
    {
      if (metric != 0) output << ',';
      output << '\"' << fusedMetricNames[metric] << "\":";
      appendFusedMetric(output, profile.distribution.metrics[metric], profile.distribution.totals[metric]);
    }
    output << "},\"ref_modes\":{";
    bool comma = false;
    for (size_t mode = 0; mode < profile.distribution.refModePredictionUnits.size(); ++mode)
    {
      if (profile.distribution.refModePredictionUnits[mode] == 0) continue;
      output << (comma ? ",\"" : "\"") << fusedRefModeNames[mode]
             << "\":{\"prediction_units\":" << profile.distribution.refModePredictionUnits[mode]
             << ",\"windows\":" << profile.distribution.refModeWindows[mode] << '}';
      comma = true;
    }
    output << "},\"rejections\":{";
    comma = false;
    for (size_t reason = 0; reason < profile.rejectedCus.size(); ++reason)
    {
      if (profile.rejectedCus[reason] == 0) continue;
      output << (comma ? ",\"" : "\"") << fusedReasonNames[reason] << "\":{\"cus\":"
             << profile.rejectedCus[reason] << ",\"luma_pixels\":" << profile.rejectedPixels[reason] << '}';
      comma = true;
    }
    output << "},\"flush_reasons\":{";
    comma = false;
    for (size_t reason = 0; reason < profile.flushReasons.size(); ++reason)
    {
      if (profile.flushReasons[reason] == 0) continue;
      output << (comma ? ",\"" : "\"") << fusedReasonNames[reason] << "\":" << profile.flushReasons[reason];
      comma = true;
    }
    const uint64_t recurringH2d = fusedCheckedAdd(
      profile.distribution.totals[static_cast<size_t>(FusedMetric::DESCRIPTOR_BYTES)],
      profile.distribution.totals[static_cast<size_t>(FusedMetric::QCOEFF_BYTES)]);
    const uint64_t dirtyD2h = profile.distribution.totals[static_cast<size_t>(FusedMetric::DIRTY_DOWNLOAD_BYTES)];
    const uint64_t amortizedShared = profile.distribution.windows == 0
                                       ? 0
                                       : fusedCeilDivide(sharedBytes, profile.distribution.windows);
    output << "},\"transfer_model\":{\"descriptor_h2d_bytes\":"
           << profile.distribution.totals[static_cast<size_t>(FusedMetric::DESCRIPTOR_BYTES)]
           << ",\"qcoeff_h2d_bytes\":"
           << profile.distribution.totals[static_cast<size_t>(FusedMetric::QCOEFF_BYTES)]
           << ",\"shared_scan_bytes_once\":" << profile.sharedScanBytesOnce
           << ",\"shared_matrix_bytes_once\":" << profile.sharedMatrixBytesOnce
           << ",\"shared_dequant_constant_bytes_once\":" << profile.sharedDequantBytesOnce
           << ",\"shared_picture_metadata_bytes_once\":" << profile.sharedPictureBytesOnce
           << ",\"shared_slice_metadata_bytes_once\":" << profile.sharedSliceBytesOnce
           << ",\"shared_reference_metadata_bytes_once\":" << profile.sharedReferenceBytesOnce
           << ",\"shared_weighted_prediction_metadata_bytes_once\":" << profile.sharedWeightedBytesOnce
           << ",\"shared_rpr_metadata_bytes_once\":" << profile.sharedRprBytesOnce
           << ",\"shared_bcw_metadata_bytes_once\":" << profile.sharedBcwBytesOnce
           << ",\"shared_bytes_once\":" << sharedBytes
           << ",\"shared_bytes_amortized_per_window\":" << amortizedShared
           << ",\"dirty_boundary_d2h_bytes\":" << dirtyD2h
           << ",\"estimated_h2d_bytes_run\":" << fusedCheckedAdd(recurringH2d, sharedBytes)
           << ",\"estimated_d2h_bytes_run\":" << dirtyD2h
           << ",\"estimated_total_transfer_bytes_run\":"
           << fusedCheckedAdd(fusedCheckedAdd(recurringH2d, sharedBytes), dirtyD2h)
           << "},\"feature_coverage\":{\"rpr_cus\":" << profile.rprCus
           << ",\"rpr_prediction_operations\":" << profile.rprPredictionOperations
           << ",\"weighted_prediction_units\":" << profile.weightedPredictionUnits
           << ",\"bcw_prediction_units\":" << profile.bcwPredictionUnits
           << "},\"ibc_deferred_fills\":{\"queued\":" << profile.ibcFills.totalQueued
           << ",\"applied\":" << profile.ibcFills.totalApplied
           << ",\"max_pending\":" << profile.ibcFills.maximumPending
           << ",\"pending\":" << profile.ibcFills.pending
           << ",\"immediate\":" << profile.ibcImmediateFills
           << ",\"disabled_sps_noops\":" << profile.ibcFills.disabledSpsNoops
           << ",\"last_queued_sequence\":" << profile.ibcFills.lastQueuedSequence
           << ",\"last_applied_sequence\":" << profile.ibcFills.lastAppliedSequence
           << ",\"boundary_checks\":" << profile.ibcBoundaryChecks
           << ",\"boundary_violations\":" << profile.ibcBoundaryViolations << "}}";
  }

  void report()
  {
    std::ostringstream output;
    std::lock_guard<std::mutex> stateLock(stateMutex);
    finishFusedCandidate();
    flushMc(McProfileReason::STREAM_END);
    flushPipeline();
    flushFused(FusedReason::STREAM_END);
    const uint64_t reportThreadHash = static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    output << "DECODER_BATCH_PROFILE {\"schema\":7,\"process_id\":" << processId
           << ",\"decoder_instance\":" << decoderInstance
           << ",\"owner_thread_hash\":" << ownerThreadHash
           << ",\"report_thread_hash\":" << reportThreadHash
           << ",\"hook_calls\":" << hookCalls
           << ",\"thread_mismatch_hooks\":" << threadMismatchHooks
           << ",\"scheduler_self_test\":{\"passed\":" << (schedulerSelfTestPassed ? "true" : "false")
           << ",\"weighted_metadata_extra_bytes\":" << sizeof(FusedWeightedPredictionMetadataModel)
           << ",\"rpr_metadata_extra_bytes\":" << sizeof(FusedRprMetadataModel) << "}"
           << ",\"first_poc\":" << firstPoc << ",\"last_poc\":" << lastPoc
           << ",\"pictures\":" << pictures << ",\"total_pixels\":" << totalPixels
           << ",\"eligible_pixels\":" << eligiblePixels << ",\"eligible_cus\":" << eligibleCus
           << ",\"paths\":{";
    bool comma = false;
    for (size_t path = 0; path < paths.size(); ++path)
    {
      if (paths[path].tasks.count == 0) continue;
      output << (comma ? ",\"" : "\"") << mcProfilePathNames[path] << "\":";
      appendDistribution(output, paths[path]);
      comma = true;
    }
    output << "},\"rejections\":{";
    comma = false;
    for (size_t reason = 0; reason < rejectedCus.size(); ++reason)
    {
      if (rejectedCus[reason] == 0 && flushReasons[reason] == 0) continue;
      output << (comma ? ",\"" : "\"") << mcProfileReasonNames[reason] << "\":{\"cus\":"
             << rejectedCus[reason] << ",\"flushes\":" << flushReasons[reason] << '}';
      comma = true;
    }
    output << "},\"inverse_transform\":";
    appendDistribution(output, inverseTransformRuns);
    output << ",\"reconstruction\":";
    appendDistribution(output, reconstructionRuns);
    output << ",\"transform_workload\":{\"totals\":";
    appendTransformAggregate(output, transformTotals);
    output << ",\"by_shape_component\":{";
    comma = false;
    for (unsigned component = 0; component < MAX_NUM_COMPONENT; ++component)
    {
      for (unsigned widthLog2 = 0; widthLog2 < transformSizeClasses; ++widthLog2)
      {
        for (unsigned heightLog2 = 0; heightLog2 < transformSizeClasses; ++heightLog2)
        {
          const size_t index = transformShapeIndex(static_cast<ComponentID>(component), widthLog2, heightLog2);
          const TransformBlockAggregate &blocks = transformBlocks[index];
          const TransformAggregate &tasks = transformShapes[index];
          if (blocks.blocks == 0 && tasks.tasks == 0) continue;
          output << (comma ? ",\"" : "\"") << transformComponentNames[component] << ':'
                 << (uint64_t{ 1 } << widthLog2) << 'x' << (uint64_t{ 1 } << heightLog2) << "\":{\"blocks\":";
          appendTransformBlockAggregate(output, blocks);
          output << ",\"transforms\":";
          appendTransformAggregate(output, tasks);
          output << '}';
          comma = true;
        }
      }
    }
    output << "},\"features\":{";
    comma = false;
    for (size_t feature = 0; feature < transformFeatures.size(); ++feature)
    {
      if (transformFeatures[feature].tasks == 0) continue;
      output << (comma ? ",\"" : "\"") << transformFeatureNames[feature] << "\":";
      appendTransformAggregate(output, transformFeatures[feature]);
      comma = true;
    }
    output << "},\"dequant_paths\":{";
    comma = false;
    for (size_t path = 0; path < dequantPaths.size(); ++path)
    {
      if (dequantPaths[path].tasks == 0) continue;
      output << (comma ? ",\"" : "\"") << dequantPathNames[path] << "\":";
      appendTransformAggregate(output, dequantPaths[path]);
      comma = true;
    }
    output << "},\"qp\":{\"underflow\":" << transformQpUnderflow << ",\"overflow\":" << transformQpOverflow
           << ",\"counts\":{";
    comma = false;
    for (size_t qpIndex = 0; qpIndex < transformQps.size(); ++qpIndex)
    {
      if (transformQps[qpIndex] == 0) continue;
      output << (comma ? ",\"" : "\"") << (static_cast<int>(qpIndex) + transformQpMinimum) << "\":"
             << transformQps[qpIndex];
      comma = true;
    }
    output << "}},\"groups\":{";
    comma = false;
    for (unsigned component = 0; component < MAX_NUM_COMPONENT; ++component)
    {
      for (unsigned widthLog2 = 0; widthLog2 < transformSizeClasses; ++widthLog2)
      {
        for (unsigned heightLog2 = 0; heightLog2 < transformSizeClasses; ++heightLog2)
        {
          for (size_t path = 0; path < static_cast<size_t>(DequantPath::NUM); ++path)
          {
            const size_t index = transformGroupIndex(static_cast<ComponentID>(component), widthLog2, heightLog2,
                                                     static_cast<DequantPath>(path));
            if (transformGroups[index].tasks.count == 0) continue;
            output << (comma ? ",\"" : "\"") << transformComponentNames[component] << ':'
                   << (uint64_t{ 1 } << widthLog2) << 'x' << (uint64_t{ 1 } << heightLog2) << ':'
                   << dequantPathNames[path] << "\":";
            appendDistribution(output, transformGroups[index]);
            comma = true;
          }
        }
      }
    }
    output << "}}";
    output << ",\"fused_windows\":{\"model\":\"mc_dequant_inverse_transform_reconstruction\""
           << ",\"assumptions\":{\"references_resident\":true,\"output_mirror_resident\":true"
           << ",\"heterogeneous_mc_paths\":true,\"heterogeneous_components\":true"
           << ",\"heterogeneous_transform_shapes\":true,\"heterogeneous_ts_dct2\":true"
           << ",\"dirty_output_downloaded_at_cpu_boundaries\":true}"
           << ",\"descriptor_layout_bytes\":{\"cu\":" << sizeof(FusedCuDescriptorModel)
           << ",\"prediction_operation\":" << sizeof(FusedPredictionOperationDescriptorModel)
           << ",\"transform_task\":" << sizeof(FusedTransformDescriptorModel)
           << ",\"qcoeff_element\":" << sizeof(TCoeff)
           << ",\"pel_element\":" << sizeof(Pel)
           << ",\"scan_element\":" << sizeof(ScanElement)
           << ",\"matrix_element\":" << sizeof(TMatrixCoeff)
           << "},\"shared_metadata_layout_bytes\":{\"picture\":" << sizeof(FusedPictureMetadataModel)
           << ",\"slice\":" << sizeof(FusedSliceMetadataModel)
           << ",\"reference\":" << sizeof(FusedReferenceMetadataModel)
           << ",\"weighted_prediction\":" << sizeof(FusedWeightedPredictionMetadataModel)
           << ",\"rpr\":" << sizeof(FusedRprMetadataModel)
           << ",\"bcw\":" << sizeof(FusedBcwMetadataModel) << "},\"cores\":{";
    for (size_t core = 0; core < fusedCores.size(); ++core)
    {
      if (core != 0) output << ',';
      output << '\"' << fusedCoreNames[core] << "\":";
      appendFusedCore(output, fusedCores[core]);
    }
    output << "}}";
    output << ",\"ibc_buffer_fills\":{\"queued\":" << ibcFills.totalQueued
           << ",\"applied\":" << ibcFills.totalApplied
           << ",\"max_pending\":" << ibcFills.maximumPending
           << ",\"disabled_sps_noops\":" << ibcFills.disabledSpsNoops
           << ",\"last_queued_sequence\":" << ibcFills.lastQueuedSequence
           << ",\"last_applied_sequence\":" << ibcFills.lastAppliedSequence
           << ",\"boundary_checks\":" << ibcBoundaryChecks
           << ",\"boundary_violations\":" << ibcBoundaryViolations
           << ",\"prepare_checks\":" << ibcPrepareChecks
           << ",\"prepare_violations\":" << ibcPrepareViolations
           << ",\"pending_at_report\":" << ibcFills.pending << '}';
    output << "}\n";
    const std::string line = output.str();
    std::lock_guard<std::mutex> outputLock(mcProfileOutputMutex);
    std::fwrite(line.data(), 1, line.size(), stderr);
    std::fflush(stderr);
  }
};
#endif

DecCu::DecCu() : m_tmpStorageCtu(nullptr)
#if VTM_ENABLE_DECODER_BATCH_PROFILING
  , m_mcProfile(mcProfileEnabledFromEnvironment() ? std::make_unique<McProfile>() : nullptr)
#endif
{
}

DecCu::~DecCu()
{
#if VTM_ENABLE_DECODER_BATCH_PROFILING
  if (m_mcProfile) m_mcProfile->report();
#endif
}

void DecCu::init( TrQuant* pcTrQuant, IntraPrediction* pcIntra, InterPrediction* pcInter)
{
  m_pcTrQuant       = pcTrQuant;
  m_pcIntraPred     = pcIntra;
  m_pcInterPred     = pcInter;
}

void DecCu::initDecCuReshaper(Reshape* pcReshape, ChromaFormat chromaFormatIdc)
{
  m_pcReshape = pcReshape;
  if (m_tmpStorageCtu == nullptr)
  {
    m_tmpStorageCtu = new PelStorage;
    m_tmpStorageCtu->create(UnitArea(chromaFormatIdc, Area(0, 0, MAX_CU_SIZE, MAX_CU_SIZE)));
  }

}
void DecCu::destoryDecCuReshaprBuf()
{
  if (m_tmpStorageCtu)
  {
    m_tmpStorageCtu->destroy();
    delete m_tmpStorageCtu;
    m_tmpStorageCtu = nullptr;
  }
}

// ====================================================================================================================
// Public member functions
// ====================================================================================================================

void DecCu::decompressCtu( CodingStructure& cs, const UnitArea& ctuArea )
{
  const int maxNumChannelType = isChromaEnabled(cs.pcv->chrFormat) && CS::isDualITree(cs) ? 2 : 1;

  if (cs.resetIBCBuffer)
  {
#if VTM_ENABLE_DECODER_BATCH_PROFILING
    if (mcProfileIbcFillObservable(cs.slice->getSPS()->getIBCFlag())) xProfileIbcBufferReset();
#endif
    m_pcInterPred->resetIBCBuffer(cs.pcv->chrFormat, cs.slice->getSPS()->getMaxCUHeight());
    cs.resetIBCBuffer = false;
  }
  for( int ch = 0; ch < maxNumChannelType; ch++ )
  {
    const ChannelType chType = ChannelType( ch );
    Position prevTmpPos;
    prevTmpPos.x = -1; prevTmpPos.y = -1;

    for( auto &currCU : cs.traverseCUs( CS::getArea( cs, ctuArea, chType ), chType ) )
    {
      if(currCU.Y().valid())
      {
        const int vSize = std::min<int>(VPDU_SIZE, cs.slice->getSPS()->getMaxCUHeight());
        if((currCU.Y().x % vSize) == 0 && (currCU.Y().y % vSize) == 0)
        {
          for(int x = currCU.Y().x; x < currCU.Y().x + currCU.Y().width; x += vSize)
          {
            for(int y = currCU.Y().y; y < currCU.Y().y + currCU.Y().height; y += vSize)
            {
#if VTM_ENABLE_DECODER_BATCH_PROFILING
              if (mcProfileIbcFillObservable(cs.slice->getSPS()->getIBCFlag())) xProfileIbcVpduReset();
#endif
              m_pcInterPred->resetVPDUforIBC(cs.pcv->chrFormat, cs.slice->getSPS()->getMaxCUHeight(), vSize,
                                             x + IBC_BUFFER_SIZE / cs.slice->getSPS()->getMaxCUHeight() / 2, y);
            }
          }
        }
      }
      if (!CU::isIntra(currCU) && !CU::isPLT(currCU) && currCU.Y().valid())
      {
#if VTM_ENABLE_DECODER_BATCH_PROFILING
        if (CU::isIBC(currCU)) xProfileIbcPreMvConsumer();
#endif
        xDeriveCuMvs(currCU);
#if K0149_BLOCK_STATISTICS
        if(currCU.geoFlag)
        {
          storeGeoMergeCtx(m_geoMrgCtx);
        }
#endif
      }
#if VTM_ENABLE_DECODER_BATCH_PROFILING
      const int mcProfilePath = xProfilePrepareMcCu(currCU);
#endif
      switch( currCU.predMode )
      {
      case MODE_INTER:
      case MODE_IBC:
        xReconInter( currCU );
#if VTM_ENABLE_DECODER_BATCH_PROFILING
        if (mcProfilePath >= 0) xProfileQueueMcCu(currCU, mcProfilePath);
#endif
        break;
      case MODE_PLT:
      case MODE_INTRA:
        xReconIntraQT( currCU );
        break;
      default:
        THROW( "Invalid prediction mode" );
        break;
      }
#if VTM_ENABLE_DECODER_BATCH_PROFILING
      xProfileFinishFusedCu();
#endif

      m_pcInterPred->xFillIBCBuffer(currCU);
#if VTM_ENABLE_DECODER_BATCH_PROFILING
      xProfileIbcFill(currCU, mcProfilePath >= 0);
#endif

      DTRACE_BLOCK_REC( cs.picture->getRecoBuf( currCU ), currCU, currCU.predMode );
    }
  }
#if K0149_BLOCK_STATISTICS
  getAndStoreBlockStatistics(cs, ctuArea);
#endif
}

#if VTM_ENABLE_DECODER_BATCH_PROFILING
int DecCu::xProfilePrepareMcCu(CodingUnit &cu)
{
  if (!m_mcProfile || !cu.Y().valid()) return -1;

  const uint64_t pixels = static_cast<uint64_t>(cu.Y().width) * static_cast<uint64_t>(cu.Y().height);
  uint64_t componentPixels = 0;
  for (unsigned component = 0; component < getNumberValidComponents(cu.chromaFormat); ++component)
  {
    const CompArea &area = cu.blocks[component];
    if (area.valid()) componentPixels += static_cast<uint64_t>(area.width) * static_cast<uint64_t>(area.height);
  }
  FusedCandidate fused;
  fused.lumaPixels = pixels;
  fused.componentPixels = componentPixels;
  fused.pictureMetadata.identity = reinterpret_cast<uintptr_t>(cu.cs->picture);
  fused.pictureMetadata.poc = cu.slice->getPOC();
  fused.pictureMetadata.width = cu.cs->pps->getPicWidthInLumaSamples();
  fused.pictureMetadata.height = cu.cs->pps->getPicHeightInLumaSamples();
  fused.pictureMetadata.chromaFormat = static_cast<unsigned>(cu.chromaFormat);
  fused.pictureMetadata.bitDepthLuma = cu.cs->sps->getBitDepth(ChannelType::LUMA);
  fused.pictureMetadata.bitDepthChroma = cu.cs->sps->getBitDepth(ChannelType::CHROMA);
  fused.sliceMetadata.identity = reinterpret_cast<uintptr_t>(cu.slice);
  fused.sliceMetadata.pictureIdentity = fused.pictureMetadata.identity;
  fused.sliceMetadata.poc = cu.slice->getPOC();
  fused.sliceMetadata.sliceType = static_cast<int>(cu.slice->getSliceType());
  fused.sliceMetadata.numRefList0 = cu.slice->getNumRefIdx(REF_PIC_LIST_0);
  fused.sliceMetadata.numRefList1 = cu.slice->getNumRefIdx(REF_PIC_LIST_1);
  fused.sliceMetadata.useWp = cu.cs->pps->getUseWP();
  fused.sliceMetadata.useWpBi = cu.cs->pps->getWPBiPred();
  const bool lmcsLuma = fusedLmcsLumaActive(cu.slice->getLmcsEnabledFlag(), m_pcReshape->getCTUFlag());
  const auto beginFused = [&](const FusedReason reason)
  {
    m_mcProfile->prepareFusedCu(fused, reason);
  };
  const auto reject = [&](const McProfileReason reason, const bool reconstructionDependency)
  {
    m_mcProfile->prepareCu(cu.slice->getPOC(), pixels, reason, reconstructionDependency);
    beginFused(fusedReasonFromMc(reason));
    return -1;
  };

  const McProfileReason modeReason = mcProfileModeReason(CU::isIBC(cu), CU::isInter(cu));
  if (modeReason != McProfileReason::NUM)
  {
    return reject(modeReason, true);
  }
  if (cu.geoFlag)
  {
    return reject(McProfileReason::GPM, false);
  }
  if (cu.affine)
  {
    return reject(McProfileReason::AFFINE_OR_PROF, false);
  }

  const bool luma = cu.Y().valid();
  const bool chroma = isChromaEnabled(cu.chromaFormat) && cu.Cb().valid();
  const bool directList0 = !(luma && (chroma || !isChromaEnabled(cu.chromaFormat)));
  int cuPath = -1;
  bool mixedEffectivePath = false;
  for (auto &pu : CU::traversePUs(cu))
  {
    if (pu.ciipFlag)
    {
      return reject(McProfileReason::CIIP, true);
    }
    if (pu.mergeType != MergeType::DEFAULT_N)
    {
      return reject(McProfileReason::SUB_PU, false);
    }

    bool scaledReference = false;
    for (int list = 0; list < NUM_REF_PIC_LIST_01; list++)
    {
      const bool listUsed = (pu.interDir & (1 << list)) != 0;
      const int refIdx = pu.refIdx[list];
      const RefPicList refList = static_cast<RefPicList>(list);
      if ((listUsed && (refIdx < 0 || refIdx >= cu.slice->getNumRefIdx(refList))) || (!listUsed && refIdx >= 0))
      {
        return reject(McProfileReason::INVALID_DPB_REFERENCE, true);
      }
      if (listUsed)
      {
        Picture *refPic = cu.slice->getRefPic(refList, refIdx);
        if (refPic == nullptr)
        {
          return reject(McProfileReason::INVALID_DPB_REFERENCE, true);
        }
        scaledReference = scaledReference || refPic->isRefScaled(cu.cs->sps, cu.cs->pps);
      }
    }

    const bool identicalMotion = !directList0 && mcProfileIdenticalMotion(pu);
    if (!directList0 && !identicalMotion)
    {
      bool bdofApplied = false;
      if (cu.cs->sps->getBDOFEnabledFlag() && !cu.cs->picHeader->getBdofDisabledFlag())
      {
        bdofApplied = PU::isSimpleSymmetricBiPred(pu) && PU::dmvrBdofSizeCheck(pu)
                      && !pu.ciipFlag && !cu.smvdMode;
        if (pu.mmvdEncOptMode == 2 && pu.mmvdMergeFlag) bdofApplied = false;
      }
      bdofApplied = bdofApplied && !scaledReference;
      if (bdofApplied)
      {
        return reject(McProfileReason::BDOF, false);
      }
      const bool dmvrApplied = PU::checkDMVRCondition(pu) && !scaledReference;
      if (dmvrApplied)
      {
        return reject(McProfileReason::DMVR, false);
      }

    }

    const McPredictionClassification classification = mcProfileClassifyPrediction(
      cu.slice->getSliceType(), cu.cs->pps->getUseWP(), cu.cs->pps->getWPBiPred(), cu.bcwIdx,
      directList0, identicalMotion, pu.interDir, pu.refIdx[REF_PIC_LIST_0], pu.refIdx[REF_PIC_LIST_1],
      scaledReference);
    const McProfilePath path = classification.path;

    const int pathIndex = static_cast<int>(path);
    if (cuPath >= 0 && cuPath != pathIndex)
    {
      mixedEffectivePath = true;
    }
    if (cuPath < 0) cuPath = pathIndex;
    const FusedRefMode mode = fusedRefMode(path);
    fused.refModes[static_cast<size_t>(mode)]++;
    fused.predictionUnits++;
    const bool weighted = mode == FusedRefMode::UNI_WEIGHTED || mode == FusedRefMode::BI_WEIGHTED;
    const bool bcw = mode == FusedRefMode::BI_BCW;
    fused.weighted = fused.weighted || weighted;
    fused.bcw = fused.bcw || bcw;

    for (int list = 0; list < NUM_REF_PIC_LIST_01; ++list)
    {
      if (!classification.effectiveLists[list]) continue;
      const RefPicList refList = static_cast<RefPicList>(list);
      const int refIdx = pu.refIdx[list];
      Picture *refPic = cu.slice->getRefPic(refList, refIdx);
      FusedReferenceMetadataKey reference;
      reference.sliceIdentity = fused.sliceMetadata.identity;
      reference.referenceIdentity = reinterpret_cast<uintptr_t>(refPic);
      reference.referencePoc = refPic->poc;
      reference.refIdx = refIdx;
      reference.refList = list;
      reference.weighted = weighted;
      reference.rpr = refPic->isRefScaled(cu.cs->sps, cu.cs->pps);
      reference.bcw = bcw;
      reference.bcwIdx = cu.bcwIdx;
      const ScalingRatio &ratio = cu.slice->getScalingRatio(refList, refIdx);
      reference.scaleX = ratio.x;
      reference.scaleY = ratio.y;
      const Window &currentWindow = cu.cs->pps->getScalingWindow();
      const Window &referenceWindow = refPic->getScalingWindow();
      reference.currentWindow = { currentWindow.getWindowLeftOffset(), currentWindow.getWindowRightOffset(),
                                  currentWindow.getWindowTopOffset(), currentWindow.getWindowBottomOffset() };
      reference.referenceWindow = { referenceWindow.getWindowLeftOffset(), referenceWindow.getWindowRightOffset(),
                                    referenceWindow.getWindowTopOffset(), referenceWindow.getWindowBottomOffset() };
      if (weighted)
      {
        const WPScalingParam *wp = cu.slice->getWpScaling(refList, refIdx);
        for (unsigned component = 0; component < MAX_NUM_COMPONENT; ++component)
        {
          reference.codedWeight[component] = wp[component].codedWeight;
          reference.codedOffset[component] = wp[component].codedOffset;
          reference.weight[component] = wp[component].w;
          reference.offset[component] = wp[component].offset;
          reference.shift[component] = wp[component].shift;
          reference.round[component] = wp[component].round;
          reference.log2WeightDenom[component] = wp[component].log2WeightDenom;
        }
      }
      fused.referenceMetadata.push_back(reference);
      fused.predictionOperations++;
      if (reference.rpr)
      {
        fused.rpr = true;
        fused.rprPredictionOperations++;
      }
    }
    if (bcw)
    {
      FusedBcwMetadataKey bcwKey;
      bcwKey.bcwIdx = cu.bcwIdx;
      bcwKey.weightList0 = getBcwWeight(cu.bcwIdx, REF_PIC_LIST_0);
      bcwKey.weightList1 = getBcwWeight(cu.bcwIdx, REF_PIC_LIST_1);
      fused.bcwMetadata.push_back(bcwKey);
    }
  }

  if (cuPath < 0)
  {
    m_mcProfile->prepareCu(cu.slice->getPOC(), pixels, McProfileReason::MIXED_EFFECTIVE_PATH, false);
    beginFused(FusedReason::SUB_PU);
    return -1;
  }
  m_mcProfile->prepareCu(cu.slice->getPOC(), pixels,
                         mixedEffectivePath ? McProfileReason::MIXED_EFFECTIVE_PATH : McProfileReason::NUM, false);
  beginFused(lmcsLuma ? FusedReason::LMCS_LUMA : FusedReason::NUM);
  return mixedEffectivePath ? -1 : cuPath;
}

void DecCu::xProfileQueueMcCu(CodingUnit &cu, const int path)
{
  if (!m_mcProfile || !cu.Y().valid()) return;
  const uint64_t pixels = static_cast<uint64_t>(cu.Y().width) * static_cast<uint64_t>(cu.Y().height);
  m_mcProfile->queueMc(static_cast<McProfilePath>(path), pixels);
}

void DecCu::xProfileLmcsChromaAdj()
{
  if (m_mcProfile) m_mcProfile->lmcsChromaAdjDependency();
}

void DecCu::xProfileIbcBufferReset()
{
  if (m_mcProfile) m_mcProfile->ibcBufferBoundary(McProfileReason::IBC_BUFFER_RESET);
}

void DecCu::xProfileIbcVpduReset()
{
  if (m_mcProfile) m_mcProfile->ibcBufferBoundary(McProfileReason::IBC_VPDU_RESET);
}

void DecCu::xProfileIbcPreMvConsumer()
{
  if (m_mcProfile) m_mcProfile->ibcBufferBoundary(McProfileReason::IBC_PRE_MV_CONSUMER);
}

void DecCu::xProfileIbcFill(CodingUnit &cu, const bool queued)
{
  if (!m_mcProfile) return;
  m_mcProfile->recordPendingIbcFill(mcProfileIbcFillObservable(cu.slice->getSPS()->getIBCFlag()), queued);
}

void DecCu::xProfileFinishFusedCu()
{
  if (m_mcProfile) m_mcProfile->finishFusedCu();
}

void DecCu::xProfileTransformBlock(TransformUnit &tu, const ComponentID compID)
{
  if (!m_mcProfile || CU::isIBC(*tu.cu) || !tu.blocks[compID].valid()) return;
  const CompArea &area = tu.blocks[compID];
  m_mcProfile->recordTransformBlock(compID, transformSizeClass(area.width), transformSizeClass(area.height),
                                    static_cast<uint64_t>(area.width) * static_cast<uint64_t>(area.height),
                                    TU::getCbf(tu, compID));
}

void DecCu::xProfileInverseTransform(TransformUnit &tu, const ComponentID compID, const QpParam &qp)
{
  if (!m_mcProfile || CU::isIBC(*tu.cu) || !tu.blocks[compID].valid()) return;

  const CompArea &area = tu.blocks[compID];
  const bool transformSkip = tu.mtsIdx[compID] == MtsType::SKIP;
  const bool mts = tu.mtsIdx[compID] != MtsType::DCT2_DCT2 && !transformSkip;
  TransType horizontal = TransType::DCT2;
  TransType vertical = TransType::DCT2;
  if (!transformSkip) m_pcTrQuant->getTrTypes(tu, compID, horizontal, vertical);
  const bool dct2 = !transformSkip && horizontal == TransType::DCT2 && vertical == TransType::DCT2;
  const bool lfnst = tu.cu->lfnstIdx > 0 && (tu.cu->isSepTree() || isLuma(compID));
  const bool sbt = tu.cu->sbtInfo != 0;
  const bool joint = tu.jointCbCr != 0 && isChroma(compID);
  const bool act = tu.cu->colorTransform;
  Slice &slice = *tu.cs->slice;
  const bool lmcs = !act && slice.getLmcsEnabledFlag() && isChroma(compID)
                 && (TU::getCbf(tu, compID) || joint)
                 && slice.getPicHeader()->getLmcsChromaResidualScaleFlag()
                 && area.width * area.height > 4;

  const bool regularResidualCoding = tu.cu->slice->getTSResidualCodingDisabledFlag() || !transformSkip;
  const bool dependentQuant = slice.getDepQuantEnabledFlag() && regularResidualCoding;
  const bool disableScalingForLfnst = slice.getExplicitScalingListUsed()
                                        ? slice.getSPS()->getDisableScalingMatrixForLfnstBlks()
                                        : false;
  const bool disableScalingForAct = slice.getSPS()->getScalingMatrixForAlternativeColourSpaceDisabledFlag()
                                 && (slice.getSPS()->getScalingMatrixDesignatedColourSpaceFlag() == act);
  const bool scalingList = m_pcTrQuant->getQuant()->getUseScalingList(area.width, area.height, transformSkip, lfnst,
                                                                      disableScalingForLfnst,
                                                                      disableScalingForAct);

  TransformProfileSample sample;
  sample.component = compID;
  sample.widthLog2 = transformSizeClass(area.width);
  sample.heightLog2 = transformSizeClass(area.height);
  sample.pixels = static_cast<uint64_t>(area.width) * static_cast<uint64_t>(area.height);
  sample.coefficients = sample.pixels;
  sample.qp = qp.Qp(transformSkip);
  sample.dequantPath = dependentQuant
                         ? (scalingList ? DequantPath::DEPENDENT_SCALING_LIST : DequantPath::DEPENDENT_FLAT)
                         : (scalingList ? DequantPath::SCALAR_SCALING_LIST : DequantPath::SCALAR_FLAT);
  const CCoeffBuf coefficients = tu.getCoeffs(compID);
  for (unsigned y = 0; y < area.height; ++y)
    for (unsigned x = 0; x < area.width; ++x)
      sample.nonzeroCoefficients += coefficients.at(x, y) != 0;

  sample.features[static_cast<size_t>(TransformFeature::DCT2)] = dct2;
  sample.features[static_cast<size_t>(TransformFeature::MTS)] = mts;
  sample.features[static_cast<size_t>(TransformFeature::TRANSFORM_SKIP)] = transformSkip;
  sample.features[static_cast<size_t>(TransformFeature::LFNST)] = lfnst;
  sample.features[static_cast<size_t>(TransformFeature::SBT)] = sbt;
  sample.features[static_cast<size_t>(TransformFeature::JOINT_CBCR)] = joint;
  sample.features[static_cast<size_t>(TransformFeature::ACT)] = act;
  sample.features[static_cast<size_t>(TransformFeature::LMCS)] = lmcs;
  sample.features[static_cast<size_t>(TransformFeature::REGULAR_DCT2_CANDIDATE)] =
    dct2 && !mts && !transformSkip && !lfnst && !sbt && !joint && !act && !lmcs;
  m_mcProfile->recordInverseTransform(sample);
}
#endif

// ====================================================================================================================
// Protected member functions
// ====================================================================================================================

void DecCu::xIntraRecBlk( TransformUnit& tu, const ComponentID compID )
{
  if( !tu.blocks[ compID ].valid() )
  {
    return;
  }

        CodingStructure &cs = *tu.cs;
  const CompArea &area      = tu.blocks[compID];

  const ChannelType chType  = toChannelType( compID );

        PelBuf piPred       = cs.getPredBuf( area );

  const PredictionUnit &pu  = *tu.cs->getPU( area.pos(), chType );

  const uint32_t chFinalMode = PU::getFinalIntraMode(pu, chType);
  PelBuf         pReco       = cs.getRecoBuf(area);

  //===== init availability pattern =====
  bool predRegDiffFromTB = CU::isPredRegDiffFromTB(*tu.cu, compID);
  bool firstTBInPredReg = CU::isFirstTBInPredReg(*tu.cu, compID, area);
  CompArea areaPredReg(COMPONENT_Y, tu.chromaFormat, area);
  if (tu.cu->ispMode != ISPType::NONE && isLuma(compID))
  {
    if (predRegDiffFromTB)
    {
      if (firstTBInPredReg)
      {
        CU::adjustPredArea(areaPredReg);
        m_pcIntraPred->initIntraPatternChTypeISP(*tu.cu, areaPredReg, pReco);
      }
    }
    else
    {
      m_pcIntraPred->initIntraPatternChTypeISP(*tu.cu, area, pReco);
    }
  }
  else
  {
    m_pcIntraPred->initIntraPatternChType(*tu.cu, area);
  }

  //===== get prediction signal =====
  if (compID != COMPONENT_Y && PU::isLMCMode(chFinalMode))
  {
    const PredictionUnit& pu = *tu.cu->firstPU;
    m_pcIntraPred->xGetLumaRecPixels( pu, area );
    m_pcIntraPred->predIntraChromaLM(compID, piPred, pu, area, chFinalMode);
  }
  else
  {
    if( PU::isMIP( pu, chType ) )
    {
      m_pcIntraPred->initIntraMip( pu, area );
      m_pcIntraPred->predIntraMip( compID, piPred, pu );
    }
    else
    {
      if (predRegDiffFromTB)
      {
        if (firstTBInPredReg)
        {
          PelBuf piPredReg = cs.getPredBuf(areaPredReg);
          m_pcIntraPred->predIntraAng(compID, piPredReg, pu);
        }
      }
      else
      {
        m_pcIntraPred->predIntraAng(compID, piPred, pu);
      }
    }
  }
  const Slice           &slice = *cs.slice;
  bool flag = slice.getLmcsEnabledFlag() && (slice.isIntra() || (!slice.isIntra() && m_pcReshape->getCTUFlag()));
  if (flag && slice.getPicHeader()->getLmcsChromaResidualScaleFlag() && (compID != COMPONENT_Y) && (tu.cbf[COMPONENT_Cb] || tu.cbf[COMPONENT_Cr]))
  {
    const Area      area  = tu.Y().valid()
                              ? tu.Y()
                              : Area(recalcPosition(tu.chromaFormat, tu.chType, ChannelType::LUMA, tu.block(tu.chType).pos()),
                                     recalcSize(tu.chromaFormat, tu.chType, ChannelType::LUMA, tu.block(tu.chType).size()));
    const CompArea &areaY = CompArea(COMPONENT_Y, tu.chromaFormat, area);
    int adj = m_pcReshape->calculateChromaAdjVpduNei(tu, areaY);
    tu.setChromaAdj(adj);
  }
  //===== inverse transform =====
  PelBuf piResi = cs.getResiBuf( area );

  const QpParam cQP( tu, compID );

  if( tu.jointCbCr && isChroma(compID) )
  {
    if( compID == COMPONENT_Cb )
    {
      PelBuf resiCr = cs.getResiBuf( tu.blocks[ COMPONENT_Cr ] );
      if( tu.jointCbCr >> 1 )
      {
        m_pcTrQuant->invTransformNxN( tu, COMPONENT_Cb, piResi, cQP );
      }
      else
      {
        const QpParam qpCr( tu, COMPONENT_Cr );
        m_pcTrQuant->invTransformNxN( tu, COMPONENT_Cr, resiCr, qpCr );
      }
      m_pcTrQuant->invTransformICT( tu, piResi, resiCr );
    }
  }
  else if (TU::getCbf(tu, compID))
  {
    m_pcTrQuant->invTransformNxN( tu, compID, piResi, cQP );
  }
  else
  {
    piResi.fill( 0 );
  }

  //===== reconstruction =====
  flag = flag && (tu.blocks[compID].width*tu.blocks[compID].height > 4);
  if (flag && (TU::getCbf(tu, compID) || tu.jointCbCr) && isChroma(compID) && slice.getPicHeader()->getLmcsChromaResidualScaleFlag())
  {
    piResi.scaleSignal(tu.getChromaAdj(), 0, tu.cu->cs->slice->clpRng(compID));
  }

  if (tu.cu->ispMode == ISPType::NONE || !isLuma(compID))
  {
    cs.setDecomp( area );
  }
  else if (tu.cu->ispMode != ISPType::NONE && isLuma(compID) && CU::isISPFirst(*tu.cu, tu.blocks[compID], compID))
  {
    cs.setDecomp( tu.cu->blocks[compID] );
  }

#if REUSE_CU_RESULTS
  CompArea    tmpArea(COMPONENT_Y, area.chromaFormat, Position(0, 0), area.size());
  PelBuf tmpPred;
#endif
  if (slice.getLmcsEnabledFlag() && (m_pcReshape->getCTUFlag() || slice.isIntra()) && compID == COMPONENT_Y)
  {
#if REUSE_CU_RESULTS
    tmpPred = m_tmpStorageCtu->getBuf(tmpArea);
    tmpPred.copyFrom(piPred);
#endif
  }
#if KEEP_PRED_AND_RESI_SIGNALS
  pReco.reconstruct( piPred, piResi, tu.cu->cs->slice->clpRng( compID ) );
#else
  piPred.reconstruct(piPred, piResi, tu.cu->cs->slice->clpRng(compID));
  pReco.copyFrom( piPred );
#endif
  if (slice.getLmcsEnabledFlag() && (m_pcReshape->getCTUFlag() || slice.isIntra()) && compID == COMPONENT_Y)
  {
#if REUSE_CU_RESULTS
    piPred.copyFrom(tmpPred);
#endif
  }
#if REUSE_CU_RESULTS
  if( cs.pcv->isEncoder )
  {
    cs.picture->getRecoBuf( area ).copyFrom( pReco );
    cs.picture->getPredBuf(area).copyFrom(piPred);
  }
#endif
}

void DecCu::xIntraRecACTBlk(TransformUnit& tu)
{
  CodingStructure      &cs = *tu.cs;
  const PredictionUnit &pu    = *tu.cs->getPU(tu.blocks[COMPONENT_Y], ChannelType::LUMA);
  const Slice          &slice = *cs.slice;

  CHECK(!tu.Y().valid() || !tu.Cb().valid() || !tu.Cr().valid(), "Invalid TU");
  CHECK(&pu != tu.cu->firstPU, "wrong PU fetch");
  CHECK(tu.cu->ispMode != ISPType::NONE, "adaptive color transform cannot be applied to ISP");
  CHECK(pu.intraDir[ChannelType::CHROMA] != DM_CHROMA_IDX, "chroma should use DM mode for adaptive color transform");

  bool flag = slice.getLmcsEnabledFlag() && (slice.isIntra() || (!slice.isIntra() && m_pcReshape->getCTUFlag()));
  if (flag && slice.getPicHeader()->getLmcsChromaResidualScaleFlag())
  {
    const Area      area  = tu.Y().valid()
                              ? tu.Y()
                              : Area(recalcPosition(tu.chromaFormat, tu.chType, ChannelType::LUMA, tu.block(tu.chType).pos()),
                                     recalcSize(tu.chromaFormat, tu.chType, ChannelType::LUMA, tu.block(tu.chType).size()));
    const CompArea &areaY = CompArea(COMPONENT_Y, tu.chromaFormat, area);
    int            adj = m_pcReshape->calculateChromaAdjVpduNei(tu, areaY);
    tu.setChromaAdj(adj);
  }

  for (int i = 0; i < getNumberValidComponents(tu.chromaFormat); i++)
  {
    ComponentID          compID = (ComponentID)i;
    const CompArea       &area = tu.blocks[compID];
    const ChannelType    chType = toChannelType(compID);

    PelBuf piPred = cs.getPredBuf(area);
    m_pcIntraPred->initIntraPatternChType(*tu.cu, area);
    if (PU::isMIP(pu, chType))
    {
      m_pcIntraPred->initIntraMip(pu, area);
      m_pcIntraPred->predIntraMip(compID, piPred, pu);
    }
    else
    {
      m_pcIntraPred->predIntraAng(compID, piPred, pu);
    }

    PelBuf piResi = cs.getResiBuf(area);

    QpParam cQP(tu, compID);

    if (tu.jointCbCr && isChroma(compID))
    {
      if (compID == COMPONENT_Cb)
      {
        PelBuf resiCr = cs.getResiBuf(tu.blocks[COMPONENT_Cr]);
        if (tu.jointCbCr >> 1)
        {
          m_pcTrQuant->invTransformNxN(tu, COMPONENT_Cb, piResi, cQP);
        }
        else
        {
          QpParam qpCr(tu, COMPONENT_Cr);

          m_pcTrQuant->invTransformNxN(tu, COMPONENT_Cr, resiCr, qpCr);
        }
        m_pcTrQuant->invTransformICT(tu, piResi, resiCr);
      }
    }
    else
    {
      if (TU::getCbf(tu, compID))
      {
        m_pcTrQuant->invTransformNxN(tu, compID, piResi, cQP);
      }
      else
      {
        piResi.fill(0);
      }
    }

    cs.setDecomp(area);
  }

  cs.getResiBuf(tu).colorSpaceConvert(cs.getResiBuf(tu), false, tu.cu->cs->slice->clpRng(COMPONENT_Y));

  for (int i = 0; i < getNumberValidComponents(tu.chromaFormat); i++)
  {
    ComponentID          compID = (ComponentID)i;
    const CompArea       &area = tu.blocks[compID];

    PelBuf piPred = cs.getPredBuf(area);
    PelBuf piResi = cs.getResiBuf(area);
    PelBuf piReco = cs.getRecoBuf(area);

    PelBuf tmpPred;
    if (slice.getLmcsEnabledFlag() && (m_pcReshape->getCTUFlag() || slice.isIntra()) && compID == COMPONENT_Y)
    {
      CompArea tmpArea(COMPONENT_Y, area.chromaFormat, Position(0, 0), area.size());
      tmpPred = m_tmpStorageCtu->getBuf(tmpArea);
      tmpPred.copyFrom(piPred);
    }

    if (flag && isChroma(compID) && (tu.blocks[compID].width*tu.blocks[compID].height > 4) && slice.getPicHeader()->getLmcsChromaResidualScaleFlag())
    {
      piResi.scaleSignal(tu.getChromaAdj(), 0, tu.cu->cs->slice->clpRng(compID));
    }
    piPred.reconstruct(piPred, piResi, tu.cu->cs->slice->clpRng(compID));
    piReco.copyFrom(piPred);

    if (slice.getLmcsEnabledFlag() && (m_pcReshape->getCTUFlag() || slice.isIntra()) && compID == COMPONENT_Y)
    {
      piPred.copyFrom(tmpPred);
    }

    if (cs.pcv->isEncoder)
    {
      cs.picture->getRecoBuf(area).copyFrom(piReco);
      cs.picture->getPredBuf(area).copyFrom(piPred);
    }
  }
}

void DecCu::xReconIntraQT( CodingUnit &cu )
{
  if (CU::isPLT(cu))
  {
    if (cu.isSepTree())
    {
      if (isLuma(cu.chType))
      {
        xReconPLT(cu, COMPONENT_Y, 1);
      }
      if (isChromaEnabled(cu.chromaFormat) && cu.chType == ChannelType::CHROMA)
      {
        xReconPLT(cu, COMPONENT_Cb, 2);
      }
    }
    else
    {
      xReconPLT(cu, COMPONENT_Y, getNumberValidComponents(cu.chromaFormat));
    }
    return;
  }

  if (cu.colorTransform)
  {
    xIntraRecACTQT(cu);
  }
  else
  {
    for (auto chType = ChannelType::LUMA; chType <= ::getLastChannel(cu.chromaFormat); chType++)
    {
      if (cu.block(chType).valid())
      {
        xIntraRecQT(cu, chType);
      }
    }
  }
}

void DecCu::xReconPLT(CodingUnit &cu, ComponentID compBegin, uint32_t numComp)
{
  const SPS&       sps = *(cu.cs->sps);
  TransformUnit&   tu = *cu.firstTU;
  PelBuf    curPLTIdx = tu.getcurPLTIdx(compBegin);

  uint32_t height = cu.block(compBegin).height;
  uint32_t width = cu.block(compBegin).width;

  //recon. pixels
  uint32_t scaleX = getComponentScaleX(COMPONENT_Cb, sps.getChromaFormatIdc());
  uint32_t scaleY = getComponentScaleY(COMPONENT_Cb, sps.getChromaFormatIdc());
  for (uint32_t y = 0; y < height; y++)
  {
    for (uint32_t x = 0; x < width; x++)
    {
      for (uint32_t compID = compBegin; compID < (compBegin + numComp); compID++)
      {
        const int  channelBitDepth = cu.cs->sps->getBitDepth(toChannelType((ComponentID)compID));
        const CompArea &area = cu.blocks[compID];

        PelBuf       picReco   = cu.cs->getRecoBuf(area);
        PLTescapeBuf escapeValue = tu.getescapeValue((ComponentID)compID);
        if (curPLTIdx.at(x, y) == cu.curPLTSize[compBegin])
        {
          TCoeff value;
          QpParam cQP(tu, (ComponentID)compID);
          int qp = cQP.Qp(true);
          int qpRem = qp % 6;
          int qpPer = qp / 6;
          if (compBegin != COMPONENT_Y || compID == COMPONENT_Y)
          {
            int invquantiserRightShift = IQUANT_SHIFT;
            int add = 1 << (invquantiserRightShift - 1);
            value = ((((escapeValue.at(x, y)*g_invQuantScales[0][qpRem]) << qpPer) + add) >> invquantiserRightShift);
            value = ClipBD<TCoeff>(value, channelBitDepth);
            picReco.at(x, y) = Pel(value);
          }
          else if (compBegin == COMPONENT_Y && compID != COMPONENT_Y && y % (1 << scaleY) == 0 && x % (1 << scaleX) == 0)
          {
            uint32_t posYC = y >> scaleY;
            uint32_t posXC = x >> scaleX;
            int invquantiserRightShift = IQUANT_SHIFT;
            int add = 1 << (invquantiserRightShift - 1);
            value = ((((escapeValue.at(posXC, posYC)*g_invQuantScales[0][qpRem]) << qpPer) + add) >> invquantiserRightShift);
            value = ClipBD<TCoeff>(value, channelBitDepth);
            picReco.at(posXC, posYC) = Pel(value);
          }
        }
        else
        {
          uint32_t curIdx = curPLTIdx.at(x, y);
          if (compBegin != COMPONENT_Y || compID == COMPONENT_Y)
          {
            picReco.at(x, y) = cu.curPLT[compID][curIdx];
          }
          else if (compBegin == COMPONENT_Y && compID != COMPONENT_Y && y % (1 << scaleY) == 0 && x % (1 << scaleX) == 0)
          {
            uint32_t posYC = y >> scaleY;
            uint32_t posXC = x >> scaleX;
            picReco.at(posXC, posYC) = cu.curPLT[compID][curIdx];
          }
        }
      }
    }
  }
  for (uint32_t compID = compBegin; compID < (compBegin + numComp); compID++)
  {
    const CompArea &area = cu.blocks[compID];
    PelBuf picReco = cu.cs->getRecoBuf(area);
    cu.cs->picture->getRecoBuf(area).copyFrom(picReco);
    cu.cs->setDecomp(area);
  }
}

/** Function for deriving reconstructed PU/CU chroma samples with QTree structure
* \param pcRecoYuv pointer to reconstructed sample arrays
* \param pcPredYuv pointer to prediction sample arrays
* \param pcResiYuv pointer to residue sample arrays
* \param chType    texture channel type (luma/chroma)
* \param rTu       reference to transform data
*
\ This function derives reconstructed PU/CU chroma samples with QTree recursive structure
*/

void DecCu::xIntraRecQT(CodingUnit &cu, const ChannelType chType)
{
  for( auto &currTU : CU::traverseTUs( cu ) )
  {
    if( isLuma( chType ) )
    {
      xIntraRecBlk( currTU, COMPONENT_Y );
    }
    else
    {
      const uint32_t numValidComp = getNumberValidComponents( cu.chromaFormat );

      for( uint32_t compID = COMPONENT_Cb; compID < numValidComp; compID++ )
      {
        xIntraRecBlk( currTU, ComponentID( compID ) );
      }
    }
  }
}

void DecCu::xIntraRecACTQT(CodingUnit &cu)
{
  for (auto &currTU : CU::traverseTUs(cu))
  {
    xIntraRecACTBlk(currTU);
  }
}

#include "CommonLib/dtrace_buffer.h"

void DecCu::xReconInter(CodingUnit &cu)
{
  if( cu.geoFlag )
  {
    m_pcInterPred->motionCompensationGeo( cu, m_geoMrgCtx );
    PU::spanGeoMotionInfo(*cu.firstPU, m_geoMrgCtx, cu.firstPU->geoSplitDir, cu.firstPU->geoMergeIdx);
  }
  else
  {
    m_pcIntraPred->geneIntrainterPred(cu);

    // inter prediction
    CHECK(CU::isIBC(cu) && cu.firstPU->ciipFlag, "IBC and Ciip cannot be used together");
    CHECK(CU::isIBC(cu) && cu.affine, "IBC and Affine cannot be used together");
    CHECK(CU::isIBC(cu) && cu.geoFlag, "IBC and geo cannot be used together");
    CHECK(CU::isIBC(cu) && cu.firstPU->mmvdMergeFlag, "IBC and MMVD cannot be used together");
    const bool luma   = cu.Y().valid();
    const bool chroma = isChromaEnabled(cu.chromaFormat) && cu.Cb().valid();
    if (luma && (chroma || !isChromaEnabled(cu.chromaFormat)))
    {
      m_pcInterPred->motionCompensateCu(cu, REF_PIC_LIST_X, true, true);
    }
    else
    {
      m_pcInterPred->motionCompensateCu(cu, REF_PIC_LIST_0, luma, chroma);
    }
  }
  if (cu.Y().valid())
  {
    CU::saveMotionForHmvp(cu);
  }

  if (cu.firstPU->ciipFlag)
  {
    if (cu.cs->slice->getLmcsEnabledFlag() && m_pcReshape->getCTUFlag())
    {
      cu.cs->getPredBuf(*cu.firstPU).Y().rspSignal(m_pcReshape->getFwdLUT());
    }
    m_pcIntraPred->geneWeightedPred(cu.cs->getPredBuf(*cu.firstPU).Y(), *cu.firstPU,
                                    m_pcIntraPred->getPredictorPtr2(COMPONENT_Y, 0));
    if (isChromaEnabled(cu.chromaFormat) && cu.chromaSize().width > 2)
    {
      m_pcIntraPred->geneWeightedPred(cu.cs->getPredBuf(*cu.firstPU).Cb(), *cu.firstPU,
                                      m_pcIntraPred->getPredictorPtr2(COMPONENT_Cb, 0));
      m_pcIntraPred->geneWeightedPred(cu.cs->getPredBuf(*cu.firstPU).Cr(), *cu.firstPU,
                                      m_pcIntraPred->getPredictorPtr2(COMPONENT_Cr, 0));
    }
  }

  DTRACE    ( g_trace_ctx, D_TMP, "pred " );
  DTRACE_CRC( g_trace_ctx, D_TMP, *cu.cs, cu.cs->getPredBuf( cu ), &cu.Y() );

  // inter recon
  xDecodeInterTexture(cu);

#if VTM_ENABLE_DECODER_BATCH_PROFILING
  if (m_mcProfile && cu.Y().valid() && !CU::isIBC(cu))
  {
    m_mcProfile->recordReconstruction(static_cast<uint64_t>(cu.Y().width) * static_cast<uint64_t>(cu.Y().height));
  }
#endif

  // clip for only non-zero cbf case
  CodingStructure &cs = *cu.cs;

  if (cu.rootCbf)
  {
#if REUSE_CU_RESULTS
    const CompArea &area = cu.blocks[COMPONENT_Y];
    CompArea    tmpArea(COMPONENT_Y, area.chromaFormat, Position(0, 0), area.size());
    PelBuf tmpPred;
#endif
    if (cs.slice->getLmcsEnabledFlag() && m_pcReshape->getCTUFlag())
    {
#if REUSE_CU_RESULTS
      if (cs.pcv->isEncoder)
      {
        tmpPred = m_tmpStorageCtu->getBuf(tmpArea);
        tmpPred.copyFrom(cs.getPredBuf(cu).get(COMPONENT_Y));
      }
#endif
      if (!cu.firstPU->ciipFlag && !CU::isIBC(cu))
      {
        cs.getPredBuf(cu).get(COMPONENT_Y).rspSignal(m_pcReshape->getFwdLUT());
      }
    }
#if KEEP_PRED_AND_RESI_SIGNALS
    cs.getRecoBuf( cu ).reconstruct( cs.getPredBuf( cu ), cs.getResiBuf( cu ), cs.slice->clpRngs() );
#else
    cs.getResiBuf( cu ).reconstruct( cs.getPredBuf( cu ), cs.getResiBuf( cu ), cs.slice->clpRngs() );
    cs.getRecoBuf( cu ).copyFrom   (                      cs.getResiBuf( cu ) );
#endif
    if (cs.slice->getLmcsEnabledFlag() && m_pcReshape->getCTUFlag())
    {
#if REUSE_CU_RESULTS
      if (cs.pcv->isEncoder)
      {
        cs.getPredBuf(cu).get(COMPONENT_Y).copyFrom(tmpPred);
      }
#endif
    }
  }
  else
  {
    cs.getRecoBuf(cu).copyClip(cs.getPredBuf(cu), cs.slice->clpRngs());
    if (cs.slice->getLmcsEnabledFlag() && m_pcReshape->getCTUFlag() && !cu.firstPU->ciipFlag && !CU::isIBC(cu))
    {
      cs.getRecoBuf(cu).get(COMPONENT_Y).rspSignal(m_pcReshape->getFwdLUT());
    }
  }

  DTRACE    ( g_trace_ctx, D_TMP, "reco " );
  DTRACE_CRC( g_trace_ctx, D_TMP, *cu.cs, cu.cs->getRecoBuf( cu ), &cu.Y() );

  cs.setDecomp(cu);
}

void DecCu::xDecodeInterTU( TransformUnit & currTU, const ComponentID compID )
{
  if (!currTU.blocks[compID].valid())
  {
    return;
  }

  const CompArea &area = currTU.blocks[compID];

  CodingStructure& cs = *currTU.cs;

  //===== inverse transform =====
  PelBuf resiBuf  = cs.getResiBuf(area);

  QpParam cQP(currTU, compID);
#if VTM_ENABLE_DECODER_BATCH_PROFILING
  xProfileTransformBlock(currTU, compID);
#endif

  if( currTU.jointCbCr && isChroma(compID) )
  {
    if( compID == COMPONENT_Cb )
    {
      PelBuf resiCr = cs.getResiBuf( currTU.blocks[ COMPONENT_Cr ] );
      if( currTU.jointCbCr >> 1 )
      {
#if VTM_ENABLE_DECODER_BATCH_PROFILING
        xProfileInverseTransform(currTU, COMPONENT_Cb, cQP);
#endif
        m_pcTrQuant->invTransformNxN( currTU, COMPONENT_Cb, resiBuf, cQP );
      }
      else
      {
        QpParam qpCr(currTU, COMPONENT_Cr);
#if VTM_ENABLE_DECODER_BATCH_PROFILING
        xProfileInverseTransform(currTU, COMPONENT_Cr, qpCr);
#endif
        m_pcTrQuant->invTransformNxN( currTU, COMPONENT_Cr, resiCr, qpCr );
      }
      m_pcTrQuant->invTransformICT( currTU, resiBuf, resiCr );
    }
  }
  else if (TU::getCbf(currTU, compID))
  {
#if VTM_ENABLE_DECODER_BATCH_PROFILING
    xProfileInverseTransform(currTU, compID, cQP);
#endif
    m_pcTrQuant->invTransformNxN( currTU, compID, resiBuf, cQP );
  }
  else
  {
    resiBuf.fill( 0 );
  }

  //===== reconstruction =====
  const Slice           &slice = *cs.slice;
  if (!currTU.cu->colorTransform && slice.getLmcsEnabledFlag() && isChroma(compID) && (TU::getCbf(currTU, compID) || currTU.jointCbCr)
   && slice.getPicHeader()->getLmcsChromaResidualScaleFlag() && currTU.blocks[compID].width * currTU.blocks[compID].height > 4)
  {
    resiBuf.scaleSignal(currTU.getChromaAdj(), 0, currTU.cu->cs->slice->clpRng(compID));
  }
}

void DecCu::xDecodeInterTexture(CodingUnit &cu)
{
  if( !cu.rootCbf )
  {
    return;
  }

  const uint32_t uiNumVaildComp = getNumberValidComponents(cu.chromaFormat);

  if (cu.colorTransform)
  {
    CodingStructure  &cs = *cu.cs;
    const Slice &slice = *cs.slice;
    for (auto& currTU : CU::traverseTUs(cu))
    {
      for (uint32_t ch = 0; ch < uiNumVaildComp; ch++)
      {
        const ComponentID compID = ComponentID(ch);
        if (slice.getLmcsEnabledFlag() && slice.getPicHeader()->getLmcsChromaResidualScaleFlag() && (compID == COMPONENT_Y))
        {
          const CompArea &areaY = currTU.blocks[COMPONENT_Y];
#if VTM_ENABLE_DECODER_BATCH_PROFILING
          if (m_pcReshape->chromaAdjVpduReadsLuma(currTU, areaY)) xProfileLmcsChromaAdj();
#endif
          int adj = m_pcReshape->calculateChromaAdjVpduNei(currTU, areaY);
          currTU.setChromaAdj(adj);
        }
        xDecodeInterTU(currTU, compID);
      }

      cs.getResiBuf(currTU).colorSpaceConvert(cs.getResiBuf(currTU), false, cu.cs->slice->clpRng(COMPONENT_Y));
      if (slice.getLmcsEnabledFlag() && slice.getPicHeader()->getLmcsChromaResidualScaleFlag() && currTU.blocks[COMPONENT_Cb].width * currTU.blocks[COMPONENT_Cb].height > 4)
      {
        cs.getResiBuf(currTU.blocks[COMPONENT_Cb]).scaleSignal(currTU.getChromaAdj(), 0, currTU.cu->cs->slice->clpRng(COMPONENT_Cb));
        cs.getResiBuf(currTU.blocks[COMPONENT_Cr]).scaleSignal(currTU.getChromaAdj(), 0, currTU.cu->cs->slice->clpRng(COMPONENT_Cr));
      }
    }
  }
  else
  {
    for (uint32_t ch = 0; ch < uiNumVaildComp; ch++)
    {
      const ComponentID compID = ComponentID(ch);

      for (auto &currTU: CU::traverseTUs(cu))
      {
        CodingStructure &cs    = *cu.cs;
        const Slice     &slice = *cs.slice;
        if (slice.getLmcsEnabledFlag() && slice.getPicHeader()->getLmcsChromaResidualScaleFlag()
            && (compID == COMPONENT_Y) && (currTU.cbf[COMPONENT_Cb] || currTU.cbf[COMPONENT_Cr]))
        {
          const CompArea &areaY = currTU.blocks[COMPONENT_Y];
#if VTM_ENABLE_DECODER_BATCH_PROFILING
          if (m_pcReshape->chromaAdjVpduReadsLuma(currTU, areaY)) xProfileLmcsChromaAdj();
#endif
          int             adj   = m_pcReshape->calculateChromaAdjVpduNei(currTU, areaY);
          currTU.setChromaAdj(adj);
        }
        xDecodeInterTU(currTU, compID);
      }
    }
  }
}

void DecCu::xDeriveCuMvs(CodingUnit &cu)
{
  for( auto &pu : CU::traversePUs( cu ) )
  {
    MergeCtx mrgCtx;

#if RExt__DECODER_DEBUG_TOOL_STATISTICS
    if( pu.cu->affine )
    {
      CodingStatistics::IncrementStatisticTool( CodingStatisticsClassType{ STATS__TOOL_AFF, pu.Y().width, pu.Y().height } );
    }
#endif

    if( pu.mergeFlag )
    {
      if (pu.mmvdMergeFlag || pu.cu->mmvdSkip)
      {
        CHECK(pu.ciipFlag, "invalid Ciip");
        if (pu.cs->sps->getSbTMVPEnabledFlag())
        {
          Size bufSize = g_miScaling.scale(pu.lumaSize());
          mrgCtx.subPuMvpMiBuf = MotionBuf(m_SubPuMiBuf, bufSize);
        }

        PU::getInterMergeCandidates(pu, mrgCtx, 1, pu.mmvdMergeIdx.pos.baseIdx + 1);
        PU::getInterMMVDMergeCandidates(pu, mrgCtx);
        mrgCtx.setMmvdMergeCandiInfo(pu, pu.mmvdMergeIdx);

        PU::spanMotionInfo(pu, mrgCtx);
      }
      else
      {
        if( pu.cu->geoFlag )
        {
          PU::getGeoMergeCandidates( pu, m_geoMrgCtx );
        }
        else if (pu.cu->affine)
        {
          AffineMergeCtx affineMergeCtx;
          if (pu.cs->sps->getSbTMVPEnabledFlag())
          {
            Size bufSize          = g_miScaling.scale(pu.lumaSize());
            mrgCtx.subPuMvpMiBuf  = MotionBuf(m_SubPuMiBuf, bufSize);
            affineMergeCtx.mrgCtx = &mrgCtx;
          }
          PU::getAffineMergeCand(pu, affineMergeCtx, pu.mergeIdx);
          pu.interDir       = affineMergeCtx.interDirNeighbours[pu.mergeIdx];
          pu.cu->affineType = affineMergeCtx.affineType[pu.mergeIdx];
          pu.cu->bcwIdx     = affineMergeCtx.bcwIdx[pu.mergeIdx];
          pu.mergeType      = affineMergeCtx.mergeType[pu.mergeIdx];
          if (pu.mergeType == MergeType::SUBPU_ATMVP)
          {
            pu.refIdx[0] = affineMergeCtx.mvFieldNeighbours[pu.mergeIdx][0][0].refIdx;
            pu.refIdx[1] = affineMergeCtx.mvFieldNeighbours[pu.mergeIdx][0][1].refIdx;
          }
          else
          {
            for (const auto l: { REF_PIC_LIST_0, REF_PIC_LIST_1 })
            {
              if (pu.cs->slice->getNumRefIdx(l) > 0)
              {
                auto &mvField = affineMergeCtx.mvFieldNeighbours[pu.mergeIdx];
                pu.mvpIdx[l]  = 0;
                pu.mvpNum[l]  = 0;
                pu.mvd[l]     = Mv();
                PU::setAllAffineMvField(pu, mvField, l);
              }
            }
          }
          PU::spanMotionInfo(pu, mrgCtx);
        }
        else
        {
          if (CU::isIBC(*pu.cu))
          {
            PU::getIBCMergeCandidates(pu, mrgCtx, pu.mergeIdx);
          }
          else
          {
            PU::getInterMergeCandidates(pu, mrgCtx, 0, pu.mergeIdx);
          }
          mrgCtx.setMergeInfo(pu, pu.mergeIdx);

          PU::spanMotionInfo(pu, mrgCtx);
        }
      }
    }
    else
    {
#if REUSE_CU_RESULTS
      if ( cu.imv && !pu.cu->affine && !cu.cs->pcv->isEncoder )
#else
      if (cu.imv && !pu.cu->affine)
#endif
      {
        PU::applyImv(pu, mrgCtx, m_pcInterPred);
      }
      else
      {
        if( pu.cu->affine )
        {
          for ( uint32_t uiRefListIdx = 0; uiRefListIdx < 2; uiRefListIdx++ )
          {
            RefPicList eRefList = RefPicList( uiRefListIdx );
            if ( pu.cs->slice->getNumRefIdx( eRefList ) > 0 && ( pu.interDir & ( 1 << uiRefListIdx ) ) )
            {
              AffineAMVPInfo affineAMVPInfo;
              PU::fillAffineMvpCand( pu, eRefList, pu.refIdx[eRefList], affineAMVPInfo );

              const unsigned mvpIdx = pu.mvpIdx[eRefList];

              pu.mvpNum[eRefList] = affineAMVPInfo.numCand;

              //    Mv mv[3];
              CHECK( pu.refIdx[eRefList] < 0, "Unexpected negative refIdx." );
              if (!cu.cs->pcv->isEncoder)
              {
                for (int i = 0; i < cu.getNumAffineMvs(); i++)
                {
                  pu.mvdAffi[eRefList][i].changeAffinePrecAmvr2Internal(pu.cu->imv);
                }
              }

              Mv mvLT = affineAMVPInfo.mvCandLT[mvpIdx] + pu.mvdAffi[eRefList][0];
              Mv mvRT = affineAMVPInfo.mvCandRT[mvpIdx] + pu.mvdAffi[eRefList][1];
              mvRT += pu.mvdAffi[eRefList][0];

              Mv mvLB;
              if (cu.affineType == AffineModel::_6_PARAMS)
              {
                mvLB = affineAMVPInfo.mvCandLB[mvpIdx] + pu.mvdAffi[eRefList][2];
                mvLB += pu.mvdAffi[eRefList][0];
              }
              PU::setAllAffineMv(pu, mvLT, mvRT, mvLB, eRefList, true);
            }
          }
        }
        else if (CU::isIBC(*pu.cu) && pu.interDir == 1)
        {
          AMVPInfo amvpInfo;
          PU::fillIBCMvpCand(pu, amvpInfo);
          pu.mvpNum[REF_PIC_LIST_0] = amvpInfo.numCand;
          Mv mvd = pu.mvd[REF_PIC_LIST_0];
#if REUSE_CU_RESULTS
          if (!cu.cs->pcv->isEncoder)
#endif
          {
            mvd.changeIbcPrecAmvr2Internal(pu.cu->imv);
          }
          if (pu.cs->sps->getMaxNumIBCMergeCand() == 1)
          {
            CHECK( pu.mvpIdx[REF_PIC_LIST_0], "mvpIdx for IBC mode should be 0" );
          }
          pu.mv[REF_PIC_LIST_0] = amvpInfo.mvCand[pu.mvpIdx[REF_PIC_LIST_0]] + mvd;
          pu.mv[REF_PIC_LIST_0].foldToStorageBitDepth();
        }
        else
        {
          for ( uint32_t uiRefListIdx = 0; uiRefListIdx < 2; uiRefListIdx++ )
          {
            RefPicList eRefList = RefPicList( uiRefListIdx );
            if ((pu.cs->slice->getNumRefIdx(eRefList) > 0 || (eRefList == REF_PIC_LIST_0 && CU::isIBC(*pu.cu))) && (pu.interDir & (1 << uiRefListIdx)))
            {
              AMVPInfo amvpInfo;
              PU::fillMvpCand(pu, eRefList, pu.refIdx[eRefList], amvpInfo);
              pu.mvpNum [eRefList] = amvpInfo.numCand;
              if (!cu.cs->pcv->isEncoder)
              {
                pu.mvd[eRefList].changeTransPrecAmvr2Internal(pu.cu->imv);
              }
              pu.mv[eRefList] = amvpInfo.mvCand[pu.mvpIdx[eRefList]] + pu.mvd[eRefList];
              pu.mv[eRefList].foldToStorageBitDepth();
            }
          }
        }
        PU::spanMotionInfo( pu, mrgCtx );
      }
    }
    if( !cu.geoFlag )
    {
      if( g_mctsDecCheckEnabled && !MCTSHelper::checkMvBufferForMCTSConstraint( pu, true ) )
      {
        printf( "DECODER: pu motion vector across tile boundaries (%d,%d,%d,%d)\n", pu.lx(), pu.ly(), pu.lwidth(), pu.lheight() );
      }
    }
    if (CU::isIBC(cu))
    {
      const int cuPelX = pu.Y().x;
      const int cuPelY = pu.Y().y;
      int roiWidth = pu.lwidth();
      int roiHeight = pu.lheight();
      const unsigned int  lcuWidth = pu.cs->slice->getSPS()->getMaxCUWidth();
      int xPred = pu.mv[0].getHor() >> MV_FRACTIONAL_BITS_INTERNAL;
      int yPred = pu.mv[0].getVer() >> MV_FRACTIONAL_BITS_INTERNAL;
      CHECK(!m_pcInterPred->isLumaBvValid(lcuWidth, cuPelX, cuPelY, roiWidth, roiHeight, xPred, yPred), "invalid block vector for IBC detected.");
    }
  }
}
//! \}
