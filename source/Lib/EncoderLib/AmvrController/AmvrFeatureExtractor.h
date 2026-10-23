/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2024, ITU/ISO/IEC
 * All rights reserved.
 */

#pragma once

#include "AmvrDataTypes.h"
#include "CommonLib/Unit.h"
#include "CommonLib/CodingStructure.h"
#include "CommonLib/Slice.h"
#include "EncCfg.h"
#include "AmvrMemoryTracker.h"

class EncCu;
class InterSearch;

class AmvrFeatureExtractor
{
public:
  static void extractFeatures(
    float outFeatures[AMVR_NUM_FEATURES],
    const CodingUnit& cu,
    const CodingStructure& tempCS,
    const CodingStructure& bestCS,
    const Partitioner& partitioner,
    EncCfg& encCfg,
    const AmvrMemoryTracker& memoryTracker,
    double bestPrelimCost,
    InterSearch* interSearch
  );

  static float calculateLumaVariance(const CPelBuf& origBuf);
};
