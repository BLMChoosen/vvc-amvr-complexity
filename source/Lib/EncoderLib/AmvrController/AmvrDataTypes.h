/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2024, ITU/ISO/IEC
 * All rights reserved.
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <limits>

static constexpr size_t AMVR_NUM_FEATURES = 50;
static constexpr const char* AMVR_DATASET_MAGIC = "AMVR";
static constexpr uint32_t AMVR_DATASET_VERSION = 1;

#pragma pack(push, 1)
struct AmvrDatasetHeader
{
  char     magic[4];          // "AMVR"
  uint32_t version;           // 1
  uint32_t numFeatures;       // 50
  uint32_t sampleSizeBytes;   // sizeof(AmvrDatasetSample)
};

struct AmvrDatasetSample
{
  float    features[AMVR_NUM_FEATURES];
  int32_t  target_best_imv;           // 0: 1/4-pel, 1: 1/2-pel, 2: 1-pel, 3: 4-pel
  int32_t  target_best_affine_amvr;   // 0: 1/4-pel, 1: 1/16-pel, 2: 1-pel
  int32_t  target_imv_combination_id; // 0 to 7: bit0: 1/2-pel, bit1: 1-pel, bit2: 4-pel
  float    cost_rd_quarter;
  float    cost_rd_half;
  float    cost_rd_1pel;
  float    cost_rd_4pel;
  float    cost_rd_affine;
};
#pragma pack(pop)

enum AmvrTargetImv
{
  AMVR_TARGET_IMV_1_4_PEL = 0,
  AMVR_TARGET_IMV_1_2_PEL = 1,
  AMVR_TARGET_IMV_1_PEL   = 2,
  AMVR_TARGET_IMV_4_PEL   = 3
};

enum AmvrTargetAffine
{
  AMVR_TARGET_AFFINE_1_4_PEL  = 0,
  AMVR_TARGET_AFFINE_1_16_PEL = 1,
  AMVR_TARGET_AFFINE_1_PEL    = 2
};

struct AmvrCuRecord
{
  bool   has14PelInter = false;
  float  features[AMVR_NUM_FEATURES] = { 0.0f };

  double cost14Pel   = std::numeric_limits<double>::max();
  double costHalfPel = std::numeric_limits<double>::max();
  double cost1Pel    = std::numeric_limits<double>::max();
  double cost4Pel    = std::numeric_limits<double>::max();
  double costAffine  = std::numeric_limits<double>::max();

  int    affineAmvrPrecision = 0; // 0: 1/4-pel, 1: 1/16-pel, 2: 1-pel

  int    cuArea = 0;
  bool   affineEvaluated = false;

  void reset()
  {
    has14PelInter = false;
    for (size_t i = 0; i < AMVR_NUM_FEATURES; ++i)
    {
      features[i] = 0.0f;
    }
    cost14Pel   = std::numeric_limits<double>::max();
    costHalfPel = std::numeric_limits<double>::max();
    cost1Pel    = std::numeric_limits<double>::max();
    cost4Pel    = std::numeric_limits<double>::max();
    costAffine  = std::numeric_limits<double>::max();
    affineAmvrPrecision = 0;
    cuArea = 0;
    affineEvaluated = false;
  }
};
