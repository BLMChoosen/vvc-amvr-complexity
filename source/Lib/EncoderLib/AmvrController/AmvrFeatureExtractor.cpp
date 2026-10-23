/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2024, ITU/ISO/IEC
 * All rights reserved.
 */

#include "AmvrFeatureExtractor.h"
#include "CommonLib/UnitTools.h"
#include <cmath>
#include <algorithm>

float AmvrFeatureExtractor::calculateLumaVariance(const CPelBuf& origBuf)
{
  const int w = origBuf.width;
  const int h = origBuf.height;
  const ptrdiff_t stride = origBuf.stride;
  const Pel* p = origBuf.buf;

  double sum = 0.0;
  double sumSq = 0.0;

  for (int y = 0; y < h; ++y)
  {
    for (int x = 0; x < w; ++x)
    {
      const Pel val = p[x];
      sum += val;
      sumSq += static_cast<double>(val) * val;
    }
    p += stride;
  }

  const double n = static_cast<double>(w * h);
  if (n <= 1.0)
  {
    return 0.0f;
  }
  const double mean = sum / n;
  const double var = (sumSq / n) - (mean * mean);
  return static_cast<float>(std::max(0.0, var));
}

void AmvrFeatureExtractor::extractFeatures(
  float outFeatures[AMVR_NUM_FEATURES],
  const CodingUnit& cu,
  const CodingStructure& tempCS,
  const CodingStructure& bestCS,
  const Partitioner& partitioner,
  EncCfg& encCfg,
  const AmvrMemoryTracker& memoryTracker,
  double bestPrelimCost,
  InterSearch* interSearch
)
{
  for (size_t i = 0; i < AMVR_NUM_FEATURES; ++i)
  {
    outFeatures[i] = 0.0f;
  }

  const Slice& slice = *tempCS.slice;
  const PredictionUnit* pu = cu.firstPU;
  const int cuWidth = cu.Y().width;
  const int cuHeight = cu.Y().height;
  const int cuArea = cuWidth * cuHeight;

  // =========================================================================
  // Categoria A: Contexto Global e Temporal (Features #01 a #08, idx 0..7)
  // =========================================================================
  const int picWidth = slice.getPPS()->getPicWidthInLumaSamples();
  const int picHeight = slice.getPPS()->getPicHeightInLumaSamples();
  outFeatures[0] = static_cast<float>(picWidth * picHeight) / (1920.0f * 1080.0f); // #01: seq_resolution_area
  outFeatures[1] = encCfg.getIsLowDelay() ? 0.0f : 1.0f;                             // #02: is_random_access
  outFeatures[2] = static_cast<float>(slice.getSliceQp());                           // #03: slice_base_qp
  outFeatures[3] = static_cast<float>(slice.getTLayer());                            // #04: temporal_layer_id

  // #05: poc_dist_min_l0
  float minDistL0 = 0.0f;
  int numRefL0 = slice.getNumRefIdx(REF_PIC_LIST_0);
  if (numRefL0 > 0)
  {
    int minDiff = 999999;
    int curPoc = slice.getPOC();
    for (int i = 0; i < numRefL0; ++i)
    {
      const Picture* refPic = slice.getRefPic(REF_PIC_LIST_0, i);
      if (refPic)
      {
        int diff = std::abs(refPic->getPOC() - curPoc);
        if (diff < minDiff) minDiff = diff;
      }
    }
    if (minDiff != 999999) minDistL0 = static_cast<float>(minDiff);
  }
  outFeatures[4] = minDistL0;

  // #06: poc_dist_min_l1
  float minDistL1 = 0.0f;
  if (slice.isInterB())
  {
    int numRefL1 = slice.getNumRefIdx(REF_PIC_LIST_1);
    if (numRefL1 > 0)
    {
      int minDiff = 999999;
      int curPoc = slice.getPOC();
      for (int i = 0; i < numRefL1; ++i)
      {
        const Picture* refPic = slice.getRefPic(REF_PIC_LIST_1, i);
        if (refPic)
        {
          int diff = std::abs(refPic->getPOC() - curPoc);
          if (diff < minDiff) minDiff = diff;
        }
      }
      if (minDiff != 999999) minDistL1 = static_cast<float>(minDiff);
    }
  }
  outFeatures[5] = minDistL1;

  // #07: num_active_refs
  outFeatures[6] = static_cast<float>(numRefL0 + (slice.isInterB() ? slice.getNumRefIdx(REF_PIC_LIST_1) : 0));

  // #08: slice_lambda_log
  outFeatures[7] = static_cast<float>(std::log(std::max(1e-6, slice.getLambdas()[0])));

  // =========================================================================
  // Categoria B: Geometria, Particionamento e Textura da CU (Features #09 a #16, idx 8..15)
  // =========================================================================
  const float widthLog2 = static_cast<float>(floorLog2(cuWidth));
  const float heightLog2 = static_cast<float>(floorLog2(cuHeight));
  outFeatures[8]  = widthLog2;                                                          // #09: cu_width_log2
  outFeatures[9]  = heightLog2;                                                         // #10: cu_height_log2
  outFeatures[10] = static_cast<float>(cuArea) / 16384.0f;                              // #11: cu_area_norm
  outFeatures[11] = widthLog2 - heightLog2;                                             // #12: cu_aspect_ratio
  outFeatures[12] = static_cast<float>(cu.qtDepth);                                     // #13: qt_depth
  outFeatures[13] = static_cast<float>(cu.mtDepth);                                     // #14: mtt_depth
  outFeatures[14] = static_cast<float>(cu.qp);                                          // #15: cu_effective_qp
  outFeatures[15] = calculateLumaVariance(tempCS.picture->getOrigBuf(cu.Y()));          // #16: luma_pixel_variance

  // =========================================================================
  // Categoria C: Modos Preliminares Já Avaliados na CU (Features #17 a #22, idx 16..21)
  // =========================================================================
  outFeatures[16] = static_cast<float>(bestPrelimCost / std::max(1, cuArea));           // #17: best_prelim_rd_cost

  bool prelimIsSkip = false;
  bool prelimIsAffine = false;
  bool prelimMmvdWon = false;
  bool prelimZeroResidual = false;
  if (!bestCS.cus.empty() && bestCS.cus.front())
  {
    const CodingUnit* bestCU = bestCS.cus.front();
    prelimIsSkip = bestCU->skip;
    prelimIsAffine = bestCU->affine;
    prelimMmvdWon = bestCU->mmvdSkip;
    prelimZeroResidual = (bestCU->rootCbf == 0);
  }
  outFeatures[17] = prelimIsSkip ? 1.0f : 0.0f;                                         // #18: prelim_best_is_skip
  outFeatures[18] = prelimIsAffine ? 1.0f : 0.0f;                                       // #19: prelim_best_is_affine
  outFeatures[19] = prelimMmvdWon ? 1.0f : 0.0f;                                        // #20: prelim_mmvd_won
  outFeatures[20] = prelimZeroResidual ? 1.0f : 0.0f;                                   // #21: prelim_zero_residual

  // #22: merge_cand_mv_spread
  float mergeSpread = 0.0f;
  if (pu)
  {
    MergeCtx mrgCtx;
    PU::getInterMergeCandidates(*pu, mrgCtx, 0);
    if (mrgCtx.numValidMergeCand >= 2)
    {
      int diffX = std::abs(mrgCtx.mvFieldNeighbours[0][0].mv.hor - mrgCtx.mvFieldNeighbours[1][0].mv.hor);
      int diffY = std::abs(mrgCtx.mvFieldNeighbours[0][0].mv.ver - mrgCtx.mvFieldNeighbours[1][0].mv.ver);
      mergeSpread = static_cast<float>(diffX + diffY);
    }
  }
  outFeatures[21] = mergeSpread;

  // =========================================================================
  // Categoria D: Busca Base (1/4-pel) e MVP (Features #23 a #34, idx 22..33)
  // =========================================================================
  outFeatures[22] = static_cast<float>(tempCS.cost);                                    // #23: quarter_pel_rd_cost
  outFeatures[23] = static_cast<float>(tempCS.dist / std::max(1, cuArea));             // #24: quarter_pel_distortion
  outFeatures[24] = static_cast<float>(tempCS.fracBits >> SCALE_BITS);                 // #25: quarter_pel_total_bits

  // #26: mvd_bits_share
  uint32_t mvdBitsEst = 0;
  if (pu)
  {
    int mvdX0 = std::abs(pu->mvd[0].hor);
    int mvdY0 = std::abs(pu->mvd[0].ver);
    mvdBitsEst += (mvdX0 > 0 ? (2 * floorLog2(mvdX0) + 1) : 1);
    mvdBitsEst += (mvdY0 > 0 ? (2 * floorLog2(mvdY0) + 1) : 1);
    if (pu->interDir & 2)
    {
      int mvdX1 = std::abs(pu->mvd[1].hor);
      int mvdY1 = std::abs(pu->mvd[1].ver);
      mvdBitsEst += (mvdX1 > 0 ? (2 * floorLog2(mvdX1) + 1) : 1);
      mvdBitsEst += (mvdY1 > 0 ? (2 * floorLog2(mvdY1) + 1) : 1);
    }
  }
  const float totalBits = std::max(1.0f, static_cast<float>(tempCS.fracBits >> SCALE_BITS));
  outFeatures[25] = static_cast<float>(mvdBitsEst) / totalBits;

  // #27: rd_ratio_1_4_vs_merge
  outFeatures[26] = static_cast<float>(tempCS.cost / std::max(1.0, bestPrelimCost));

  // #28: inter_pred_dir
  outFeatures[27] = pu ? static_cast<float>(pu->interDir) : 1.0f;

  // #29, #30: mv magnitudes
  outFeatures[28] = pu ? static_cast<float>(std::abs(pu->mv[0].hor) + std::abs(pu->mv[0].ver)) : 0.0f;
  outFeatures[29] = (pu && (pu->interDir & 2)) ? static_cast<float>(std::abs(pu->mv[1].hor) + std::abs(pu->mv[1].ver)) : 0.0f;

  // #31, #32: mvd magnitudes
  outFeatures[30] = pu ? static_cast<float>(std::abs(pu->mvd[0].hor) + std::abs(pu->mvd[0].ver)) : 0.0f;
  outFeatures[31] = (pu && (pu->interDir & 2)) ? static_cast<float>(std::abs(pu->mvd[1].hor) + std::abs(pu->mvd[1].ver)) : 0.0f;

  // #33: amvp_cand_diff
  float amvpDiff = 0.0f;
  if (pu)
  {
    AMVPInfo amvpInfo;
    PU::fillMvpCand(const_cast<PredictionUnit&>(*pu), REF_PIC_LIST_0, pu->refIdx[0], amvpInfo);
    if (amvpInfo.numCand >= 2)
    {
      amvpDiff = static_cast<float>(std::abs(amvpInfo.mvCand[0].hor - amvpInfo.mvCand[1].hor) + std::abs(amvpInfo.mvCand[0].ver - amvpInfo.mvCand[1].ver));
    }
  }
  outFeatures[32] = amvpDiff;

  // #34: subpel_refine_gain
  float subpelFrac = 0.0f;
  if (pu)
  {
    int fracX = std::abs(pu->mv[0].hor) & 3;
    int fracY = std::abs(pu->mv[0].ver) & 3;
    subpelFrac = static_cast<float>(fracX + fracY) / 6.0f;
  }
  outFeatures[33] = subpelFrac;

  // =========================================================================
  // Categoria E: Contexto Espacial (Vizinhos/Pai) (Features #35 a #42, idx 34..41)
  // =========================================================================
  int availCount = 0;
  int imvCount = 0;
  int maxImv = 0;
  int affineCount = 0;
  int affineAmvrCount = 0;
  int totalDepth = 0;
  int depthCount = 0;

  Mv mvAbove(0, 0);
  Mv mvLeft(0, 0);
  bool hasMvAbove = false;
  bool hasMvLeft = false;

  const Position posL = cu.lumaPos().offset(-1, 0);
  const Position posA = cu.lumaPos().offset(0, -1);
  const Position posAL = cu.lumaPos().offset(-1, -1);
  const Position posAR = cu.lumaPos().offset(cuWidth, -1);
  const Position posBL = cu.lumaPos().offset(-1, cuHeight);

  const Position neighPositions[5] = { posL, posA, posAL, posAR, posBL };

  for (int i = 0; i < 5; ++i)
  {
    if (!pu) break;
    const PredictionUnit* puN = cu.cs->getPURestricted(neighPositions[i], *pu, partitioner.chType);
    if (puN && puN->cu)
    {
      availCount++;
      const CodingUnit* cuN = puN->cu;
      if (cuN->imv != 0)
      {
        imvCount++;
      }
      if (cuN->imv > maxImv)
      {
        maxImv = cuN->imv;
      }
      if (cuN->affine)
      {
        affineCount++;
        if (cuN->imv != 0)
        {
          affineAmvrCount++;
        }
      }
      if (i == 0) // Left
      {
        mvLeft = puN->mv[0];
        hasMvLeft = true;
        totalDepth += (cuN->qtDepth + cuN->mtDepth);
        depthCount++;
      }
      else if (i == 1) // Above
      {
        mvAbove = puN->mv[0];
        hasMvAbove = true;
        totalDepth += (cuN->qtDepth + cuN->mtDepth);
        depthCount++;
      }
    }
  }

  outFeatures[34] = static_cast<float>(availCount);                                                         // #35: neigh_avail_count
  outFeatures[35] = availCount > 0 ? static_cast<float>(imvCount) / static_cast<float>(availCount) : 0.0f; // #36: neigh_imv_usage_rate
  outFeatures[36] = static_cast<float>(maxImv);                                                             // #37: neigh_max_imv_idx
  outFeatures[37] = availCount > 0 ? static_cast<float>(affineCount) / static_cast<float>(availCount) : 0.0f; // #38: neigh_affine_rate
  outFeatures[38] = availCount > 0 ? static_cast<float>(affineAmvrCount) / static_cast<float>(availCount) : 0.0f; // #39: neigh_affine_amvr_rate
  outFeatures[39] = depthCount > 0 ? static_cast<float>(totalDepth) / static_cast<float>(depthCount) : 0.0f; // #40: neigh_avg_split_depth

  // #41: neigh_mv_spatial_grad
  float mvGrad = 0.0f;
  if (hasMvAbove && hasMvLeft)
  {
    mvGrad = static_cast<float>(std::abs(mvAbove.hor - mvLeft.hor) + std::abs(mvAbove.ver - mvLeft.ver));
  }
  outFeatures[40] = mvGrad;

  // #42: parent_cu_imv_mode
  outFeatures[41] = 0.0f; // Default if root or unknown

  // =========================================================================
  // Categoria F: Memória Contextual Online via EWMA (Features #43 a #50, idx 42..49)
  // =========================================================================
  memoryTracker.fillEwmaFeatures(&outFeatures[42], cuArea);
}
