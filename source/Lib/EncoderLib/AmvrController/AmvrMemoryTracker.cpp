/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2024, ITU/ISO/IEC
 * All rights reserved.
 */

#include "AmvrMemoryTracker.h"

AmvrMemoryTracker::AmvrMemoryTracker()
  : m_alpha(0.05)
  , m_targetPruneRate(0.35)
  , m_ewmaImvWinGlobal(0.15)
  , m_ewmaImvRdSaving(0.0)
  , m_ewmaCoarseMvShare(0.5)
  , m_ewmaAffineAmvrWin(0.1)
  , m_ewmaAffineRdSaving(0.0)
  , m_ewmaFrameMvdEnergy(0.0)
  , m_onlineBudgetPressure(0.0)
  , m_totalCuEvaluated(0)
  , m_totalCuPruned(0)
  , m_coarseWins(1.0)
  , m_hpelWins(1.0)
{
  for (int i = 0; i < 4; ++i)
  {
    m_ewmaImvWinByArea[i] = 0.15;
  }
}

void AmvrMemoryTracker::init(double alpha, double targetPruneRate)
{
  m_alpha = (alpha > 0.0 && alpha <= 1.0) ? alpha : 0.05;
  m_targetPruneRate = (targetPruneRate >= 0.0 && targetPruneRate <= 1.0) ? targetPruneRate : 0.35;
  reset();
}

void AmvrMemoryTracker::reset()
{
  m_ewmaImvWinGlobal = 0.15;
  for (int i = 0; i < 4; ++i)
  {
    m_ewmaImvWinByArea[i] = 0.15;
  }
  m_ewmaImvRdSaving = 0.0;
  m_ewmaCoarseMvShare = 0.5;
  m_ewmaAffineAmvrWin = 0.1;
  m_ewmaAffineRdSaving = 0.0;
  m_ewmaFrameMvdEnergy = 0.0;
  m_onlineBudgetPressure = 0.0;
  m_totalCuEvaluated = 0;
  m_totalCuPruned = 0;
  m_coarseWins = 1.0;
  m_hpelWins = 1.0;
}

int AmvrMemoryTracker::getAreaBin(int cuArea) const
{
  if (cuArea <= 64)
  {
    return 0; // <= 64
  }
  else if (cuArea <= 256)
  {
    return 1; // 128 - 256
  }
  else if (cuArea <= 1024)
  {
    return 2; // 512 - 1024
  }
  else
  {
    return 3; // >= 2048
  }
}

void AmvrMemoryTracker::fillEwmaFeatures(float* outFeatures8, int cuArea) const
{
  int bin = getAreaBin(cuArea);

  // Features #43 to #50
  outFeatures8[0] = static_cast<float>(m_ewmaImvWinGlobal);
  outFeatures8[1] = static_cast<float>(m_ewmaImvWinByArea[bin]);
  outFeatures8[2] = static_cast<float>(m_ewmaImvRdSaving);
  outFeatures8[3] = static_cast<float>(m_ewmaCoarseMvShare);
  outFeatures8[4] = static_cast<float>(m_ewmaAffineAmvrWin);
  outFeatures8[5] = static_cast<float>(m_ewmaAffineRdSaving);
  outFeatures8[6] = static_cast<float>(m_ewmaFrameMvdEnergy);
  outFeatures8[7] = static_cast<float>(m_onlineBudgetPressure);
}

void AmvrMemoryTracker::updateCU(const AmvrCuRecord& record, int bestImv, int bestAffineAmvr, double winningCost)
{
  if (!record.has14PelInter)
  {
    return;
  }

  const double alpha = m_alpha;
  const int bin = getAreaBin(record.cuArea);

  // 1. Translational IMV win (best != 1/4-pel)
  const double imvWon = (bestImv != AMVR_TARGET_IMV_1_4_PEL) ? 1.0 : 0.0;
  m_ewmaImvWinGlobal = alpha * imvWon + (1.0 - alpha) * m_ewmaImvWinGlobal;
  m_ewmaImvWinByArea[bin] = alpha * imvWon + (1.0 - alpha) * m_ewmaImvWinByArea[bin];

  // 2. RD saving when IMV wins
  if (imvWon > 0.5 && record.cost14Pel < std::numeric_limits<double>::max() && record.cost14Pel > 0.0)
  {
    double saving = (record.cost14Pel - winningCost) / record.cost14Pel;
    saving = std::max(0.0, std::min(1.0, saving));
    m_ewmaImvRdSaving = alpha * saving + (1.0 - alpha) * m_ewmaImvRdSaving;
  }

  // 3. Coarse (1-pel / 4-pel) vs Fine (1/2-pel) share
  if (bestImv == AMVR_TARGET_IMV_1_PEL || bestImv == AMVR_TARGET_IMV_4_PEL)
  {
    m_coarseWins = alpha * 1.0 + (1.0 - alpha) * m_coarseWins;
    m_hpelWins   = (1.0 - alpha) * m_hpelWins;
  }
  else if (bestImv == AMVR_TARGET_IMV_1_2_PEL)
  {
    m_hpelWins   = alpha * 1.0 + (1.0 - alpha) * m_hpelWins;
    m_coarseWins = (1.0 - alpha) * m_coarseWins;
  }
  double totalWins = m_coarseWins + m_hpelWins;
  if (totalWins > 1e-6)
  {
    m_ewmaCoarseMvShare = m_coarseWins / totalWins;
  }

  // 4. Affine AMVR win and saving
  if (record.affineEvaluated)
  {
    const double affAmvrWon = (bestAffineAmvr != AMVR_TARGET_AFFINE_1_4_PEL) ? 1.0 : 0.0;
    m_ewmaAffineAmvrWin = alpha * affAmvrWon + (1.0 - alpha) * m_ewmaAffineAmvrWin;

    if (affAmvrWon > 0.5 && record.costAffine < std::numeric_limits<double>::max() && record.costAffine > winningCost)
    {
      double affSaving = (record.costAffine - winningCost) / std::max(1.0, record.costAffine);
      affSaving = std::max(0.0, std::min(1.0, affSaving));
      m_ewmaAffineRdSaving = alpha * affSaving + (1.0 - alpha) * m_ewmaAffineRdSaving;
    }
  }

  // 5. MVD energy (magnitude |MV - MVP| from feature #31 / #32)
  float mvdL0 = record.features[30]; // feature #31 (0-indexed: 30)
  float mvdL1 = record.features[31]; // feature #32 (0-indexed: 31)
  double currentMvdEnergy = static_cast<double>(mvdL0 + mvdL1);
  m_ewmaFrameMvdEnergy = alpha * currentMvdEnergy + (1.0 - alpha) * m_ewmaFrameMvdEnergy;
}

void AmvrMemoryTracker::recordPruneDecision(bool pruned)
{
  m_totalCuEvaluated++;
  if (pruned)
  {
    m_totalCuPruned++;
  }

  if (m_totalCuEvaluated > 0)
  {
    double currentPruneRate = static_cast<double>(m_totalCuPruned) / static_cast<double>(m_totalCuEvaluated);
    m_onlineBudgetPressure = currentPruneRate - m_targetPruneRate;
  }
}

double AmvrMemoryTracker::getDynamicThreshold(double baseThreshold, int cuArea) const
{
  int bin = getAreaBin(cuArea);
  double localWinRate = m_ewmaImvWinByArea[bin];

  // Adjust threshold: if IMV rarely wins in this area bin, increase threshold (prune more aggressively).
  // If budget pressure is negative (pruning less than target), increase threshold.
  double deltaThreshold = (m_onlineBudgetPressure < 0.0 ? 0.05 : -0.05) * std::abs(m_onlineBudgetPressure);
  if (localWinRate < 0.05)
  {
    deltaThreshold += 0.05;
  }
  else if (localWinRate > 0.30)
  {
    deltaThreshold -= 0.05;
  }

  return std::max(0.05, std::min(0.95, baseThreshold + deltaThreshold));
}
