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
#include <algorithm>
#include <cmath>

class AmvrMemoryTracker
{
public:
  AmvrMemoryTracker();
  ~AmvrMemoryTracker() = default;

  void init(double alpha, double targetPruneRate);
  void reset();

  int  getAreaBin(int cuArea) const;
  void fillEwmaFeatures(float* outFeatures8, int cuArea) const;

  void updateCU(const AmvrCuRecord& record, int bestImv, int bestAffineAmvr, double winningCost);
  void recordPruneDecision(bool pruned);

  double getDynamicThreshold(double baseThreshold, int cuArea) const;

  double getAlpha() const { return m_alpha; }
  double getTargetPruneRate() const { return m_targetPruneRate; }

private:
  double m_alpha;
  double m_targetPruneRate;

  // Online EWMA metrics
  double m_ewmaImvWinGlobal;
  double m_ewmaImvWinByArea[4];
  double m_ewmaImvRdSaving;
  double m_ewmaCoarseMvShare;
  double m_ewmaAffineAmvrWin;
  double m_ewmaAffineRdSaving;
  double m_ewmaFrameMvdEnergy;
  double m_onlineBudgetPressure;

  // Pruning statistics
  uint64_t m_totalCuEvaluated;
  uint64_t m_totalCuPruned;

  // Coarse vs fine win counters for ratio smoothing
  double m_coarseWins;
  double m_hpelWins;
};
