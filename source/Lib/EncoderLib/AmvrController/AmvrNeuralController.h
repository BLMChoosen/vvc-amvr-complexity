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
#include "amvr_weights.h"
#include <vector>

class AmvrNeuralController
{
public:
  AmvrNeuralController();
  ~AmvrNeuralController() = default;

  void normalizeFeatures(const float inFeatures[AMVR_NUM_FEATURES], float outNormFeatures[AMVR_NUM_FEATURES]) const;

  // Performs forward pass: computes Head 1 (Translational Softmax [8]) and Head 2 (Affine Sigmoid [2])
  void forward(const float inFeatures[AMVR_NUM_FEATURES], float outHead1[8], float outHead2[2]) const;

  // Pruning decisions
  bool shouldPruneImvMode(int imvMode, const float head1Probs[8], double threshold) const;
  bool shouldPruneAffinePrecision(int affinePrec, const float head2Probs[2], double threshold) const;

private:
  static void denseRelu(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    size_t inDim,
    size_t outDim
  );

  static void denseLinear(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    size_t inDim,
    size_t outDim
  );

  static void softmax(const float* input, float* output, size_t dim);
  static void sigmoid(const float* input, float* output, size_t dim);
};
