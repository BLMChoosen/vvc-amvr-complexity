/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2024, ITU/ISO/IEC
 * All rights reserved.
 */

#include "AmvrNeuralController.h"
#include <cmath>
#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

AmvrNeuralController::AmvrNeuralController()
{
}

void AmvrNeuralController::normalizeFeatures(const float inFeatures[AMVR_NUM_FEATURES], float outNormFeatures[AMVR_NUM_FEATURES]) const
{
  for (size_t i = 0; i < AMVR_NUM_FEATURES; ++i)
  {
    const float stdVal = (AmvrWeights::FEATURE_STD[i] > 1e-6f) ? AmvrWeights::FEATURE_STD[i] : 1.0f;
    outNormFeatures[i] = (inFeatures[i] - AmvrWeights::FEATURE_MEAN[i]) / stdVal;
  }
}

void AmvrNeuralController::denseRelu(
  const float* input,
  const float* weights,
  const float* bias,
  float* output,
  size_t inDim,
  size_t outDim
)
{
  for (size_t j = 0; j < outDim; ++j)
  {
    const float* wRow = &weights[j * inDim];
    float sum = bias[j];

#if defined(__AVX2__)
    size_t i = 0;
    __m256 vSum = _mm256_setzero_ps();
    for (; i + 7 < inDim; i += 8)
    {
      __m256 vIn = _mm256_loadu_ps(&input[i]);
      __m256 vW  = _mm256_loadu_ps(&wRow[i]);
#if defined(__FMA__)
      vSum = _mm256_fmadd_ps(vIn, vW, vSum);
#else
      vSum = _mm256_add_ps(vSum, _mm256_mul_ps(vIn, vW));
#endif
    }
    // Horizontal add of 8 floats
    alignas(32) float buffer[8];
    _mm256_storeu_ps(buffer, vSum);
    for (int k = 0; k < 8; ++k)
    {
      sum += buffer[k];
    }
    for (; i < inDim; ++i)
    {
      sum += input[i] * wRow[i];
    }
#else
    for (size_t i = 0; i < inDim; ++i)
    {
      sum += input[i] * wRow[i];
    }
#endif

    // ReLU
    output[j] = std::max(0.0f, sum);
  }
}

void AmvrNeuralController::denseLinear(
  const float* input,
  const float* weights,
  const float* bias,
  float* output,
  size_t inDim,
  size_t outDim
)
{
  for (size_t j = 0; j < outDim; ++j)
  {
    const float* wRow = &weights[j * inDim];
    float sum = bias[j];

#if defined(__AVX2__)
    size_t i = 0;
    __m256 vSum = _mm256_setzero_ps();
    for (; i + 7 < inDim; i += 8)
    {
      __m256 vIn = _mm256_loadu_ps(&input[i]);
      __m256 vW  = _mm256_loadu_ps(&wRow[i]);
#if defined(__FMA__)
      vSum = _mm256_fmadd_ps(vIn, vW, vSum);
#else
      vSum = _mm256_add_ps(vSum, _mm256_mul_ps(vIn, vW));
#endif
    }
    alignas(32) float buffer[8];
    _mm256_storeu_ps(buffer, vSum);
    for (int k = 0; k < 8; ++k)
    {
      sum += buffer[k];
    }
    for (; i < inDim; ++i)
    {
      sum += input[i] * wRow[i];
    }
#else
    for (size_t i = 0; i < inDim; ++i)
    {
      sum += input[i] * wRow[i];
    }
#endif

    output[j] = sum;
  }
}

void AmvrNeuralController::softmax(const float* input, float* output, size_t dim)
{
  float maxVal = input[0];
  for (size_t i = 1; i < dim; ++i)
  {
    if (input[i] > maxVal)
    {
      maxVal = input[i];
    }
  }

  float sumExp = 0.0f;
  for (size_t i = 0; i < dim; ++i)
  {
    output[i] = std::exp(input[i] - maxVal);
    sumExp += output[i];
  }

  const float invSum = (sumExp > 1e-9f) ? (1.0f / sumExp) : 1.0f;
  for (size_t i = 0; i < dim; ++i)
  {
    output[i] *= invSum;
  }
}

void AmvrNeuralController::sigmoid(const float* input, float* output, size_t dim)
{
  for (size_t i = 0; i < dim; ++i)
  {
    output[i] = 1.0f / (1.0f + std::exp(-input[i]));
  }
}

void AmvrNeuralController::forward(const float inFeatures[AMVR_NUM_FEATURES], float outHead1[8], float outHead2[2]) const
{
  // 1. Normalize
  alignas(32) float normFeat[AmvrWeights::INPUT_DIM];
  normalizeFeatures(inFeatures, normFeat);

  // 2. Hidden Layer 1 (50 -> 64, ReLU)
  alignas(32) float hidden1[AmvrWeights::HIDDEN1_DIM];
  denseRelu(normFeat, AmvrWeights::W1, AmvrWeights::B1, hidden1, AmvrWeights::INPUT_DIM, AmvrWeights::HIDDEN1_DIM);

  // 3. Hidden Layer 2 (64 -> 64, ReLU)
  alignas(32) float hidden2[AmvrWeights::HIDDEN2_DIM];
  denseRelu(hidden1, AmvrWeights::W2, AmvrWeights::B2, hidden2, AmvrWeights::HIDDEN1_DIM, AmvrWeights::HIDDEN2_DIM);

  // 4. Head 1: Translational (64 -> 8, Softmax)
  alignas(32) float logitsHead1[AmvrWeights::HEAD1_DIM];
  denseLinear(hidden2, AmvrWeights::W_HEAD1, AmvrWeights::B_HEAD1, logitsHead1, AmvrWeights::HIDDEN2_DIM, AmvrWeights::HEAD1_DIM);
  softmax(logitsHead1, outHead1, AmvrWeights::HEAD1_DIM);

  // 5. Head 2: Affine AMVR (64 -> 2, Sigmoid)
  alignas(32) float logitsHead2[AmvrWeights::HEAD2_DIM];
  denseLinear(hidden2, AmvrWeights::W_HEAD2, AmvrWeights::B_HEAD2, logitsHead2, AmvrWeights::HIDDEN2_DIM, AmvrWeights::HEAD2_DIM);
  sigmoid(logitsHead2, outHead2, AmvrWeights::HEAD2_DIM);
}

bool AmvrNeuralController::shouldPruneImvMode(int imvMode, const float head1Probs[8], double threshold) const
{
  // head1Probs represents probabilities of 8 discrete combinations of (bit0: 1/2-pel, bit1: 1-pel, bit2: 4-pel)
  // Check active bit for the requested mode:
  float activeProb = 0.0f;
  if (imvMode == AMVR_TARGET_IMV_1_2_PEL) // bit 0 (combinations 1, 3, 5, 7)
  {
    activeProb = head1Probs[1] + head1Probs[3] + head1Probs[5] + head1Probs[7];
  }
  else if (imvMode == AMVR_TARGET_IMV_1_PEL) // bit 1 (combinations 2, 3, 6, 7)
  {
    activeProb = head1Probs[2] + head1Probs[3] + head1Probs[6] + head1Probs[7];
  }
  else if (imvMode == AMVR_TARGET_IMV_4_PEL) // bit 2 (combinations 4, 5, 6, 7)
  {
    activeProb = head1Probs[4] + head1Probs[5] + head1Probs[6] + head1Probs[7];
  }
  else
  {
    return false; // 1/4-pel is base mode, never pruned
  }

  return (static_cast<double>(activeProb) < threshold);
}

bool AmvrNeuralController::shouldPruneAffinePrecision(int affinePrec, const float head2Probs[2], double threshold) const
{
  if (affinePrec == AMVR_TARGET_AFFINE_1_16_PEL) // Head2 output 0
  {
    return (static_cast<double>(head2Probs[0]) < threshold);
  }
  else if (affinePrec == AMVR_TARGET_AFFINE_1_PEL) // Head2 output 1
  {
    return (static_cast<double>(head2Probs[1]) < threshold);
  }

  return false;
}
