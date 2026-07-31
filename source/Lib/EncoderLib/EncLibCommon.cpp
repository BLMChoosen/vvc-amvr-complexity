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

/** \file     EncLibCommon.cpp
    \brief    Common encoder library class
*/

#include "CommonDef.h"
#include "EncLibCommon.h"
#include "CudaBackend/ComputeBackend.h"
#include "CudaBackend/CudaContext.h"
#include "CudaBackend/CudaDistortion.h"
#include "CudaBackend/CudaQpa.h"

#include <algorithm>

struct EncLibCommon::ComputeState
{
  vtm::ComputeConfig config;
  vtm::CudaContext   cudaContext;
  unsigned           users = 0;
  bool               configured = false;
  bool               synchronized = false;
  std::vector<std::uint64_t> sadResults;
  std::vector<vtm::CudaQpaResult> qpaChunkResults;
  std::uint64_t qpaFallbacks = 0;
};

EncLibCommon::EncLibCommon()
  : m_spsMap(MAX_NUM_SPS), m_ppsMap(MAX_NUM_PPS), m_computeState(new ComputeState)
{
  std::memset( m_layerDecPicBuffering, 0, sizeof( m_layerDecPicBuffering ) );
  for (const auto t: { ApsType::ALF, ApsType::LMCS, ApsType::SCALING_LIST })
  {
    m_apsMaps[t] = ParameterSetMap<APS>(MAX_NUM_APS(t));
  }
}

EncLibCommon::~EncLibCommon()
{
}

void EncLibCommon::configureComputeBackend(const vtm::ComputeConfig &config)
{
  if (m_computeState->configured)
  {
    CHECK(m_computeState->config.backend != config.backend || m_computeState->config.device != config.device
            || m_computeState->config.enableExperimentalSad != config.enableExperimentalSad
            || m_computeState->config.enableExperimentalQpa != config.enableExperimentalQpa,
          "All encoder layers must use the same GPUBackend and GPUDevice");
    return;
  }

  m_computeState->config = config;
  m_computeState->configured = true;
}

void EncLibCommon::acquireComputeBackend()
{
  CHECK(!m_computeState->configured, "Compute backend must be configured before encoder creation");
  if (m_computeState->users == 0 && m_computeState->config.backend == vtm::ComputeBackend::CUDA)
  {
    m_computeState->cudaContext.create(m_computeState->config.device);
    m_computeState->synchronized = false;
  }
  ++m_computeState->users;
}

void EncLibCommon::synchronizeComputeBackend()
{
  CHECK(m_computeState->users == 0, "Cannot synchronize an encoder compute backend without an owner");
  if (m_computeState->cudaContext.isCreated() && !m_computeState->synchronized)
  {
    m_computeState->cudaContext.synchronize();
    m_computeState->synchronized = true;
  }
}

void EncLibCommon::releaseComputeBackend()
{
  CHECK(m_computeState->users == 0, "Encoder compute backend owner count underflow");

  --m_computeState->users;
  if (m_computeState->users == 0 && m_computeState->cudaContext.isCreated())
  {
    m_computeState->cudaContext.shutdown(!m_computeState->synchronized);
    m_computeState->synchronized = false;
  }
}

bool EncLibCommon::isCudaBackendActive() const
{
  return m_computeState->cudaContext.isCreated();
}

void EncLibCommon::registerPictureMirror(const void *owner, const vtm::CudaPictureRole role,
                                         const vtm::CudaHostPictureDesc &picture)
{
  if (m_computeState->cudaContext.isCreated())
  {
    m_computeState->cudaContext.registerPictureMirror(owner, role, picture);
  }
}

void EncLibCommon::bindPictureMirror(const void *owner, const vtm::CudaPictureRole role,
                                     const vtm::CudaHostPictureDesc &picture)
{
  if (m_computeState->cudaContext.isCreated())
  {
    if (m_computeState->cudaContext.hasPictureMirror(owner, role))
    {
      m_computeState->cudaContext.rebindHostPicture(owner, role, picture);
    }
    else
    {
      m_computeState->cudaContext.registerPictureMirror(owner, role, picture);
    }
  }
}

void EncLibCommon::releasePictureMirrors(const void *owner)
{
  if (m_computeState->cudaContext.isCreated())
  {
    m_computeState->cudaContext.releasePictureMirrors(owner);
  }
}

bool EncLibCommon::isCudaSadBatchAvailable(const void *sourceOwner, const void *referenceOwner) const
{
  return m_computeState->config.enableExperimentalSad
         && m_computeState->cudaContext.isDistortionAccelerationAvailable()
         && m_computeState->cudaContext.hasPictureMirror(sourceOwner, vtm::CudaPictureRole::Original)
         && m_computeState->cudaContext.hasPictureMirror(referenceOwner, vtm::CudaPictureRole::Reconstruction);
}

bool EncLibCommon::computeSadGrid(const void *sourceOwner, const void *source, const void *referenceOwner,
                                  const void *reference, const std::uint32_t columns, const std::uint32_t rows,
                                  const std::uint32_t width, const std::uint32_t height,
                                  const std::uint8_t elementSize, const std::uint8_t bitDepth,
                                  const std::uint8_t subShift, const std::uint64_t *&results)
{
  results = nullptr;
  if (!isCudaSadBatchAvailable(sourceOwner, referenceOwner)
      || !m_computeState->cudaContext.hasPictureMirror(sourceOwner, vtm::CudaPictureRole::Original)
      || !m_computeState->cudaContext.hasPictureMirror(referenceOwner, vtm::CudaPictureRole::Reconstruction))
  {
    return false;
  }

  vtm::CudaDistortionBatchDesc batch{};
  batch.sourceMirror = m_computeState->cudaContext.pictureMirrorHandle(sourceOwner, vtm::CudaPictureRole::Original);
  batch.referenceMirror =
    m_computeState->cudaContext.pictureMirrorHandle(referenceOwner, vtm::CudaPictureRole::Reconstruction);
  batch.source = source;
  batch.candidateGrid.reference = reference;
  batch.candidateGrid.columns = columns;
  batch.candidateGrid.rows = rows;
  batch.candidateGrid.stepX = 1;
  batch.candidateGrid.stepY = 1;
  batch.width = width;
  batch.height = height;
  batch.sourcePlane = 0;
  batch.referencePlane = 0;
  batch.elementSize = elementSize;
  batch.bitDepth = bitDepth;
  batch.subShift = subShift;
  batch.metric = vtm::CudaDistortionMetric::Sad;
  const std::uint64_t candidateCount = static_cast<std::uint64_t>(columns) * rows;
  if (candidateCount == 0 || candidateCount > vtm::CUDA_MAX_DISTORTION_CANDIDATES)
  {
    return false;
  }
  if (m_computeState->sadResults.size() < candidateCount)
  {
    m_computeState->sadResults.resize(static_cast<std::size_t>(candidateCount));
  }
  if (!m_computeState->cudaContext.computeDistortionBatch(batch, m_computeState->sadResults.data()))
  {
    return false;
  }
  results = m_computeState->sadResults.data();
  return true;
}

vtm::CudaSadStats EncLibCommon::cudaSadStats() const
{
  return { m_computeState->cudaContext.isCreated() ? m_computeState->cudaContext.distortionBatchDispatchCount() : 0,
           m_computeState->cudaContext.distortionBatchFailureCount(),
           m_computeState->cudaContext.isCreated()
             && !m_computeState->cudaContext.isDistortionAccelerationAvailable() };
}

bool EncLibCommon::isCudaQpaBatchAvailable(const void *sourceOwner) const
{
  return m_computeState->config.enableExperimentalQpa
         && m_computeState->cudaContext.isQpaAccelerationAvailable()
         && m_computeState->cudaContext.hasPictureMirror(sourceOwner, vtm::CudaPictureRole::Original);
}

bool EncLibCommon::prepareQpaTasks(const void *sourceOwner)
{
  if (!m_computeState->config.enableExperimentalQpa)
  {
    return false;
  }
  if (!isCudaQpaBatchAvailable(sourceOwner))
  {
    ++m_computeState->qpaFallbacks;
    return false;
  }
  return true;
}

bool EncLibCommon::computeQpaTasks(const void *sourceOwner, const vtm::CudaQpaTask *tasks,
                                   const std::size_t taskCount, std::vector<vtm::CudaQpaResult> &results)
{
  if (tasks == nullptr || taskCount == 0)
  {
    return false;
  }
  if (!isCudaQpaBatchAvailable(sourceOwner))
  {
    ++m_computeState->qpaFallbacks;
    return false;
  }
  for (std::size_t index = 1; index < taskCount; ++index)
  {
    if (tasks[index].ticket <= tasks[index - 1].ticket)
    {
      return false;
    }
  }

  // Keep publication transactional across chunks: callers only see the completed vector after every chunk succeeds.
  std::vector<vtm::CudaQpaResult> completed(taskCount);
  if (m_computeState->qpaChunkResults.size() < vtm::CUDA_MAX_QPA_TASKS)
  {
    m_computeState->qpaChunkResults.resize(vtm::CUDA_MAX_QPA_TASKS);
  }
  const vtm::CudaMirrorHandle mirror =
    m_computeState->cudaContext.pictureMirrorHandle(sourceOwner, vtm::CudaPictureRole::Original);
  for (std::size_t offset = 0; offset < taskCount; offset += vtm::CUDA_MAX_QPA_TASKS)
  {
    const std::size_t remaining = taskCount - offset;
    const std::uint32_t chunk = static_cast<std::uint32_t>(
      std::min<std::size_t>(remaining, vtm::CUDA_MAX_QPA_TASKS));
    if (!m_computeState->cudaContext.computeQpaBatch(mirror, tasks + offset, chunk,
                                                     m_computeState->qpaChunkResults.data()))
    {
      ++m_computeState->qpaFallbacks;
      return false;
    }
    std::copy_n(m_computeState->qpaChunkResults.begin(), chunk, completed.begin() + offset);
  }
  results.swap(completed);
  return true;
}

vtm::CudaQpaStats EncLibCommon::cudaQpaStats() const
{
  return { m_computeState->cudaContext.isCreated() ? m_computeState->cudaContext.qpaBatchDispatchCount() : 0,
           m_computeState->cudaContext.isCreated() ? m_computeState->cudaContext.qpaTaskCount() : 0,
           m_computeState->cudaContext.qpaBatchFailureCount(), m_computeState->qpaFallbacks,
           m_computeState->config.enableExperimentalQpa,
           m_computeState->config.enableExperimentalQpa && m_computeState->cudaContext.qpaBatchFailureCount() != 0
             && !m_computeState->cudaContext.isQpaAccelerationAvailable(),
           !m_computeState->config.enableExperimentalQpa };
}
