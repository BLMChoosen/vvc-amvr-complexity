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

struct EncLibCommon::ComputeState
{
  vtm::ComputeConfig config;
  vtm::CudaContext   cudaContext;
  unsigned           users = 0;
  bool               configured = false;
  bool               synchronized = false;
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
    CHECK(m_computeState->config.backend != config.backend || m_computeState->config.device != config.device,
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
