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

 /** \file     SEIGreenMetadataApply.cpp
     \brief    green metadata SEI
 */

#include "SEIGreenMetadataApply.h"

#if GREEN_METADATA_SEI_ENABLED && GREEN_METADATA_SEI_AMI_ENABLED_WG03_N01464

SEIGreenMetadataApply::SEIGreenMetadataApply()
  : m_width(0), m_height(0), m_chromaFormat(ChromaFormat::UNDEFINED), m_bitDepth(0), m_greenMetadataInfo(nullptr)
{
}

void SEIGreenMetadataApply::create(uint32_t width, uint32_t height, ChromaFormat fmt, uint8_t bitDepth, bool fullRangeFlag)
{
  m_width              = width;
  m_height             = height;
  m_chromaFormat       = fmt;
  m_bitDepth           = bitDepth;
  m_fullRangeFlag      = fullRangeFlag;
  m_greenMetadataInfo = new SEIGreenMetadataInfo;
}

SEIGreenMetadataApply::~SEIGreenMetadataApply()
{
  delete m_greenMetadataInfo;
}

void SEIGreenMetadataApply::apply(PelStorage* attenuationMap, PelStorage* attenuatedBuf)
{
  PelBuf y0 = attenuatedBuf->Y();
  PelBuf y1 = attenuationMap->Y();

  for (int i = 0; i < y0.height; i++)
  {
    Pel* y0_row = y0.buf + i * y0.stride;
    const Pel* y1_row = y1.buf + i * y1.stride;

    if (m_greenMetadataInfo->m_greenMetadataAMIAttenuationCompIdc[0] == 0)  /* Apply luma component of the attenuation map to the luma component of the primary picture */
    {
      if (m_greenMetadataInfo->m_greenMetadataAMIAttenuationUseIdc[0] == 0) /* Subtraction */
      {
        for (int j = 0; j < y0.width; j++)
        {
          y0_row[j] -= y1_row[j];
          y0_row[j] = std::max((int) y0_row[j], m_fullRangeFlag ? 0 : 16 * (1 << (m_bitDepth - 8)));
        }
      }
    }
  }
}

#endif
