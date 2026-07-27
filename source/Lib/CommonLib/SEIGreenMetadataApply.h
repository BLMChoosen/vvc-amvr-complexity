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

 /** \file     SEIGreenMetadataApply.h
     \brief    Green Metadata SEI
 */

#ifndef __SEIGREENMETADATAAPPLY__
#define __SEIGREENMETADATAAPPLY__

#include "Picture.h"

#if GREEN_METADATA_SEI_ENABLED && GREEN_METADATA_SEI_AMI_ENABLED_WG03_N01464

//! \ingroup CommonLib
//! \{

class SEIGreenMetadataApply
{
private:
  uint32_t              m_width;
  uint32_t              m_height;
  ChromaFormat          m_chromaFormat;
  uint8_t               m_bitDepth;
  bool                  m_fullRangeFlag;

public:
  SEIGreenMetadataInfo* m_greenMetadataInfo;

public:
  SEIGreenMetadataApply();
  virtual ~SEIGreenMetadataApply();

  void create (uint32_t width, uint32_t height, ChromaFormat fmt, uint8_t bitDepth, bool fullRangeFlag);
  bool getMap (PicList* pcListPic, int layerId, PelStorage* attenuationMap);
  void apply  (PelStorage* attenuationMap, PelStorage* attenuatedBuf);

};// END CLASS DEFINITION SEIGreenMetadataApply

#endif

#endif
