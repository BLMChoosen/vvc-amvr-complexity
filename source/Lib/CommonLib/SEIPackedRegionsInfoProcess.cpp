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

 /** \file     SEIPackedRegionsInfoProcess.cpp
     \brief    Packed regions info SEI processing
 */

#include "SequenceParameterSet.h"
#include "Picture.h"
#include "SEIPackedRegionsInfoProcess.h"


void SEIPackedRegionsInfoProcess::init(SEIPackedRegionsInfo& sei, const SPS& sps, uint32_t picWidth, uint32_t picHeight)
{
  m_enabled = true;
  m_persistence = sei.m_persistenceFlag;
  m_layerId = sei.m_layerId;
  m_multilayerFlag = sei.m_multilayerFlag;
  m_priUseMaxDimensionsFlag = sei.m_useMaxDimensionsFlag;
  m_priUnitSize = 1 << sei.m_log2UnitSize;
  m_picSize                 = Size(picWidth, picHeight);
  m_maxPicSize              = Size(sps.getMaxPicWidthInLumaSamples(), sps.getMaxPicHeightInLumaSamples());
  m_targetPicSize           = Size();
  if (sei.m_targetPicParamsPresentFlag)
  {
    m_targetPicSize.width  = sei.m_targetPicWidthMinus1 + 1;
    m_targetPicSize.height = sei.m_targetPicHeightMinus1 + 1;
  }
  m_bitDepths    = sps.getBitDepths();
  m_chromaFormat = sps.getChromaFormatIdc();
  m_subWidthC = SPS::getWinUnitX(sps.getChromaFormatIdc());
  m_subHeightC = SPS::getWinUnitY(sps.getChromaFormatIdc());

  if (sei.m_targetPicParamsPresentFlag)
  {
    CHECK(m_targetPicSize.width % m_subWidthC != 0,
          "The value of (pri_target_pic_width_minus1 + 1) % SubWidthC shall be equal to 0");
    CHECK(m_targetPicSize.height % m_subHeightC != 0,
          "The value of (pri_target_pic_height_minus1 + 1) % SubHeightC shall be equal to 0");
  }

  m_priNumRegions = sei.m_numRegionsMinus1 + 1;
  m_priRegionTopLeftX.resize(m_priNumRegions);
  m_priRegionTopLeftY.resize(m_priNumRegions);
  m_priRegionSize.resize(m_priNumRegions);
  m_priResampleWidthNum.resize(m_priNumRegions);
  m_priResampleWidthDenom.resize(m_priNumRegions);
  m_priResampleHeightNum.resize(m_priNumRegions);
  m_priResampleHeightDenom.resize(m_priNumRegions);
  m_priTargetRegion.resize(m_priNumRegions);
  m_priRegionId.resize(m_priNumRegions);
  if (sei.m_multilayerFlag)
  {
    m_regionLayerId = sei.m_regionLayerId;
    m_regionIsALayerFlag = sei.m_regionIsALayerFlag;
  }
  for (uint32_t i = 0; i < m_priNumRegions; i++)
  {
    if (!sei.m_useMaxDimensionsFlag)
    {
      m_priRegionTopLeftX[i] = sei.m_regionTopLeftInUnitsX[i] * m_priUnitSize;
      m_priRegionTopLeftY[i] = sei.m_regionTopLeftInUnitsY[i] * m_priUnitSize;
      m_priRegionSize[i].width  = (sei.m_regionWidthInUnitsMinus1[i] + 1) * m_priUnitSize;
      m_priRegionSize[i].height = (sei.m_regionHeightInUnitsMinus1[i] + 1) * m_priUnitSize;
    }
    else
    {
      m_priRegionTopLeftX[i] =
        Fraction(sei.m_regionTopLeftInUnitsX[i] * m_priUnitSize * m_picSize.width, m_maxPicSize.width).getIntValRound();
      m_priRegionTopLeftY[i] =
        Fraction(sei.m_regionTopLeftInUnitsY[i] * m_priUnitSize * m_picSize.height, m_maxPicSize.height)
          .getIntValRound();
      m_priRegionSize[i].width =
        Fraction((sei.m_regionWidthInUnitsMinus1[i] + 1) * m_priUnitSize * m_picSize.width, m_maxPicSize.width)
          .getIntValRound();
      m_priRegionSize[i].height =
        Fraction((sei.m_regionHeightInUnitsMinus1[i] + 1) * m_priUnitSize * m_picSize.height, m_maxPicSize.height)
          .getIntValRound();
    }
    uint32_t resamplingRatioIdx = 0;
    if (sei.m_numResamplingRatiosMinus1 > 0)
    {
      resamplingRatioIdx = sei.m_resamplingRatioIdx[i];
    }
    m_priResampleWidthNum[i] = sei.m_resamplingWidthNumMinus1[resamplingRatioIdx] + 1;
    m_priResampleWidthDenom[i] = sei.m_resamplingWidthDenomMinus1[resamplingRatioIdx] + 1;
    m_priResampleHeightNum[i] = sei.m_resamplingHeightNumMinus1[resamplingRatioIdx] + 1;
    m_priResampleHeightDenom[i] = sei.m_resamplingHeightDenomMinus1[resamplingRatioIdx] + 1;
    m_priTargetRegion[i].width =
      Fraction(m_priRegionSize[i].width * m_priResampleWidthNum[i], m_priResampleWidthDenom[i] * m_subWidthC)
        .getIntValRound()
      * m_subWidthC;
    m_priTargetRegion[i].height =
      Fraction(m_priRegionSize[i].height * m_priResampleHeightNum[i], m_priResampleHeightDenom[i] * m_subHeightC)
        .getIntValRound()
      * m_subHeightC;
  }
  if (sei.m_targetPicParamsPresentFlag)
  {
    m_priTargetRegionTopLeftX.resize(m_priNumRegions);
    m_priTargetRegionTopLeftY.resize(m_priNumRegions);
    for (uint32_t i = 0; i < m_priNumRegions; i++)
    {
      m_priTargetRegionTopLeftX[i] = sei.m_targetRegionTopLeftInUnitsX[i] * m_priUnitSize;
      m_priTargetRegionTopLeftY[i] = sei.m_targetRegionTopLeftInUnitsY[i] * m_priUnitSize;
    }
  }
  m_priRegionId = sei.m_regionId;
}

void SEIPackedRegionsInfoProcess::packRegions(const CPelUnitBuf& src, int layerId, PelUnitBuf& dst, const SPS& sps)
{
  for (int comp = 0; comp < ::getNumberValidComponents(m_chromaFormat); comp++)
  {
    ComponentID compID = ComponentID(comp);
    dst.get(compID).fill(1 << (m_bitDepths[toChannelType(compID)] - 1));
  }
  for (uint32_t i = 0; i < m_priNumRegions; i++)
  {
    if (m_multilayerFlag && (m_regionLayerId[i] != layerId || m_regionIsALayerFlag[i]))
    {
      continue;
    }
    const int xScale =
      Fraction(m_priTargetRegion[i].width << ScalingRatio::BITS, m_priRegionSize[i].width).getIntValRound();
    const int yScale =
      Fraction(m_priTargetRegion[i].height << ScalingRatio::BITS, m_priRegionSize[i].height).getIntValRound();
    ScalingRatio scalingRatio = { xScale, yScale };

    for (int comp = 0; comp < ::getNumberValidComponents(m_chromaFormat); comp++)
    {
      ComponentID compID = ComponentID(comp);
      const CPelBuf& beforeScale = src.get(compID);
      PelBuf& afterScale = dst.get(compID);
      int cx = isLuma(compID) ? 1 : m_subWidthC;
      int cy = isLuma(compID) ? 1 : m_subHeightC;
      CPelBuf beforeScaleSub = beforeScale.subBuf(m_priTargetRegionTopLeftX[i] / cx, m_priTargetRegionTopLeftY[i] / cy,
                                                  m_priTargetRegion[i].width / cx, m_priTargetRegion[i].height / cy);
      PelBuf  afterScaleSub  = afterScale.subBuf(m_priRegionTopLeftX[i] / cx, m_priRegionTopLeftY[i] / cy,
                                                 m_priRegionSize[i].width / cx, m_priRegionSize[i].height / cy);
      bool    downsampling   = (m_priTargetRegion[i].width > m_priRegionSize[i].width)
                          || (m_priTargetRegion[i].height > m_priRegionSize[i].height);
      bool useLumaFilter = downsampling;
      Picture::sampleRateConv(scalingRatio, ::getComponentScaleX(compID, m_chromaFormat),
                              ::getComponentScaleY(compID, m_chromaFormat), beforeScaleSub, 0, 0, afterScaleSub, 0, 0,
                              m_bitDepths[toChannelType(compID)], downsampling || useLumaFilter ? true : isLuma(compID),
                              downsampling, isLuma(compID) ? 1 : sps.getHorCollocatedChromaFlag(),
                              isLuma(compID) ? 1 : sps.getVerCollocatedChromaFlag(), false, false);
    }
  }
}

void SEIPackedRegionsInfoProcess::reconstruct(PicList* pcListPic, Picture* currentPic, PelUnitBuf& dst, const SPS& sps)
{
  for (int comp = 0; comp < ::getNumberValidComponents(m_chromaFormat); comp++)
  {
    ComponentID compID = ComponentID(comp);
    dst.get(compID).fill(1 << (m_bitDepths[toChannelType(compID)] - 1));
  }

  uint32_t maxId = *std::max_element(m_priRegionId.begin(), m_priRegionId.end());
  for (uint32_t regionId = 0; regionId <= maxId; regionId++)
  {
    auto it = std::find(m_priRegionId.begin(), m_priRegionId.end(), regionId);
    if (it != m_priRegionId.end())
    {
      const ptrdiff_t i = std::distance(m_priRegionId.begin(), it);

      Picture* picSrc = currentPic;
      if (m_multilayerFlag && m_regionLayerId[i] != currentPic->layerId)
      {
        const VPS* vps = currentPic->cs->vps;
        CHECK(vps == nullptr, "vps does not exist");
        CHECK(currentPic->layerId == 0, "current layer is 0");
        CHECK(vps->getIndependentLayerFlag(vps->getGeneralLayerIdx(currentPic->layerId)), "current layer is independent");
        CHECK(!vps->getDirectRefLayerFlag(vps->getGeneralLayerIdx(currentPic->layerId), vps->getGeneralLayerIdx(m_regionLayerId[i])), "src layer is not a reference layer for current layer");

        bool found = false;
        for (auto& checkPic : *pcListPic)
        {
          if (checkPic != nullptr && checkPic->layerId == m_regionLayerId[i] && checkPic->getPOC() == currentPic->getPOC())
          {
            picSrc = checkPic;
            found = true;
            break;
          }
        }
        CHECK(!found, "did not found picture of other layer");
      }
      const PelUnitBuf& src = picSrc->getRecoBuf();
      Window win = picSrc->getConformanceWindow();

      int winLeftOffset   = win.getWindowLeftOffset() * SPS::getWinUnitX(picSrc->m_chromaFormatIdc);
      int winTopOffset    = win.getWindowTopOffset() * SPS::getWinUnitY(picSrc->m_chromaFormatIdc);
      int winRightOffset  = win.getWindowRightOffset() * SPS::getWinUnitX(picSrc->m_chromaFormatIdc);
      int winBottomOffset = win.getWindowBottomOffset() * SPS::getWinUnitY(picSrc->m_chromaFormatIdc);

      uint32_t picWidthInLumaSamples = src.get(COMPONENT_Y).width - winLeftOffset - winRightOffset;
      uint32_t picHeightInLumaSamples = src.get(COMPONENT_Y).height - winTopOffset - winBottomOffset;
      if (m_multilayerFlag && m_regionIsALayerFlag[i] != 0)
      {
        m_priRegionTopLeftX[i] =0;
        m_priRegionTopLeftY[i] =0;
        m_priRegionSize[i]     = Size(picWidthInLumaSamples, picHeightInLumaSamples);
      }
      if (m_priUseMaxDimensionsFlag)
      {
        m_priTargetRegion[i].width =
          ((uint32_t) (((double) m_priRegionSize[i].width * m_priResampleWidthNum[i] * m_maxPicSize.width)
                         / (m_priResampleWidthDenom[i] * picWidthInLumaSamples * m_subWidthC)
                       + 0.5))
          * m_subWidthC;
        m_priTargetRegion[i].height =
          ((uint32_t) (((double) m_priRegionSize[i].height * m_priResampleHeightNum[i] * m_maxPicSize.height)
                         / (m_priResampleHeightDenom[i] * picHeightInLumaSamples * m_subHeightC)
                       + 0.5))
          * m_subHeightC;
      }
      CHECK(m_priRegionTopLeftX[i] % m_subWidthC != 0, "priRegionTopLeftX[i] must be a multiple of SubWidthC");
      CHECK(m_priRegionTopLeftY[i] % m_subHeightC != 0, "priRegionTopLeftY[i] must be a multiple of SubHeightC");
      CHECK(m_priRegionSize[i].width % m_subWidthC != 0, "priRegionWidth[i] must be a multiple of SubWidthC");
      CHECK(m_priRegionSize[i].height % m_subHeightC != 0, "priRegionHeight[i] must be a multiple of SubHeightC");
      CHECK(m_priRegionTopLeftX[i] + m_priRegionSize[i].width > picWidthInLumaSamples,
            "priRegionTopLeftX[i] + priRegionWidth[i] shall be less than or equal to picWidthInLumaSamples");
      CHECK(m_priRegionTopLeftY[i] + m_priRegionSize[i].height > picHeightInLumaSamples,
            "priRegionTopLeftY[i] + priRegionHeight[i] shall be less than or equal to picHightInLumaSamples");
      int xScale = ((m_priRegionSize[i].width << ScalingRatio::BITS) + (m_priTargetRegion[i].width >> 1))
                   / m_priTargetRegion[i].width;
      int yScale = ((m_priRegionSize[i].height << ScalingRatio::BITS) + (m_priTargetRegion[i].height >> 1))
                   / m_priTargetRegion[i].height;
      ScalingRatio scalingRatio = { xScale, yScale };
      for (int comp = 0; comp < ::getNumberValidComponents(m_chromaFormat); comp++)
      {
        ComponentID compID = ComponentID(comp);
        const CPelBuf& beforeScale = src.get(compID);
        PelBuf& afterScale = dst.get(compID);
        int cx = isLuma(compID) ? 1 : m_subWidthC;
        int cy = isLuma(compID) ? 1 : m_subHeightC;
        CPelBuf        beforeScaleSub = beforeScale.subBuf((winLeftOffset + m_priRegionTopLeftX[i]) / cx,
                                                           (winTopOffset + m_priRegionTopLeftY[i]) / cy,
                                                           m_priRegionSize[i].width / cx, m_priRegionSize[i].height / cy);
        PelBuf afterScaleSub = afterScale.subBuf(m_priTargetRegionTopLeftX[i] / cx, m_priTargetRegionTopLeftY[i] / cy,
                                                 m_priTargetRegion[i].width / cx, m_priTargetRegion[i].height / cy);
        bool   downsampling  = (m_priRegionSize[i].width > m_priTargetRegion[i].width)
                            || (m_priRegionSize[i].height > m_priTargetRegion[i].height);
        bool useLumaFilter = downsampling;
        Picture::sampleRateConv(scalingRatio, ::getComponentScaleX(compID, m_chromaFormat),
                                ::getComponentScaleY(compID, m_chromaFormat), beforeScaleSub, 0, 0, afterScaleSub, 0, 0,
                                m_bitDepths[toChannelType(compID)],
                                downsampling || useLumaFilter ? true : isLuma(compID), downsampling,
                                isLuma(compID) ? 1 : sps.getHorCollocatedChromaFlag(),
                                isLuma(compID) ? 1 : sps.getVerCollocatedChromaFlag(), false, false);
      }
    }
  }
}
