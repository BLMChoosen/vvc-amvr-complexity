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

/** \file     EncSearch.cpp
 *  \brief    encoder inter search class
 */

#include "InterSearch.h"


#include "CommonLib/CommonDef.h"
#include "CommonLib/Rom.h"
#include "CommonLib/MotionInfo.h"
#include "CommonLib/Picture.h"
#include "CommonLib/UnitTools.h"
#include "CommonLib/dtrace_next.h"
#include "CommonLib/dtrace_buffer.h"
#include "CommonLib/MCTS.h"

#include "EncModeCtrl.h"
#include "EncLib.h"
#include "EncLibCommon.h"
#include "CudaBackend/CudaDistortion.h"

#include <math.h>
#include <limits>

//! \ingroup EncoderLib
//! \{

static const Mv s_acMvRefineH[9] =
{
  Mv(  0,  0 ), // 0
  Mv(  0, -1 ), // 1
  Mv(  0,  1 ), // 2
  Mv( -1,  0 ), // 3
  Mv(  1,  0 ), // 4
  Mv( -1, -1 ), // 5
  Mv(  1, -1 ), // 6
  Mv( -1,  1 ), // 7
  Mv(  1,  1 )  // 8
};

static const Mv s_acMvRefineQ[9] =
{
  Mv(  0,  0 ), // 0
  Mv(  0, -1 ), // 1
  Mv(  0,  1 ), // 2
  Mv( -1, -1 ), // 5
  Mv(  1, -1 ), // 6
  Mv( -1,  0 ), // 3
  Mv(  1,  0 ), // 4
  Mv( -1,  1 ), // 7
  Mv(  1,  1 )  // 8
};

InterSearch::InterSearch()
  : m_modeCtrl(nullptr)
  , m_pSplitCS(nullptr)
  , m_pFullCS(nullptr)
  , m_pcEncCfg(nullptr)
  , m_encLibCommon(nullptr)
  , m_pcTrQuant(nullptr)
  , m_pcReshape(nullptr)
  , m_searchRange(0)
  , m_bipredSearchRange(0)
  , m_motionEstimationSearchMethod(MESearchMethod::FULL)
  , m_CABACEstimator(nullptr)
  , m_ctxPool(nullptr)
  , m_pTempPel(nullptr)
  , m_isInitialized(false)
{
  for (int i=0; i<MAX_NUM_REF_LIST_ADAPT_SR; i++)
  {
    memset(m_adaptSR[i], 0, MAX_IDX_ADAPT_SR * sizeof(int));
  }
  for (int i=0; i<AMVP_MAX_NUM_CANDS+1; i++)
  {
    memset (m_auiMVPIdxCost[i], 0, (AMVP_MAX_NUM_CANDS+1) * sizeof (uint32_t) );
  }

  setWpScalingDistParam( -1, REF_PIC_LIST_X, nullptr );
  m_affMVList = nullptr;
#if GDR_ENABLED
  m_affMVListSolid = nullptr;
#endif
  m_affMVListSize = 0;
  m_affMVListIdx = 0;
  m_uniMvList = nullptr;
  m_uniMvListSize = 0;
  m_uniMvListIdx = 0;
  m_histBestSbt    = MAX_UCHAR;
  m_histBestMtsIdx = MtsType::NONE;
}


void InterSearch::destroy()
{
  CHECK(!m_isInitialized, "Not initialized");
  if ( m_pTempPel )
  {
    delete [] m_pTempPel;
    m_pTempPel = nullptr;
  }

  m_pSplitCS = m_pFullCS = nullptr;

  m_pSaveCS = nullptr;

  for(uint32_t i = 0; i < NUM_REF_PIC_LIST_01; i++)
  {
    m_tmpPredStorage[i].destroy();
  }
  m_tmpStorageCtu.destroy();
  m_tmpAffiStorage.destroy();

  if (m_tmpAffiError != nullptr)
  {
    delete[] m_tmpAffiError;
  }
  if (m_tmpAffiDeri[0] != nullptr)
  {
    delete[] m_tmpAffiDeri[0];
  }
  if (m_tmpAffiDeri[1] != nullptr)
  {
    delete[] m_tmpAffiDeri[1];
  }
  if (m_affMVList)
  {
    delete[] m_affMVList;
    m_affMVList = nullptr;
  }
#if GDR_ENABLED
  if (m_affMVListSolid)
  {
    delete[] m_affMVListSolid;
    m_affMVListSolid = nullptr;
  }
#endif

  m_affMVListIdx = 0;
  m_affMVListSize = 0;
  if (m_uniMvList)
  {
    delete[] m_uniMvList;
    m_uniMvList = nullptr;
  }
  m_uniMvListIdx = 0;
  m_uniMvListSize = 0;
  m_isInitialized = false;
}

void InterSearch::setTempBuffers( CodingStructure ****pSplitCS, CodingStructure ****pFullCS, CodingStructure **pSaveCS )
{
  m_pSplitCS = pSplitCS;
  m_pFullCS  = pFullCS;
  m_pSaveCS  = pSaveCS;
}

InterSearch::~InterSearch()
{
  if (m_isInitialized)
  {
    destroy();
  }
}

void InterSearch::init(EncCfg *pcEncCfg, TrQuant *pcTrQuant, int searchRange, int bipredSearchRange,
                       MESearchMethod motionEstimationSearchMethod, bool useCompositeRef, const uint32_t maxCUWidth,
                       const uint32_t maxCUHeight, const uint32_t maxTotalCUDepth, RdCost *pcRdCost,
                       CABACWriter *CABACEstimator, CtxPool *ctxPool, EncReshape *pcReshape,
                       EncLibCommon *encLibCommon)
{
  CHECK(m_isInitialized, "Already initialized");
  m_defaultCachedBvs.clear();
  m_pcEncCfg                     = pcEncCfg;
  m_encLibCommon                 = encLibCommon;
  m_pcTrQuant                    = pcTrQuant;
  m_searchRange                  = searchRange;
  m_bipredSearchRange            = bipredSearchRange;
  m_motionEstimationSearchMethod = motionEstimationSearchMethod;
  m_CABACEstimator               = CABACEstimator;
  m_ctxPool                      = ctxPool;
  m_useCompositeRef              = useCompositeRef;
  m_pcReshape                    = pcReshape;

  for (uint32_t dir = 0; dir < MAX_NUM_REF_LIST_ADAPT_SR; dir++)
  {
    for (uint32_t refIdx = 0; refIdx < MAX_IDX_ADAPT_SR; refIdx++)
    {
      m_adaptSR[dir][refIdx] = searchRange;
    }
  }

  // initialize motion cost
  for (int num = 0; num < AMVP_MAX_NUM_CANDS + 1; num++)
  {
    for (int idx = 0; idx < AMVP_MAX_NUM_CANDS; idx++)
    {
      if (idx < num)
      {
        m_auiMVPIdxCost[idx][num] = xGetMvpIdxBits(idx, num);
      }
      else
      {
        m_auiMVPIdxCost[idx][num] = MAX_UINT;
      }
    }
  }

  const ChromaFormat cform = pcEncCfg->getChromaFormatIdc();
  InterPrediction::init( pcRdCost, cform, maxCUHeight );

  for( uint32_t i = 0; i < NUM_REF_PIC_LIST_01; i++ )
  {
    m_tmpPredStorage[i].create( UnitArea( cform, Area( 0, 0, MAX_CU_SIZE, MAX_CU_SIZE ) ) );
  }
  m_tmpStorageCtu.create(UnitArea(cform, Area(0, 0, MAX_CU_SIZE, MAX_CU_SIZE)));
  m_tmpAffiStorage.create( UnitArea( cform, Area( 0, 0, MAX_CU_SIZE, MAX_CU_SIZE ) ) );
  m_tmpAffiError = new Pel[MAX_CU_SIZE * MAX_CU_SIZE];
  m_tmpAffiDeri[0] = new int[MAX_CU_SIZE * MAX_CU_SIZE];
  m_tmpAffiDeri[1] = new int[MAX_CU_SIZE * MAX_CU_SIZE];
  m_pTempPel = new Pel[maxCUWidth*maxCUHeight];
  m_affMVListMaxSize = pcEncCfg->getIsLowDelay() ? AFFINE_ME_LIST_SIZE_LD : AFFINE_ME_LIST_SIZE;
  if (!m_affMVList)
  {
    m_affMVList = new AffineMVInfo[m_affMVListMaxSize];
#if GDR_ENABLED
    if (!m_affMVListSolid)
    {
      m_affMVListSolid = new AffineMVInfoSolid[m_affMVListMaxSize];
    }
#endif
  }
  m_affMVListIdx = 0;
  m_affMVListSize = 0;
  m_uniMvListMaxSize = 15;
  if (!m_uniMvList)
  {
    m_uniMvList = new BlkUniMvInfo[m_uniMvListMaxSize];
  }
  m_uniMvListIdx = 0;
  m_uniMvListSize = 0;
  m_isInitialized = true;
}

void InterSearch::resetSavedAffineMotion()
{
  for ( int i = 0; i < 2; i++ )
  {
    for ( int j = 0; j < 2; j++ )
    {
      m_affineMotion.acMvAffine4Para[i][j] = Mv( 0, 0 );
      m_affineMotion.acMvAffine6Para[i][j] = Mv( 0, 0 );
#if GDR_ENABLED
      m_affineMotion.acMvAffine4ParaSolid[i][j] = true;
      m_affineMotion.acMvAffine6ParaSolid[i][j] = true;
#endif
    }
    m_affineMotion.acMvAffine6Para[i][2] = Mv( 0, 0 );
#if GDR_ENABLED
    m_affineMotion.acMvAffine6ParaSolid[i][2] = true;
#endif

    m_affineMotion.affine4ParaRefIdx[i] = -1;
    m_affineMotion.affine6ParaRefIdx[i] = -1;
  }
  for ( int i = 0; i < 3; i++ )
  {
    m_affineMotion.hevcCost[i] = std::numeric_limits<Distortion>::max();
  }
  m_affineMotion.affine4ParaAvail = false;
  m_affineMotion.affine6ParaAvail = false;
}

#if GDR_ENABLED
void InterSearch::storeAffineMotion(Mv acAffineMv[2][3], bool acAffineMvSolid[2][3], int8_t affineRefIdx[2],
                                    AffineModel affineType, int bcwIdx)
#else
void InterSearch::storeAffineMotion(Mv acAffineMv[2][3], int8_t affineRefIdx[2], AffineModel affineType, int bcwIdx)
#endif
{
  if ((bcwIdx == BCW_DEFAULT || !m_affineMotion.affine6ParaAvail) && affineType == AffineModel::_6_PARAMS)
  {
    for ( int i = 0; i < 2; i++ )
    {
      for ( int j = 0; j < 3; j++ )
      {
        m_affineMotion.acMvAffine6Para[i][j] = acAffineMv[i][j];
#if GDR_ENABLED
        m_affineMotion.acMvAffine6ParaSolid[i][j] = acAffineMvSolid[i][j];
#endif
      }
      m_affineMotion.affine6ParaRefIdx[i] = affineRefIdx[i];
    }
    m_affineMotion.affine6ParaAvail = true;
  }

  if ((bcwIdx == BCW_DEFAULT || !m_affineMotion.affine4ParaAvail) && affineType == AffineModel::_4_PARAMS)
  {
    for ( int i = 0; i < 2; i++ )
    {
      for ( int j = 0; j < 2; j++ )
      {
        m_affineMotion.acMvAffine4Para[i][j] = acAffineMv[i][j];
#if GDR_ENABLED
        m_affineMotion.acMvAffine4ParaSolid[i][j] = acAffineMvSolid[i][j];
#endif
      }
      m_affineMotion.affine4ParaRefIdx[i] = affineRefIdx[i];
    }
    m_affineMotion.affine4ParaAvail = true;
  }
}

inline void InterSearch::xTZSearchHelp( IntTZSearchStruct& rcStruct, const int iSearchX, const int iSearchY, const uint8_t ucPointNr, const uint32_t uiDistance )
{
  Distortion  uiSad = 0;

//  CHECK(!( !( rcStruct.searchRange.left > iSearchX || rcStruct.searchRange.right < iSearchX || rcStruct.searchRange.top > iSearchY || rcStruct.searchRange.bottom < iSearchY )), "Unspecified error");

  const Pel* const  piRefSrch = rcStruct.piRefY + iSearchY * rcStruct.iRefStride + iSearchX;

  m_cDistParam.cur.buf = piRefSrch;

  if( 1 == rcStruct.subShiftMode )
  {
    // motion cost
    Distortion uiBitCost = m_pcRdCost->getCostOfVectorWithPredictor( iSearchX, iSearchY, rcStruct.imvShift );

    // Skip search if bit cost is already larger than best SAD
    if (uiBitCost < rcStruct.uiBestSad)
    {
      Distortion uiTempSad = m_cDistParam.distFunc( m_cDistParam );

      if((uiTempSad + uiBitCost) < rcStruct.uiBestSad)
      {
        // it's not supposed that any member of DistParams is manipulated beside cur.buf
        int subShift = m_cDistParam.subShift;
        const Pel* pOrgCpy = m_cDistParam.org.buf;
        uiSad += uiTempSad >> m_cDistParam.subShift;

        while( m_cDistParam.subShift > 0 )
        {
          int isubShift           = m_cDistParam.subShift -1;
          m_cDistParam.org.buf = rcStruct.pcPatternKey->buf + (rcStruct.pcPatternKey->stride << isubShift);
          m_cDistParam.cur.buf = piRefSrch + (rcStruct.iRefStride << isubShift);
          uiTempSad            = m_cDistParam.distFunc( m_cDistParam );
          uiSad               += uiTempSad >> m_cDistParam.subShift;

          if(((uiSad << isubShift) + uiBitCost) > rcStruct.uiBestSad)
          {
            break;
          }

          m_cDistParam.subShift--;
        }

        if(m_cDistParam.subShift == 0)
        {
          uiSad += uiBitCost;

          if( uiSad < rcStruct.uiBestSad )
          {
            rcStruct.uiBestSad      = uiSad;
            rcStruct.iBestX         = iSearchX;
            rcStruct.iBestY         = iSearchY;
            rcStruct.uiBestDistance = uiDistance;
            rcStruct.uiBestRound    = 0;
            rcStruct.ucPointNr      = ucPointNr;
            m_cDistParam.maximumDistortionForEarlyExit = uiSad;
          }
        }

        // restore org ptr
        m_cDistParam.org.buf  = pOrgCpy;
        m_cDistParam.subShift = subShift;
      }
    }
  }
  else
  {
    uiSad = m_cDistParam.distFunc( m_cDistParam );

    // only add motion cost if uiSad is smaller than best. Otherwise pointless
    // to add motion cost.
    if( uiSad < rcStruct.uiBestSad )
    {
      // motion cost
      uiSad += m_pcRdCost->getCostOfVectorWithPredictor( iSearchX, iSearchY, rcStruct.imvShift );

      if( uiSad < rcStruct.uiBestSad )
      {
        rcStruct.uiBestSad      = uiSad;
        rcStruct.iBestX         = iSearchX;
        rcStruct.iBestY         = iSearchY;
        rcStruct.uiBestDistance = uiDistance;
        rcStruct.uiBestRound    = 0;
        rcStruct.ucPointNr      = ucPointNr;
        m_cDistParam.maximumDistortionForEarlyExit = uiSad;
      }
    }
  }
}



inline void InterSearch::xTZ2PointSearch( IntTZSearchStruct& rcStruct )
{
  const SearchRange& sr = rcStruct.searchRange;

  static const int xOffset[2][9] = { {  0, -1, -1,  0, -1, +1, -1, -1, +1 }, {  0,  0, +1, +1, -1, +1,  0, +1,  0 } };
  static const int yOffset[2][9] = { {  0,  0, -1, -1, +1, -1,  0, +1,  0 }, {  0, -1, -1,  0, -1, +1, +1, +1, +1 } };

  // 2 point search,                   //   1 2 3
  // check only the 2 untested points  //   4 0 5
  // around the start point            //   6 7 8
  const int iX1 = rcStruct.iBestX + xOffset[0][rcStruct.ucPointNr];
  const int iX2 = rcStruct.iBestX + xOffset[1][rcStruct.ucPointNr];

  const int iY1 = rcStruct.iBestY + yOffset[0][rcStruct.ucPointNr];
  const int iY2 = rcStruct.iBestY + yOffset[1][rcStruct.ucPointNr];

  if( iX1 >= sr.left && iX1 <= sr.right && iY1 >= sr.top && iY1 <= sr.bottom )
  {
    xTZSearchHelp( rcStruct, iX1, iY1, 0, 2 );
  }

  if( iX2 >= sr.left && iX2 <= sr.right && iY2 >= sr.top && iY2 <= sr.bottom )
  {
    xTZSearchHelp( rcStruct, iX2, iY2, 0, 2 );
  }
}


inline void InterSearch::xTZ8PointSquareSearch( IntTZSearchStruct& rcStruct, const int iStartX, const int iStartY, const int iDist )
{
  const SearchRange& sr = rcStruct.searchRange;
  // 8 point search,                   //   1 2 3
  // search around the start point     //   4 0 5
  // with the required  distance       //   6 7 8
  CHECK( iDist == 0 , "Invalid distance");
  const int iTop        = iStartY - iDist;
  const int iBottom     = iStartY + iDist;
  const int iLeft       = iStartX - iDist;
  const int iRight      = iStartX + iDist;
  rcStruct.uiBestRound += 1;

  if ( iTop >= sr.top ) // check top
  {
    if ( iLeft >= sr.left ) // check top left
    {
      xTZSearchHelp( rcStruct, iLeft, iTop, 1, iDist );
    }
    // top middle
    xTZSearchHelp( rcStruct, iStartX, iTop, 2, iDist );

    if ( iRight <= sr.right ) // check top right
    {
      xTZSearchHelp( rcStruct, iRight, iTop, 3, iDist );
    }
  } // check top
  if ( iLeft >= sr.left ) // check middle left
  {
    xTZSearchHelp( rcStruct, iLeft, iStartY, 4, iDist );
  }
  if ( iRight <= sr.right ) // check middle right
  {
    xTZSearchHelp( rcStruct, iRight, iStartY, 5, iDist );
  }
  if ( iBottom <= sr.bottom ) // check bottom
  {
    if ( iLeft >= sr.left ) // check bottom left
    {
      xTZSearchHelp( rcStruct, iLeft, iBottom, 6, iDist );
    }
    // check bottom middle
    xTZSearchHelp( rcStruct, iStartX, iBottom, 7, iDist );

    if ( iRight <= sr.right ) // check bottom right
    {
      xTZSearchHelp( rcStruct, iRight, iBottom, 8, iDist );
    }
  } // check bottom
}

inline void InterSearch::xTZ8PointDiamondSearch( IntTZSearchStruct& rcStruct,
                                                 const int iStartX,
                                                 const int iStartY,
                                                 const int iDist,
                                                 const bool bCheckCornersAtDist1 )
{
  const SearchRange& sr = rcStruct.searchRange;
  // 8 point search,                   //   1 2 3
  // search around the start point     //   4 0 5
  // with the required  distance       //   6 7 8
  CHECK( iDist == 0, "Invalid distance" );
  const int iTop        = iStartY - iDist;
  const int iBottom     = iStartY + iDist;
  const int iLeft       = iStartX - iDist;
  const int iRight      = iStartX + iDist;
  rcStruct.uiBestRound += 1;

  if ( iDist == 1 )
  {
    if ( iTop >= sr.top ) // check top
    {
      if (bCheckCornersAtDist1)
      {
        if ( iLeft >= sr.left) // check top-left
        {
          xTZSearchHelp( rcStruct, iLeft, iTop, 1, iDist );
        }
        xTZSearchHelp( rcStruct, iStartX, iTop, 2, iDist );
        if ( iRight <= sr.right ) // check middle right
        {
          xTZSearchHelp( rcStruct, iRight, iTop, 3, iDist );
        }
      }
      else
      {
        xTZSearchHelp( rcStruct, iStartX, iTop, 2, iDist );
      }
    }
    if ( iLeft >= sr.left ) // check middle left
    {
      xTZSearchHelp( rcStruct, iLeft, iStartY, 4, iDist );
    }
    if ( iRight <= sr.right ) // check middle right
    {
      xTZSearchHelp( rcStruct, iRight, iStartY, 5, iDist );
    }
    if ( iBottom <= sr.bottom ) // check bottom
    {
      if (bCheckCornersAtDist1)
      {
        if ( iLeft >= sr.left) // check top-left
        {
          xTZSearchHelp( rcStruct, iLeft, iBottom, 6, iDist );
        }
        xTZSearchHelp( rcStruct, iStartX, iBottom, 7, iDist );
        if ( iRight <= sr.right ) // check middle right
        {
          xTZSearchHelp( rcStruct, iRight, iBottom, 8, iDist );
        }
      }
      else
      {
        xTZSearchHelp( rcStruct, iStartX, iBottom, 7, iDist );
      }
    }
  }
  else
  {
    if ( iDist <= 8 )
    {
      const int iTop_2      = iStartY - (iDist>>1);
      const int iBottom_2   = iStartY + (iDist>>1);
      const int iLeft_2     = iStartX - (iDist>>1);
      const int iRight_2    = iStartX + (iDist>>1);

      if (  iTop >= sr.top && iLeft >= sr.left &&
           iRight <= sr.right && iBottom <= sr.bottom ) // check border
      {
        xTZSearchHelp( rcStruct, iStartX,  iTop,      2, iDist    );
        xTZSearchHelp( rcStruct, iLeft_2,  iTop_2,    1, iDist>>1 );
        xTZSearchHelp( rcStruct, iRight_2, iTop_2,    3, iDist>>1 );
        xTZSearchHelp( rcStruct, iLeft,    iStartY,   4, iDist    );
        xTZSearchHelp( rcStruct, iRight,   iStartY,   5, iDist    );
        xTZSearchHelp( rcStruct, iLeft_2,  iBottom_2, 6, iDist>>1 );
        xTZSearchHelp( rcStruct, iRight_2, iBottom_2, 8, iDist>>1 );
        xTZSearchHelp( rcStruct, iStartX,  iBottom,   7, iDist    );
      }
      else // check border
      {
        if ( iTop >= sr.top ) // check top
        {
          xTZSearchHelp( rcStruct, iStartX, iTop, 2, iDist );
        }
        if ( iTop_2 >= sr.top ) // check half top
        {
          if ( iLeft_2 >= sr.left ) // check half left
          {
            xTZSearchHelp( rcStruct, iLeft_2, iTop_2, 1, (iDist>>1) );
          }
          if ( iRight_2 <= sr.right ) // check half right
          {
            xTZSearchHelp( rcStruct, iRight_2, iTop_2, 3, (iDist>>1) );
          }
        } // check half top
        if ( iLeft >= sr.left ) // check left
        {
          xTZSearchHelp( rcStruct, iLeft, iStartY, 4, iDist );
        }
        if ( iRight <= sr.right ) // check right
        {
          xTZSearchHelp( rcStruct, iRight, iStartY, 5, iDist );
        }
        if ( iBottom_2 <= sr.bottom ) // check half bottom
        {
          if ( iLeft_2 >= sr.left ) // check half left
          {
            xTZSearchHelp( rcStruct, iLeft_2, iBottom_2, 6, (iDist>>1) );
          }
          if ( iRight_2 <= sr.right ) // check half right
          {
            xTZSearchHelp( rcStruct, iRight_2, iBottom_2, 8, (iDist>>1) );
          }
        } // c…92257 tokens truncated…s now made. Fully encode the CU, including the headers:
  m_CABACEstimator->getCtx() = ctxStart;

  uint64_t finalFracBits = xGetSymbolFracBitsInter( cs, partitioner );
  // we've now encoded the CU, and so have a valid bit cost
  if (!cu.rootCbf)
  {
    if (luma)
    {
      cs.getResiBuf().bufs[0].fill(0); // Clear the residual image, if we didn't code it.
    }
    if (chroma && isChromaEnabled(cs.pcv->chrFormat))
    {
      cs.getResiBuf().bufs[1].fill(0); // Clear the residual image, if we didn't code it.
      cs.getResiBuf().bufs[2].fill(0); // Clear the residual image, if we didn't code it.
    }
  }

  if (luma)
  {
    if (cu.rootCbf && cs.slice->getLmcsEnabledFlag() && m_pcReshape->getCTUFlag())
    {
      const CompArea &areaY = cu.Y();
      CompArea      tmpArea(COMPONENT_Y, areaY.chromaFormat, Position(0, 0), areaY.size());
      PelBuf          tmpPred = m_tmpStorageCtu.getBuf(tmpArea);
      tmpPred.copyFrom(cs.getPredBuf(COMPONENT_Y));

      if (!cu.firstPU->ciipFlag && !CU::isIBC(cu))
      {
        tmpPred.rspSignal(m_pcReshape->getFwdLUT());
      }

      cs.getRecoBuf(COMPONENT_Y).reconstruct(tmpPred, cs.getResiBuf(COMPONENT_Y), cs.slice->clpRng(COMPONENT_Y));
    }
    else
    {
      cs.getRecoBuf().bufs[0].reconstruct(cs.getPredBuf().bufs[0], cs.getResiBuf().bufs[0], cs.slice->clpRngs().comp[0]);
      if (cs.slice->getLmcsEnabledFlag() && m_pcReshape->getCTUFlag() && !cu.firstPU->ciipFlag && !CU::isIBC(cu))
      {
        cs.getRecoBuf().bufs[0].rspSignal(m_pcReshape->getFwdLUT());
      }
    }
  }
  if (chroma && isChromaEnabled(cs.pcv->chrFormat))
  {
    cs.getRecoBuf().bufs[1].reconstruct(cs.getPredBuf().bufs[1], cs.getResiBuf().bufs[1], cs.slice->clpRngs().comp[1]);
    cs.getRecoBuf().bufs[2].reconstruct(cs.getPredBuf().bufs[2], cs.getResiBuf().bufs[2], cs.slice->clpRngs().comp[2]);
  }

  // update with clipped distortion and cost (previously unclipped reconstruction values were used)
  Distortion finalDistortion = 0;

  for (int comp = 0; comp < numValidComponents; comp++)
  {
    const ComponentID compID = ComponentID(comp);
    if (compID == COMPONENT_Y && !luma)
    {
      continue;
    }
    if (compID != COMPONENT_Y && !chroma)
    {
      continue;
    }
    CPelBuf reco = cs.getRecoBuf (compID);
    CPelBuf org  = cs.getOrgBuf  (compID);

#if WCG_EXT
    if (m_pcEncCfg->getLumaLevelToDeltaQPMapping().isEnabled() || (
      m_pcEncCfg->getLmcs() && (cs.slice->getLmcsEnabledFlag() && m_pcReshape->getCTUFlag())))
    {
      const CPelBuf orgLuma = cs.getOrgBuf( cs.area.blocks[COMPONENT_Y] );
      if (compID == COMPONENT_Y && !(m_pcEncCfg->getLumaLevelToDeltaQPMapping().isEnabled()) )
      {
        const CompArea &areaY = cu.Y();

        CompArea tmpArea1(COMPONENT_Y, areaY.chromaFormat, Position(0, 0), areaY.size());
        PelBuf   tmpRecLuma = m_tmpStorageCtu.getBuf(tmpArea1);
        tmpRecLuma.copyFrom(reco);
        tmpRecLuma.rspSignal(m_pcReshape->getInvLUT());
        finalDistortion += m_pcRdCost->getDistPart(org, tmpRecLuma, sps.getBitDepth(toChannelType(compID)), compID,
                                                   DFuncWtd::SSE_WTD, orgLuma);
      }
      else
      {
        finalDistortion +=
          m_pcRdCost->getDistPart(org, reco, sps.getBitDepth(toChannelType(compID)), compID, DFuncWtd::SSE_WTD, orgLuma);
      }
    }
    else
#endif
    {
      finalDistortion += m_pcRdCost->getDistPart(org, reco, sps.getBitDepth(toChannelType(compID)), compID, DFunc::SSE);
    }
  }

  cs.dist     = finalDistortion;
  cs.fracBits = finalFracBits;
  cs.cost     = m_pcRdCost->calcRdCost(cs.fracBits, cs.dist);
  if (cs.slice->getSPS()->getUseColorTrans())
  {
    if (cs.cost < cs.tmpColorSpaceCost)
    {
      cs.tmpColorSpaceCost = cs.cost;
      if (m_pcEncCfg->getRGBFormatFlag())
      {
        cs.firstColorSpaceSelected = cu.colorTransform || !cu.rootCbf;
      }
      else
      {
        cs.firstColorSpaceSelected = !cu.colorTransform || !cu.rootCbf;
      }
    }
  }

  CHECK(cs.tus.size() == 0, "No TUs present");
}

uint64_t InterSearch::xGetSymbolFracBitsInter(CodingStructure &cs, Partitioner &partitioner)
{
  uint64_t fracBits   = 0;
  CodingUnit &cu    = *cs.getCU( partitioner.chType );

  m_CABACEstimator->resetBits();

  if( cu.firstPU->mergeFlag && !cu.rootCbf )
  {
    cu.skip = true;
    CHECK(cu.colorTransform, "ACT should not be enabled for skip mode");
    m_CABACEstimator->cu_skip_flag  ( cu );
    if (cu.firstPU->ciipFlag)
    {
      // CIIP shouldn't be skip, the upper level function will deal with it, i.e. setting the overall cost to MAX_DOUBLE
    }
    else
    {
      m_CABACEstimator->merge_data(*cu.firstPU);
    }
    fracBits   += m_CABACEstimator->getEstFracBits();
  }
  else
  {
    CHECK( cu.skip, "Skip flag has to be off at this point!" );

    if (cu.Y().valid())
    m_CABACEstimator->cu_skip_flag( cu );
    m_CABACEstimator->pred_mode   ( cu );
    m_CABACEstimator->cu_pred_data( cu );
    CUCtx cuCtx;
    cuCtx.isDQPCoded = true;
    cuCtx.isChromaQpAdjCoded = true;
    m_CABACEstimator->cu_residual ( cu, partitioner, cuCtx );
    fracBits       += m_CABACEstimator->getEstFracBits();
  }

  return fracBits;
}

double InterSearch::xGetMEDistortionWeight(uint8_t bcwIdx, RefPicList eRefPicList)
{
  if( bcwIdx != BCW_DEFAULT )
  {
    return (double) abs(getBcwWeight(bcwIdx, eRefPicList)) / BCW_WEIGHT_BASE;
  }
  else
  {
    return 0.5;
  }
}

#if GDR_ENABLED
bool InterSearch::xReadBufferedUniMv(PredictionUnit &pu, RefPicList eRefPicList, int32_t refIdx, Mv &pcMvPred, Mv &rcMv,
                                     bool &rcMvSolid, uint32_t &ruiBits, Distortion &ruiCost)
#else
bool InterSearch::xReadBufferedUniMv(PredictionUnit &pu, RefPicList eRefPicList, int32_t refIdx, Mv &pcMvPred, Mv &rcMv,
                                     uint32_t &ruiBits, Distortion &ruiCost)
#endif
{
  if (m_uniMotions.isReadMode((uint32_t) eRefPicList, (uint32_t) refIdx))
  {
#if GDR_ENABLED
    m_uniMotions.copyTo(rcMv, rcMvSolid, ruiCost, (uint32_t) eRefPicList, (uint32_t) refIdx);
#else
    m_uniMotions.copyTo(rcMv, ruiCost, (uint32_t) eRefPicList, (uint32_t) refIdx);
#endif

    Mv pred = pcMvPred;
    pred.changeTransPrecInternal2Amvr(pu.cu->imv);
    m_pcRdCost->setPredictor(pred);
    m_pcRdCost->setCostScale(0);

    Mv mv = rcMv;
    mv.changeTransPrecInternal2Amvr(pu.cu->imv);
    uint32_t mvBits = m_pcRdCost->getBitsOfVectorWithPredictor(mv.getHor(), mv.getVer(), 0);

    ruiBits += mvBits;
    ruiCost += m_pcRdCost->getCost(ruiBits);
    return true;
  }
  return false;
}

#if GDR_ENABLED
bool InterSearch::xReadBufferedAffineUniMv(PredictionUnit &pu, RefPicList eRefPicList, int32_t refIdx, Mv acMvPred[3],
                                           Mv acMv[3], bool acMvSolid[3], uint32_t &ruiBits, Distortion &ruiCost,
                                           int &mvpIdx, const AffineAMVPInfo &aamvpi)
#else
bool InterSearch::xReadBufferedAffineUniMv(PredictionUnit &pu, RefPicList eRefPicList, int32_t refIdx, Mv acMvPred[3],
                                           Mv acMv[3], uint32_t &ruiBits, Distortion &ruiCost, int &mvpIdx,
                                           const AffineAMVPInfo &aamvpi)
#endif
{
  if (m_uniMotions.isReadModeAffine((uint32_t) eRefPicList, (uint32_t) refIdx, pu.cu->affineType))
  {
#if GDR_ENABLED
    m_uniMotions.copyAffineMvTo(acMv, acMvSolid, ruiCost, (uint32_t) eRefPicList, (uint32_t) refIdx, pu.cu->affineType,
                                mvpIdx);
#else
    m_uniMotions.copyAffineMvTo(acMv, ruiCost, (uint32_t) eRefPicList, (uint32_t) refIdx, pu.cu->affineType, mvpIdx);
#endif
    m_pcRdCost->setCostScale(0);
    acMvPred[0] = aamvpi.mvCandLT[mvpIdx];
    acMvPred[1] = aamvpi.mvCandRT[mvpIdx];
    acMvPred[2] = aamvpi.mvCandLB[mvpIdx];

    uint32_t mvBits = 0;
    for (int verIdx = 0; verIdx < pu.cu->getNumAffineMvs(); verIdx++)
    {
      Mv pred = verIdx ? acMvPred[verIdx] + acMv[0] - acMvPred[0] : acMvPred[verIdx];
      pred.changePrecision(MvPrecision::INTERNAL, MvPrecision::QUARTER);
      m_pcRdCost->setPredictor(pred);
      Mv mv = acMv[verIdx];
      mv.changePrecision(MvPrecision::INTERNAL, MvPrecision::QUARTER);
      mvBits += m_pcRdCost->getBitsOfVectorWithPredictor(mv.getHor(), mv.getVer(), 0);
    }
    ruiBits += mvBits;
    ruiCost += m_pcRdCost->getCost(ruiBits);
    return true;
  }
  return false;
}

void InterSearch::initWeightIdxBits()
{
  for (int n = 0; n < BCW_NUM; ++n)
  {
    m_estWeightIdxBits[n] = deriveWeightIdxBits(n);
  }
}

void InterSearch::xClipMv( Mv& rcMv, const Position& pos, const struct Size& size, const SPS& sps, const PPS& pps )
{
  int mvShift = MV_FRACTIONAL_BITS_INTERNAL;
  int offset = 8;

  int horMax = ( pps.getPicWidthInLumaSamples() + offset - (int)pos.x - 1 ) << mvShift;
  int horMin = (-(int) sps.getMaxCUWidth() - offset - (int) pos.x + 1) * (1 << mvShift);

  int verMax = ( pps.getPicHeightInLumaSamples() + offset - (int)pos.y - 1 ) << mvShift;
  int verMin = (-(int) sps.getMaxCUHeight() - offset - (int) pos.y + 1) * (1 << mvShift);

  const SubPic &curSubPic = pps.getSubPicFromPos(pos);
  if (curSubPic.getTreatedAsPicFlag() && m_clipMvInSubPic)
  {
    horMax = ((curSubPic.getSubPicRight() + 1)  + offset - (int)pos.x - 1) << mvShift;
    horMin = (-(int) sps.getMaxCUWidth() - offset - ((int) pos.x - curSubPic.getSubPicLeft()) + 1) * (1 << mvShift);

    verMax = ((curSubPic.getSubPicBottom() + 1) + offset -  (int)pos.y - 1) << mvShift;
    verMin = (-(int) sps.getMaxCUHeight() - offset - ((int) pos.y - curSubPic.getSubPicTop()) + 1) * (1 << mvShift);
  }
  if( pps.getWrapAroundEnabledFlag() )
  {
    int horMax = ( pps.getPicWidthInLumaSamples() + sps.getMaxCUWidth() - size.width + offset - (int)pos.x - 1 ) << mvShift;
    int horMin = (-(int) sps.getMaxCUWidth() - offset - (int) pos.x + 1) * (1 << mvShift);
    rcMv.setHor( std::min( horMax, std::max( horMin, rcMv.getHor() ) ) );
    rcMv.setVer( std::min( verMax, std::max( verMin, rcMv.getVer() ) ) );
    return;
  }

  rcMv.setHor( std::min( horMax, std::max( horMin, rcMv.getHor() ) ) );
  rcMv.setVer( std::min( verMax, std::max( verMin, rcMv.getVer() ) ) );
}

uint32_t InterSearch::xDetermineBestMvp( PredictionUnit& pu, Mv acMvTemp[3], int& mvpIdx, const AffineAMVPInfo& aamvpi )
{
  bool mvpUpdated  = false;
  uint32_t minBits = std::numeric_limits<uint32_t>::max();
#if GDR_ENABLED
  const CodingStructure &cs = *pu.cs;
  const bool             isEncodeGdrClean =
    cs.sps->getGDREnabledFlag() && cs.pcv->isEncoder
    && ((cs.picture->gdrParam.inGdrInterval && cs.isClean(pu.Y().topRight(), ChannelType::LUMA))
        || (cs.picture->gdrParam.verBoundary == -1));
#endif

  for ( int i = 0; i < aamvpi.numCand; i++ )
  {
    Mv mvPred[3] = { aamvpi.mvCandLT[i], aamvpi.mvCandRT[i], aamvpi.mvCandLB[i] };
    uint32_t candBits = m_auiMVPIdxCost[i][aamvpi.numCand];
    candBits += xCalcAffineMVBits( pu, acMvTemp, mvPred );

#if GDR_ENABLED
    bool isSolid = true;
    if (isEncodeGdrClean)
    {
      isSolid = aamvpi.mvSolidLT[i] && aamvpi.mvSolidRT[i];
      if (pu.cu->affineType == AffineModel::_6_PARAMS)
      {
        isSolid = isSolid && aamvpi.mvSolidLB[i];
      }
    }

    if ((candBits < minBits) && isSolid)
#else
    if ( candBits < minBits )
#endif
    {
      minBits    = candBits;
      mvpIdx     = i;
      mvpUpdated = true;
    }
  }

#if GDR_ENABLED
  mvpUpdated = true; // do not check mvp update for GDR
#endif

  CHECK( !mvpUpdated, "xDetermineBestMvp() error" );

  return minBits;
}

void InterSearch::symmvdCheckBestMvp(PredictionUnit &pu, PelUnitBuf &origBuf, Mv curMv, RefPicList curRefList,
                                     RefSetArray<AMVPInfo> &amvpInfo, int32_t bcwIdx,
                                     Mv cMvPredSym[NUM_REF_PIC_LIST_01],
#if GDR_ENABLED
                                     bool cMvPredSymSolid[NUM_REF_PIC_LIST_01],
#endif
                                     int32_t mvpIdxSym[NUM_REF_PIC_LIST_01], Distortion &bestCost, bool skip)
{
#if GDR_ENABLED
  CodingStructure &cs = *pu.cs;
  const bool       isEncodeGdrClean =
    cs.sps->getGDREnabledFlag() && cs.pcv->isEncoder
    && ((cs.picture->gdrParam.inGdrInterval && cs.isClean(pu.Y().topRight(), ChannelType::LUMA))
        || (cs.picture->gdrParam.verBoundary == -1));
  bool bestCostOk = true;
  bool costOk = true;
  bool allOk;
#endif

  RefPicList tarRefList = (RefPicList)(1 - curRefList);
  int32_t refIdxCur = pu.cu->slice->getSymRefIdx(curRefList);
  int32_t refIdxTar = pu.cu->slice->getSymRefIdx(tarRefList);

  MvField cCurMvField, cTarMvField;
  cCurMvField.setMvField(curMv, refIdxCur);
  AMVPInfo& amvpCur = amvpInfo[curRefList][refIdxCur];
  AMVPInfo& amvpTar = amvpInfo[tarRefList][refIdxTar];
  m_pcRdCost->setCostScale(0);


  // get prediction of eCurRefPicList
  PelUnitBuf predBufA = m_tmpPredStorage[curRefList].getBuf(UnitAreaRelative(*pu.cu, pu));
  const Picture* picRefA = pu.cu->slice->getRefPic(curRefList, cCurMvField.refIdx);
  Mv mvA = cCurMvField.mv;
  clipMv( mvA, pu.cu->lumaPos(), pu.cu->lumaSize(), *pu.cs->sps, *pu.cs->pps );
  if ( (mvA.hor & 15) == 0 && (mvA.ver & 15) == 0 )
  {
    Position offset = pu.blocks[COMPONENT_Y].pos().offset( mvA.getHor() >> 4, mvA.getVer() >> 4 );
    CPelBuf pelBufA = picRefA->getRecoBuf( CompArea( COMPONENT_Y, pu.chromaFormat, offset, pu.blocks[COMPONENT_Y].size() ), false );
    predBufA.bufs[0].buf = const_cast<Pel *>(pelBufA.buf);
    predBufA.bufs[0].stride = pelBufA.stride;
  }
  else
  {
    xPredInterBlk(COMPONENT_Y, pu, picRefA, mvA, predBufA, false, pu.cu->slice->clpRng(COMPONENT_Y), false, false,
                  curRefList);
  }
  PelUnitBuf bufTmp = m_tmpStorageCtu.getBuf(UnitAreaRelative(*pu.cu, pu));
  bufTmp.copyFrom( origBuf );
  bufTmp.removeHighFreq(predBufA, m_pcEncCfg->getClipForBiPredMeEnabled(), pu.cu->slice->clpRngs(),
                        getBcwWeight(pu.cu->bcwIdx, tarRefList));

  double fWeight = xGetMEDistortionWeight(pu.cu->bcwIdx, tarRefList);

  int32_t skipMvpIdx[2];
  skipMvpIdx[0] = skip ? mvpIdxSym[0] : -1;
  skipMvpIdx[1] = skip ? mvpIdxSym[1] : -1;

  for (int i = 0; i < amvpCur.numCand; i++)
  {
    for (int j = 0; j < amvpTar.numCand; j++)
    {
      if (skipMvpIdx[curRefList] == i && skipMvpIdx[tarRefList] == j)
      {
        continue;
      }

      cTarMvField.setMvField(curMv.getSymmvdMv(amvpCur.mvCand[i], amvpTar.mvCand[j]), refIdxTar);

      // get prediction of eTarRefPicList
      PelUnitBuf predBufB = m_tmpPredStorage[tarRefList].getBuf(UnitAreaRelative(*pu.cu, pu));
      const Picture* picRefB = pu.cu->slice->getRefPic(tarRefList, cTarMvField.refIdx);
      Mv mvB = cTarMvField.mv;
      clipMv( mvB, pu.cu->lumaPos(), pu.cu->lumaSize(), *pu.cs->sps, *pu.cs->pps );
      if ( (mvB.hor & 15) == 0 && (mvB.ver & 15) == 0 )
      {
        Position offset = pu.blocks[COMPONENT_Y].pos().offset( mvB.getHor() >> 4, mvB.getVer() >> 4 );
        CPelBuf pelBufB = picRefB->getRecoBuf( CompArea( COMPONENT_Y, pu.chromaFormat, offset, pu.blocks[COMPONENT_Y].size() ), false );
        predBufB.bufs[0].buf = const_cast<Pel *>(pelBufB.buf);
        predBufB.bufs[0].stride = pelBufB.stride;
      }
      else
      {
        xPredInterBlk(COMPONENT_Y, pu, picRefB, mvB, predBufB, false, pu.cu->slice->clpRng(COMPONENT_Y), false, false,
                      tarRefList);
      }
      // calc distortion
      const DFunc distFunc = (!pu.cu->slice->getDisableSATDForRD()) ? DFunc::HAD : DFunc::SAD;
      Distortion cost     = (Distortion) floor(
            fWeight
            * (double) m_pcRdCost->getDistPart(bufTmp.Y(), predBufB.Y(), pu.cs->sps->getBitDepth(ChannelType::LUMA),
                                               COMPONENT_Y, distFunc));

      Mv pred = amvpCur.mvCand[i];
      pred.changeTransPrecInternal2Amvr(pu.cu->imv);
      m_pcRdCost->setPredictor(pred);
      Mv mv = curMv;
      mv.changeTransPrecInternal2Amvr(pu.cu->imv);
      uint32_t bits = m_pcRdCost->getBitsOfVectorWithPredictor(mv.hor, mv.ver, 0);
      bits += m_auiMVPIdxCost[i][AMVP_MAX_NUM_CANDS];
      bits += m_auiMVPIdxCost[j][AMVP_MAX_NUM_CANDS];
      cost += m_pcRdCost->getCost(bits);
#if GDR_ENABLED
      if (isEncodeGdrClean)
      {
        bool curSolid = amvpCur.mvSolid[i];
        bool tarSolid = amvpTar.mvSolid[j];
        costOk = curSolid && tarSolid;
      }
#endif


#if GDR_ENABLED
      allOk = (cost < bestCost);
      if (isEncodeGdrClean)
      {
        if (costOk)
        {
          allOk = (bestCostOk) ? (cost < bestCost) : true;
        }
        else
        {
          allOk = false;
        }
      }
#endif

#if GDR_ENABLED
      if (allOk)
#else
      if (cost < bestCost)
#endif
      {
        bestCost = cost;
        cMvPredSym[curRefList] = amvpCur.mvCand[i];
        cMvPredSym[tarRefList] = amvpTar.mvCand[j];
#if GDR_ENABLED
        if (isEncodeGdrClean)
        {
          bestCostOk = costOk;
          cMvPredSymSolid[curRefList] = amvpCur.mvSolid[i];
          cMvPredSymSolid[tarRefList] = amvpTar.mvSolid[j];
        }
#endif
        mvpIdxSym[curRefList] = i;
        mvpIdxSym[tarRefList] = j;
      }
    }
  }
}

uint64_t InterSearch::xCalcPuMeBits(PredictionUnit& pu)
{
  assert(pu.mergeFlag);
  assert(!CU::isIBC(*pu.cu));
  m_CABACEstimator->resetBits();
  m_CABACEstimator->merge_flag(pu);
  if (pu.mergeFlag)
  {
    m_CABACEstimator->merge_data(pu);
  }
  return m_CABACEstimator->getEstFracBits();
}

bool InterSearch::isValidBv(PredictionUnit& pu, int xPos, int yPos, int width, int height, int picWidth, int picHeight,
                            int xBv, int yBv, int ctuSize)
{
  const int refRightX  = xPos + xBv + width - 1;
  const int refBottomY = yPos + yBv + height - 1;

  // check whether bottom-right sample is definitely not yet reconstructed
  if (refRightX >= xPos && refBottomY >= yPos)
  {
    return false;
  }

  const int ctuSizeLog2 = floorLog2(ctuSize);

  const int refLeftX = xPos + xBv;
  const int refTopY  = yPos + yBv;

  const int curCtuCol = xPos >> ctuSizeLog2;
  const int curCtuRow = yPos >> ctuSizeLog2;

  const int refTopCtuRow    = refTopY >> ctuSizeLog2;
  const int refBottomCtuRow = refBottomY >> ctuSizeLog2;
  const int refLeftCtuCol   = refLeftX >> ctuSizeLog2;
  const int refRightCtuCol  = refRightX >> ctuSizeLog2;

  // check whether top or bottom is in different CTU row
  if (curCtuRow != refTopCtuRow || curCtuRow != refBottomCtuRow)
  {
    return false;
  }

  // number of CTUs to the left that may be referenced. When CTU size is 128x128, this includes a CTU to the
  // left that may be partially referenced
  constexpr int IBC_REF_WINDOW_SIZE = 1 << (2 * 7);
  const int     numLeftCTUs = std::min((IBC_REF_WINDOW_SIZE >> 2 * ctuSizeLog2) - (ctuSizeLog2 < 7 ? 1 : 0), curCtuCol);
  if (refRightCtuCol > curCtuCol || refLeftCtuCol < curCtuCol - numLeftCTUs)
  {
    return false;
  }

  // check whether in same tile
  if (refLeftCtuCol != curCtuCol)
  {
    const TileIdx curTileIdx = pu.cs->pps->getTileIdx(curCtuCol, curCtuRow);
    const TileIdx refTileIdx = pu.cs->pps->getTileIdx(refLeftCtuCol, curCtuRow);
    if (curTileIdx != refTileIdx)
    {
      return false;
    }
  }

  // if part of ref block is in the left CTU, some area can be referred from the not-yet updated local CTU buffer
  if (ctuSizeLog2 == 7 && refLeftCtuCol == curCtuCol - 1)
  {
    // ref block's collocated block in current CTU
    const Position refPosCol      = pu.Y().topLeft().offset(xBv + ctuSize, yBv);
    const Position refPosCol64x64 = { refPosCol.x & ~63, refPosCol.y & ~63 };
    if (pu.cs->isDecomp(refPosCol64x64, ChannelType::LUMA))
    {
      return false;
    }
    if (refPosCol64x64 == pu.Y().topLeft())
    {
      return false;
    }
  }

  // in the same CTU, or valid area from left CTU. Check if the reference block is already coded
  const Position refPosBR = pu.Y().bottomRight().offset(xBv, yBv);
  return pu.cs->isDecomp(refPosBR, ChannelType::LUMA);
}

//! \}
