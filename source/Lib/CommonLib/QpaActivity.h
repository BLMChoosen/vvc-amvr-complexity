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

#ifndef VTM_QPA_ACTIVITY_H
#define VTM_QPA_ACTIVITY_H

#include <cstddef>
#include <cstdint>

namespace vtm
{

struct QpaActivityArea
{
  std::uint32_t x = 0;
  std::uint32_t y = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
};

struct QpaActivitySums
{
  std::uint64_t highpass = 0;
  std::int64_t  luma = 0;
};

template<typename Sample>
inline QpaActivitySums computeQpaActivitySums(const Sample *luma, const std::ptrdiff_t strideSamples,
                                              const QpaActivityArea &filterArea,
                                              const QpaActivityArea &lumaArea)
{
  QpaActivitySums sums{};
  const Sample *filterBase = luma + static_cast<std::ptrdiff_t>(filterArea.y) * strideSamples + filterArea.x;
  for (std::uint32_t y = 1; y + 1 < filterArea.height; ++y)
  {
    const Sample *row = filterBase + static_cast<std::ptrdiff_t>(y) * strideSamples;
    const Sample *previousRow = row - strideSamples;
    const Sample *nextRow = row + strideSamples;
    for (std::uint32_t x = 1; x + 1 < filterArea.width; ++x)
    {
      const int f = 12 * int(row[x])
                    - 2 * (int(row[x - 1]) + int(row[x + 1]) + int(previousRow[x]) + int(nextRow[x]))
                    - int(previousRow[x - 1]) - int(previousRow[x + 1])
                    - int(nextRow[x - 1]) - int(nextRow[x + 1]);
      sums.highpass += static_cast<std::uint64_t>(f < 0 ? -f : f);
    }
  }

  const Sample *areaBase = luma + static_cast<std::ptrdiff_t>(lumaArea.y) * strideSamples + lumaArea.x;
  for (std::uint32_t y = 0; y < lumaArea.height; ++y)
  {
    const Sample *row = areaBase + static_cast<std::ptrdiff_t>(y) * strideSamples;
    for (std::uint32_t x = 0; x < lumaArea.width; ++x)
    {
      sums.luma += row[x];
    }
  }
  return sums;
}

}   // namespace vtm

#endif   // VTM_QPA_ACTIVITY_H
