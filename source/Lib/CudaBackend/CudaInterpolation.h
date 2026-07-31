/* The copyright in this software is being made available under the BSD
 * License. See the copyright notice in the source tree for details.
 */

#ifndef VTM_CUDA_INTERPOLATION_H
#define VTM_CUDA_INTERPOLATION_H

#include "CudaPictureMirror.h"

#include <cstdint>
#include <type_traits>

namespace vtm
{

enum class CudaFractionalStage : std::uint8_t
{
  Half,
  Quarter
};

constexpr std::uint32_t CUDA_FRACTIONAL_CANDIDATE_COUNT = 9;

// The reference pointer denotes the integer-pel centre used by InterSearch. The source pointer denotes the
// original block. Both pointers must belong to the registered luma planes; the picture mirrors remain read-only.
struct CudaFractionalSadBatchDesc
{
  CudaMirrorHandle     sourceMirror;
  CudaMirrorHandle     referenceMirror;
  const void          *source;
  const void          *reference;
  std::uint32_t        width;
  std::uint32_t        height;
  std::uint8_t         elementSize;
  std::uint8_t         bitDepth;
  CudaFractionalStage  stage;
  std::int8_t          centreHorQuarter;
  std::int8_t          centreVerQuarter;
  bool                 useAltHalfFilter;
};

static_assert(std::is_standard_layout<CudaFractionalSadBatchDesc>::value
                && std::is_trivial<CudaFractionalSadBatchDesc>::value,
              "CUDA fractional SAD descriptor must remain POD");

}   // namespace vtm

#endif   // VTM_CUDA_INTERPOLATION_H
