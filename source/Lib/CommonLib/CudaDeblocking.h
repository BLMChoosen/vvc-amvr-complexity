/* The copyright in this software is being made available under the BSD
 * License. See COPYING for details. */

#ifndef VTM_CUDA_DEBLOCKING_H
#define VTM_CUDA_DEBLOCKING_H

#include <cstdint>
#include <type_traits>

namespace vtm
{

constexpr std::uint64_t CUDA_DBF_MIN_FRAME_PIXELS = UINT64_C(1920) * 1080;
constexpr std::uint32_t CUDA_DBF_MAX_TASKS = 4 * 1024 * 1024;

enum class CudaDbfDispatchResult : std::uint8_t
{
  NotEligible,
  NoOp,
  Executed
};

enum class CudaDbfTestFailurePoint : std::uint8_t
{
  None,
  Allocation,
  ParameterUpload,
  SnapshotCopy,
  VerticalLaunch,
  VerticalCompletion,
  HorizontalLaunch,
  HorizontalCompletion,
  CommitCopy,
  CommitCompletion
};

// One normative four-sample luma edge segment. CPU code derives all syntax-
// dependent state. CUDA only performs the sample-dependent decisions and filter arithmetic.
struct CudaDbfLumaTask
{
  std::uint32_t x;
  std::uint32_t y;
  std::int32_t  tc;
  std::int32_t  beta;
  std::int32_t  minSample;
  std::int32_t  maxSample;
  std::uint8_t  direction;       // 0 vertical, 1 horizontal
  std::uint8_t  maxFilterLenP;   // normative length: 1, 2, 3, 5, or 7
  std::uint8_t  maxFilterLenQ;
  std::uint8_t  flags;
};

enum CudaDbfTaskFlag : std::uint8_t
{
  CUDA_DBF_SIDE_P_LARGE = 1 << 0,
  CUDA_DBF_SIDE_Q_LARGE = 1 << 1,
  CUDA_DBF_PART_P_NO_FILTER = 1 << 2,
  CUDA_DBF_PART_Q_NO_FILTER = 1 << 3
};

struct CudaDbfFrame
{
  std::uint32_t width;
  std::uint32_t height;
  std::uint8_t  bitDepth;
  std::uint8_t  elementSize;
  std::uint8_t  reserved[2];
};

static_assert(std::is_standard_layout<CudaDbfLumaTask>::value && std::is_trivial<CudaDbfLumaTask>::value,
              "CUDA DBF task must remain POD");
static_assert(std::is_standard_layout<CudaDbfFrame>::value && std::is_trivial<CudaDbfFrame>::value,
              "CUDA DBF frame must remain POD");

} // namespace vtm

#endif // VTM_CUDA_DEBLOCKING_H
