/* The copyright in this software is being made available under the BSD License. */
#ifndef VTM_CUDA_DECODER_TRANSFORM_H
#define VTM_CUDA_DECODER_TRANSFORM_H

#include <cstdint>
#include <type_traits>

namespace vtm
{

constexpr std::uint32_t CUDA_MAX_DECODER_TRANSFORM_TASKS = 4096;
constexpr std::uint32_t CUDA_MAX_DECODER_TRANSFORM_SAMPLES = CUDA_MAX_DECODER_TRANSFORM_TASKS * 32 * 32;

enum class CudaDecoderTransformMode : std::uint8_t
{
  Dct2 = 0,
  TransformSkip = 1
};

// Address-free descriptor: offsets address the packed arrays in CudaDecoderTransformBatch.
struct CudaDecoderTransformTask
{
  std::uint64_t ticket = 0;
  std::uint32_t coefficientOffset = 0;
  std::uint32_t predictionOffset = 0;
  std::uint32_t outputOffset = 0;
  std::uint32_t scanOffset = 0;
  std::uint32_t horizontalMatrixOffset = 0;
  std::uint32_t verticalMatrixOffset = 0;
  std::uint16_t width = 0;
  std::uint16_t height = 0;
  CudaDecoderTransformMode mode = CudaDecoderTransformMode::Dct2;
  std::uint8_t dependentQuant = 0;
  std::int8_t dequantShift = 0;
  std::uint8_t firstTransformShift = 0;
  std::uint8_t secondTransformShift = 0;
  std::uint8_t reserved = 0;
  std::int32_t inverseQuantScale = 0;
  std::int32_t coefficientMinimum = 0;
  std::int32_t coefficientMaximum = 0;
  std::int32_t residualMinimum = 0;
  std::int32_t residualMaximum = 0;
  std::int32_t sampleMinimum = 0;
  std::int32_t sampleMaximum = 0;
};

struct CudaDecoderTransformBatch
{
  const CudaDecoderTransformTask *tasks = nullptr;
  std::uint32_t taskCount = 0;
  const std::int32_t *quantizedCoefficients = nullptr;
  std::uint32_t coefficientCount = 0;
  const std::int32_t *prediction = nullptr;
  std::uint32_t predictionCount = 0;
  const std::uint16_t *scan = nullptr;
  std::uint32_t scanCount = 0;
  const std::int16_t *matrices = nullptr;
  std::uint32_t matrixCoefficientCount = 0;
  std::int32_t *output = nullptr;
  std::uint32_t outputCount = 0;
};

static_assert(std::is_standard_layout<CudaDecoderTransformTask>::value
                && std::is_trivially_copyable<CudaDecoderTransformTask>::value,
              "CUDA decoder transform tasks must remain POD");
static_assert(std::is_standard_layout<CudaDecoderTransformBatch>::value
                && std::is_trivially_copyable<CudaDecoderTransformBatch>::value,
              "CUDA decoder transform batches must remain POD");

} // namespace vtm

#endif
