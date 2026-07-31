/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2026, ITU/ISO/IEC
 * All rights reserved.
 */

#ifndef VTM_CUDA_PICTURE_MIRROR_H
#define VTM_CUDA_PICTURE_MIRROR_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace vtm
{

constexpr std::size_t CUDA_PICTURE_PLANE_COUNT = 3;

using CudaMirrorHandle = std::uint64_t;

enum class CudaPictureRole : std::uint8_t
{
  Original,
  Reconstruction
};

enum class CudaMirrorState : std::uint8_t
{
  HostValid,
  DeviceValid,
  Synchronized
};

// Host-only description of an active plane. data points at sample (0, 0), not at the allocation base.
// Margins are expressed in samples and are copied together with the active picture.
struct CudaHostPlaneDesc
{
  void          *data          = nullptr;
  std::uint32_t  width         = 0;
  std::uint32_t  height        = 0;
  std::ptrdiff_t strideBytes   = 0;
  std::uint16_t  marginLeft    = 0;
  std::uint16_t  marginRight   = 0;
  std::uint16_t  marginTop     = 0;
  std::uint16_t  marginBottom  = 0;
  std::uint8_t   elementSize   = 0;
  std::uint8_t   bitDepth      = 0;
};

struct CudaHostPictureDesc
{
  std::array<CudaHostPlaneDesc, CUDA_PICTURE_PLANE_COUNT> planes{};
  std::uint8_t planeCount = 0;
};

// POD-only descriptors exposed to future batched kernels. No host pointer, codec object, or STL container is present.
struct CudaDevicePlaneDesc
{
  void          *data          = nullptr;
  std::size_t    pitchBytes    = 0;
  std::uint32_t  width         = 0;
  std::uint32_t  height        = 0;
  std::uint16_t  marginLeft    = 0;
  std::uint16_t  marginRight   = 0;
  std::uint16_t  marginTop     = 0;
  std::uint16_t  marginBottom  = 0;
  std::uint8_t   elementSize   = 0;
  std::uint8_t   bitDepth      = 0;
};

struct CudaDevicePictureDesc
{
  std::array<CudaDevicePlaneDesc, CUDA_PICTURE_PLANE_COUNT> planes{};
  std::uint8_t planeCount = 0;
};

static_assert(std::is_standard_layout<CudaHostPlaneDesc>::value && std::is_trivial<CudaHostPlaneDesc>::value,
              "CUDA host plane descriptor must remain POD");
static_assert(std::is_standard_layout<CudaDevicePlaneDesc>::value && std::is_trivial<CudaDevicePlaneDesc>::value,
              "CUDA device plane descriptor must remain POD");
static_assert(std::is_standard_layout<CudaDevicePictureDesc>::value && std::is_trivial<CudaDevicePictureDesc>::value,
              "CUDA picture descriptor must remain POD");

}   // namespace vtm

#endif   // VTM_CUDA_PICTURE_MIRROR_H
