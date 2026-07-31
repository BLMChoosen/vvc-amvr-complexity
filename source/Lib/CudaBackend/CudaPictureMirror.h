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
  void          *data;
  std::uint32_t  width;
  std::uint32_t  height;
  std::ptrdiff_t strideBytes;
  std::uint16_t  marginLeft;
  std::uint16_t  marginRight;
  std::uint16_t  marginTop;
  std::uint16_t  marginBottom;
  std::uint8_t   elementSize;
  std::uint8_t   bitDepth;
};

struct CudaHostPictureDesc
{
  std::array<CudaHostPlaneDesc, CUDA_PICTURE_PLANE_COUNT> planes;
  std::uint8_t planeCount;
};

// POD-only descriptors exposed to future batched kernels. No host pointer, codec object, or STL container is present.
struct CudaDevicePlaneDesc
{
  void          *data;
  std::size_t    pitchBytes;
  std::uint32_t  width;
  std::uint32_t  height;
  std::uint16_t  marginLeft;
  std::uint16_t  marginRight;
  std::uint16_t  marginTop;
  std::uint16_t  marginBottom;
  std::uint8_t   elementSize;
  std::uint8_t   bitDepth;
};

struct CudaDevicePictureDesc
{
  std::array<CudaDevicePlaneDesc, CUDA_PICTURE_PLANE_COUNT> planes;
  std::uint8_t planeCount;
};

static_assert(std::is_standard_layout<CudaHostPlaneDesc>::value && std::is_trivial<CudaHostPlaneDesc>::value,
              "CUDA host plane descriptor must remain POD");
static_assert(std::is_standard_layout<CudaDevicePlaneDesc>::value && std::is_trivial<CudaDevicePlaneDesc>::value,
              "CUDA device plane descriptor must remain POD");
static_assert(std::is_standard_layout<CudaDevicePictureDesc>::value && std::is_trivial<CudaDevicePictureDesc>::value,
              "CUDA picture descriptor must remain POD");

}   // namespace vtm

#endif   // VTM_CUDA_PICTURE_MIRROR_H
