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

#include "CudaContext.h"

#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <thread>
#include <exception>
#include <unordered_map>
#include <vector>

#if VTM_ENABLE_CUDA
#include "CudaRuntime.h"
#endif

namespace vtm
{

struct CudaPinnedBuffer::Impl
{
  void       *allocation = nullptr;
  std::size_t bytes      = 0;
};

CudaPinnedBuffer::CudaPinnedBuffer() : m_impl(new Impl)
{
}

CudaPinnedBuffer::CudaPinnedBuffer(const std::size_t bytes) : CudaPinnedBuffer()
{
  allocate(bytes);
}

CudaPinnedBuffer::~CudaPinnedBuffer() noexcept
{
  reset();
}

CudaPinnedBuffer::CudaPinnedBuffer(CudaPinnedBuffer &&other) noexcept = default;

CudaPinnedBuffer &CudaPinnedBuffer::operator=(CudaPinnedBuffer &&other) noexcept
{
  if (this != &other)
  {
    reset();
    m_impl = std::move(other.m_impl);
  }
  return *this;
}

void CudaPinnedBuffer::allocate(const std::size_t bytes)
{
  if (bytes == 0)
  {
    throw std::runtime_error("CUDA pinned allocation size must be greater than zero");
  }
#if VTM_ENABLE_CUDA
  std::unique_ptr<Impl> replacement;
  if (!m_impl)
  {
    replacement.reset(new Impl);
  }
  void *allocation = cuda_backend::allocatePinnedHost(bytes);
  reset();
  if (replacement)
  {
    m_impl = std::move(replacement);
  }
  m_impl->allocation = allocation;
  m_impl->bytes      = bytes;
#else
  (void) bytes;
  throw std::runtime_error("CUDA pinned memory requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

void CudaPinnedBuffer::reset() noexcept
{
  if (!m_impl)
  {
    return;
  }
#if VTM_ENABLE_CUDA
  cuda_backend::releasePinnedHost(m_impl->allocation);
#endif
  m_impl->allocation = nullptr;
  m_impl->bytes      = 0;
}

void *CudaPinnedBuffer::data() noexcept
{
  return m_impl ? m_impl->allocation : nullptr;
}

const void *CudaPinnedBuffer::data() const noexcept
{
  return m_impl ? m_impl->allocation : nullptr;
}

std::size_t CudaPinnedBuffer::size() const noexcept
{
  return m_impl ? m_impl->bytes : 0;
}

CudaPinnedBuffer::operator bool() const noexcept
{
  return data() != nullptr;
}

namespace
{

std::size_t roleIndex(const CudaPictureRole role)
{
  switch (role)
  {
  case CudaPictureRole::Original: return 0;
  case CudaPictureRole::Reconstruction: return 1;
  }
  throw std::runtime_error("Invalid CUDA picture role");
}

#if VTM_ENABLE_CUDA
struct PictureMirrorPlane
{
  CudaMirrorState       state = CudaMirrorState::HostValid;
  CudaDevicePlaneDesc  device{};
  void                *deviceBase = nullptr;
  CudaPinnedBuffer     staging{};
  std::size_t          fullRowBytes = 0;
  std::size_t          fullHeight = 0;
  std::size_t          deviceBytes = 0;
  std::size_t          pinnedBytes = 0;
  bool                 uploadPending = false;
  bool                 requested = false;
};

struct PictureMirror
{
  const void            *owner = nullptr;
  CudaPictureRole        role  = CudaPictureRole::Reconstruction;
  CudaHostPictureDesc    host{};
  std::array<PictureMirrorPlane, CUDA_PICTURE_PLANE_COUNT> planes{};
};
#endif

}   // namespace

struct CudaContext::Impl
{
#if VTM_ENABLE_CUDA
  cuda_backend::RuntimeContext *runtime = nullptr;
  std::unordered_map<CudaMirrorHandle, std::unique_ptr<PictureMirror>> mirrors;
  std::unordered_map<const void *, std::array<CudaMirrorHandle, 2>> owners;
  std::uint32_t generation = 0;
  std::uint32_t nextHandle = 1;
  bool distortionAccelerationEnabled = true;
  std::uint64_t distortionFailures = 0;
  bool qpaAccelerationEnabled = true;
  std::uint64_t qpaFailures = 0;
  CudaMirrorMemoryStats mirrorMemory{};
#if VTM_CUDA_TESTING
  std::uint8_t mirrorFailurePlane = 0xff;
  unsigned mirrorAllocationFailureStep = 0;
  unsigned mirrorUploadFailures = 0;
  unsigned mirrorDownloadFailures = 0;
#endif
#endif
  std::thread::id ownerThread;
};

namespace
{

void requireOwnerThread(const std::thread::id &ownerThread)
{
  if (ownerThread != std::this_thread::get_id())
  {
    throw std::runtime_error("CUDA context APIs must be called from the thread that created the context");
  }
}

#if VTM_ENABLE_CUDA

std::atomic<std::uint32_t> nextContextGeneration{ 1 };

template<typename ContextImpl>
void requireRuntime(const ContextImpl *impl)
{
  if (impl->runtime == nullptr)
  {
    throw std::runtime_error("CUDA context is not initialized");
  }
  requireOwnerThread(impl->ownerThread);
}

std::size_t checkedAdd(const std::size_t first, const std::size_t second, const char *what)
{
  if (second > std::numeric_limits<std::size_t>::max() - first)
  {
    throw std::runtime_error(std::string("CUDA picture ") + what + " overflows size_t");
  }
  return first + second;
}

std::size_t checkedMultiply(const std::size_t first, const std::size_t second, const char *what)
{
  if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first)
  {
    throw std::runtime_error(std::string("CUDA picture ") + what + " overflows size_t");
  }
  return first * second;
}

void validatePicture(const CudaHostPictureDesc &picture)
{
  if (picture.planeCount != CUDA_PICTURE_PLANE_COUNT)
  {
    throw std::runtime_error("CUDA picture mirrors require Y, Cb, and Cr planes");
  }

  for (std::size_t index = 0; index < picture.planeCount; ++index)
  {
    const CudaHostPlaneDesc &plane = picture.planes[index];
    if (plane.data == nullptr || plane.width == 0 || plane.height == 0)
    {
      throw std::runtime_error("CUDA picture plane has no active host storage");
    }
    if (plane.elementSize != 1 && plane.elementSize != 2 && plane.elementSize != 4)
    {
      throw std::runtime_error("CUDA picture plane element size must be one, two, or four bytes");
    }
    if (plane.bitDepth != 8 && plane.bitDepth != 10)
    {
      throw std::runtime_error("CUDA picture mirrors currently support only 8-bit and 10-bit samples");
    }
    if (plane.bitDepth > plane.elementSize * 8)
    {
      throw std::runtime_error("CUDA picture plane element size cannot represent its bit depth");
    }
    if (plane.strideBytes <= 0)
    {
      throw std::runtime_error("CUDA picture plane stride must be positive");
    }

    const std::size_t fullWidth = checkedAdd(
      checkedAdd(plane.marginLeft, plane.width, "width"), plane.marginRight, "width");
    const std::size_t rowBytes = checkedMultiply(fullWidth, plane.elementSize, "row size");
    if (rowBytes > static_cast<std::size_t>(plane.strideBytes))
    {
      throw std::runtime_error("CUDA picture plane including margins exceeds its host stride");
    }
    (void) checkedAdd(checkedAdd(plane.marginTop, plane.height, "height"), plane.marginBottom, "height");
  }

  const CudaHostPlaneDesc &luma = picture.planes[0];
  for (std::size_t index = 1; index < picture.planeCount; ++index)
  {
    const CudaHostPlaneDesc &chroma = picture.planes[index];
    if (chroma.width != (luma.width + 1) / 2 || chroma.height != (luma.height + 1) / 2)
    {
      throw std::runtime_error("CUDA picture mirrors currently require 4:2:0 plane dimensions");
    }
  }
}

template<typename ContextImpl>
PictureMirror &findMirror(ContextImpl *impl, const CudaMirrorHandle handle)
{
  const auto found = impl->mirrors.find(handle);
  if (handle == 0 || found == impl->mirrors.end())
  {
    throw std::runtime_error("CUDA picture mirror handle is invalid or has been released");
  }
  return *found->second;
}

const void *mapHostBlock(const PictureMirror &mirror, const std::uint8_t planeIndex, const void *hostPointer,
                         const std::uint32_t width, const std::uint32_t height)
{
  if (planeIndex >= mirror.host.planeCount || hostPointer == nullptr || width == 0 || height == 0)
  {
    throw std::runtime_error("CUDA distortion block descriptor is invalid");
  }

  const CudaHostPlaneDesc &host = mirror.host.planes[planeIndex];
  const auto activeAddress = reinterpret_cast<std::uintptr_t>(host.data);
  const std::size_t activePrefix = static_cast<std::size_t>(host.marginTop) * host.strideBytes
                                   + static_cast<std::size_t>(host.marginLeft) * host.elementSize;
  if (activePrefix > activeAddress)
  {
    throw std::runtime_error("CUDA distortion mirror host address underflows");
  }
  const std::uintptr_t baseAddress = activeAddress - activePrefix;
  const std::uintptr_t pointerAddress = reinterpret_cast<std::uintptr_t>(hostPointer);
  if (pointerAddress < baseAddress)
  {
    throw std::runtime_error("CUDA distortion block starts before its registered mirror");
  }

  const std::size_t offset = static_cast<std::size_t>(pointerAddress - baseAddress);
  const std::size_t row = offset / static_cast<std::size_t>(host.strideBytes);
  const std::size_t columnBytes = offset % static_cast<std::size_t>(host.strideBytes);
  if (columnBytes % host.elementSize != 0)
  {
    throw std::runtime_error("CUDA distortion block is not sample aligned");
  }
  const std::size_t column = columnBytes / host.elementSize;
  const std::size_t fullWidth = static_cast<std::size_t>(host.marginLeft) + host.width + host.marginRight;
  const std::size_t fullHeight = static_cast<std::size_t>(host.marginTop) + host.height + host.marginBottom;
  if (row >= fullHeight || height > fullHeight - row || column >= fullWidth || width > fullWidth - column)
  {
    throw std::runtime_error("CUDA distortion block exceeds its registered mirror");
  }

  const PictureMirrorPlane &mirrorPlane = mirror.planes[planeIndex];
  if (mirrorPlane.deviceBase == nullptr)
  {
    throw std::runtime_error("CUDA picture plane has not been allocated");
  }
  const CudaDevicePlaneDesc &device = mirrorPlane.device;
  const auto *deviceBase = static_cast<const unsigned char *>(mirrorPlane.deviceBase);
  return deviceBase + row * device.pitchBytes + column * host.elementSize;
}

unsigned char *hostBase(const CudaHostPlaneDesc &plane)
{
  auto *active = static_cast<unsigned char *>(plane.data);
  return active - static_cast<std::ptrdiff_t>(plane.marginTop) * plane.strideBytes
         - static_cast<std::ptrdiff_t>(plane.marginLeft) * plane.elementSize;
}

void stageHostPlane(PictureMirror &mirror, const std::size_t index)
{
  const CudaHostPlaneDesc &plane = mirror.host.planes[index];
  PictureMirrorPlane &mirrorPlane = mirror.planes[index];
  const unsigned char *source = hostBase(plane);
  auto *destination = static_cast<unsigned char *>(mirrorPlane.staging.data());
  for (std::size_t row = 0; row < mirrorPlane.fullHeight; ++row)
  {
    std::memcpy(destination + row * mirrorPlane.fullRowBytes,
                source + static_cast<std::ptrdiff_t>(row) * plane.strideBytes,
                mirrorPlane.fullRowBytes);
  }
}

void unstageHostPlane(PictureMirror &mirror, const std::size_t index)
{
  const CudaHostPlaneDesc &plane = mirror.host.planes[index];
  PictureMirrorPlane &mirrorPlane = mirror.planes[index];
  unsigned char *destination = hostBase(plane);
  const auto *source = static_cast<const unsigned char *>(mirrorPlane.staging.data());
  for (std::size_t row = 0; row < mirrorPlane.fullHeight; ++row)
  {
    std::memcpy(destination + static_cast<std::ptrdiff_t>(row) * plane.strideBytes,
                source + row * mirrorPlane.fullRowBytes, mirrorPlane.fullRowBytes);
  }
}

bool samePlaneAllocationLayout(const CudaHostPlaneDesc &a, const CudaHostPlaneDesc &b)
{
  return a.width == b.width && a.height == b.height && a.marginLeft == b.marginLeft
         && a.marginRight == b.marginRight && a.marginTop == b.marginTop && a.marginBottom == b.marginBottom
         && a.elementSize == b.elementSize && a.bitDepth == b.bitDepth;
}

CudaPlaneMask activePlaneMask(const PictureMirror &mirror)
{
  return static_cast<CudaPlaneMask>((CudaPlaneMask{ 1 } << mirror.host.planeCount) - 1);
}

CudaPlaneMask planeBit(const std::uint8_t plane)
{
  if (plane >= CUDA_PICTURE_PLANE_COUNT)
  {
    throw std::runtime_error("CUDA picture plane index is invalid");
  }
  return static_cast<CudaPlaneMask>(CudaPlaneMask{ 1 } << plane);
}

void validatePlaneMask(const PictureMirror &mirror, const CudaPlaneMask planes)
{
  if (planes == 0 || (planes & ~activePlaneMask(mirror)) != 0)
  {
    throw std::runtime_error("CUDA picture plane mask is empty or invalid");
  }
}

void updateMemoryPeak(CudaMirrorMemoryUsage &usage)
{
  usage.peakDeviceBytes = std::max(usage.peakDeviceBytes, usage.currentDeviceBytes);
  usage.peakPinnedBytes = std::max(usage.peakPinnedBytes, usage.currentPinnedBytes);
}

template<typename ContextImpl>
void accountAllocation(ContextImpl *impl, const CudaPictureRole role, const std::size_t plane,
                       const std::size_t deviceBytes, const std::size_t pinnedBytes)
{
  CudaMirrorMemoryUsage &total = impl->mirrorMemory.total;
  CudaMirrorMemoryUsage &detail = impl->mirrorMemory.byRoleAndPlane[roleIndex(role)][plane];
  total.currentDeviceBytes += deviceBytes;
  total.currentPinnedBytes += pinnedBytes;
  detail.currentDeviceBytes += deviceBytes;
  detail.currentPinnedBytes += pinnedBytes;
  updateMemoryPeak(total);
  updateMemoryPeak(detail);
}

template<typename ContextImpl>
void accountDeviceRelease(ContextImpl *impl, const CudaPictureRole role, const std::size_t plane,
                          const std::size_t bytes)
{
  impl->mirrorMemory.total.currentDeviceBytes -= bytes;
  impl->mirrorMemory.byRoleAndPlane[roleIndex(role)][plane].currentDeviceBytes -= bytes;
}

template<typename ContextImpl>
void accountPinnedRelease(ContextImpl *impl, const CudaPictureRole role, const std::size_t plane,
                          const std::size_t bytes)
{
  impl->mirrorMemory.total.currentPinnedBytes -= bytes;
  impl->mirrorMemory.byRoleAndPlane[roleIndex(role)][plane].currentPinnedBytes -= bytes;
}

template<typename ContextImpl>
void allocateMirrorPlane(ContextImpl *impl, PictureMirror &mirror, const std::size_t index)
{
  PictureMirrorPlane &mirrorPlane = mirror.planes[index];
  if (mirrorPlane.deviceBase != nullptr)
  {
    mirrorPlane.requested = true;
    return;
  }
  const CudaHostPlaneDesc &host = mirror.host.planes[index];
  const std::size_t fullWidth = static_cast<std::size_t>(host.marginLeft) + host.width + host.marginRight;
  const std::size_t fullHeight = static_cast<std::size_t>(host.marginTop) + host.height + host.marginBottom;
  const std::size_t rowBytes = checkedMultiply(fullWidth, host.elementSize, "row size");
  const std::size_t pitch = checkedAdd(rowBytes, 127, "device pitch") & ~std::size_t(127);
  const std::size_t deviceBytes = checkedMultiply(pitch, fullHeight, "device allocation");
  const std::size_t pinnedBytes = checkedMultiply(rowBytes, fullHeight, "staging allocation");

  CudaPinnedBuffer staging;
#if VTM_CUDA_TESTING
  if (impl->mirrorFailurePlane == index && impl->mirrorAllocationFailureStep == 1)
  {
    impl->mirrorAllocationFailureStep = 0;
    throw std::runtime_error("Injected CUDA mirror pinned allocation failure");
  }
#endif
  staging.allocate(pinnedBytes);
#if VTM_CUDA_TESTING
  if (impl->mirrorFailurePlane == index && impl->mirrorAllocationFailureStep == 2)
  {
    impl->mirrorAllocationFailureStep = 0;
    throw std::runtime_error("Injected CUDA mirror device allocation failure");
  }
#endif
  void *deviceBase = cuda_backend::allocateDevice(impl->runtime, deviceBytes, CudaQueue::Upload);
#if VTM_CUDA_TESTING
  if (impl->mirrorFailurePlane == index && impl->mirrorAllocationFailureStep == 3)
  {
    impl->mirrorAllocationFailureStep = 0;
    try
    {
      cuda_backend::releaseDevice(impl->runtime, deviceBase, CudaQueue::Upload);
      cuda_backend::synchronizeQueue(impl->runtime, CudaQueue::Upload);
    }
    catch (...)
    {
      try { cuda_backend::releaseDeviceImmediate(impl->runtime, deviceBase); } catch (...) {}
    }
    throw std::runtime_error("Injected CUDA mirror post-device allocation failure");
  }
#endif

  mirrorPlane.deviceBase = deviceBase;
  mirrorPlane.staging = std::move(staging);
  mirrorPlane.fullRowBytes = rowBytes;
  mirrorPlane.fullHeight = fullHeight;
  mirrorPlane.deviceBytes = deviceBytes;
  mirrorPlane.pinnedBytes = pinnedBytes;
  mirrorPlane.requested = true;
  CudaDevicePlaneDesc &device = mirrorPlane.device;
  device.data = static_cast<unsigned char *>(deviceBase)
                + static_cast<std::size_t>(host.marginTop) * pitch
                + static_cast<std::size_t>(host.marginLeft) * host.elementSize;
  device.pitchBytes   = pitch;
  device.width        = host.width;
  device.height       = host.height;
  device.marginLeft   = host.marginLeft;
  device.marginRight  = host.marginRight;
  device.marginTop    = host.marginTop;
  device.marginBottom = host.marginBottom;
  device.elementSize  = host.elementSize;
  device.bitDepth     = host.bitDepth;
  accountAllocation(impl, mirror.role, index, deviceBytes, pinnedBytes);
}

template<typename ContextImpl>
void releaseMirrorResources(ContextImpl *impl, PictureMirror &mirror)
{
  std::exception_ptr firstError;
  try
  {
    cuda_backend::synchronizeRuntimeContext(impl->runtime);
  }
  catch (...)
  {
    firstError = std::current_exception();
  }
  for (std::size_t index = 0; index < mirror.planes.size(); ++index)
  {
    PictureMirrorPlane &plane = mirror.planes[index];
    void *&allocation = plane.deviceBase;
    if (allocation != nullptr)
    {
      try
      {
        cuda_backend::releaseDevice(impl->runtime, allocation, CudaQueue::Compute);
        allocation = nullptr;
        accountDeviceRelease(impl, mirror.role, index, plane.deviceBytes);
        plane.deviceBytes = 0;
        plane.device = CudaDevicePlaneDesc{};
      }
      catch (...)
      {
        if (!firstError)
        {
          firstError = std::current_exception();
        }
        // cudaFreeAsync did not accept ownership. After the synchronization attempt above, a synchronous
        // cudaFree is the safe fallback. If it also fails, retain the pointer in the mirror so teardown can retry.
        try
        {
          cuda_backend::releaseDeviceImmediate(impl->runtime, allocation);
          allocation = nullptr;
          accountDeviceRelease(impl, mirror.role, index, plane.deviceBytes);
          plane.deviceBytes = 0;
          plane.device = CudaDevicePlaneDesc{};
        }
        catch (...)
        {
        }
      }
    }
    if (plane.staging)
    {
      plane.staging.reset();
      accountPinnedRelease(impl, mirror.role, index, plane.pinnedBytes);
      plane.pinnedBytes = 0;
    }
    plane.uploadPending = false;
  }
  try
  {
    cuda_backend::synchronizeQueue(impl->runtime, CudaQueue::Compute);
  }
  catch (...)
  {
    if (!firstError)
    {
      firstError = std::current_exception();
    }
  }
  if (firstError)
  {
    std::rethrow_exception(firstError);
  }
}

bool hasDeviceAllocations(const PictureMirror &mirror)
{
  for (const PictureMirrorPlane &plane : mirror.planes)
  {
    if (plane.deviceBase != nullptr)
    {
      return true;
    }
  }
  return false;
}

template<typename ContextImpl>
void releaseAllMirrorsNoThrow(ContextImpl *impl) noexcept
{
  if (impl->runtime == nullptr)
  {
    impl->mirrors.clear();
    impl->owners.clear();
    return;
  }
  try
  {
    cuda_backend::synchronizeRuntimeContext(impl->runtime);
  }
  catch (...)
  {
  }
  for (auto &entry : impl->mirrors)
  {
    PictureMirror &mirror = *entry.second;
    for (std::size_t index = 0; index < mirror.planes.size(); ++index)
    {
      PictureMirrorPlane &plane = mirror.planes[index];
      void *&allocation = plane.deviceBase;
      if (allocation != nullptr)
      {
        try
        {
          cuda_backend::releaseDevice(impl->runtime, allocation, CudaQueue::Compute);
          allocation = nullptr;
          accountDeviceRelease(impl, mirror.role, index, plane.deviceBytes);
          plane.deviceBytes = 0;
        }
        catch (...)
        {
          try
          {
            cuda_backend::releaseDeviceImmediate(impl->runtime, allocation);
            allocation = nullptr;
            accountDeviceRelease(impl, mirror.role, index, plane.deviceBytes);
            plane.deviceBytes = 0;
          }
          catch (...)
          {
            // Keep ownership in the mirror until destroyRuntimeContext tears down the owning memory pool.
          }
        }
      }
      if (plane.staging)
      {
        plane.staging.reset();
        accountPinnedRelease(impl, mirror.role, index, plane.pinnedBytes);
        plane.pinnedBytes = 0;
      }
    }
  }
  try
  {
    cuda_backend::synchronizeQueue(impl->runtime, CudaQueue::Compute);
  }
  catch (...)
  {
  }
}

#endif

}   // namespace

CudaContext::CudaContext() : m_impl(new Impl)
{
}

CudaContext::~CudaContext() noexcept
{
#if VTM_ENABLE_CUDA
  if (m_impl->runtime != nullptr)
  {
    if (m_impl->ownerThread != std::this_thread::get_id())
    {
      std::terminate();
    }
    releaseAllMirrorsNoThrow(m_impl.get());
    cuda_backend::destroyRuntimeContext(m_impl->runtime);
    m_impl->runtime = nullptr;
    m_impl->ownerThread = std::thread::id{};
  }
#endif
}

void CudaContext::create(const int device)
{
  if (isCreated())
  {
    throw std::runtime_error("CUDA context is already initialized");
  }
  if (device < 0)
  {
    throw std::runtime_error("GPUDevice must be zero or greater");
  }

#if VTM_ENABLE_CUDA
  m_impl->runtime = cuda_backend::createRuntimeContext(device);
  m_impl->distortionAccelerationEnabled = true;
  m_impl->distortionFailures = 0;
  m_impl->qpaAccelerationEnabled = true;
  m_impl->qpaFailures = 0;
  m_impl->mirrorMemory = CudaMirrorMemoryStats{};
  m_impl->ownerThread = std::this_thread::get_id();
  m_impl->generation = nextContextGeneration.fetch_add(1, std::memory_order_relaxed);
  if (m_impl->generation == 0)
  {
    m_impl->generation = nextContextGeneration.fetch_add(1, std::memory_order_relaxed);
  }
  m_impl->nextHandle = 1;
#else
  (void) device;
  throw std::runtime_error("CUDA backend requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

void CudaContext::synchronize()
{
#if VTM_ENABLE_CUDA
  if (m_impl->runtime != nullptr)
  {
    requireOwnerThread(m_impl->ownerThread);
    cuda_backend::synchronizeRuntimeContext(m_impl->runtime);
  }
#endif
}

void CudaContext::shutdown(const bool synchronize)
{
#if VTM_ENABLE_CUDA
  if (m_impl->runtime != nullptr)
  {
    requireOwnerThread(m_impl->ownerThread);
    std::exception_ptr firstError;
    try
    {
      releaseAllPictureMirrors();
    }
    catch (...)
    {
      firstError = std::current_exception();
    }
    cuda_backend::RuntimeContext *runtime = m_impl->runtime;
    m_impl->runtime = nullptr;
    m_impl->ownerThread = std::thread::id{};
    try
    {
      cuda_backend::shutdownRuntimeContext(runtime, synchronize);
    }
    catch (...)
    {
      if (!firstError)
      {
        firstError = std::current_exception();
      }
    }
    // shutdownRuntimeContext destroys the owning memory pool even when it reports an error. Only now is it safe
    // to discard mirrors whose async and synchronous free attempts both failed.
    m_impl->mirrors.clear();
    m_impl->owners.clear();
    if (firstError)
    {
      std::rethrow_exception(firstError);
    }
  }
#endif
}

void CudaContext::destroy()
{
  shutdown();
}

void CudaContext::recordFence(const CudaQueue queue, const CudaFence fence)
{
#if VTM_ENABLE_CUDA
  if (m_impl->runtime == nullptr)
  {
    throw std::runtime_error("CUDA context is not initialized");
  }
  requireOwnerThread(m_impl->ownerThread);
  cuda_backend::recordFence(m_impl->runtime, queue, fence);
#else
  (void) queue;
  (void) fence;
  throw std::runtime_error("CUDA backend requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

void CudaContext::waitFence(const CudaQueue queue, const CudaFence fence)
{
#if VTM_ENABLE_CUDA
  if (m_impl->runtime == nullptr)
  {
    throw std::runtime_error("CUDA context is not initialized");
  }
  requireOwnerThread(m_impl->ownerThread);
  cuda_backend::waitFence(m_impl->runtime, queue, fence);
#else
  (void) queue;
  (void) fence;
  throw std::runtime_error("CUDA backend requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

void *CudaContext::allocateDevice(const std::size_t bytes, const CudaQueue queue)
{
#if VTM_ENABLE_CUDA
  if (m_impl->runtime == nullptr)
  {
    throw std::runtime_error("CUDA context is not initialized");
  }
  requireOwnerThread(m_impl->ownerThread);
  return cuda_backend::allocateDevice(m_impl->runtime, bytes, queue);
#else
  (void) bytes;
  (void) queue;
  throw std::runtime_error("CUDA backend requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

void CudaContext::releaseDevice(void *allocation, const CudaQueue queue)
{
#if VTM_ENABLE_CUDA
  if (m_impl->runtime == nullptr)
  {
    throw std::runtime_error("CUDA context is not initialized");
  }
  requireOwnerThread(m_impl->ownerThread);
  cuda_backend::releaseDevice(m_impl->runtime, allocation, queue);
#else
  (void) allocation;
  (void) queue;
  throw std::runtime_error("CUDA backend requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

CudaMirrorHandle CudaContext::registerPictureMirror(const void *owner, const CudaPictureRole role,
                                                     const CudaHostPictureDesc &picture)
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  if (owner == nullptr)
  {
    throw std::runtime_error("CUDA picture mirror owner cannot be null");
  }
  validatePicture(picture);

  const std::size_t ownerRole = roleIndex(role);
  const auto ownerFound = m_impl->owners.find(owner);
  if (ownerFound != m_impl->owners.end() && ownerFound->second[ownerRole] != 0)
  {
    throw std::runtime_error("CUDA picture mirror is already registered for this owner and role");
  }

  CudaMirrorHandle handle = 0;
  decltype(m_impl->mirrors)::iterator slot;
  for (;;)
  {
    std::uint32_t localHandle = m_impl->nextHandle++;
    if (localHandle == 0)
    {
      continue;
    }
    handle = (static_cast<CudaMirrorHandle>(m_impl->generation) << 32) | localHandle;
    const auto inserted = m_impl->mirrors.emplace(handle, nullptr);
    if (inserted.second)
    {
      slot = inserted.first;
      break;
    }
  }

  try
  {
    // The potentially-throwing unordered_map insertion happens before any CUDA allocation. Once allocation starts,
    // the registry owns the mirror so even a failed rollback remains reachable by shutdown for another release try.
    slot->second.reset(new PictureMirror);
    slot->second->owner = owner;
    slot->second->role  = role;
    slot->second->host = picture;
    m_impl->owners[owner][ownerRole] = handle;
  }
  catch (...)
  {
    const std::exception_ptr registrationError = std::current_exception();
    if (slot->second)
    {
      try
      {
        releaseMirrorResources(m_impl.get(), *slot->second);
      }
      catch (...)
      {
      }
    }
    if (!slot->second || !hasDeviceAllocations(*slot->second))
    {
      m_impl->mirrors.erase(slot);
    }
    std::rethrow_exception(registrationError);
  }
  return handle;
#else
  (void) owner;
  (void) role;
  (void) picture;
  throw std::runtime_error("CUDA picture mirror requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

void CudaContext::rebindHostPicture(const CudaMirrorHandle handle, const CudaHostPictureDesc &picture)
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  validatePicture(picture);
  PictureMirror &mirror = findMirror(m_impl.get(), handle);

  // A PelStorage swap can occur while an earlier upload is still queued. Complete all users of the old host
  // and device storage before changing either pointer set.
  cuda_backend::synchronizeRuntimeContext(m_impl->runtime);
  bool sameLayout = mirror.host.planeCount == picture.planeCount;
  if (sameLayout)
  {
    for (std::size_t index = 0; index < picture.planeCount; ++index)
    {
      sameLayout = sameLayout && samePlaneAllocationLayout(mirror.host.planes[index], picture.planes[index]);
    }
  }
  if (sameLayout)
  {
    mirror.host = picture;
    for (PictureMirrorPlane &plane : mirror.planes)
    {
      plane.state = CudaMirrorState::HostValid;
      plane.uploadPending = false;
    }
    return;
  }

  PictureMirror replacement;
  replacement.owner = mirror.owner;
  replacement.role  = mirror.role;
  replacement.host  = picture;
  try
  {
    for (std::size_t index = 0; index < picture.planeCount; ++index)
    {
      if (mirror.planes[index].requested)
      {
        allocateMirrorPlane(m_impl.get(), replacement, index);
      }
    }
  }
  catch (...)
  {
    const std::exception_ptr allocationError = std::current_exception();
    try { releaseMirrorResources(m_impl.get(), replacement); } catch (...) {}
    std::rethrow_exception(allocationError);
  }

  std::exception_ptr releaseError;
  try
  {
    releaseMirrorResources(m_impl.get(), mirror);
  }
  catch (...)
  {
    releaseError = std::current_exception();
  }
  mirror = std::move(replacement);
  if (releaseError)
  {
    std::rethrow_exception(releaseError);
  }
#else
  (void) handle;
  (void) picture;
  throw std::runtime_error("CUDA picture mirror requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

void CudaContext::rebindHostPicture(const void *owner, const CudaPictureRole role,
                                    const CudaHostPictureDesc &picture)
{
#if VTM_ENABLE_CUDA
  rebindHostPicture(pictureMirrorHandle(owner, role), picture);
#else
  (void) owner;
  (void) role;
  (void) picture;
  throw std::runtime_error("CUDA picture mirror requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

void CudaContext::releasePictureMirror(const CudaMirrorHandle handle)
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  const auto mirrorFound = m_impl->mirrors.find(handle);
  if (handle == 0 || mirrorFound == m_impl->mirrors.end())
  {
    throw std::runtime_error("CUDA picture mirror handle is invalid or has been released");
  }
  PictureMirror &mirror = *mirrorFound->second;
  std::exception_ptr releaseError;
  try
  {
    releaseMirrorResources(m_impl.get(), mirror);
  }
  catch (...)
  {
    releaseError = std::current_exception();
  }

  if (!hasDeviceAllocations(mirror))
  {
    const void *owner = mirror.owner;
    const std::size_t ownerRole = roleIndex(mirror.role);
    m_impl->mirrors.erase(mirrorFound);
    auto ownerFound = m_impl->owners.find(owner);
    if (ownerFound != m_impl->owners.end())
    {
      ownerFound->second[ownerRole] = 0;
      if (ownerFound->second[0] == 0 && ownerFound->second[1] == 0)
      {
        m_impl->owners.erase(ownerFound);
      }
    }
  }
  if (releaseError)
  {
    std::rethrow_exception(releaseError);
  }
#else
  (void) handle;
  throw std::runtime_error("CUDA picture mirror requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

void CudaContext::releasePictureMirrors(const void *owner)
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  const auto found = m_impl->owners.find(owner);
  if (found == m_impl->owners.end())
  {
    return;
  }
  const std::array<CudaMirrorHandle, 2> handles = found->second;
  std::exception_ptr firstError;
  for (const CudaMirrorHandle handle : handles)
  {
    if (handle != 0)
    {
      try
      {
        releasePictureMirror(handle);
      }
      catch (...)
      {
        if (!firstError)
        {
          firstError = std::current_exception();
        }
      }
    }
  }
  if (firstError)
  {
    std::rethrow_exception(firstError);
  }
#else
  (void) owner;
#endif
}

void CudaContext::releaseAllPictureMirrors()
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  std::exception_ptr firstError;
  std::vector<CudaMirrorHandle> handles;
  handles.reserve(m_impl->mirrors.size());
  for (const auto &entry : m_impl->mirrors)
  {
    handles.push_back(entry.first);
  }
  for (const CudaMirrorHandle handle : handles)
  {
    try
    {
      releasePictureMirror(handle);
    }
    catch (...)
    {
      if (!firstError)
      {
        firstError = std::current_exception();
      }
    }
  }
  if (firstError)
  {
    std::rethrow_exception(firstError);
  }
#endif
}

bool CudaContext::hasPictureMirror(const void *owner, const CudaPictureRole role) const
{
#if VTM_ENABLE_CUDA
  if (m_impl->runtime == nullptr)
  {
    return false;
  }
  requireOwnerThread(m_impl->ownerThread);
  const auto found = m_impl->owners.find(owner);
  return found != m_impl->owners.end() && found->second[roleIndex(role)] != 0;
#else
  (void) owner;
  (void) role;
  return false;
#endif
}

CudaMirrorHandle CudaContext::pictureMirrorHandle(const void *owner, const CudaPictureRole role) const
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  const auto found = m_impl->owners.find(owner);
  if (found == m_impl->owners.end() || found->second[roleIndex(role)] == 0)
  {
    throw std::runtime_error("CUDA picture mirror owner and role are not registered");
  }
  return found->second[roleIndex(role)];
#else
  (void) owner;
  (void) role;
  throw std::runtime_error("CUDA picture mirror requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

std::size_t CudaContext::pictureMirrorCount() const
{
#if VTM_ENABLE_CUDA
  if (m_impl->runtime != nullptr)
  {
    requireOwnerThread(m_impl->ownerThread);
  }
  return m_impl->mirrors.size();
#else
  return 0;
#endif
}

CudaMirrorState CudaContext::pictureMirrorState(const CudaMirrorHandle handle) const
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  const PictureMirror &mirror = findMirror(m_impl.get(), handle);
  const CudaMirrorState first = mirror.planes[0].state;
  for (std::size_t index = 1; index < mirror.host.planeCount; ++index)
  {
    if (mirror.planes[index].state != first)
    {
      return CudaMirrorState::Mixed;
    }
  }
  return first;
#else
  (void) handle;
  throw std::runtime_error("CUDA picture mirror requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

CudaMirrorState CudaContext::pictureMirrorPlaneState(const CudaMirrorHandle handle,
                                                     const std::uint8_t plane) const
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  const PictureMirror &mirror = findMirror(m_impl.get(), handle);
  validatePlaneMask(mirror, planeBit(plane));
  return mirror.planes[plane].state;
#else
  (void) handle;
  (void) plane;
  throw std::runtime_error("CUDA picture mirror requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

CudaPlaneMask CudaContext::pictureMirrorAllocatedPlanes(const CudaMirrorHandle handle) const
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  const PictureMirror &mirror = findMirror(m_impl.get(), handle);
  CudaPlaneMask allocated = 0;
  for (std::uint8_t index = 0; index < mirror.host.planeCount; ++index)
  {
    if (mirror.planes[index].deviceBase != nullptr)
    {
      allocated |= planeBit(index);
    }
  }
  return allocated;
#else
  (void) handle;
  return 0;
#endif
}

CudaDevicePictureDesc CudaContext::devicePicture(const CudaMirrorHandle handle) const
{
#if VTM_ENABLE_CUDA
  return devicePicturePlanes(handle, CUDA_PLANE_ALL);
#else
  (void) handle;
  throw std::runtime_error("CUDA picture mirror requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

CudaDevicePictureDesc CudaContext::devicePicturePlanes(const CudaMirrorHandle handle,
                                                       const CudaPlaneMask planes) const
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  const PictureMirror &mirror = findMirror(m_impl.get(), handle);
  validatePlaneMask(mirror, planes);
  CudaDevicePictureDesc picture{};
  picture.planeCount = mirror.host.planeCount;
  for (std::uint8_t index = 0; index < mirror.host.planeCount; ++index)
  {
    if ((planes & planeBit(index)) != 0)
    {
      const PictureMirrorPlane &mirrorPlane = mirror.planes[index];
      if (mirrorPlane.deviceBase == nullptr || mirrorPlane.state == CudaMirrorState::HostValid)
      {
        throw std::runtime_error("CUDA device picture plane is unavailable or stale; call ensureDevicePlanes first");
      }
      picture.planes[index] = mirrorPlane.device;
    }
  }
  return picture;
#else
  (void) handle;
  (void) planes;
  throw std::runtime_error("CUDA picture mirror requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

void CudaContext::markHostModified(const CudaMirrorHandle handle)
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  PictureMirror &mirror = findMirror(m_impl.get(), handle);
  markHostPlanesModified(handle, activePlaneMask(mirror));
#else
  (void) handle;
  throw std::runtime_error("CUDA picture mirror requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

void CudaContext::markHostPlanesModified(const CudaMirrorHandle handle, const CudaPlaneMask planes)
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  PictureMirror &mirror = findMirror(m_impl.get(), handle);
  validatePlaneMask(mirror, planes);
  bool needsFence = false;
  for (std::uint8_t index = 0; index < mirror.host.planeCount; ++index)
  {
    if ((planes & planeBit(index)) == 0) continue;
    if (mirror.planes[index].state == CudaMirrorState::DeviceValid)
    {
      throw std::runtime_error("CUDA picture mirror conflict: device data must be downloaded before host modification");
    }
    needsFence = needsFence || mirror.planes[index].state == CudaMirrorState::Synchronized;
  }
  if (needsFence)
  {
    cuda_backend::recordFence(m_impl->runtime, CudaQueue::Compute, CudaFence::ComputeComplete);
    cuda_backend::waitFence(m_impl->runtime, CudaQueue::Upload, CudaFence::ComputeComplete);
  }
  for (std::uint8_t index = 0; index < mirror.host.planeCount; ++index)
  {
    if ((planes & planeBit(index)) != 0) mirror.planes[index].state = CudaMirrorState::HostValid;
  }
#else
  (void) handle;
  (void) planes;
  throw std::runtime_error("CUDA picture mirror requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

void CudaContext::markHostPlaneModified(const CudaMirrorHandle handle, const std::uint8_t plane)
{
  if (plane >= CUDA_PICTURE_PLANE_COUNT) throw std::runtime_error("CUDA picture plane index is invalid");
  markHostPlanesModified(handle, static_cast<CudaPlaneMask>(CudaPlaneMask{ 1 } << plane));
}

void CudaContext::markDeviceModified(const CudaMirrorHandle handle)
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  PictureMirror &mirror = findMirror(m_impl.get(), handle);
  markDevicePlanesModified(handle, activePlaneMask(mirror));
#else
  (void) handle;
  throw std::runtime_error("CUDA picture mirror requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

void CudaContext::markDevicePlanesModified(const CudaMirrorHandle handle, const CudaPlaneMask planes)
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  PictureMirror &mirror = findMirror(m_impl.get(), handle);
  validatePlaneMask(mirror, planes);
  for (std::uint8_t index = 0; index < mirror.host.planeCount; ++index)
  {
    if ((planes & planeBit(index)) != 0
        && (mirror.planes[index].deviceBase == nullptr
            || mirror.planes[index].state == CudaMirrorState::HostValid))
    {
      throw std::runtime_error("CUDA picture mirror conflict: host data must be uploaded before device modification");
    }
  }
  cuda_backend::recordFence(m_impl->runtime, CudaQueue::Compute, CudaFence::ComputeComplete);
  cuda_backend::waitFence(m_impl->runtime, CudaQueue::Download, CudaFence::ComputeComplete);
  for (std::uint8_t index = 0; index < mirror.host.planeCount; ++index)
  {
    if ((planes & planeBit(index)) != 0) mirror.planes[index].state = CudaMirrorState::DeviceValid;
  }
#else
  (void) handle;
  (void) planes;
  throw std::runtime_error("CUDA picture mirror requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

void CudaContext::markDevicePlaneModified(const CudaMirrorHandle handle, const std::uint8_t plane)
{
  if (plane >= CUDA_PICTURE_PLANE_COUNT) throw std::runtime_error("CUDA picture plane index is invalid");
  markDevicePlanesModified(handle, static_cast<CudaPlaneMask>(CudaPlaneMask{ 1 } << plane));
}

void CudaContext::ensureDevice(const CudaMirrorHandle handle)
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  PictureMirror &mirror = findMirror(m_impl.get(), handle);
  ensureDevicePlanes(handle, activePlaneMask(mirror));
#else
  (void) handle;
  throw std::runtime_error("CUDA picture mirror requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

void CudaContext::ensureDevicePlanes(const CudaMirrorHandle handle, const CudaPlaneMask planes)
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  PictureMirror &mirror = findMirror(m_impl.get(), handle);
  validatePlaneMask(mirror, planes);
  bool submitted = false;
  for (std::uint8_t index = 0; index < mirror.host.planeCount; ++index)
  {
    if ((planes & planeBit(index)) == 0) continue;
    PictureMirrorPlane &mirrorPlane = mirror.planes[index];
    allocateMirrorPlane(m_impl.get(), mirror, index);
    if (mirrorPlane.state != CudaMirrorState::HostValid) continue;
    if (mirrorPlane.uploadPending)
    {
      cuda_backend::synchronizeQueue(m_impl->runtime, CudaQueue::Upload);
      mirrorPlane.uploadPending = false;
    }
#if VTM_CUDA_TESTING
    if (m_impl->mirrorFailurePlane == index && m_impl->mirrorUploadFailures > 0)
    {
      --m_impl->mirrorUploadFailures;
      throw std::runtime_error("Injected CUDA mirror upload failure");
    }
#endif
    stageHostPlane(mirror, index);
    cuda_backend::copy2DToDeviceAsync(m_impl->runtime, mirrorPlane.deviceBase,
                                      mirrorPlane.device.pitchBytes, mirrorPlane.staging.data(),
                                      mirrorPlane.fullRowBytes, mirrorPlane.fullRowBytes,
                                      mirrorPlane.fullHeight, CudaQueue::Upload);
    mirrorPlane.uploadPending = true;
    mirrorPlane.state = CudaMirrorState::Synchronized;
    submitted = true;
  }
  if (submitted)
  {
    cuda_backend::recordFence(m_impl->runtime, CudaQueue::Upload, CudaFence::UploadComplete);
    cuda_backend::waitFence(m_impl->runtime, CudaQueue::Compute, CudaFence::UploadComplete);
  }
#else
  (void) handle;
  (void) planes;
  throw std::runtime_error("CUDA picture mirror requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

void CudaContext::ensureDevicePlane(const CudaMirrorHandle handle, const std::uint8_t plane)
{
  if (plane >= CUDA_PICTURE_PLANE_COUNT) throw std::runtime_error("CUDA picture plane index is invalid");
  ensureDevicePlanes(handle, static_cast<CudaPlaneMask>(CudaPlaneMask{ 1 } << plane));
}

void CudaContext::ensureHost(const CudaMirrorHandle handle)
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  PictureMirror &mirror = findMirror(m_impl.get(), handle);
  ensureHostPlanes(handle, activePlaneMask(mirror));
#else
  (void) handle;
  throw std::runtime_error("CUDA picture mirror requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

void CudaContext::ensureHostPlanes(const CudaMirrorHandle handle, const CudaPlaneMask planes)
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  PictureMirror &mirror = findMirror(m_impl.get(), handle);
  validatePlaneMask(mirror, planes);
  CudaPlaneMask submitted = 0;
  for (std::uint8_t index = 0; index < mirror.host.planeCount; ++index)
  {
    if ((planes & planeBit(index)) == 0 || mirror.planes[index].state != CudaMirrorState::DeviceValid) continue;
    PictureMirrorPlane &mirrorPlane = mirror.planes[index];
#if VTM_CUDA_TESTING
    if (m_impl->mirrorFailurePlane == index && m_impl->mirrorDownloadFailures > 0)
    {
      --m_impl->mirrorDownloadFailures;
      throw std::runtime_error("Injected CUDA mirror download failure");
    }
#endif
    cuda_backend::copy2DToHostAsync(m_impl->runtime, mirrorPlane.staging.data(), mirrorPlane.fullRowBytes,
                                    mirrorPlane.deviceBase, mirrorPlane.device.pitchBytes,
                                    mirrorPlane.fullRowBytes, mirrorPlane.fullHeight, CudaQueue::Download);
    submitted |= planeBit(index);
  }
  if (submitted != 0)
  {
    cuda_backend::synchronizeQueue(m_impl->runtime, CudaQueue::Download);
    for (std::uint8_t index = 0; index < mirror.host.planeCount; ++index)
    {
      if ((submitted & planeBit(index)) == 0) continue;
      unstageHostPlane(mirror, index);
      mirror.planes[index].uploadPending = false;
      mirror.planes[index].state = CudaMirrorState::Synchronized;
    }
  }
#else
  (void) handle;
  (void) planes;
  throw std::runtime_error("CUDA picture mirror requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

void CudaContext::ensureHostPlane(const CudaMirrorHandle handle, const std::uint8_t plane)
{
  if (plane >= CUDA_PICTURE_PLANE_COUNT) throw std::runtime_error("CUDA picture plane index is invalid");
  ensureHostPlanes(handle, static_cast<CudaPlaneMask>(CudaPlaneMask{ 1 } << plane));
}

CudaMirrorMemoryStats CudaContext::pictureMirrorMemoryStats() const
{
#if VTM_ENABLE_CUDA
  if (m_impl->runtime != nullptr) requireOwnerThread(m_impl->ownerThread);
  return m_impl->mirrorMemory;
#else
  return CudaMirrorMemoryStats{};
#endif
}

bool CudaContext::isDistortionAccelerationAvailable() const noexcept
{
#if VTM_ENABLE_CUDA
  return m_impl->runtime != nullptr && m_impl->distortionAccelerationEnabled;
#else
  return false;
#endif
}

bool CudaContext::computeDistortionBatch(const CudaDistortionBatchDesc &batch, std::uint64_t *results) noexcept
{
#if VTM_ENABLE_CUDA
  if (!isDistortionAccelerationAvailable() || results == nullptr || batch.candidateGrid.reference == nullptr
      || batch.candidateGrid.columns == 0 || batch.candidateGrid.rows == 0
      || batch.candidateGrid.stepX == 0 || batch.candidateGrid.stepY == 0)
  {
    return false;
  }
  const std::uint64_t candidateCount64 = static_cast<std::uint64_t>(batch.candidateGrid.columns)
                                         * batch.candidateGrid.rows;
  if (candidateCount64 > CUDA_MAX_DISTORTION_CANDIDATES || batch.metric != CudaDistortionMetric::Sad
      || (batch.elementSize != 2 && batch.elementSize != 4) || (batch.bitDepth != 8 && batch.bitDepth != 10)
      || batch.subShift > 4 || batch.width == 0 || batch.height == 0
      || (batch.height % (1u << batch.subShift)) != 0)
  {
    return false;
  }
  try
  {
    requireRuntime(m_impl.get());
    PictureMirror &sourceMirror = findMirror(m_impl.get(), batch.sourceMirror);
    PictureMirror &referenceMirror = findMirror(m_impl.get(), batch.referenceMirror);
    if (batch.sourcePlane >= sourceMirror.host.planeCount || batch.referencePlane >= referenceMirror.host.planeCount)
    {
      return false;
    }
    const CudaHostPlaneDesc &sourcePlane = sourceMirror.host.planes[batch.sourcePlane];
    const CudaHostPlaneDesc &referencePlane = referenceMirror.host.planes[batch.referencePlane];
    if (sourcePlane.elementSize != batch.elementSize || referencePlane.elementSize != batch.elementSize
        || sourcePlane.bitDepth != batch.bitDepth || referencePlane.bitDepth != batch.bitDepth)
    {
      return false;
    }
    const std::uint64_t referenceWidth64 = batch.width
      + static_cast<std::uint64_t>(batch.candidateGrid.columns - 1) * batch.candidateGrid.stepX;
    const std::uint64_t referenceHeight64 = batch.height
      + static_cast<std::uint64_t>(batch.candidateGrid.rows - 1) * batch.candidateGrid.stepY;
    if (referenceWidth64 > std::numeric_limits<std::uint32_t>::max()
        || referenceHeight64 > std::numeric_limits<std::uint32_t>::max())
    {
      return false;
    }
    try
    {
      ensureDevicePlane(batch.sourceMirror, batch.sourcePlane);
      if (batch.referenceMirror != batch.sourceMirror)
      {
        ensureDevicePlane(batch.referenceMirror, batch.referencePlane);
      }
      else if (batch.referencePlane != batch.sourcePlane)
      {
        ensureDevicePlane(batch.referenceMirror, batch.referencePlane);
      }
      const void *sourceDevice = mapHostBlock(sourceMirror, batch.sourcePlane, batch.source,
                                              batch.width, batch.height);
      const void *referenceDevice = mapHostBlock(referenceMirror, batch.referencePlane,
                                                 batch.candidateGrid.reference,
                                                 static_cast<std::uint32_t>(referenceWidth64),
                                                 static_cast<std::uint32_t>(referenceHeight64));
      cuda_backend::computeDistortionBatch(m_impl->runtime, batch, sourceDevice,
                                           sourceMirror.planes[batch.sourcePlane].device.pitchBytes,
                                           referenceDevice,
                                           referenceMirror.planes[batch.referencePlane].device.pitchBytes, results);
      return true;
    }
    catch (...)
    {
      ++m_impl->distortionFailures;
      m_impl->distortionAccelerationEnabled = false;
      cuda_backend::recoverDistortionRuntime(m_impl->runtime);
      return false;
    }
  }
  catch (...)
  {
    return false;
  }
#else
  (void) batch;
  (void) results;
  return false;
#endif
}

bool CudaContext::isQpaAccelerationAvailable() const noexcept
{
#if VTM_ENABLE_CUDA
  return m_impl->runtime != nullptr && m_impl->qpaAccelerationEnabled;
#else
  return false;
#endif
}

bool CudaContext::computeQpaBatch(const CudaMirrorHandle sourceHandle, const CudaQpaTask *tasks,
                                  const std::uint32_t taskCount, CudaQpaResult *results) noexcept
{
#if VTM_ENABLE_CUDA
  if (!isQpaAccelerationAvailable() || tasks == nullptr || results == nullptr
      || taskCount == 0 || taskCount > CUDA_MAX_QPA_TASKS)
  {
    return false;
  }
  try
  {
    requireRuntime(m_impl.get());
    PictureMirror &sourceMirror = findMirror(m_impl.get(), sourceHandle);
    if (sourceMirror.role != CudaPictureRole::Original || sourceMirror.host.planeCount == 0)
    {
      return false;
    }
    const CudaHostPlaneDesc &host = sourceMirror.host.planes[0];
    if ((host.elementSize != 2 && host.elementSize != 4) || (host.bitDepth != 8 && host.bitDepth != 10)
        || host.strideBytes <= 0 || (host.strideBytes % host.elementSize) != 0)
    {
      return false;
    }

    const std::uint64_t maxSample = (std::uint64_t{ 1 } << host.bitDepth) - 1;
    for (std::uint32_t index = 0; index < taskCount; ++index)
    {
      const CudaQpaTask &task = tasks[index];
      const auto areaInsidePlane = [&host](const CudaQpaRect &area) {
        return area.width != 0 && area.height != 0
               && static_cast<std::uint64_t>(area.x) + area.width <= host.width
               && static_cast<std::uint64_t>(area.y) + area.height <= host.height;
      };
      if (!areaInsidePlane(task.filterArea) || !areaInsidePlane(task.lumaArea)
          || task.filterArea.width < 3 || task.filterArea.height < 3)
      {
        return false;
      }

      const std::uint64_t filterSamples = static_cast<std::uint64_t>(task.filterArea.width - 2)
                                          * (task.filterArea.height - 2);
      const std::uint64_t areaSamples = static_cast<std::uint64_t>(task.lumaArea.width) * task.lumaArea.height;
      // The 3x3 high-pass has positive and negative coefficient sums of 12. These guards prove that
      // every integer reduction remains defined even for the largest descriptor accepted by this API.
      if (filterSamples > std::numeric_limits<std::uint64_t>::max() / (12 * maxSample)
          || areaSamples > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) / maxSample)
      {
        return false;
      }
    }

    try
    {
      ensureDevicePlane(sourceHandle, 0);
      cuda_backend::waitFence(m_impl->runtime, CudaQueue::Qpa, CudaFence::UploadComplete);
      cuda_backend::computeQpaBatch(m_impl->runtime, sourceMirror.planes[0].device, tasks, taskCount, results);
      for (std::uint32_t index = 0; index < taskCount; ++index)
      {
        if (results[index].ticket != tasks[index].ticket || results[index].ctuAddr != tasks[index].ctuAddr)
        {
          throw std::runtime_error("CUDA QPA result ticket/order mismatch");
        }
      }
      return true;
    }
    catch (...)
    {
      ++m_impl->qpaFailures;
      m_impl->qpaAccelerationEnabled = false;
      cuda_backend::recoverQpaRuntime(m_impl->runtime);
      return false;
    }
  }
  catch (...)
  {
    return false;
  }
#else
  (void) sourceHandle;
  (void) tasks;
  (void) taskCount;
  (void) results;
  return false;
#endif
}

std::uint64_t CudaContext::distortionBatchDispatchCount() const
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  return cuda_backend::distortionBatchDispatchCount(m_impl->runtime);
#else
  return 0;
#endif
}

std::uint64_t CudaContext::qpaBatchDispatchCount() const
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  return cuda_backend::qpaBatchDispatchCount(m_impl->runtime);
#else
  return 0;
#endif
}

std::uint64_t CudaContext::qpaTaskCount() const
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  return cuda_backend::qpaTaskCount(m_impl->runtime);
#else
  return 0;
#endif
}

std::uint64_t CudaContext::qpaBatchFailureCount() const noexcept
{
#if VTM_ENABLE_CUDA
  return m_impl->qpaFailures;
#else
  return 0;
#endif
}

std::uint64_t CudaContext::distortionBatchFailureCount() const noexcept
{
#if VTM_ENABLE_CUDA
  return m_impl->distortionFailures;
#else
  return 0;
#endif
}

#if VTM_CUDA_TESTING
void CudaContext::injectReleaseFailuresForTesting(const unsigned asyncFailures, const unsigned immediateFailures)
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  cuda_backend::injectReleaseFailures(m_impl->runtime, asyncFailures, immediateFailures);
#else
  (void) asyncFailures;
  (void) immediateFailures;
  throw std::runtime_error("CUDA release failure injection requires ENABLE_CUDA=ON");
#endif
}


void CudaContext::injectDistortionFailuresForTesting(const unsigned allocationFailureStep,
                                                      const unsigned executionFailures)
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  cuda_backend::injectDistortionFailures(m_impl->runtime, allocationFailureStep, executionFailures);
#else
  (void) allocationFailureStep;
  (void) executionFailures;
#endif
}

void CudaContext::injectQpaFailuresForTesting(const unsigned allocationFailureStep,
                                              const unsigned executionFailures)
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  cuda_backend::injectQpaFailures(m_impl->runtime, allocationFailureStep, executionFailures);
#else
  (void) allocationFailureStep;
  (void) executionFailures;
#endif
}

void CudaContext::injectMirrorPlaneFailuresForTesting(const std::uint8_t plane,
                                                       const unsigned allocationFailureStep,
                                                       const unsigned uploadFailures,
                                                       const unsigned downloadFailures)
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  if (plane >= CUDA_PICTURE_PLANE_COUNT || allocationFailureStep > 3)
  {
    throw std::runtime_error("CUDA mirror plane failure injection arguments are invalid");
  }
  m_impl->mirrorFailurePlane = plane;
  m_impl->mirrorAllocationFailureStep = allocationFailureStep;
  m_impl->mirrorUploadFailures = uploadFailures;
  m_impl->mirrorDownloadFailures = downloadFailures;
#else
  (void) plane;
  (void) allocationFailureStep;
  (void) uploadFailures;
  (void) downloadFailures;
#endif
}
#endif

bool CudaContext::isCreated() const noexcept
{
#if VTM_ENABLE_CUDA
  return m_impl->runtime != nullptr;
#else
  return false;
#endif
}

bool CudaContext::isCompiled() noexcept
{
#if VTM_ENABLE_CUDA
  return true;
#else
  return false;
#endif
}

}   // namespace vtm
