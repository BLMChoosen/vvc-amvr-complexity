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
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <algorithm>
#include <stdexcept>
#include <string>
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
  if (!reset())
  {
    (void) reset();
  }
}

CudaPinnedBuffer::CudaPinnedBuffer(CudaPinnedBuffer &&other) noexcept = default;

CudaPinnedBuffer &CudaPinnedBuffer::operator=(CudaPinnedBuffer &&other) noexcept
{
  if (this != &other)
  {
    if (reset())
    {
      m_impl = std::move(other.m_impl);
    }
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
  if (!reset())
  {
    (void) cuda_backend::releasePinnedHost(allocation);
    throw std::runtime_error("CUDA pinned buffer replacement could not release its previous allocation");
  }
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

bool CudaPinnedBuffer::reset() noexcept
{
  if (!m_impl)
  {
    return true;
  }
#if VTM_ENABLE_CUDA
  if (!cuda_backend::releasePinnedHost(m_impl->allocation))
  {
    return false;
  }
#endif
  m_impl->allocation = nullptr;
  m_impl->bytes      = 0;
  return true;
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
  bool                 downloadPending = false;
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
  bool alfAccelerationEnabled = true;
  std::uint64_t alfFailures = 0;
  std::uint64_t alfMirrorUploadBytes = 0;
  std::uint64_t alfMirrorDownloadBytes = 0;
  std::uint64_t alfIntegrationSynchronizations = 0;
  std::uint64_t alfUploadSubmissionNanoseconds = 0;
  std::uint64_t alfDownloadNanoseconds = 0;
  std::uint64_t alfIntegrationNanoseconds = 0;
  bool dbfAccelerationEnabled = true;
  std::uint64_t dbfFailures = 0;
  std::uint64_t dbfNoOpFrames = 0;
  std::uint64_t dbfMirrorUploadBytes = 0;
  std::uint64_t dbfMirrorDownloadBytes = 0;
  std::uint64_t dbfMirrorSynchronizations = 0;
  std::uint64_t dbfIntegrationSynchronizations = 0;
  std::uint64_t dbfDescriptorCollectionNanoseconds = 0;
  std::uint64_t dbfIntegrationNanoseconds = 0;
  bool loopFilterChainAccelerationEnabled = true;
  std::uint64_t loopFilterChainFailures = 0;
  std::uint64_t loopFilterChainNotEligible = 0;
  std::uint64_t loopFilterChainMirrorUploadBytes = 0;
  std::uint64_t loopFilterChainMirrorDownloadBytes = 0;
  std::uint64_t loopFilterChainIntegrationSynchronizations = 0;
  std::uint64_t loopFilterChainCollectionNanoseconds = 0;
  std::uint64_t loopFilterChainIntegrationNanoseconds = 0;
  std::uint64_t mirrorSynchronizationOperations = 0;
  CudaMirrorMemoryStats mirrorMemory{};
#if VTM_CUDA_TESTING
  bool distortionDiagnosticConstructionFailure = false;
  bool qpaDiagnosticConstructionFailure = false;
  std::uint8_t mirrorFailurePlane = 0xff;
  unsigned mirrorAllocationFailureStep = 0;
  unsigned mirrorUploadFailures = 0;
  unsigned mirrorDownloadFailures = 0;
  CudaLoopFilterChainTestFailurePoint loopFilterChainIntegrationFailure =
    CudaLoopFilterChainTestFailurePoint::None;
#endif
#endif
  std::thread::id ownerThread;
};

namespace
{

class CudaBatchExecutionError final : public std::runtime_error
{
public:
  explicit CudaBatchExecutionError(const std::string &message) : std::runtime_error(message) {}
};

#if VTM_CUDA_TESTING
CudaBatchTestFailurePoint batchFailurePointFromEnvironment(const char *name)
{
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0') return CudaBatchTestFailurePoint::None;
  if (std::strcmp(value, "upload") == 0) return CudaBatchTestFailurePoint::Upload;
  if (std::strcmp(value, "kernel") == 0) return CudaBatchTestFailurePoint::KernelLaunch;
  if (std::strcmp(value, "download") == 0) return CudaBatchTestFailurePoint::ResultDownload;
  if (std::strcmp(value, "sync") == 0) return CudaBatchTestFailurePoint::Completion;
  if (std::strcmp(value, "commit") == 0) return CudaBatchTestFailurePoint::Publication;
  throw std::runtime_error(std::string("Invalid ") + name + " value: " + value);
}
#endif

void requireOwnerThread(const std::thread::id &ownerThread)
{
  if (ownerThread != std::this_thread::get_id())
  {
    throw std::runtime_error("CUDA context APIs must be called from the thread that created the context");
  }
}

bool isPlaneDescriptorSupported(const CudaHostPlaneDesc &plane) noexcept
{
  if (plane.data == nullptr || plane.width == 0 || plane.height == 0 || plane.strideBytes <= 0
      || (plane.elementSize != 1 && plane.elementSize != 2 && plane.elementSize != 4)
      || (plane.bitDepth != 8 && plane.bitDepth != 10) || plane.bitDepth > plane.elementSize * 8)
  {
    return false;
  }

  const std::uint64_t fullWidth = static_cast<std::uint64_t>(plane.marginLeft) + plane.width + plane.marginRight;
  const std::uint64_t fullHeight = static_cast<std::uint64_t>(plane.marginTop) + plane.height + plane.marginBottom;
  if (fullWidth > std::numeric_limits<std::uint64_t>::max() / plane.elementSize)
  {
    return false;
  }
  const std::uint64_t rowBytes = fullWidth * plane.elementSize;
  const std::uint64_t stride = static_cast<std::uint64_t>(plane.strideBytes);
  if (rowBytes > stride || fullHeight == 0)
  {
    return false;
  }
  const std::uint64_t lastRow = fullHeight - 1;
  if (lastRow > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) / stride)
  {
    return false;
  }
  const std::uint64_t lastRowOffset = lastRow * stride;
  if (rowBytes > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) - lastRowOffset)
  {
    return false;
  }
  if (plane.marginTop != 0
      && stride > std::numeric_limits<std::uint64_t>::max() / plane.marginTop)
  {
    return false;
  }
  const std::uint64_t topOffset = static_cast<std::uint64_t>(plane.marginTop) * stride;
  const std::uint64_t leftOffset = static_cast<std::uint64_t>(plane.marginLeft) * plane.elementSize;
  if (leftOffset > std::numeric_limits<std::uint64_t>::max() - topOffset)
  {
    return false;
  }
  const std::uint64_t prefix = topOffset + leftOffset;
  return prefix <= reinterpret_cast<std::uintptr_t>(plane.data);
}

bool isPictureDescriptorSupported(const CudaHostPictureDesc &picture) noexcept
{
  if (picture.planeCount != 1 && picture.planeCount != CUDA_PICTURE_PLANE_COUNT)
  {
    return false;
  }
  for (std::size_t index = 0; index < picture.planeCount; ++index)
  {
    if (!isPlaneDescriptorSupported(picture.planes[index]))
    {
      return false;
    }
  }
  if (picture.planeCount == 1)
  {
    return true;
  }

  const CudaHostPlaneDesc &luma = picture.planes[0];
  const CudaHostPlaneDesc &cb = picture.planes[1];
  const CudaHostPlaneDesc &cr = picture.planes[2];
  if (cb.width != cr.width || cb.height != cr.height)
  {
    return false;
  }
  const std::uint32_t halfWidth = (luma.width >> 1) + (luma.width & 1);
  const std::uint32_t halfHeight = (luma.height >> 1) + (luma.height & 1);
  const bool is420 = cb.width == halfWidth && cb.height == halfHeight;
  const bool is422 = cb.width == halfWidth && cb.height == luma.height;
  const bool is444 = cb.width == luma.width && cb.height == luma.height;
  return is420 || is422 || is444;
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
  if (!isPictureDescriptorSupported(picture))
  {
    throw std::runtime_error("CUDA picture mirror descriptor or chroma geometry is unsupported");
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

template<typename ContextImpl>
void quarantineLumaMirrorNoexcept(ContextImpl *impl, const CudaMirrorHandle handle) noexcept
{
  if (impl == nullptr) return;
  const auto found = impl->mirrors.find(handle);
  if (handle == 0 || found == impl->mirrors.end() || found->second->host.planeCount == 0) return;
  PictureMirrorPlane &plane = found->second->planes[0];
  plane.state = CudaMirrorState::HostValid;
  plane.uploadPending = false;
  plane.downloadPending = false;
}

template<typename ContextImpl>
void recoverAndQuarantineAlfNoexcept(ContextImpl *impl, const CudaMirrorHandle handle) noexcept
{
  if (impl == nullptr || impl->runtime == nullptr)
  {
    quarantineLumaMirrorNoexcept(impl, handle);
    return;
  }
  cuda_backend::recoverAlfRuntime(impl->runtime);
  try
  {
    // Drain upload/download work as well as the recovered ALF queue before publishing HostValid.
    // This stays local to the failing context and deliberately cannot replace the original error.
    cuda_backend::synchronizeRuntimeContext(impl->runtime);
  }
  catch (...)
  {
  }
  quarantineLumaMirrorNoexcept(impl, handle);
}

template<typename ContextImpl>
void recoverAndQuarantineDbfNoexcept(ContextImpl *impl, const CudaMirrorHandle handle) noexcept
{
  if (impl == nullptr || impl->runtime == nullptr)
  {
    quarantineLumaMirrorNoexcept(impl, handle);
    return;
  }
  cuda_backend::recoverDbfRuntime(impl->runtime);
  try { cuda_backend::synchronizeRuntimeContext(impl->runtime); } catch (...) {}
  quarantineLumaMirrorNoexcept(impl, handle);
}

template<typename ContextImpl>
void recoverAndQuarantineLoopFilterChainNoexcept(ContextImpl *impl, const CudaMirrorHandle handle) noexcept
{
  if (impl == nullptr || impl->runtime == nullptr)
  {
    quarantineLumaMirrorNoexcept(impl, handle);
    return;
  }
  cuda_backend::recoverLoopFilterChainRuntime(impl->runtime);
  cuda_backend::recoverDbfRuntime(impl->runtime);
  cuda_backend::recoverAlfRuntime(impl->runtime);
  try { cuda_backend::synchronizeRuntimeContext(impl->runtime); } catch (...) {}
  quarantineLumaMirrorNoexcept(impl, handle);
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
  const std::size_t stride = static_cast<std::size_t>(host.strideBytes);
  const std::size_t activePrefix = checkedAdd(
    checkedMultiply(host.marginTop, stride, "active row prefix"),
    checkedMultiply(host.marginLeft, host.elementSize, "active column prefix"), "active prefix");
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
  const std::size_t row = offset / stride;
  const std::size_t columnBytes = offset % stride;
  if (columnBytes % host.elementSize != 0)
  {
    throw std::runtime_error("CUDA distortion block is not sample aligned");
  }
  const std::size_t column = columnBytes / host.elementSize;
  const std::size_t fullWidth = checkedAdd(checkedAdd(host.marginLeft, host.width, "width"),
                                           host.marginRight, "width");
  const std::size_t fullHeight = checkedAdd(checkedAdd(host.marginTop, host.height, "height"),
                                            host.marginBottom, "height");
  if (row >= fullHeight || height > fullHeight - row || column >= fullWidth || width > fullWidth - column)
  {
    throw std::runtime_error("CUDA distortion block exceeds its registered mirror");
  }

  const PictureMirrorPlane &mirrorPlane = mirror.planes[planeIndex];
  if (mirrorPlane.deviceBase == nullptr)
  {
    return nullptr;
  }
  const CudaDevicePlaneDesc &device = mirrorPlane.device;
  const std::size_t deviceOffset = checkedAdd(checkedMultiply(row, device.pitchBytes, "device row offset"),
                                               checkedMultiply(column, host.elementSize,
                                                               "device column offset"),
                                               "device block offset");
  if (deviceOffset >= mirrorPlane.deviceBytes
      || deviceOffset > std::numeric_limits<std::uintptr_t>::max()
                          - reinterpret_cast<std::uintptr_t>(mirrorPlane.deviceBase))
  {
    throw std::runtime_error("CUDA distortion device block address overflows");
  }
  return reinterpret_cast<const void *>(reinterpret_cast<std::uintptr_t>(mirrorPlane.deviceBase) + deviceOffset);
}

unsigned char *hostBase(const CudaHostPlaneDesc &plane)
{
  const std::size_t prefix = checkedAdd(
    checkedMultiply(plane.marginTop, static_cast<std::size_t>(plane.strideBytes), "host row prefix"),
    checkedMultiply(plane.marginLeft, plane.elementSize, "host column prefix"), "host prefix");
  const std::uintptr_t active = reinterpret_cast<std::uintptr_t>(plane.data);
  if (prefix > active)
  {
    throw std::runtime_error("CUDA picture host base address underflows");
  }
  return reinterpret_cast<unsigned char *>(active - prefix);
}

void stageHostPlane(PictureMirror &mirror, const std::size_t index)
{
  const CudaHostPlaneDesc &plane = mirror.host.planes[index];
  PictureMirrorPlane &mirrorPlane = mirror.planes[index];
  const std::uintptr_t sourceBase = reinterpret_cast<std::uintptr_t>(hostBase(plane));
  const std::uintptr_t destinationBase = reinterpret_cast<std::uintptr_t>(mirrorPlane.staging.data());
  for (std::size_t row = 0; row < mirrorPlane.fullHeight; ++row)
  {
    const std::size_t sourceOffset = checkedMultiply(row, static_cast<std::size_t>(plane.strideBytes),
                                                     "host staging source offset");
    const std::size_t destinationOffset = checkedMultiply(row, mirrorPlane.fullRowBytes,
                                                          "host staging destination offset");
    if (sourceOffset > std::numeric_limits<std::uintptr_t>::max() - sourceBase
        || destinationOffset > std::numeric_limits<std::uintptr_t>::max() - destinationBase)
    {
      throw std::runtime_error("CUDA picture staging address overflows");
    }
    std::memcpy(reinterpret_cast<void *>(destinationBase + destinationOffset),
                reinterpret_cast<const void *>(sourceBase + sourceOffset), mirrorPlane.fullRowBytes);
  }
}

void unstageHostPlane(PictureMirror &mirror, const std::size_t index)
{
  const CudaHostPlaneDesc &plane = mirror.host.planes[index];
  PictureMirrorPlane &mirrorPlane = mirror.planes[index];
  const std::uintptr_t destinationBase = reinterpret_cast<std::uintptr_t>(hostBase(plane));
  const std::uintptr_t sourceBase = reinterpret_cast<std::uintptr_t>(mirrorPlane.staging.data());
  for (std::size_t row = 0; row < mirrorPlane.fullHeight; ++row)
  {
    const std::size_t destinationOffset = checkedMultiply(row, static_cast<std::size_t>(plane.strideBytes),
                                                          "host unstage destination offset");
    const std::size_t sourceOffset = checkedMultiply(row, mirrorPlane.fullRowBytes,
                                                     "host unstage source offset");
    if (destinationOffset > std::numeric_limits<std::uintptr_t>::max() - destinationBase
        || sourceOffset > std::numeric_limits<std::uintptr_t>::max() - sourceBase)
    {
      throw std::runtime_error("CUDA picture unstaging address overflows");
    }
    std::memcpy(reinterpret_cast<void *>(destinationBase + destinationOffset),
                reinterpret_cast<const void *>(sourceBase + sourceOffset), mirrorPlane.fullRowBytes);
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
void accountTransfer(ContextImpl *impl, const CudaPictureRole role, const std::size_t plane,
                     const std::size_t bytes, const bool upload)
{
  CudaMirrorMemoryUsage &total = impl->mirrorMemory.total;
  CudaMirrorMemoryUsage &detail = impl->mirrorMemory.byRoleAndPlane[roleIndex(role)][plane];
  if (upload)
  {
    total.uploadedBytes += bytes;
    detail.uploadedBytes += bytes;
  }
  else
  {
    total.downloadedBytes += bytes;
    detail.downloadedBytes += bytes;
  }
}

template<typename ContextImpl>
void ensureMirrorPlaneResources(ContextImpl *impl, PictureMirror &mirror, const std::size_t index)
{
  PictureMirrorPlane &mirrorPlane = mirror.planes[index];
  const bool needsDevice = mirrorPlane.deviceBase == nullptr;
  const bool needsPinned = !mirrorPlane.staging;
  mirrorPlane.requested = true;
  if (!needsDevice && !needsPinned)
  {
    return;
  }
  const CudaHostPlaneDesc &host = mirror.host.planes[index];
  const std::size_t fullWidth = static_cast<std::size_t>(host.marginLeft) + host.width + host.marginRight;
  const std::size_t fullHeight = static_cast<std::size_t>(host.marginTop) + host.height + host.marginBottom;
  const std::size_t rowBytes = checkedMultiply(fullWidth, host.elementSize, "row size");
  const std::size_t pitch = checkedAdd(rowBytes, 127, "device pitch") & ~std::size_t(127);
  const std::size_t deviceBytes = checkedMultiply(pitch, fullHeight, "device allocation");
  const std::size_t pinnedBytes = checkedMultiply(rowBytes, fullHeight, "staging allocation");
  const std::size_t requiredBytes = checkedAdd(needsDevice ? deviceBytes : 0,
                                                needsPinned ? pinnedBytes : 0,
                                                "resource budget requirement");
  const std::size_t currentBytes = checkedAdd(
    static_cast<std::size_t>(impl->mirrorMemory.total.currentDeviceBytes),
    static_cast<std::size_t>(impl->mirrorMemory.total.currentPinnedBytes), "current resource budget usage");
  if (currentBytes > impl->mirrorMemory.budgetBytes
      || requiredBytes > impl->mirrorMemory.budgetBytes - currentBytes)
  {
    ++impl->mirrorMemory.budgetRejections;
    throw std::runtime_error("CUDA picture mirror memory budget exceeded");
  }

  CudaPinnedBuffer staging;
  if (needsPinned)
  {
#if VTM_CUDA_TESTING
    if (impl->mirrorFailurePlane == index && impl->mirrorAllocationFailureStep == 1)
    {
      impl->mirrorAllocationFailureStep = 0;
      throw std::runtime_error("Injected CUDA mirror pinned allocation failure");
    }
#endif
    staging.allocate(pinnedBytes);
  }
  void *deviceBase = nullptr;
  if (needsDevice)
  {
#if VTM_CUDA_TESTING
    if (impl->mirrorFailurePlane == index && impl->mirrorAllocationFailureStep == 2)
    {
      impl->mirrorAllocationFailureStep = 0;
      throw std::runtime_error("Injected CUDA mirror device allocation failure");
    }
#endif
    deviceBase = cuda_backend::allocateDevice(impl->runtime, deviceBytes, CudaQueue::Upload);
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
  }

  mirrorPlane.fullRowBytes = rowBytes;
  mirrorPlane.fullHeight = fullHeight;
  if (needsDevice)
  {
    CudaDevicePlaneDesc device{};
    const std::size_t activeOffset = checkedAdd(
      checkedMultiply(host.marginTop, pitch, "device active row offset"),
      checkedMultiply(host.marginLeft, host.elementSize, "device active column offset"),
      "device active offset");
    const std::uintptr_t deviceAddress = reinterpret_cast<std::uintptr_t>(deviceBase);
    if (activeOffset >= deviceBytes
        || activeOffset > std::numeric_limits<std::uintptr_t>::max() - deviceAddress)
    {
      throw std::runtime_error("CUDA picture device active address overflows");
    }
    device.data = reinterpret_cast<void *>(deviceAddress + activeOffset);
    device.pitchBytes   = pitch;
    device.width        = host.width;
    device.height       = host.height;
    device.marginLeft   = host.marginLeft;
    device.marginRight  = host.marginRight;
    device.marginTop    = host.marginTop;
    device.marginBottom = host.marginBottom;
    device.elementSize  = host.elementSize;
    device.bitDepth     = host.bitDepth;
    mirrorPlane.deviceBase = deviceBase;
    mirrorPlane.deviceBytes = deviceBytes;
    mirrorPlane.device = device;
    accountAllocation(impl, mirror.role, index, deviceBytes, 0);
  }
  if (needsPinned)
  {
    mirrorPlane.staging = std::move(staging);
    mirrorPlane.pinnedBytes = pinnedBytes;
    accountAllocation(impl, mirror.role, index, 0, pinnedBytes);
  }
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
      if (plane.staging.reset())
      {
        accountPinnedRelease(impl, mirror.role, index, plane.pinnedBytes);
        plane.pinnedBytes = 0;
      }
      else if (!firstError)
      {
        firstError = std::make_exception_ptr(std::runtime_error("CUDA pinned mirror release failed"));
      }
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

bool hasMirrorAllocations(const PictureMirror &mirror)
{
  for (const PictureMirrorPlane &plane : mirror.planes)
  {
    if (plane.deviceBase != nullptr || plane.staging)
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
        if (plane.staging.reset())
        {
          accountPinnedRelease(impl, mirror.role, index, plane.pinnedBytes);
          plane.pinnedBytes = 0;
        }
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
#if VTM_CUDA_TESTING
  m_impl->distortionDiagnosticConstructionFailure = false;
  m_impl->qpaDiagnosticConstructionFailure = false;
#endif
  m_impl->alfAccelerationEnabled = true;
  m_impl->alfFailures = 0;
  m_impl->alfMirrorUploadBytes = 0;
  m_impl->alfMirrorDownloadBytes = 0;
  m_impl->alfIntegrationSynchronizations = 0;
  m_impl->alfUploadSubmissionNanoseconds = 0;
  m_impl->alfDownloadNanoseconds = 0;
  m_impl->alfIntegrationNanoseconds = 0;
  m_impl->dbfAccelerationEnabled = true;
  m_impl->dbfFailures = 0;
  m_impl->dbfNoOpFrames = 0;
  m_impl->dbfMirrorUploadBytes = 0;
  m_impl->dbfMirrorDownloadBytes = 0;
  m_impl->dbfMirrorSynchronizations = 0;
  m_impl->dbfIntegrationSynchronizations = 0;
  m_impl->dbfDescriptorCollectionNanoseconds = 0;
  m_impl->dbfIntegrationNanoseconds = 0;
  m_impl->loopFilterChainAccelerationEnabled = true;
  m_impl->loopFilterChainFailures = 0;
  m_impl->loopFilterChainNotEligible = 0;
  m_impl->loopFilterChainMirrorUploadBytes = 0;
  m_impl->loopFilterChainMirrorDownloadBytes = 0;
  m_impl->loopFilterChainIntegrationSynchronizations = 0;
  m_impl->loopFilterChainCollectionNanoseconds = 0;
  m_impl->loopFilterChainIntegrationNanoseconds = 0;
#if VTM_CUDA_TESTING
  m_impl->loopFilterChainIntegrationFailure = CudaLoopFilterChainTestFailurePoint::None;
#endif
  m_impl->mirrorSynchronizationOperations = 0;
  m_impl->mirrorMemory = CudaMirrorMemoryStats{};
  m_impl->mirrorMemory.budgetBytes = CUDA_DEFAULT_MIRROR_MEMORY_BUDGET_BYTES;
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
    m_impl->mirrorMemory.total.currentDeviceBytes = 0;
    m_impl->mirrorMemory.total.currentPinnedBytes = 0;
    for (auto &role : m_impl->mirrorMemory.byRoleAndPlane)
    {
      for (CudaMirrorMemoryUsage &plane : role)
      {
        plane.currentDeviceBytes = 0;
        plane.currentPinnedBytes = 0;
      }
    }
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
    if (!slot->second || !hasMirrorAllocations(*slot->second))
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
    for (std::size_t index = 0; index < mirror.host.planeCount; ++index)
    {
      PictureMirrorPlane &plane = mirror.planes[index];
      plane.state = CudaMirrorState::HostValid;
      plane.uploadPending = false;
      plane.downloadPending = false;
      if (plane.requested)
      {
        ensureMirrorPlaneResources(m_impl.get(), mirror, index);
      }
    }
    return;
  }

  // Release before changing the host descriptor so a failed async+immediate free remains owned by the registered
  // mirror and can be retried. Once every old plane is released, partial allocation of the new layout is likewise
  // retained by the mirror and remains retryable without leaking either layout.
  try
  {
    releaseMirrorResources(m_impl.get(), mirror);
  }
  catch (...)
  {
    throw;
  }
  mirror.host = picture;
  for (std::size_t index = 0; index < picture.planeCount; ++index)
  {
    PictureMirrorPlane &plane = mirror.planes[index];
    plane.state = CudaMirrorState::HostValid;
    plane.uploadPending = false;
    plane.downloadPending = false;
    if (plane.requested)
    {
      ensureMirrorPlaneResources(m_impl.get(), mirror, index);
    }
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

  if (!hasMirrorAllocations(mirror))
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
  requireRuntime(m_impl.get());
  const PictureMirror &mirror = findMirror(m_impl.get(), handle);
  return devicePicturePlanes(handle, activePlaneMask(mirror));
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
  // Allocate every requested plane before submitting any transfer. An allocation failure can leave lazy resources
  // owned by the mirror, but never exposes partially uploaded device data.
  for (std::uint8_t index = 0; index < mirror.host.planeCount; ++index)
  {
    if ((planes & planeBit(index)) == 0) continue;
    ensureMirrorPlaneResources(m_impl.get(), mirror, index);
  }
  bool needsUploadSynchronization = false;
  for (std::uint8_t index = 0; index < mirror.host.planeCount; ++index)
  {
    needsUploadSynchronization = needsUploadSynchronization
                                 || ((planes & planeBit(index)) != 0 && mirror.planes[index].uploadPending);
  }
  if (needsUploadSynchronization)
  {
    cuda_backend::synchronizeQueue(m_impl->runtime, CudaQueue::Upload);
    ++m_impl->mirrorSynchronizationOperations;
    for (PictureMirrorPlane &mirrorPlane : mirror.planes)
    {
      mirrorPlane.uploadPending = false;
    }
  }

  CudaPlaneMask submitted = 0;
  try
  {
    for (std::uint8_t index = 0; index < mirror.host.planeCount; ++index)
    {
      if ((planes & planeBit(index)) == 0) continue;
      PictureMirrorPlane &mirrorPlane = mirror.planes[index];
      if (mirrorPlane.state != CudaMirrorState::HostValid) continue;
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
      accountTransfer(m_impl.get(), mirror.role, index, mirrorPlane.pinnedBytes, true);
      submitted |= planeBit(index);
    }
    if (submitted != 0)
    {
      cuda_backend::recordFence(m_impl->runtime, CudaQueue::Upload, CudaFence::UploadComplete);
      cuda_backend::waitFence(m_impl->runtime, CudaQueue::Compute, CudaFence::UploadComplete);
      for (std::uint8_t index = 0; index < mirror.host.planeCount; ++index)
      {
        if ((submitted & planeBit(index)) == 0) continue;
        mirror.planes[index].state = CudaMirrorState::Synchronized;
        mirror.planes[index].uploadPending = true;
      }
    }
  }
  catch (...)
  {
    const std::exception_ptr firstError = std::current_exception();
    if (submitted != 0)
    {
      bool synchronized = false;
      try
      {
        cuda_backend::synchronizeQueue(m_impl->runtime, CudaQueue::Upload);
        ++m_impl->mirrorSynchronizationOperations;
        synchronized = true;
      }
      catch (...)
      {
      }
      for (std::uint8_t index = 0; index < mirror.host.planeCount; ++index)
      {
        if ((submitted & planeBit(index)) == 0) continue;
        mirror.planes[index].state = synchronized ? CudaMirrorState::Synchronized
                                                   : CudaMirrorState::HostValid;
        mirror.planes[index].uploadPending = !synchronized;
      }
    }
    std::rethrow_exception(firstError);
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
  bool hasPendingDownload = false;
  for (const PictureMirrorPlane &mirrorPlane : mirror.planes)
  {
    hasPendingDownload = hasPendingDownload || mirrorPlane.downloadPending;
  }
  if (hasPendingDownload)
  {
    cuda_backend::synchronizeQueue(m_impl->runtime, CudaQueue::Download);
    ++m_impl->mirrorSynchronizationOperations;
    for (PictureMirrorPlane &mirrorPlane : mirror.planes) mirrorPlane.downloadPending = false;
  }
  CudaPlaneMask submitted = 0;
  try
  {
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
      accountTransfer(m_impl.get(), mirror.role, index, mirrorPlane.pinnedBytes, false);
      mirrorPlane.downloadPending = true;
      submitted |= planeBit(index);
    }
    if (submitted != 0)
    {
      cuda_backend::synchronizeQueue(m_impl->runtime, CudaQueue::Download);
      ++m_impl->mirrorSynchronizationOperations;
      for (std::uint8_t index = 0; index < mirror.host.planeCount; ++index)
      {
        if ((submitted & planeBit(index)) == 0) continue;
        unstageHostPlane(mirror, index);
        mirror.planes[index].uploadPending = false;
        mirror.planes[index].downloadPending = false;
        mirror.planes[index].state = CudaMirrorState::Synchronized;
      }
    }
  }
  catch (...)
  {
    const std::exception_ptr firstError = std::current_exception();
    if (submitted != 0)
    {
      bool synchronized = false;
      try
      {
        cuda_backend::synchronizeQueue(m_impl->runtime, CudaQueue::Download);
        ++m_impl->mirrorSynchronizationOperations;
        synchronized = true;
      }
      catch (...)
      {
      }
      for (std::uint8_t index = 0; index < mirror.host.planeCount; ++index)
      {
        if ((submitted & planeBit(index)) == 0) continue;
        if (synchronized)
        {
          unstageHostPlane(mirror, index);
          mirror.planes[index].state = CudaMirrorState::Synchronized;
        }
        mirror.planes[index].downloadPending = !synchronized;
      }
    }
    std::rethrow_exception(firstError);
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

void CudaContext::setPictureMirrorMemoryBudget(const std::uint64_t bytes)
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  const std::uint64_t current = m_impl->mirrorMemory.total.currentDeviceBytes
                                + m_impl->mirrorMemory.total.currentPinnedBytes;
  if (bytes == 0 || bytes < current)
  {
    throw std::runtime_error("CUDA picture mirror memory budget is zero or below current usage");
  }
  m_impl->mirrorMemory.budgetBytes = bytes;
#else
  (void) bytes;
  throw std::runtime_error("CUDA picture mirror requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

bool CudaContext::isPictureMirrorDescriptorSupported(const CudaHostPictureDesc &picture) noexcept
{
  return isPictureDescriptorSupported(picture);
}

bool CudaContext::isDistortionAccelerationAvailable() const noexcept
{
#if VTM_ENABLE_CUDA
  return m_impl->runtime != nullptr && m_impl->distortionAccelerationEnabled;
#else
  return false;
#endif
}

bool CudaContext::computeDistortionBatch(const CudaDistortionBatchDesc &batch, std::uint64_t *results)
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
  // Mapping validates the operational mirror state. It deliberately throws instead of reporting NotEligible.
  (void) mapHostBlock(sourceMirror, batch.sourcePlane, batch.source, batch.width, batch.height);
  (void) mapHostBlock(referenceMirror, batch.referencePlane, batch.candidateGrid.reference,
                      static_cast<std::uint32_t>(referenceWidth64),
                      static_cast<std::uint32_t>(referenceHeight64));
  try
  {
#if VTM_CUDA_TESTING
    const CudaBatchTestFailurePoint environmentFailure =
      batchFailurePointFromEnvironment("VTM_CUDA_SAD_TEST_FAILURE");
    if (environmentFailure != CudaBatchTestFailurePoint::None)
    {
      if (environmentFailure == CudaBatchTestFailurePoint::Upload)
      {
        m_impl->mirrorFailurePlane = batch.sourcePlane;
        m_impl->mirrorUploadFailures = 1;
      }
      else
      {
        cuda_backend::injectDistortionFailurePoint(m_impl->runtime, environmentFailure);
      }
    }
#endif
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
  catch (const std::exception &error)
  {
    ++m_impl->distortionFailures;
    m_impl->distortionAccelerationEnabled = false;
    cuda_backend::recoverDistortionRuntime(m_impl->runtime);
#if VTM_CUDA_TESTING
    if (m_impl->distortionDiagnosticConstructionFailure)
    {
      m_impl->distortionDiagnosticConstructionFailure = false;
      throw std::bad_alloc();
    }
#endif
    throw CudaBatchExecutionError(std::string("CUDA SAD execution failed after selection: ") + error.what());
  }
  catch (...)
  {
    ++m_impl->distortionFailures;
    m_impl->distortionAccelerationEnabled = false;
    cuda_backend::recoverDistortionRuntime(m_impl->runtime);
    throw CudaBatchExecutionError("CUDA SAD execution failed after selection: unknown error");
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

bool CudaContext::isAlfAccelerationAvailable() const noexcept
{
#if VTM_ENABLE_CUDA
  return m_impl->runtime != nullptr && m_impl->alfAccelerationEnabled;
#else
  return false;
#endif
}

CudaAlfDispatchResult CudaContext::filterAlfLumaFrame(const CudaMirrorHandle reconstructionHandle,
                                                       const CudaAlfLumaFrame &frame,
                                                       const CudaAlfCtuParam *ctus,
                                                       const std::uint32_t ctuCount,
                                                       CudaAlfClassifier *classifiers)
{
#if VTM_ENABLE_CUDA
  const std::uint64_t pixels = static_cast<std::uint64_t>(frame.width) * frame.height;
  const std::uint64_t expectedCtusInWidth = frame.ctuWidth == 0 ? 0
    : (static_cast<std::uint64_t>(frame.width) + frame.ctuWidth - 1) / frame.ctuWidth;
  const std::uint64_t expectedCtusInHeight = frame.ctuHeight == 0 ? 0
    : (static_cast<std::uint64_t>(frame.height) + frame.ctuHeight - 1) / frame.ctuHeight;
  if (!isAlfAccelerationAvailable() || ctus == nullptr || ctuCount == 0 || ctuCount > CUDA_ALF_MAX_CTUS
      || frame.width == 0 || frame.height == 0 || frame.ctuWidth == 0 || frame.ctuHeight == 0
      || frame.ctusInWidth == 0 || frame.ctusInHeight == 0
      || frame.ctusInWidth != expectedCtusInWidth || frame.ctusInHeight != expectedCtusInHeight
      || expectedCtusInWidth * expectedCtusInHeight != ctuCount
      || pixels < CUDA_ALF_MIN_FRAME_PIXELS
      || (frame.width & 3) != 0 || (frame.height & 3) != 0
      || frame.vbCtuHeight <= 0 || (frame.vbCtuHeight & (frame.vbCtuHeight - 1)) != 0
      || frame.vbCtuHeight != static_cast<std::int32_t>(frame.ctuHeight)
      || frame.vbPos != frame.vbCtuHeight - 4
      || (frame.bitDepth != 8 && frame.bitDepth != 10)
      || (frame.elementSize != 2 && frame.elementSize != 4)
      || pixels > std::numeric_limits<std::size_t>::max() / frame.elementSize
      || frame.minSample != 0 || frame.maxSample != (std::int32_t{ 1 } << frame.bitDepth) - 1)
  {
    return CudaAlfDispatchResult::NotEligible;
  }
  for (std::uint32_t index = 0; index < ctuCount; ++index)
  {
    const CudaAlfCtuParam &ctu = ctus[index];
    const std::uint64_t expectedX64 = static_cast<std::uint64_t>(index % frame.ctusInWidth) * frame.ctuWidth;
    const std::uint64_t expectedY64 = static_cast<std::uint64_t>(index / frame.ctusInWidth) * frame.ctuHeight;
    const std::uint32_t expectedWidth = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(frame.ctuWidth, static_cast<std::uint64_t>(frame.width) - expectedX64));
    const std::uint32_t expectedHeight = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(frame.ctuHeight, static_cast<std::uint64_t>(frame.height) - expectedY64));
    if (ctu.x != expectedX64 || ctu.y != expectedY64 || ctu.width != expectedWidth
        || ctu.height != expectedHeight || ctu.enabled > 1)
    {
      return CudaAlfDispatchResult::NotEligible;
    }
    if (ctu.enabled != 0)
    {
      const std::int32_t maximumClip = std::int32_t{ 1 } << frame.bitDepth;
      for (std::uint32_t classIndex = 0; classIndex < CUDA_ALF_CLASSES; ++classIndex)
      {
        for (std::uint32_t coefficientIndex = 0; coefficientIndex < CUDA_ALF_COEFFICIENTS;
             ++coefficientIndex)
        {
          const std::uint32_t parameterIndex = classIndex * CUDA_ALF_COEFFICIENTS + coefficientIndex;
          const std::int32_t clip = ctu.clipValues[parameterIndex];
          const std::int16_t coefficient = ctu.coefficients[parameterIndex];
          if (clip < 0 || clip > maximumClip
              || (coefficientIndex + 1 == CUDA_ALF_COEFFICIENTS
                    ? coefficient != 128
                    : coefficient < -128 || coefficient > 127))
          {
            return CudaAlfDispatchResult::NotEligible;
          }
        }
      }
    }
  }

  requireRuntime(m_impl.get());
  PictureMirror &mirror = findMirror(m_impl.get(), reconstructionHandle);
  if (mirror.role != CudaPictureRole::Reconstruction || mirror.host.planeCount == 0)
  {
    return CudaAlfDispatchResult::NotEligible;
  }
  const CudaHostPlaneDesc &host = mirror.host.planes[0];
  if (host.width != frame.width || host.height != frame.height || host.bitDepth != frame.bitDepth
      || host.elementSize != frame.elementSize || host.data == nullptr || host.strideBytes <= 0
      || static_cast<std::uint64_t>(host.strideBytes)
           < static_cast<std::uint64_t>(host.width) * host.elementSize)
  {
    return CudaAlfDispatchResult::NotEligible;
  }

  const auto integrationStart = std::chrono::steady_clock::now();
  const CudaMirrorMemoryUsage mirrorBefore = m_impl->mirrorMemory.total;
  const std::uint64_t mirrorSynchronizationsBefore = m_impl->mirrorSynchronizationOperations;
  try
  {
#if VTM_CUDA_TESTING
    if (const char *failure = std::getenv("VTM_CUDA_ALF_TEST_FAILURE");
        failure != nullptr && std::strcmp(failure, "kernel-launch") == 0)
    {
      cuda_backend::injectAlfFailure(m_impl->runtime, CudaAlfTestFailurePoint::KernelLaunch);
    }
#endif
    const auto uploadStart = std::chrono::steady_clock::now();
    ensureDevicePlane(reconstructionHandle, 0);
    cuda_backend::waitFence(m_impl->runtime, CudaQueue::Alf, CudaFence::UploadComplete);
    const auto uploadEnd = std::chrono::steady_clock::now();
    const AlfAccelerationStats runtimeBefore = cuda_backend::alfStats(m_impl->runtime);
    cuda_backend::filterAlfLumaFrame(m_impl->runtime, mirror.planes[0].device, frame, ctus, ctuCount,
                                     classifiers);
    const AlfAccelerationStats runtimeAfter = cuda_backend::alfStats(m_impl->runtime);

    mirror.planes[0].state = CudaMirrorState::DeviceValid;
    mirror.planes[0].uploadPending = false;
    const auto downloadStart = std::chrono::steady_clock::now();
    ensureHostPlane(reconstructionHandle, 0);
    const auto integrationEnd = std::chrono::steady_clock::now();
    const CudaMirrorMemoryUsage mirrorAfter = m_impl->mirrorMemory.total;
    const std::uint64_t uploadBytes = mirrorAfter.uploadedBytes - mirrorBefore.uploadedBytes;
    const std::uint64_t downloadBytes = mirrorAfter.downloadedBytes - mirrorBefore.downloadedBytes;
    const std::uint64_t runtimeSynchronizations =
      runtimeAfter.runtimeSynchronizations - runtimeBefore.runtimeSynchronizations;
    const std::uint64_t mirrorSynchronizations =
      m_impl->mirrorSynchronizationOperations - mirrorSynchronizationsBefore;
    m_impl->alfMirrorUploadBytes += uploadBytes;
    m_impl->alfMirrorDownloadBytes += downloadBytes;
    // Include the runtime's host synchronizations, actual mirror host synchronizations, and the
    // upload-to-ALF stream dependency.  Keep runtimeSynchronizations independently reportable.
    m_impl->alfIntegrationSynchronizations += runtimeSynchronizations + mirrorSynchronizations + 1;
    // This interval measures host-side staging/submission and dependency enqueue only. The ALF
    // runtime synchronizes its stream, so runtimeNanoseconds includes any device wait for upload.
    m_impl->alfUploadSubmissionNanoseconds += static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(uploadEnd - uploadStart).count());
    m_impl->alfDownloadNanoseconds += static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(integrationEnd - downloadStart).count());
    m_impl->alfIntegrationNanoseconds += static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(integrationEnd - integrationStart).count());
    return CudaAlfDispatchResult::Executed;
  }
  catch (const std::exception &error)
  {
    ++m_impl->alfFailures;
    m_impl->alfAccelerationEnabled = false;
    // The host reconstruction remains authoritative. Recover outstanding work, then invalidate the local
    // mirror state without calling a potentially throwing publication API before propagating the first error.
    recoverAndQuarantineAlfNoexcept(m_impl.get(), reconstructionHandle);
    throw std::runtime_error(std::string("CUDA ALF execution failed: ") + error.what());
  }
  catch (...)
  {
    ++m_impl->alfFailures;
    m_impl->alfAccelerationEnabled = false;
    recoverAndQuarantineAlfNoexcept(m_impl.get(), reconstructionHandle);
    throw std::runtime_error("CUDA ALF execution failed with an unknown error");
  }
#else
  (void) reconstructionHandle;
  (void) frame;
  (void) ctus;
  (void) ctuCount;
  (void) classifiers;
  return CudaAlfDispatchResult::NotEligible;
#endif
}

AlfAccelerationStats CudaContext::alfStats() const noexcept
{
  AlfAccelerationStats stats{};
#if VTM_ENABLE_CUDA
  stats = cuda_backend::alfStats(m_impl->runtime);
  stats.mirrorUploadBytes = m_impl->alfMirrorUploadBytes;
  stats.mirrorDownloadBytes = m_impl->alfMirrorDownloadBytes;
  stats.integrationSynchronizations = m_impl->alfIntegrationSynchronizations;
  stats.uploadSubmissionNanoseconds = m_impl->alfUploadSubmissionNanoseconds;
  stats.downloadNanoseconds = m_impl->alfDownloadNanoseconds;
  stats.integrationNanoseconds = m_impl->alfIntegrationNanoseconds;
  stats.failures = m_impl->alfFailures;
  stats.enabled = m_impl->runtime != nullptr && m_impl->alfAccelerationEnabled;
  stats.poisoned = m_impl->runtime != nullptr && !m_impl->alfAccelerationEnabled;
#endif
  return stats;
}

bool CudaContext::isDbfAccelerationAvailable() const noexcept
{
#if VTM_ENABLE_CUDA
  return m_impl->runtime != nullptr && m_impl->dbfAccelerationEnabled;
#else
  return false;
#endif
}

CudaDbfDispatchResult CudaContext::filterDbfLumaFrame(const CudaMirrorHandle reconstructionHandle,
                                                       const CudaDbfFrame &frame,
                                                       const CudaDbfLumaTask *tasks,
                                                       const std::uint32_t taskCount)
{
#if VTM_ENABLE_CUDA
  const std::uint64_t pixels = std::uint64_t(frame.width) * frame.height;
  if (!isDbfAccelerationAvailable() || (tasks == nullptr && taskCount != 0) || taskCount > CUDA_DBF_MAX_TASKS
      || frame.width == 0 || frame.height == 0 || pixels < CUDA_DBF_MIN_FRAME_PIXELS
      || (frame.bitDepth != 8 && frame.bitDepth != 10)
      || (frame.elementSize != 2 && frame.elementSize != 4))
    return CudaDbfDispatchResult::NotEligible;
  for (std::uint32_t i = 0; i < taskCount; ++i)
  {
    const CudaDbfLumaTask &t = tasks[i];
    const bool validLenP = t.maxFilterLenP == 1 || t.maxFilterLenP == 2 || t.maxFilterLenP == 3
                           || t.maxFilterLenP == 5 || t.maxFilterLenP == 7;
    const bool validLenQ = t.maxFilterLenQ == 1 || t.maxFilterLenQ == 2 || t.maxFilterLenQ == 3
                           || t.maxFilterLenQ == 5 || t.maxFilterLenQ == 7;
    const bool segmentInside = t.direction == 0
      ? t.x < frame.width && std::uint64_t(t.y) + 4 <= frame.height
      : t.y < frame.height && std::uint64_t(t.x) + 4 <= frame.width;
    const bool pLarge = (t.flags & CUDA_DBF_SIDE_P_LARGE) != 0;
    const bool qLarge = (t.flags & CUDA_DBF_SIDE_Q_LARGE) != 0;
    if (t.direction > 1 || !validLenP || !validLenQ || !segmentInside
        || (t.x & 3) != 0 || (t.y & 3) != 0
        || t.tc < 0 || t.tc > cudaDbfMaximumTc(frame.bitDepth)
        || t.beta < 0 || t.beta > cudaDbfMaximumBeta(frame.bitDepth)
        || t.minSample != 0 || t.maxSample != (std::int32_t{1} << frame.bitDepth) - 1
        || (t.flags & ~(CUDA_DBF_SIDE_P_LARGE | CUDA_DBF_SIDE_Q_LARGE
                        | CUDA_DBF_PART_P_NO_FILTER | CUDA_DBF_PART_Q_NO_FILTER)) != 0
        || pLarge != (t.maxFilterLenP > 3) || qLarge != (t.maxFilterLenQ > 3))
      return CudaDbfDispatchResult::NotEligible;
  }
  requireRuntime(m_impl.get());
  PictureMirror &mirror = findMirror(m_impl.get(), reconstructionHandle);
  if (mirror.role != CudaPictureRole::Reconstruction || mirror.host.planeCount == 0)
    return CudaDbfDispatchResult::NotEligible;
  const CudaHostPlaneDesc &host = mirror.host.planes[0];
  if (host.width != frame.width || host.height != frame.height || host.bitDepth != frame.bitDepth
      || host.elementSize != frame.elementSize || host.data == nullptr || host.strideBytes <= 0
      || host.marginLeft < 8 || host.marginRight < 8 || host.marginTop < 8 || host.marginBottom < 8)
    return CudaDbfDispatchResult::NotEligible;
  if (taskCount == 0)
  {
    ++m_impl->dbfNoOpFrames;
    return CudaDbfDispatchResult::NoOp;
  }
  const CudaMirrorMemoryUsage before = m_impl->mirrorMemory.total;
  const std::uint64_t mirrorSynchronizationsBefore = m_impl->mirrorSynchronizationOperations;
  const auto integrationStart = std::chrono::steady_clock::now();
  try
  {
#if VTM_CUDA_TESTING
    if (const char *failure = std::getenv("VTM_CUDA_DBF_TEST_FAILURE"))
    {
      CudaDbfTestFailurePoint point = CudaDbfTestFailurePoint::None;
      if (std::strcmp(failure, "allocation") == 0) point = CudaDbfTestFailurePoint::Allocation;
      else if (std::strcmp(failure, "upload") == 0) point = CudaDbfTestFailurePoint::ParameterUpload;
      else if (std::strcmp(failure, "snapshot") == 0) point = CudaDbfTestFailurePoint::SnapshotCopy;
      else if (std::strcmp(failure, "vertical-launch") == 0) point = CudaDbfTestFailurePoint::VerticalLaunch;
      else if (std::strcmp(failure, "vertical-sync") == 0) point = CudaDbfTestFailurePoint::VerticalCompletion;
      else if (std::strcmp(failure, "horizontal-launch") == 0) point = CudaDbfTestFailurePoint::HorizontalLaunch;
      else if (std::strcmp(failure, "horizontal-sync") == 0) point = CudaDbfTestFailurePoint::HorizontalCompletion;
      else if (std::strcmp(failure, "commit") == 0) point = CudaDbfTestFailurePoint::CommitCopy;
      else if (std::strcmp(failure, "commit-sync") == 0) point = CudaDbfTestFailurePoint::CommitCompletion;
      else throw std::runtime_error(std::string("Invalid VTM_CUDA_DBF_TEST_FAILURE value: ") + failure);
      cuda_backend::injectDbfFailure(m_impl->runtime, point);
    }
#endif
    ensureDevicePlane(reconstructionHandle, 0);
    cuda_backend::waitFence(m_impl->runtime, CudaQueue::Dbf, CudaFence::UploadComplete);
    const DbfAccelerationStats runtimeBefore = cuda_backend::dbfStats(m_impl->runtime);
    cuda_backend::filterDbfLumaFrame(m_impl->runtime, mirror.planes[0].device, frame, tasks, taskCount);
    const DbfAccelerationStats runtimeAfter = cuda_backend::dbfStats(m_impl->runtime);
    mirror.planes[0].state = CudaMirrorState::DeviceValid;
    mirror.planes[0].uploadPending = false;
    ensureHostPlane(reconstructionHandle, 0);
    const CudaMirrorMemoryUsage after = m_impl->mirrorMemory.total;
    m_impl->dbfMirrorUploadBytes += after.uploadedBytes - before.uploadedBytes;
    m_impl->dbfMirrorDownloadBytes += after.downloadedBytes - before.downloadedBytes;
    const std::uint64_t mirrorSynchronizations =
      m_impl->mirrorSynchronizationOperations - mirrorSynchronizationsBefore;
    const std::uint64_t runtimeSynchronizations =
      runtimeAfter.runtimeSynchronizations - runtimeBefore.runtimeSynchronizations;
    m_impl->dbfMirrorSynchronizations += mirrorSynchronizations;
    // Runtime host synchronizations, actual mirror synchronizations, and the upload-to-DBF stream dependency.
    m_impl->dbfIntegrationSynchronizations += runtimeSynchronizations + mirrorSynchronizations + 1;
    m_impl->dbfIntegrationNanoseconds += static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - integrationStart).count());
    return CudaDbfDispatchResult::Executed;
  }
  catch (const std::exception &error)
  {
    ++m_impl->dbfFailures;
    m_impl->dbfAccelerationEnabled = false;
    recoverAndQuarantineDbfNoexcept(m_impl.get(), reconstructionHandle);
    throw std::runtime_error(std::string("CUDA DBF execution failed after selection: ") + error.what());
  }
  catch (...)
  {
    ++m_impl->dbfFailures;
    m_impl->dbfAccelerationEnabled = false;
    recoverAndQuarantineDbfNoexcept(m_impl.get(), reconstructionHandle);
    throw std::runtime_error("CUDA DBF execution failed after selection: unknown error");
  }
#else
  (void) reconstructionHandle; (void) frame; (void) tasks; (void) taskCount;
  return CudaDbfDispatchResult::NotEligible;
#endif
}

DbfAccelerationStats CudaContext::dbfStats() const noexcept
{
  DbfAccelerationStats stats{};
#if VTM_ENABLE_CUDA
  stats = cuda_backend::dbfStats(m_impl->runtime);
  stats.mirrorUploadBytes = m_impl->dbfMirrorUploadBytes;
  stats.mirrorDownloadBytes = m_impl->dbfMirrorDownloadBytes;
  stats.mirrorSynchronizations = m_impl->dbfMirrorSynchronizations;
  stats.integrationSynchronizations = m_impl->dbfIntegrationSynchronizations;
  stats.failures = m_impl->dbfFailures;
  stats.noOpFrames = m_impl->dbfNoOpFrames;
  stats.descriptorCollectionNanoseconds = m_impl->dbfDescriptorCollectionNanoseconds;
  stats.integrationNanoseconds = m_impl->dbfIntegrationNanoseconds;
  stats.enabled = m_impl->runtime != nullptr && m_impl->dbfAccelerationEnabled;
  stats.poisoned = m_impl->runtime != nullptr && !m_impl->dbfAccelerationEnabled;
#endif
  return stats;
}

void CudaContext::recordDbfDescriptorCollection(const std::uint64_t nanoseconds) noexcept
{
#if VTM_ENABLE_CUDA
  m_impl->dbfDescriptorCollectionNanoseconds += nanoseconds;
#else
  (void) nanoseconds;
#endif
}

bool CudaContext::isLoopFilterChainAccelerationAvailable() const noexcept
{
#if VTM_ENABLE_CUDA
  return m_impl->runtime != nullptr && m_impl->loopFilterChainAccelerationEnabled;
#else
  return false;
#endif
}

CudaLoopFilterChainDispatchResult CudaContext::preflightLoopFilterChain(
  const CudaMirrorHandle reconstructionHandle, const CudaLoopFilterChainFrame &frame,
  const std::int32_t *lmcsLut, const CudaDbfLumaTask *dbfTasks, const std::uint32_t dbfTaskCount,
  const CudaSaoLumaCtuParam *saoCtus, const std::uint32_t saoCtuCount,
  const CudaAlfLumaFrame *alfFrame, const CudaAlfCtuParam *alfCtus, const std::uint32_t alfCtuCount)
{
#if VTM_ENABLE_CUDA
  const auto reject = [this](const char *reason = "unspecified") {
    (void) reason;
    ++m_impl->loopFilterChainNotEligible;
    return CudaLoopFilterChainDispatchResult::NotEligible;
  };
#if VTM_CUDA_TESTING
  if (std::getenv("VTM_CUDA_LOOP_FILTER_CHAIN_TEST_NOT_ELIGIBLE") != nullptr)
    return reject("injected preflight rejection");
#endif
  const std::uint8_t validStages = CUDA_LOOP_FILTER_LMCS | CUDA_LOOP_FILTER_DBF
                                   | CUDA_LOOP_FILTER_SAO | CUDA_LOOP_FILTER_ALF;
  const std::uint64_t pixels = std::uint64_t(frame.width) * frame.height;
  const std::uint64_t expectedWidth = frame.ctuWidth == 0 ? 0
    : (std::uint64_t(frame.width) + frame.ctuWidth - 1) / frame.ctuWidth;
  const std::uint64_t expectedHeight = frame.ctuHeight == 0 ? 0
    : (std::uint64_t(frame.height) + frame.ctuHeight - 1) / frame.ctuHeight;
  if (!isLoopFilterChainAccelerationAvailable() || frame.width == 0 || frame.height == 0
      || frame.ctuWidth == 0 || frame.ctuHeight == 0 || frame.ctusInWidth != expectedWidth
      || frame.ctusInHeight != expectedHeight || expectedWidth * expectedHeight != frame.ctuCount
      || frame.ctuCount == 0 || frame.ctuCount > CUDA_LOOP_FILTER_CHAIN_MAX_CTUS
      || pixels < CUDA_LOOP_FILTER_CHAIN_MIN_FRAME_PIXELS
      || (frame.bitDepth != 8 && frame.bitDepth != 10)
      || (frame.elementSize != 2 && frame.elementSize != 4)
      || frame.minSample != 0 || frame.maxSample != (std::int32_t{ 1 } << frame.bitDepth) - 1
      || (frame.stages & ~validStages) != 0 || frame.unsupportedFeatures != 0)
    return reject("frame descriptor or unsupported feature");

  const bool runLmcs = (frame.stages & CUDA_LOOP_FILTER_LMCS) != 0;
  const bool runDbf = (frame.stages & CUDA_LOOP_FILTER_DBF) != 0;
  const bool runSao = (frame.stages & CUDA_LOOP_FILTER_SAO) != 0;
  const bool runAlf = (frame.stages & CUDA_LOOP_FILTER_ALF) != 0;
  if (runLmcs && (lmcsLut == nullptr || frame.lmcsLutSize != (std::uint32_t{ 1 } << frame.bitDepth)))
    return reject("enabled LMCS pointer or LUT size");
  if (!runLmcs && frame.lmcsLutSize != 0) return reject("disabled LMCS has a LUT");
  if (runDbf && dbfTaskCount != 0 && dbfTasks == nullptr) return reject("enabled DBF task pointer");
  if (dbfTaskCount > CUDA_DBF_MAX_TASKS) return reject("DBF task count limit");
  if (runSao && (saoCtus == nullptr || saoCtuCount != frame.ctuCount))
    return reject("enabled SAO pointer or CTU count");
  if (!runSao && saoCtuCount != 0) return reject("disabled SAO has CTUs");
  if (runAlf && (alfFrame == nullptr || alfCtus == nullptr || alfCtuCount != frame.ctuCount))
    return reject("enabled ALF pointer or CTU count");
  if (!runAlf && alfCtuCount != 0) return reject("disabled ALF has CTUs");
  if (runLmcs)
    for (std::uint32_t i = 0; i < frame.lmcsLutSize; ++i)
      if (lmcsLut[i] < frame.minSample || lmcsLut[i] > frame.maxSample) return reject("LMCS LUT sample range");

  for (std::uint32_t i = 0; i < dbfTaskCount; ++i)
  {
    const CudaDbfLumaTask &t = dbfTasks[i];
    const bool pLarge = (t.flags & CUDA_DBF_SIDE_P_LARGE) != 0;
    const bool qLarge = (t.flags & CUDA_DBF_SIDE_Q_LARGE) != 0;
    const bool validLenP = t.maxFilterLenP == 1 || t.maxFilterLenP == 2 || t.maxFilterLenP == 3
                           || t.maxFilterLenP == 5 || t.maxFilterLenP == 7;
    const bool validLenQ = t.maxFilterLenQ == 1 || t.maxFilterLenQ == 2 || t.maxFilterLenQ == 3
                           || t.maxFilterLenQ == 5 || t.maxFilterLenQ == 7;
    const bool segmentInside = t.direction == 0
      ? t.x < frame.width && std::uint64_t(t.y) + 4 <= frame.height
      : t.y < frame.height && std::uint64_t(t.x) + 4 <= frame.width;
    if (!runDbf || t.direction > 1 || !validLenP || !validLenQ || !segmentInside
        || (t.x & 3) != 0 || (t.y & 3) != 0 || t.tc < 0 || t.tc > cudaDbfMaximumTc(frame.bitDepth)
        || t.beta < 0 || t.beta > cudaDbfMaximumBeta(frame.bitDepth)
        || t.minSample != frame.minSample || t.maxSample != frame.maxSample
        || (t.flags & ~(CUDA_DBF_SIDE_P_LARGE | CUDA_DBF_SIDE_Q_LARGE
                        | CUDA_DBF_PART_P_NO_FILTER | CUDA_DBF_PART_Q_NO_FILTER)) != 0
        || pLarge != (t.maxFilterLenP > 3) || qLarge != (t.maxFilterLenQ > 3))
      return reject("DBF task descriptor");
  }
  if (runSao)
  {
    const std::int32_t maximumOffset = (std::int32_t{ 1 } << (frame.bitDepth - 5)) - 1;
    for (std::uint32_t i = 0; i < saoCtuCount; ++i)
    {
      const CudaSaoLumaCtuParam &ctu = saoCtus[i];
      const std::uint32_t x = (i % frame.ctusInWidth) * frame.ctuWidth;
      const std::uint32_t y = (i / frame.ctusInWidth) * frame.ctuHeight;
      const std::uint32_t width = std::min(frame.ctuWidth, frame.width - x);
      const std::uint32_t height = std::min(frame.ctuHeight, frame.height - y);
      if (ctu.x != x || ctu.y != y || ctu.width != width || ctu.height != height
          || ctu.enabled > 1 || (ctu.enabled && (ctu.type < 0 || ctu.type > 4)))
        return reject("SAO CTU geometry or type");
      if (ctu.enabled)
        for (const std::int32_t offset : ctu.offsets)
          if (offset < -maximumOffset || offset > maximumOffset) return reject("SAO offset range");
    }
  }
  if (runAlf)
  {
    if (alfFrame->width != frame.width || alfFrame->height != frame.height
        || alfFrame->ctuWidth != frame.ctuWidth || alfFrame->ctuHeight != frame.ctuHeight
        || alfFrame->ctusInWidth != frame.ctusInWidth || alfFrame->ctusInHeight != frame.ctusInHeight
        || alfFrame->bitDepth != frame.bitDepth || alfFrame->elementSize != frame.elementSize
        || alfFrame->minSample != frame.minSample || alfFrame->maxSample != frame.maxSample
        || (frame.width & 3) != 0 || (frame.height & 3) != 0
        || alfFrame->vbCtuHeight != static_cast<std::int32_t>(frame.ctuHeight)
        || alfFrame->vbPos != alfFrame->vbCtuHeight - 4)
      return reject("ALF frame descriptor");
    for (std::uint32_t i = 0; i < alfCtuCount; ++i)
    {
      const CudaAlfCtuParam &ctu = alfCtus[i];
      const std::uint32_t x = (i % frame.ctusInWidth) * frame.ctuWidth;
      const std::uint32_t y = (i / frame.ctusInWidth) * frame.ctuHeight;
      if (ctu.x != x || ctu.y != y || ctu.width != std::min(frame.ctuWidth, frame.width - x)
          || ctu.height != std::min(frame.ctuHeight, frame.height - y) || ctu.enabled > 1)
        return reject("ALF CTU geometry");
      if (ctu.enabled)
        for (std::uint32_t j = 0; j < CUDA_ALF_CLASSES * CUDA_ALF_COEFFICIENTS; ++j)
        {
          const std::uint32_t coefficientIndex = j % CUDA_ALF_COEFFICIENTS;
          if (ctu.clipValues[j] < 0 || ctu.clipValues[j] > (std::int32_t{ 1 } << frame.bitDepth)
              || (coefficientIndex + 1 == CUDA_ALF_COEFFICIENTS
                    ? ctu.coefficients[j] != 128
                    : ctu.coefficients[j] < -128 || ctu.coefficients[j] > 127))
            return reject("ALF coefficient or clip range");
        }
    }
  }

  // A signalled stage set can collapse to no luma work after normative parameter reconstruction.
  // Complete before resolving the mirror or touching the runtime.
  if (frame.stages == 0 || (frame.stages == CUDA_LOOP_FILTER_DBF && dbfTaskCount == 0))
    return CudaLoopFilterChainDispatchResult::NoOp;

  requireRuntime(m_impl.get());
  const auto mirrorFound = m_impl->mirrors.find(reconstructionHandle);
  if (reconstructionHandle == 0 || mirrorFound == m_impl->mirrors.end())
    return reject("reconstruction mirror handle");
  PictureMirror &mirror = *mirrorFound->second;
  if (mirror.role != CudaPictureRole::Reconstruction || mirror.host.planeCount == 0)
    return reject("reconstruction mirror role or plane count");
  const CudaHostPlaneDesc &host = mirror.host.planes[0];
  if (host.width != frame.width || host.height != frame.height || host.bitDepth != frame.bitDepth
      || host.elementSize != frame.elementSize || host.data == nullptr || host.strideBytes <= 0
      || (runDbf && dbfTaskCount != 0
          && (host.marginLeft < 8 || host.marginRight < 8 || host.marginTop < 8 || host.marginBottom < 8)))
    return reject("host plane descriptor or DBF margins");

  return CudaLoopFilterChainDispatchResult::Executed;
#else
  (void) reconstructionHandle; (void) frame; (void) lmcsLut; (void) dbfTasks; (void) dbfTaskCount;
  (void) saoCtus; (void) saoCtuCount; (void) alfFrame; (void) alfCtus; (void) alfCtuCount;
  return CudaLoopFilterChainDispatchResult::NotEligible;
#endif
}

CudaLoopFilterChainDispatchResult CudaContext::filterLoopFilterChain(
  const CudaMirrorHandle reconstructionHandle, const CudaLoopFilterChainFrame &frame,
  const std::int32_t *lmcsLut, const CudaDbfLumaTask *dbfTasks, const std::uint32_t dbfTaskCount,
  const CudaSaoLumaCtuParam *saoCtus, const std::uint32_t saoCtuCount,
  const CudaAlfLumaFrame *alfFrame, const CudaAlfCtuParam *alfCtus, const std::uint32_t alfCtuCount)
{
#if VTM_ENABLE_CUDA
  const CudaLoopFilterChainDispatchResult preflight = preflightLoopFilterChain(
    reconstructionHandle, frame, lmcsLut, dbfTasks, dbfTaskCount, saoCtus, saoCtuCount,
    alfFrame, alfCtus, alfCtuCount);
  if (preflight != CudaLoopFilterChainDispatchResult::Executed)
  {
    return preflight;
  }
  const bool runLmcs = (frame.stages & CUDA_LOOP_FILTER_LMCS) != 0;
  const bool runDbf = (frame.stages & CUDA_LOOP_FILTER_DBF) != 0;
  const bool runSao = (frame.stages & CUDA_LOOP_FILTER_SAO) != 0;
  PictureMirror &mirror = findMirror(m_impl.get(), reconstructionHandle);

  const auto integrationStart = std::chrono::steady_clock::now();
  const CudaMirrorMemoryUsage before = m_impl->mirrorMemory.total;
  const std::uint64_t mirrorSyncBefore = m_impl->mirrorSynchronizationOperations;
  try
  {
#if VTM_CUDA_TESTING
    if (const char *failure = std::getenv("VTM_CUDA_LOOP_FILTER_CHAIN_TEST_FAILURE"))
    {
      CudaLoopFilterChainTestFailurePoint point = CudaLoopFilterChainTestFailurePoint::None;
      if (std::strcmp(failure, "allocation") == 0) point = CudaLoopFilterChainTestFailurePoint::GrowLutDevice;
      else if (std::strcmp(failure, "upload") == 0) point = CudaLoopFilterChainTestFailurePoint::Upload;
      else if (std::strcmp(failure, "lmcs-launch") == 0) point = CudaLoopFilterChainTestFailurePoint::LmcsLaunch;
      else if (std::strcmp(failure, "lmcs-sync") == 0) point = CudaLoopFilterChainTestFailurePoint::LmcsCompletion;
      else if (std::strcmp(failure, "dbf-stage") == 0) point = CudaLoopFilterChainTestFailurePoint::DbfStage;
      else if (std::strcmp(failure, "sao-snapshot") == 0) point = CudaLoopFilterChainTestFailurePoint::SaoSnapshot;
      else if (std::strcmp(failure, "sao-launch") == 0) point = CudaLoopFilterChainTestFailurePoint::SaoLaunch;
      else if (std::strcmp(failure, "sao-sync") == 0) point = CudaLoopFilterChainTestFailurePoint::SaoCompletion;
      else if (std::strcmp(failure, "alf-stage") == 0) point = CudaLoopFilterChainTestFailurePoint::AlfStage;
      else if (std::strcmp(failure, "download") == 0) point = CudaLoopFilterChainTestFailurePoint::Download;
      else if (std::strcmp(failure, "commit") == 0) point = CudaLoopFilterChainTestFailurePoint::Commit;
      else throw std::runtime_error(std::string("Invalid VTM_CUDA_LOOP_FILTER_CHAIN_TEST_FAILURE value: ") + failure);
      if (point == CudaLoopFilterChainTestFailurePoint::Download
          || point == CudaLoopFilterChainTestFailurePoint::Commit)
        m_impl->loopFilterChainIntegrationFailure = point;
      else
        cuda_backend::injectLoopFilterChainFailure(m_impl->runtime, point);
    }
#endif
    ensureDevicePlane(reconstructionHandle, 0);
    const CudaQueue firstConsumer = runLmcs || (runDbf && dbfTaskCount != 0) || runSao
      ? CudaQueue::Dbf : CudaQueue::Alf;
    cuda_backend::waitFence(m_impl->runtime, firstConsumer, CudaFence::UploadComplete);
    const LoopFilterChainAccelerationStats runtimeBefore = cuda_backend::loopFilterChainStats(m_impl->runtime);
    cuda_backend::filterLoopFilterChain(m_impl->runtime, mirror.planes[0].device, frame, lmcsLut,
                                         dbfTasks, dbfTaskCount, saoCtus, saoCtuCount,
                                         alfFrame, alfCtus, alfCtuCount);
    const LoopFilterChainAccelerationStats runtimeAfter = cuda_backend::loopFilterChainStats(m_impl->runtime);
    mirror.planes[0].state = CudaMirrorState::DeviceValid;
    mirror.planes[0].uploadPending = false;
#if VTM_CUDA_TESTING
    if (m_impl->loopFilterChainIntegrationFailure == CudaLoopFilterChainTestFailurePoint::Download
        || m_impl->loopFilterChainIntegrationFailure == CudaLoopFilterChainTestFailurePoint::Commit)
    {
      const CudaLoopFilterChainTestFailurePoint point = m_impl->loopFilterChainIntegrationFailure;
      m_impl->loopFilterChainIntegrationFailure = CudaLoopFilterChainTestFailurePoint::None;
      throw std::runtime_error(point == CudaLoopFilterChainTestFailurePoint::Download
                                 ? "Injected CUDA loop-filter chain download failure"
                                 : "Injected CUDA loop-filter chain commit failure");
    }
#endif
    ensureHostPlane(reconstructionHandle, 0);
    const CudaMirrorMemoryUsage after = m_impl->mirrorMemory.total;
    m_impl->loopFilterChainMirrorUploadBytes += after.uploadedBytes - before.uploadedBytes;
    m_impl->loopFilterChainMirrorDownloadBytes += after.downloadedBytes - before.downloadedBytes;
    m_impl->loopFilterChainIntegrationSynchronizations +=
      runtimeAfter.runtimeSynchronizations - runtimeBefore.runtimeSynchronizations
      + (m_impl->mirrorSynchronizationOperations - mirrorSyncBefore) + 1;
    m_impl->loopFilterChainIntegrationNanoseconds += static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - integrationStart).count());
    return CudaLoopFilterChainDispatchResult::Executed;
  }
  catch (const std::exception &error)
  {
    ++m_impl->loopFilterChainFailures;
    m_impl->loopFilterChainAccelerationEnabled = false;
    recoverAndQuarantineLoopFilterChainNoexcept(m_impl.get(), reconstructionHandle);
    throw std::runtime_error(std::string("CUDA loop-filter chain execution failed after selection: ") + error.what());
  }
  catch (...)
  {
    ++m_impl->loopFilterChainFailures;
    m_impl->loopFilterChainAccelerationEnabled = false;
    recoverAndQuarantineLoopFilterChainNoexcept(m_impl.get(), reconstructionHandle);
    throw std::runtime_error("CUDA loop-filter chain execution failed after selection: unknown error");
  }
#else
  (void) reconstructionHandle; (void) frame; (void) lmcsLut; (void) dbfTasks; (void) dbfTaskCount;
  (void) saoCtus; (void) saoCtuCount; (void) alfFrame; (void) alfCtus; (void) alfCtuCount;
  return CudaLoopFilterChainDispatchResult::NotEligible;
#endif
}

LoopFilterChainAccelerationStats CudaContext::loopFilterChainStats() const noexcept
{
  LoopFilterChainAccelerationStats stats{};
#if VTM_ENABLE_CUDA
  stats = cuda_backend::loopFilterChainStats(m_impl->runtime);
  stats.mirrorUploadBytes = m_impl->loopFilterChainMirrorUploadBytes;
  stats.mirrorDownloadBytes = m_impl->loopFilterChainMirrorDownloadBytes;
  stats.integrationSynchronizations = m_impl->loopFilterChainIntegrationSynchronizations;
  stats.collectionNanoseconds = m_impl->loopFilterChainCollectionNanoseconds;
  stats.integrationNanoseconds = m_impl->loopFilterChainIntegrationNanoseconds;
  stats.failures = m_impl->loopFilterChainFailures;
  stats.notEligible = m_impl->loopFilterChainNotEligible;
  stats.enabled = m_impl->runtime != nullptr && m_impl->loopFilterChainAccelerationEnabled;
  stats.poisoned = m_impl->runtime != nullptr && !m_impl->loopFilterChainAccelerationEnabled;
#endif
  return stats;
}

void CudaContext::recordLoopFilterChainCollection(const std::uint64_t nanoseconds) noexcept
{
#if VTM_ENABLE_CUDA
  m_impl->loopFilterChainCollectionNanoseconds += nanoseconds;
#else
  (void) nanoseconds;
#endif
}

bool CudaContext::computeQpaBatch(const CudaMirrorHandle sourceHandle, const CudaQpaTask *tasks,
                                  const std::uint32_t taskCount, CudaQpaResult *results)
{
#if VTM_ENABLE_CUDA
  if (!isQpaAccelerationAvailable() || tasks == nullptr || results == nullptr
      || taskCount == 0 || taskCount > CUDA_MAX_QPA_TASKS)
  {
    return false;
  }
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
#if VTM_CUDA_TESTING
    const CudaBatchTestFailurePoint environmentFailure =
      batchFailurePointFromEnvironment("VTM_CUDA_QPA_TEST_FAILURE");
    if (environmentFailure != CudaBatchTestFailurePoint::None)
    {
      cuda_backend::injectQpaFailurePoint(m_impl->runtime, environmentFailure);
    }
#endif
    ensureDevicePlane(sourceHandle, 0);
    cuda_backend::waitFence(m_impl->runtime, CudaQueue::Qpa, CudaFence::UploadComplete);
    cuda_backend::computeQpaBatch(m_impl->runtime, sourceMirror.planes[0].device, tasks, taskCount, results);
    return true;
  }
  catch (const std::exception &error)
  {
    ++m_impl->qpaFailures;
    m_impl->qpaAccelerationEnabled = false;
    cuda_backend::recoverQpaRuntime(m_impl->runtime);
#if VTM_CUDA_TESTING
    if (m_impl->qpaDiagnosticConstructionFailure)
    {
      m_impl->qpaDiagnosticConstructionFailure = false;
      throw std::bad_alloc();
    }
#endif
    throw CudaBatchExecutionError(std::string("CUDA QPA execution failed after selection: ") + error.what());
  }
  catch (...)
  {
    ++m_impl->qpaFailures;
    m_impl->qpaAccelerationEnabled = false;
    cuda_backend::recoverQpaRuntime(m_impl->runtime);
    throw CudaBatchExecutionError("CUDA QPA execution failed after selection: unknown error");
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

void CudaContext::injectDistortionFailurePointForTesting(const CudaBatchTestFailurePoint failurePoint)
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  if (failurePoint == CudaBatchTestFailurePoint::Upload)
  {
    m_impl->mirrorFailurePlane = 0;
    m_impl->mirrorUploadFailures = 1;
  }
  else if (failurePoint == CudaBatchTestFailurePoint::DiagnosticConstruction)
  {
    m_impl->distortionDiagnosticConstructionFailure = true;
    cuda_backend::injectDistortionFailures(m_impl->runtime, 0, 1);
  }
  else
  {
    cuda_backend::injectDistortionFailurePoint(m_impl->runtime, failurePoint);
  }
#else
  (void) failurePoint;
#endif
}

void CudaContext::injectQpaFailurePointForTesting(const CudaBatchTestFailurePoint failurePoint)
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  if (failurePoint == CudaBatchTestFailurePoint::DiagnosticConstruction)
  {
    m_impl->qpaDiagnosticConstructionFailure = true;
    cuda_backend::injectQpaFailures(m_impl->runtime, 0, 1);
  }
  else
  {
    cuda_backend::injectQpaFailurePoint(m_impl->runtime, failurePoint);
  }
#else
  (void) failurePoint;
#endif
}

void CudaContext::injectAlfFailureForTesting(const CudaAlfTestFailurePoint failurePoint)
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  cuda_backend::injectAlfFailure(m_impl->runtime, failurePoint);
#else
  (void) failurePoint;
#endif
}

void CudaContext::injectDbfFailureForTesting(const CudaDbfTestFailurePoint failurePoint)
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  cuda_backend::injectDbfFailure(m_impl->runtime, failurePoint);
#else
  (void) failurePoint;
#endif
}

void CudaContext::injectLoopFilterChainFailureForTesting(
  const CudaLoopFilterChainTestFailurePoint failurePoint)
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  if (failurePoint == CudaLoopFilterChainTestFailurePoint::Download
      || failurePoint == CudaLoopFilterChainTestFailurePoint::Commit)
    m_impl->loopFilterChainIntegrationFailure = failurePoint;
  else if (failurePoint == CudaLoopFilterChainTestFailurePoint::RecoveryDeviceRelease
           || failurePoint == CudaLoopFilterChainTestFailurePoint::RecoveryPinnedRelease)
  {
    // Force entry into recovery while preserving the requested release fault for that recovery pass.
    cuda_backend::injectLoopFilterChainFailure(m_impl->runtime, failurePoint);
    m_impl->loopFilterChainIntegrationFailure = CudaLoopFilterChainTestFailurePoint::Download;
  }
  else
    cuda_backend::injectLoopFilterChainFailure(m_impl->runtime, failurePoint);
#else
  (void) failurePoint;
#endif
}

std::uint64_t CudaContext::dbfLiveDeviceAllocationsForTesting() noexcept
{
#if VTM_ENABLE_CUDA
  return cuda_backend::dbfLiveDeviceAllocationsForTesting();
#else
  return 0;
#endif
}

std::uint64_t CudaContext::dbfLivePinnedAllocationsForTesting() noexcept
{
#if VTM_ENABLE_CUDA
  return cuda_backend::dbfLivePinnedAllocationsForTesting();
#else
  return 0;
#endif
}

std::uint64_t CudaContext::chainLiveDeviceAllocationsForTesting() noexcept
{
#if VTM_ENABLE_CUDA
  return cuda_backend::chainLiveDeviceAllocationsForTesting();
#else
  return 0;
#endif
}

std::uint64_t CudaContext::chainLivePinnedAllocationsForTesting() noexcept
{
#if VTM_ENABLE_CUDA
  return cuda_backend::chainLivePinnedAllocationsForTesting();
#else
  return 0;
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

void CudaContext::injectPinnedReleaseFailuresForTesting(const unsigned failures)
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  cuda_backend::injectPinnedReleaseFailures(failures);
#else
  (void) failures;
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
