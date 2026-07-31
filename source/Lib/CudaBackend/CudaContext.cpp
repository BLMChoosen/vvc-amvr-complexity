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
struct PictureMirror
{
  const void            *owner = nullptr;
  CudaPictureRole        role  = CudaPictureRole::Reconstruction;
  CudaMirrorState        state = CudaMirrorState::HostValid;
  CudaHostPictureDesc    host{};
  CudaDevicePictureDesc  device{};
  std::array<void *, CUDA_PICTURE_PLANE_COUNT> deviceBases{};
  std::array<CudaPinnedBuffer, CUDA_PICTURE_PLANE_COUNT> staging{};
  std::array<std::size_t, CUDA_PICTURE_PLANE_COUNT> fullRowBytes{};
  std::array<std::size_t, CUDA_PICTURE_PLANE_COUNT> fullHeights{};
  bool uploadPending = false;
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

unsigned char *hostBase(const CudaHostPlaneDesc &plane)
{
  auto *active = static_cast<unsigned char *>(plane.data);
  return active - static_cast<std::ptrdiff_t>(plane.marginTop) * plane.strideBytes
         - static_cast<std::ptrdiff_t>(plane.marginLeft) * plane.elementSize;
}

void stageHostPicture(PictureMirror &mirror)
{
  for (std::size_t index = 0; index < mirror.host.planeCount; ++index)
  {
    const CudaHostPlaneDesc &plane = mirror.host.planes[index];
    const unsigned char *source = hostBase(plane);
    auto *destination = static_cast<unsigned char *>(mirror.staging[index].data());
    for (std::size_t row = 0; row < mirror.fullHeights[index]; ++row)
    {
      std::memcpy(destination + row * mirror.fullRowBytes[index],
                  source + static_cast<std::ptrdiff_t>(row) * plane.strideBytes,
                  mirror.fullRowBytes[index]);
    }
  }
}

void unstageHostPicture(PictureMirror &mirror)
{
  for (std::size_t index = 0; index < mirror.host.planeCount; ++index)
  {
    const CudaHostPlaneDesc &plane = mirror.host.planes[index];
    unsigned char *destination = hostBase(plane);
    const auto *source = static_cast<const unsigned char *>(mirror.staging[index].data());
    for (std::size_t row = 0; row < mirror.fullHeights[index]; ++row)
    {
      std::memcpy(destination + static_cast<std::ptrdiff_t>(row) * plane.strideBytes,
                  source + row * mirror.fullRowBytes[index], mirror.fullRowBytes[index]);
    }
  }
}

bool sameAllocationLayout(const CudaHostPictureDesc &first, const CudaHostPictureDesc &second)
{
  if (first.planeCount != second.planeCount)
  {
    return false;
  }
  for (std::size_t index = 0; index < first.planeCount; ++index)
  {
    const CudaHostPlaneDesc &a = first.planes[index];
    const CudaHostPlaneDesc &b = second.planes[index];
    if (a.width != b.width || a.height != b.height || a.marginLeft != b.marginLeft
        || a.marginRight != b.marginRight || a.marginTop != b.marginTop || a.marginBottom != b.marginBottom
        || a.elementSize != b.elementSize || a.bitDepth != b.bitDepth)
    {
      return false;
    }
  }
  return true;
}

template<typename ContextImpl>
void allocateMirrorResources(ContextImpl *impl, PictureMirror &mirror, const CudaHostPictureDesc &picture)
{
  mirror.host = picture;
  mirror.device = CudaDevicePictureDesc{};
  mirror.device.planeCount = picture.planeCount;
  mirror.state = CudaMirrorState::HostValid;
  mirror.uploadPending = false;

  for (std::size_t index = 0; index < picture.planeCount; ++index)
  {
    const CudaHostPlaneDesc &host = picture.planes[index];
    const std::size_t fullWidth = static_cast<std::size_t>(host.marginLeft) + host.width + host.marginRight;
    const std::size_t fullHeight = static_cast<std::size_t>(host.marginTop) + host.height + host.marginBottom;
    const std::size_t rowBytes = checkedMultiply(fullWidth, host.elementSize, "row size");
    const std::size_t pitch = checkedAdd(rowBytes, 127, "device pitch") & ~std::size_t(127);
    const std::size_t allocationBytes = checkedMultiply(pitch, fullHeight, "device allocation");
    const std::size_t stagingBytes = checkedMultiply(rowBytes, fullHeight, "staging allocation");

    mirror.staging[index].allocate(stagingBytes);
    mirror.deviceBases[index] = cuda_backend::allocateDevice(impl->runtime, allocationBytes, CudaQueue::Upload);
    mirror.fullRowBytes[index] = rowBytes;
    mirror.fullHeights[index]  = fullHeight;

    CudaDevicePlaneDesc &device = mirror.device.planes[index];
    device.data = static_cast<unsigned char *>(mirror.deviceBases[index])
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
  for (void *&allocation : mirror.deviceBases)
  {
    if (allocation != nullptr)
    {
      try
      {
        cuda_backend::releaseDevice(impl->runtime, allocation, CudaQueue::Compute);
        allocation = nullptr;
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
        }
        catch (...)
        {
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
  for (const void *allocation : mirror.deviceBases)
  {
    if (allocation != nullptr)
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
    for (void *&allocation : entry.second->deviceBases)
    {
      if (allocation != nullptr)
      {
        try
        {
          cuda_backend::releaseDevice(impl->runtime, allocation, CudaQueue::Compute);
          allocation = nullptr;
        }
        catch (...)
        {
          try
          {
            cuda_backend::releaseDeviceImmediate(impl->runtime, allocation);
            allocation = nullptr;
          }
          catch (...)
          {
            // Keep ownership in the mirror until destroyRuntimeContext tears down the owning memory pool.
          }
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
    allocateMirrorResources(m_impl.get(), *slot->second, picture);
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
  mirror.uploadPending = false;

  if (sameAllocationLayout(mirror.host, picture))
  {
    mirror.host  = picture;
    mirror.state = CudaMirrorState::HostValid;
    return;
  }

  PictureMirror replacement;
  replacement.owner = mirror.owner;
  replacement.role  = mirror.role;
  try
  {
    allocateMirrorResources(m_impl.get(), replacement, picture);
  }
  catch (...)
  {
    try
    {
      releaseMirrorResources(m_impl.get(), replacement);
    }
    catch (...)
    {
    }
    throw;
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
  return findMirror(m_impl.get(), handle).state;
#else
  (void) handle;
  throw std::runtime_error("CUDA picture mirror requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

CudaDevicePictureDesc CudaContext::devicePicture(const CudaMirrorHandle handle) const
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  const PictureMirror &mirror = findMirror(m_impl.get(), handle);
  if (mirror.state == CudaMirrorState::HostValid)
  {
    throw std::runtime_error("CUDA device picture is stale; call ensureDevice before submitting work");
  }
  return mirror.device;
#else
  (void) handle;
  throw std::runtime_error("CUDA picture mirror requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

void CudaContext::markHostModified(const CudaMirrorHandle handle)
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  PictureMirror &mirror = findMirror(m_impl.get(), handle);
  if (mirror.state == CudaMirrorState::DeviceValid)
  {
    throw std::runtime_error("CUDA picture mirror conflict: device data must be downloaded before host modification");
  }
  if (mirror.state == CudaMirrorState::Synchronized)
  {
    cuda_backend::recordFence(m_impl->runtime, CudaQueue::Compute, CudaFence::ComputeComplete);
    cuda_backend::waitFence(m_impl->runtime, CudaQueue::Upload, CudaFence::ComputeComplete);
  }
  mirror.state = CudaMirrorState::HostValid;
#else
  (void) handle;
  throw std::runtime_error("CUDA picture mirror requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

void CudaContext::markDeviceModified(const CudaMirrorHandle handle)
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  PictureMirror &mirror = findMirror(m_impl.get(), handle);
  if (mirror.state == CudaMirrorState::HostValid)
  {
    throw std::runtime_error("CUDA picture mirror conflict: host data must be uploaded before device modification");
  }
  cuda_backend::recordFence(m_impl->runtime, CudaQueue::Compute, CudaFence::ComputeComplete);
  cuda_backend::waitFence(m_impl->runtime, CudaQueue::Download, CudaFence::ComputeComplete);
  mirror.state = CudaMirrorState::DeviceValid;
#else
  (void) handle;
  throw std::runtime_error("CUDA picture mirror requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

void CudaContext::ensureDevice(const CudaMirrorHandle handle)
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  PictureMirror &mirror = findMirror(m_impl.get(), handle);
  if (mirror.state != CudaMirrorState::HostValid)
  {
    return;
  }
  if (mirror.uploadPending)
  {
    cuda_backend::synchronizeQueue(m_impl->runtime, CudaQueue::Upload);
    mirror.uploadPending = false;
  }
  stageHostPicture(mirror);
  for (std::size_t index = 0; index < mirror.host.planeCount; ++index)
  {
    cuda_backend::copy2DToDeviceAsync(m_impl->runtime, mirror.deviceBases[index],
                                      mirror.device.planes[index].pitchBytes, mirror.staging[index].data(),
                                      mirror.fullRowBytes[index], mirror.fullRowBytes[index],
                                      mirror.fullHeights[index], CudaQueue::Upload);
  }
  cuda_backend::recordFence(m_impl->runtime, CudaQueue::Upload, CudaFence::UploadComplete);
  cuda_backend::waitFence(m_impl->runtime, CudaQueue::Compute, CudaFence::UploadComplete);
  mirror.uploadPending = true;
  mirror.state = CudaMirrorState::Synchronized;
#else
  (void) handle;
  throw std::runtime_error("CUDA picture mirror requested, but this binary was built with ENABLE_CUDA=OFF");
#endif
}

void CudaContext::ensureHost(const CudaMirrorHandle handle)
{
#if VTM_ENABLE_CUDA
  requireRuntime(m_impl.get());
  PictureMirror &mirror = findMirror(m_impl.get(), handle);
  if (mirror.state != CudaMirrorState::DeviceValid)
  {
    return;
  }
  for (std::size_t index = 0; index < mirror.host.planeCount; ++index)
  {
    cuda_backend::copy2DToHostAsync(m_impl->runtime, mirror.staging[index].data(), mirror.fullRowBytes[index],
                                    mirror.deviceBases[index], mirror.device.planes[index].pitchBytes,
                                    mirror.fullRowBytes[index], mirror.fullHeights[index], CudaQueue::Download);
  }
  cuda_backend::synchronizeQueue(m_impl->runtime, CudaQueue::Download);
  mirror.uploadPending = false;
  unstageHostPicture(mirror);
  mirror.state = CudaMirrorState::Synchronized;
#else
  (void) handle;
  throw std::runtime_error("CUDA picture mirror requested, but this binary was built with ENABLE_CUDA=OFF");
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
