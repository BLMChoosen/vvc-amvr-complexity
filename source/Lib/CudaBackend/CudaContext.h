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

#ifndef VTM_CUDA_CONTEXT_H
#define VTM_CUDA_CONTEXT_H

#include "CudaPictureMirror.h"
#include "CudaDistortion.h"

#include <memory>
#include <cstddef>

namespace vtm
{

enum class CudaQueue
{
  Upload,
  Compute,
  Download
};

enum class CudaFence
{
  UploadComplete,
  ComputeComplete,
  DownloadComplete
};

class CudaPinnedBuffer
{
public:
  CudaPinnedBuffer();
  explicit CudaPinnedBuffer(std::size_t bytes);
  ~CudaPinnedBuffer() noexcept;

  CudaPinnedBuffer(const CudaPinnedBuffer &) = delete;
  CudaPinnedBuffer &operator=(const CudaPinnedBuffer &) = delete;
  CudaPinnedBuffer(CudaPinnedBuffer &&other) noexcept;
  CudaPinnedBuffer &operator=(CudaPinnedBuffer &&other) noexcept;

  void allocate(std::size_t bytes);
  void reset() noexcept;
  void *data() noexcept;
  const void *data() const noexcept;
  std::size_t size() const noexcept;
  explicit operator bool() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

class CudaContext
{
public:
  CudaContext();
  ~CudaContext() noexcept;

  CudaContext(const CudaContext &) = delete;
  CudaContext &operator=(const CudaContext &) = delete;

  void create(int device);
  void synchronize();
  void shutdown(bool synchronize = true);
  void destroy();

  // A context and all of its queue and teardown APIs are confined to the thread that called create().
  // The noexcept destructor terminates if an owner violates this rule; owners must call shutdown()/destroy()
  // on the creating thread so CUDA teardown is never attempted concurrently from an arbitrary thread.
  // These host-only primitives are the submission contract for future batched codec work.
  void  recordFence(CudaQueue queue, CudaFence fence);
  void  waitFence(CudaQueue queue, CudaFence fence);
  void *allocateDevice(std::size_t bytes, CudaQueue queue = CudaQueue::Compute);
  void  releaseDevice(void *allocation, CudaQueue queue = CudaQueue::Compute);

  CudaMirrorHandle registerPictureMirror(const void *owner, CudaPictureRole role,
                                         const CudaHostPictureDesc &picture);
  void rebindHostPicture(CudaMirrorHandle handle, const CudaHostPictureDesc &picture);
  void rebindHostPicture(const void *owner, CudaPictureRole role, const CudaHostPictureDesc &picture);
  void releasePictureMirror(CudaMirrorHandle handle);
  void releasePictureMirrors(const void *owner);
  void releaseAllPictureMirrors();
  bool hasPictureMirror(const void *owner, CudaPictureRole role) const;
  CudaMirrorHandle pictureMirrorHandle(const void *owner, CudaPictureRole role) const;
  std::size_t pictureMirrorCount() const;
  CudaMirrorState pictureMirrorState(CudaMirrorHandle handle) const;
  CudaDevicePictureDesc devicePicture(CudaMirrorHandle handle) const;
  void markHostModified(CudaMirrorHandle handle);
  void markDeviceModified(CudaMirrorHandle handle);
  void ensureDevice(CudaMirrorHandle handle);
  // Future GPU writers must call ensureHost before any CPU-side hash, YUV writer, or other host consumer.
  void ensureHost(CudaMirrorHandle handle);
  bool isDistortionAccelerationAvailable() const noexcept;
  bool computeDistortionBatch(const CudaDistortionBatchDesc &batch, std::uint64_t *results) noexcept;
  std::uint64_t distortionBatchDispatchCount() const;
  std::uint64_t distortionBatchFailureCount() const noexcept;
#if VTM_CUDA_TESTING
  void injectReleaseFailuresForTesting(unsigned asyncFailures, unsigned immediateFailures);
  void injectDistortionFailuresForTesting(unsigned allocationFailureStep, unsigned executionFailures);
#endif

  bool isCreated() const noexcept;
  static bool isCompiled() noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

}   // namespace vtm

#endif   // VTM_CUDA_CONTEXT_H
