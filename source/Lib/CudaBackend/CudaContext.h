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
#include "CudaDecoderTransform.h"
#include "CudaQpa.h"
#include "CudaAlf.h"
#include "CommonLib/CudaDeblocking.h"
#include "CommonLib/CudaLoopFilterChain.h"

#include <memory>
#include <cstddef>

namespace vtm
{

enum class CudaQueue
{
  Upload,
  Compute,
  Download,
  Qpa,
  Alf,
  Dbf
};

enum class CudaFence
{
  UploadComplete,
  ComputeComplete,
  DownloadComplete
};

enum class CudaBatchTestFailurePoint : std::uint8_t
{
  None,
  Upload,
  KernelLaunch,
  ResultDownload,
  Completion,
  Publication,
  ResultCorruption,
  DiagnosticConstruction
};

constexpr std::uint64_t CUDA_DEFAULT_MIRROR_MEMORY_BUDGET_BYTES = std::uint64_t{ 512 } * 1024 * 1024;

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
  bool reset() noexcept;
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
  CudaMirrorState pictureMirrorPlaneState(CudaMirrorHandle handle, std::uint8_t plane) const;
  CudaPlaneMask pictureMirrorAllocatedPlanes(CudaMirrorHandle handle) const;
  CudaDevicePictureDesc devicePicture(CudaMirrorHandle handle) const;
  CudaDevicePictureDesc devicePicturePlanes(CudaMirrorHandle handle, CudaPlaneMask planes) const;
  void markHostModified(CudaMirrorHandle handle);
  void markHostPlanesModified(CudaMirrorHandle handle, CudaPlaneMask planes);
  void markHostPlaneModified(CudaMirrorHandle handle, std::uint8_t plane);
  void markDeviceModified(CudaMirrorHandle handle);
  void markDevicePlanesModified(CudaMirrorHandle handle, CudaPlaneMask planes);
  void markDevicePlaneModified(CudaMirrorHandle handle, std::uint8_t plane);
  void ensureDevice(CudaMirrorHandle handle);
  void ensureDevicePlanes(CudaMirrorHandle handle, CudaPlaneMask planes);
  void ensureDevicePlane(CudaMirrorHandle handle, std::uint8_t plane);
  // Future GPU writers must call ensureHost before any CPU-side hash, YUV writer, or other host consumer.
  void ensureHost(CudaMirrorHandle handle);
  void ensureHostPlanes(CudaMirrorHandle handle, CudaPlaneMask planes);
  void ensureHostPlane(CudaMirrorHandle handle, std::uint8_t plane);
  CudaMirrorMemoryStats pictureMirrorMemoryStats() const;
  void setPictureMirrorMemoryBudget(std::uint64_t bytes);
  static bool isPictureMirrorDescriptorSupported(const CudaHostPictureDesc &picture) noexcept;
  bool isDistortionAccelerationAvailable() const noexcept;
  // false means an explicitly unsupported descriptor/NotEligible. Once CUDA selection starts, failures throw.
  bool computeDistortionBatch(const CudaDistortionBatchDesc &batch, std::uint64_t *results);
  std::uint64_t distortionBatchDispatchCount() const;
  std::uint64_t distortionBatchFailureCount() const noexcept;
  bool isQpaAccelerationAvailable() const noexcept;
  // false means an explicitly unsupported descriptor/NotEligible. Once CUDA selection starts, failures throw.
  bool computeQpaBatch(CudaMirrorHandle sourceMirror, const CudaQpaTask *tasks,
                       std::uint32_t taskCount, CudaQpaResult *results);
  std::uint64_t qpaBatchDispatchCount() const;
  std::uint64_t qpaTaskCount() const;
  std::uint64_t qpaBatchFailureCount() const noexcept;
  bool computeDecoderTransformBatch(const CudaDecoderTransformBatch &batch);
  std::uint64_t decoderTransformBatchDispatchCount() const;
  bool isAlfAccelerationAvailable() const noexcept;
  CudaAlfDispatchResult filterAlfLumaFrame(CudaMirrorHandle reconstructionMirror,
                                           const CudaAlfLumaFrame &frame,
                                           const CudaAlfCtuParam *ctus, std::uint32_t ctuCount,
                                           CudaAlfClassifier *classifiers = nullptr);
  AlfAccelerationStats alfStats() const noexcept;
  bool isDbfAccelerationAvailable() const noexcept;
  CudaDbfDispatchResult filterDbfLumaFrame(CudaMirrorHandle reconstructionMirror,
                                            const CudaDbfFrame &frame,
                                            const CudaDbfLumaTask *tasks, std::uint32_t taskCount);
  DbfAccelerationStats dbfStats() const noexcept;
  void recordDbfDescriptorCollection(std::uint64_t nanoseconds) noexcept;
  bool isLoopFilterChainAccelerationAvailable() const noexcept;
  // Executed means every backend, mirror, and POD contract is valid and execution may be selected.
  // NoOp and NotEligible never allocate, transfer, synchronize, or mutate the picture.
  CudaLoopFilterChainDispatchResult preflightLoopFilterChain(
    CudaMirrorHandle reconstructionMirror, const CudaLoopFilterChainFrame &frame,
    const std::int32_t *lmcsLut, const CudaDbfLumaTask *dbfTasks, std::uint32_t dbfTaskCount,
    const CudaSaoLumaCtuParam *saoCtus, std::uint32_t saoCtuCount,
    const CudaAlfLumaFrame *alfFrame, const CudaAlfCtuParam *alfCtus, std::uint32_t alfCtuCount);
  CudaLoopFilterChainDispatchResult filterLoopFilterChain(
    CudaMirrorHandle reconstructionMirror, const CudaLoopFilterChainFrame &frame,
    const std::int32_t *lmcsLut, const CudaDbfLumaTask *dbfTasks, std::uint32_t dbfTaskCount,
    const CudaSaoLumaCtuParam *saoCtus, std::uint32_t saoCtuCount,
    const CudaAlfLumaFrame *alfFrame, const CudaAlfCtuParam *alfCtus, std::uint32_t alfCtuCount);
  LoopFilterChainAccelerationStats loopFilterChainStats() const noexcept;
  void recordLoopFilterChainCollection(std::uint64_t nanoseconds) noexcept;
#if VTM_CUDA_TESTING
  void injectReleaseFailuresForTesting(unsigned asyncFailures, unsigned immediateFailures);
  void injectDistortionFailuresForTesting(unsigned allocationFailureStep, unsigned executionFailures);
  void injectQpaFailuresForTesting(unsigned allocationFailureStep, unsigned executionFailures);
  void injectDistortionFailurePointForTesting(CudaBatchTestFailurePoint failurePoint);
  void injectQpaFailurePointForTesting(CudaBatchTestFailurePoint failurePoint);
  void injectAlfFailureForTesting(CudaAlfTestFailurePoint failurePoint);
  void injectDbfFailureForTesting(CudaDbfTestFailurePoint failurePoint);
  void injectLoopFilterChainFailureForTesting(CudaLoopFilterChainTestFailurePoint failurePoint);
  static std::uint64_t dbfLiveDeviceAllocationsForTesting() noexcept;
  static std::uint64_t dbfLivePinnedAllocationsForTesting() noexcept;
  static std::uint64_t chainLiveDeviceAllocationsForTesting() noexcept;
  static std::uint64_t chainLivePinnedAllocationsForTesting() noexcept;
  void injectMirrorPlaneFailuresForTesting(std::uint8_t plane, unsigned allocationFailureStep,
                                           unsigned uploadFailures, unsigned downloadFailures);
  void injectPinnedReleaseFailuresForTesting(unsigned failures);
#endif

  bool isCreated() const noexcept;
  static bool isCompiled() noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

}   // namespace vtm

#endif   // VTM_CUDA_CONTEXT_H
