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

#ifndef VTM_CUDA_RUNTIME_H
#define VTM_CUDA_RUNTIME_H

#include "CudaContext.h"

#include <cstddef>

namespace vtm::cuda_backend
{

struct RuntimeContext;

RuntimeContext *createRuntimeContext(int device);
void synchronizeRuntimeContext(RuntimeContext *context);
void shutdownRuntimeContext(RuntimeContext *context, bool synchronize);
void destroyRuntimeContext(RuntimeContext *context) noexcept;
void recordFence(RuntimeContext *context, CudaQueue queue, CudaFence fence);
void waitFence(RuntimeContext *context, CudaQueue queue, CudaFence fence);
void *allocateDevice(RuntimeContext *context, std::size_t bytes, CudaQueue queue);
void releaseDevice(RuntimeContext *context, void *allocation, CudaQueue queue);
void releaseDeviceImmediate(RuntimeContext *context, void *allocation);
void *allocatePinnedHost(std::size_t bytes);
bool releasePinnedHost(void *allocation) noexcept;
void copy2DToDeviceAsync(RuntimeContext *context, void *destination, std::size_t destinationPitch,
                         const void *source, std::size_t sourcePitch, std::size_t widthBytes,
                         std::size_t height, CudaQueue queue);
void copy2DToHostAsync(RuntimeContext *context, void *destination, std::size_t destinationPitch,
                       const void *source, std::size_t sourcePitch, std::size_t widthBytes,
                       std::size_t height, CudaQueue queue);
void synchronizeQueue(RuntimeContext *context, CudaQueue queue);
void computeDistortionBatch(RuntimeContext *context, const CudaDistortionBatchDesc &batch,
                            const void *sourceDevice, std::size_t sourcePitchBytes,
                            const void *referenceDevice, std::size_t referencePitchBytes,
                            std::uint64_t *results);
std::uint64_t distortionBatchDispatchCount(const RuntimeContext *context);
void computeQpaBatch(RuntimeContext *context, const CudaDevicePlaneDesc &source,
                     const CudaQpaTask *tasks, std::uint32_t taskCount, CudaQpaResult *results);
void filterAlfLumaFrame(RuntimeContext *context, const CudaDevicePlaneDesc &plane,
                        const CudaAlfLumaFrame &frame, const CudaAlfCtuParam *ctus,
                        std::uint32_t ctuCount, CudaAlfClassifier *classifiers);
void filterDbfLumaFrame(RuntimeContext *context, const CudaDevicePlaneDesc &plane,
                        const CudaDbfFrame &frame, const CudaDbfLumaTask *tasks,
                        std::uint32_t taskCount);
void filterLoopFilterChain(RuntimeContext *context, const CudaDevicePlaneDesc &plane,
                           const CudaLoopFilterChainFrame &frame, const std::int32_t *lmcsLut,
                           const CudaDbfLumaTask *dbfTasks, std::uint32_t dbfTaskCount,
                           const CudaSaoLumaCtuParam *saoCtus, std::uint32_t saoCtuCount,
                           const CudaAlfLumaFrame *alfFrame, const CudaAlfCtuParam *alfCtus,
                           std::uint32_t alfCtuCount);
AlfAccelerationStats alfStats(const RuntimeContext *context) noexcept;
DbfAccelerationStats dbfStats(const RuntimeContext *context) noexcept;
LoopFilterChainAccelerationStats loopFilterChainStats(const RuntimeContext *context) noexcept;
std::uint64_t qpaBatchDispatchCount(const RuntimeContext *context);
std::uint64_t qpaTaskCount(const RuntimeContext *context);
void recoverDistortionRuntime(RuntimeContext *context) noexcept;
void recoverQpaRuntime(RuntimeContext *context) noexcept;
void recoverAlfRuntime(RuntimeContext *context) noexcept;
void recoverDbfRuntime(RuntimeContext *context) noexcept;
void recoverLoopFilterChainRuntime(RuntimeContext *context) noexcept;
void injectReleaseFailures(RuntimeContext *context, unsigned asyncFailures, unsigned immediateFailures);
void injectDistortionFailures(RuntimeContext *context, unsigned allocationFailureStep, unsigned executionFailures);
void injectQpaFailures(RuntimeContext *context, unsigned allocationFailureStep, unsigned executionFailures);
void injectDistortionFailurePoint(RuntimeContext *context, CudaBatchTestFailurePoint failurePoint);
void injectQpaFailurePoint(RuntimeContext *context, CudaBatchTestFailurePoint failurePoint);
void injectAlfFailure(RuntimeContext *context, CudaAlfTestFailurePoint failurePoint);
void injectDbfFailure(RuntimeContext *context, CudaDbfTestFailurePoint failurePoint);
void injectLoopFilterChainFailure(RuntimeContext *context, CudaLoopFilterChainTestFailurePoint failurePoint);
void injectPinnedReleaseFailures(unsigned failures);
std::uint64_t dbfLiveDeviceAllocationsForTesting() noexcept;
std::uint64_t dbfLivePinnedAllocationsForTesting() noexcept;
std::uint64_t chainLiveDeviceAllocationsForTesting() noexcept;
std::uint64_t chainLivePinnedAllocationsForTesting() noexcept;

}   // namespace vtm::cuda_backend

#endif   // VTM_CUDA_RUNTIME_H
