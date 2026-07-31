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
void *allocatePinnedHost(std::size_t bytes);
void releasePinnedHost(void *allocation) noexcept;
void copy2DToDeviceAsync(RuntimeContext *context, void *destination, std::size_t destinationPitch,
                         const void *source, std::size_t sourcePitch, std::size_t widthBytes,
                         std::size_t height, CudaQueue queue);
void copy2DToHostAsync(RuntimeContext *context, void *destination, std::size_t destinationPitch,
                       const void *source, std::size_t sourcePitch, std::size_t widthBytes,
                       std::size_t height, CudaQueue queue);
void synchronizeQueue(RuntimeContext *context, CudaQueue queue);

}   // namespace vtm::cuda_backend

#endif   // VTM_CUDA_RUNTIME_H
