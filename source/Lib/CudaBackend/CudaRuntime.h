/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2026, ITU/ISO/IEC
 * All rights reserved.
 */

#ifndef VTM_CUDA_RUNTIME_H
#define VTM_CUDA_RUNTIME_H

namespace vtm::cuda_backend
{

struct RuntimeContext;

RuntimeContext *createRuntimeContext(int device);
void synchronizeRuntimeContext(RuntimeContext *context);
void destroyRuntimeContext(RuntimeContext *context) noexcept;
bool supportsMain10(const RuntimeContext *context) noexcept;

}   // namespace vtm::cuda_backend

#endif   // VTM_CUDA_RUNTIME_H
