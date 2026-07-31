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

#include "CudaBackend/ComputeBackend.h"
#include "CudaBackend/CudaContext.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace
{

int fail(const std::string &message)
{
  std::cerr << message << '\n';
  return EXIT_FAILURE;
}

}   // namespace

int main(const int argc, char *argv[])
{
  vtm::ComputeConfig config;
  if (config.backend != vtm::ComputeBackend::CPU || config.device != 0)
  {
    return fail("ComputeConfig defaults are invalid");
  }

  vtm::ComputeBackend backend;
  if (!vtm::parseComputeBackend("cpu", backend) || backend != vtm::ComputeBackend::CPU)
  {
    return fail("Failed to parse cpu backend");
  }
  if (!vtm::parseComputeBackend("cuda", backend) || backend != vtm::ComputeBackend::CUDA)
  {
    return fail("Failed to parse cuda backend");
  }
  if (vtm::parseComputeBackend("invalid", backend))
  {
    return fail("Invalid backend was accepted");
  }

  if (argc == 1)
  {
    return EXIT_SUCCESS;
  }
  if (argc != 3 || std::string(argv[1]) != "--cuda")
  {
    return fail("Usage: CudaBackendTest [--cuda device]");
  }
  if (!vtm::CudaContext::isCompiled())
  {
    return fail("CUDA runtime test requested from an ENABLE_CUDA=OFF build");
  }

  try
  {
    vtm::CudaContext context;
    context.create(std::stoi(argv[2]));
    if (!context.isCreated())
    {
      return fail("CUDA context was not created");
    }
    if (!context.supportsMain10())
    {
      return fail("CUDA device does not support Main 10 processing");
    }
    context.synchronize();
    context.destroy();
    if (context.isCreated())
    {
      return fail("CUDA context was not destroyed");
    }
  }
  catch (const std::exception &error)
  {
    return fail(error.what());
  }

  return EXIT_SUCCESS;
}
