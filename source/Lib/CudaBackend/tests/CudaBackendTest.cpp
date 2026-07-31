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
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <thread>

namespace
{

int fail(const std::string &message)
{
  std::cerr << message << '\n';
  return EXIT_FAILURE;
}

bool throws(const std::function<void()> &operation)
{
  try
  {
    operation();
  }
  catch (const std::exception &)
  {
    return true;
  }
  return false;
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
    vtm::CudaContext context;
    context.destroy();
    context.destroy();
    return EXIT_SUCCESS;
  }
  if (argc == 2 && std::string(argv[1]) == "--disabled")
  {
    if (vtm::CudaContext::isCompiled())
    {
      return fail("Disabled-backend test requires ENABLE_CUDA=OFF");
    }
    vtm::CudaContext context;
    if (!throws([&context]() { context.create(0); }))
    {
      return fail("ENABLE_CUDA=OFF build accepted CUDA context creation");
    }
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
    if (!throws([]() {
          vtm::CudaContext invalid;
          invalid.create(-1);
        }))
    {
      return fail("Negative CUDA device was accepted");
    }
    if (!throws([]() {
          vtm::CudaContext invalid;
          invalid.create(std::numeric_limits<int>::max());
        }))
    {
      return fail("Out-of-range CUDA device was accepted");
    }

    vtm::CudaContext context;
    context.create(std::stoi(argv[2]));
    if (!context.isCreated())
    {
      return fail("CUDA context was not created");
    }
    if (!throws([&context, argv]() { context.create(std::stoi(argv[2])); }))
    {
      return fail("Double CUDA context creation was accepted");
    }

    bool crossThreadSyncRejected = false;
    bool crossThreadDestroyRejected = false;
    bool crossThreadShutdownRejected = false;
    std::thread wrongThread([&context, &crossThreadSyncRejected, &crossThreadDestroyRejected,
                             &crossThreadShutdownRejected]() {
      crossThreadSyncRejected = throws([&context]() { context.synchronize(); });
      crossThreadDestroyRejected = throws([&context]() { context.destroy(); });
      crossThreadShutdownRejected = throws([&context]() { context.shutdown(); });
    });
    wrongThread.join();
    if (!crossThreadSyncRejected || !crossThreadDestroyRejected || !crossThreadShutdownRejected)
    {
      return fail("Cross-thread CUDA context access or teardown was accepted");
    }

    if (!throws([&context]() { context.allocateDevice(0); }))
    {
      return fail("Zero-byte CUDA allocation was accepted");
    }
    void *allocation = context.allocateDevice(4096, vtm::CudaQueue::Upload);
    context.recordFence(vtm::CudaQueue::Upload, vtm::CudaFence::UploadComplete);
    context.waitFence(vtm::CudaQueue::Compute, vtm::CudaFence::UploadComplete);
    context.releaseDevice(allocation, vtm::CudaQueue::Compute);
    context.recordFence(vtm::CudaQueue::Compute, vtm::CudaFence::ComputeComplete);
    context.waitFence(vtm::CudaQueue::Download, vtm::CudaFence::ComputeComplete);
    context.synchronize();
    context.shutdown();
    if (context.isCreated())
    {
      return fail("CUDA context was not shut down");
    }
    context.shutdown();

    context.create(std::stoi(argv[2]));
    allocation = context.allocateDevice(4096, vtm::CudaQueue::Compute);
    context.releaseDevice(allocation, vtm::CudaQueue::Compute);
    context.shutdown();   // verifies teardown with pending asynchronous work
    context.create(std::stoi(argv[2]));
    context.destroy();
    context.destroy();

    vtm::CudaContext first;
    vtm::CudaContext second;
    first.create(std::stoi(argv[2]));
    second.create(std::stoi(argv[2]));
    first.shutdown();
    second.shutdown();
  }
  catch (const std::exception &error)
  {
    return fail(error.what());
  }

  return EXIT_SUCCESS;
}
