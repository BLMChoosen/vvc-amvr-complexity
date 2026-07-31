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
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

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

std::int32_t readSample(const void *address, const std::uint8_t elementSize)
{
  if (elementSize == 2)
  {
    std::int16_t value = 0;
    std::memcpy(&value, address, sizeof(value));
    return value;
  }
  std::int32_t value = 0;
  std::memcpy(&value, address, sizeof(value));
  return value;
}

void writeSample(void *address, const std::uint8_t elementSize, const std::int32_t value)
{
  if (elementSize == 2)
  {
    const std::int16_t stored = static_cast<std::int16_t>(value);
    std::memcpy(address, &stored, sizeof(stored));
    return;
  }
  std::memcpy(address, &value, sizeof(value));
}

std::uint64_t referenceSad(const void *source, const std::ptrdiff_t sourceStrideBytes, const void *reference,
                           const std::ptrdiff_t referenceStrideBytes, const std::uint32_t width,
                           const std::uint32_t height, const std::uint8_t elementSize,
                           const std::uint8_t bitDepth, const std::uint8_t subShift)
{
  std::uint64_t sum = 0;
  const std::uint32_t step = 1u << subShift;
  for (std::uint32_t y = 0; y < height; y += step)
  {
    const auto *sourceRow = static_cast<const std::uint8_t *>(source) + y * sourceStrideBytes;
    const auto *referenceRow = static_cast<const std::uint8_t *>(reference) + y * referenceStrideBytes;
    for (std::uint32_t x = 0; x < width; ++x)
    {
      const std::int32_t difference = readSample(sourceRow + x * elementSize, elementSize)
                                      - readSample(referenceRow + x * elementSize, elementSize);
      sum += static_cast<std::uint64_t>(difference < 0 ? -difference : difference);
    }
  }
  (void) bitDepth;
  return sum << subShift;
}

struct TestPicture
{
  struct Plane
  {
    std::vector<std::uint8_t> storage;
    std::vector<std::uint8_t> expected;
    std::size_t stride = 0;
    std::size_t rowBytes = 0;
    std::size_t fullHeight = 0;
  };

  std::array<Plane, vtm::CUDA_PICTURE_PLANE_COUNT> planes;
  vtm::CudaHostPictureDesc descriptor{};

  TestPicture(const std::uint32_t width, const std::uint32_t height, const std::uint8_t elementSize,
              const std::uint8_t bitDepth, const std::uint8_t salt = 0)
  {
    descriptor.planeCount = vtm::CUDA_PICTURE_PLANE_COUNT;
    for (std::size_t index = 0; index < planes.size(); ++index)
    {
      const std::uint32_t planeWidth = index == 0 ? width : (width + 1) / 2;
      const std::uint32_t planeHeight = index == 0 ? height : (height + 1) / 2;
      const std::uint16_t horizontalMargin = index == 0 ? 3 : 1;
      const std::uint16_t verticalMargin = index == 0 ? 2 : 1;
      Plane &plane = planes[index];
      plane.rowBytes = (horizontalMargin + planeWidth + horizontalMargin) * elementSize;
      plane.stride = plane.rowBytes + 11 + index;
      plane.fullHeight = verticalMargin + planeHeight + verticalMargin;
      plane.storage.resize(plane.stride * plane.fullHeight, 0xcd);
      for (std::size_t row = 0; row < plane.fullHeight; ++row)
      {
        for (std::size_t column = 0; column < plane.rowBytes; ++column)
        {
          plane.storage[row * plane.stride + column] =
            static_cast<std::uint8_t>((row * 37 + column * 13 + index * 53 + bitDepth + salt) & 0xff);
        }
      }
      plane.expected = plane.storage;

      vtm::CudaHostPlaneDesc &target = descriptor.planes[index];
      target.data = plane.storage.data() + verticalMargin * plane.stride + horizontalMargin * elementSize;
      target.width = planeWidth;
      target.height = planeHeight;
      target.strideBytes = static_cast<std::ptrdiff_t>(plane.stride);
      target.marginLeft = horizontalMargin;
      target.marginRight = horizontalMargin;
      target.marginTop = verticalMargin;
      target.marginBottom = verticalMargin;
      target.elementSize = elementSize;
      target.bitDepth = bitDepth;
    }
  }

  void clearTransferredBytes()
  {
    for (Plane &plane : planes)
    {
      for (std::size_t row = 0; row < plane.fullHeight; ++row)
      {
        std::memset(plane.storage.data() + row * plane.stride, 0, plane.rowBytes);
      }
    }
  }

  bool matchesExpected() const
  {
    for (std::size_t index = 0; index < planes.size(); ++index)
    {
      if (planes[index].storage != planes[index].expected)
      {
        return false;
      }
    }
    return true;
  }


  void fillSamples(const std::uint8_t salt)
  {
    for (std::size_t planeIndex = 0; planeIndex < planes.size(); ++planeIndex)
    {
      Plane &plane = planes[planeIndex];
      const vtm::CudaHostPlaneDesc &description = descriptor.planes[planeIndex];
      const std::uint32_t maximum = (1u << description.bitDepth) - 1;
      const std::size_t sampleCount = plane.rowBytes / description.elementSize;
      for (std::size_t row = 0; row < plane.fullHeight; ++row)
      {
        for (std::size_t column = 0; column < sampleCount; ++column)
        {
          std::uint32_t value = static_cast<std::uint32_t>(row * 131 + column * 47 + planeIndex * 29 + salt);
          value &= maximum;
          if ((row + column) % 19 == 0)
          {
            value = 0;
          }
          else if ((row * 3 + column) % 23 == 0)
          {
            value = maximum;
          }
          writeSample(plane.storage.data() + row * plane.stride + column * description.elementSize,
                      description.elementSize, static_cast<std::int32_t>(value));
        }
      }
      plane.expected = plane.storage;
    }
  }
};

bool runSadBatchCase(vtm::CudaContext &context, const std::uint8_t elementSize, const std::uint8_t bitDepth,
                     const std::uint32_t blockWidth, const std::uint32_t blockHeight,
                     const std::uint8_t subShift)
{
  TestPicture source(96, 80, elementSize, bitDepth, 3);
  TestPicture reference(96, 80, elementSize, bitDepth, 71);
  source.fillSamples(3);
  reference.fillSamples(71);
  int sourceOwner = 0;
  int referenceOwner = 0;
  const vtm::CudaMirrorHandle sourceMirror = context.registerPictureMirror(
    &sourceOwner, vtm::CudaPictureRole::Original, source.descriptor);
  const vtm::CudaMirrorHandle referenceMirror = context.registerPictureMirror(
    &referenceOwner, vtm::CudaPictureRole::Reconstruction, reference.descriptor);

  const auto &sourcePlane = source.descriptor.planes[0];
  const auto &referencePlane = reference.descriptor.planes[0];
  const auto *sourceBlock = static_cast<const std::uint8_t *>(sourcePlane.data)
                            + 5 * sourcePlane.strideBytes + 7 * elementSize;
  const auto *referenceOrigin = static_cast<const std::uint8_t *>(referencePlane.data);
  constexpr std::uint32_t candidateColumns = 8;
  constexpr std::uint32_t candidateRows = 8;
  constexpr std::uint32_t candidateCount = candidateColumns * candidateRows;

  vtm::CudaDistortionBatchDesc batch{};
  batch.sourceMirror = sourceMirror;
  batch.referenceMirror = referenceMirror;
  batch.source = sourceBlock;
  batch.candidateGrid = { referenceOrigin, candidateColumns, candidateRows, 1, 1 };
  batch.width = blockWidth;
  batch.height = blockHeight;
  batch.sourcePlane = 0;
  batch.referencePlane = 0;
  batch.elementSize = elementSize;
  batch.bitDepth = bitDepth;
  batch.subShift = subShift;
  batch.metric = vtm::CudaDistortionMetric::Sad;

  const std::uint64_t dispatchesBefore = context.distortionBatchDispatchCount();
  std::vector<std::uint64_t> results(candidateCount);
  if (!context.computeDistortionBatch(batch, results.data()))
  {
    return false;
  }
  if (context.distortionBatchDispatchCount() != dispatchesBefore + 1)
  {
    return false;
  }
  for (std::uint32_t candidate = 0; candidate < candidateCount; ++candidate)
  {
    const std::uint32_t candidateX = candidate % candidateColumns;
    const std::uint32_t candidateY = candidate / candidateColumns;
    const void *candidateReference = referenceOrigin + candidateY * referencePlane.strideBytes
                                     + candidateX * elementSize;
    const std::uint64_t expected = referenceSad(sourceBlock, sourcePlane.strideBytes,
                                                candidateReference, referencePlane.strideBytes,
                                                blockWidth, blockHeight, elementSize, bitDepth, subShift);
    if (results[candidate] != expected)
    {
      return false;
    }
  }
  std::size_t best = 0;
  for (std::size_t candidate = 1; candidate < results.size(); ++candidate)
  {
    if (results[candidate] < results[best])
    {
      best = candidate;
    }
  }
  std::vector<std::uint64_t> repeated = results;
  std::size_t repeatedBest = 0;
  for (std::size_t candidate = 1; candidate < repeated.size(); ++candidate)
  {
    if (repeated[candidate] < repeated[repeatedBest])
    {
      repeatedBest = candidate;
    }
  }
  if (best != repeatedBest)
  {
    return false;
  }

  const std::uint64_t dispatchesAfterValidBatch = context.distortionBatchDispatchCount();
  vtm::CudaDistortionBatchDesc invalidFormat = batch;
  invalidFormat.elementSize = 1;
  if (context.computeDistortionBatch(invalidFormat, results.data())
      || context.distortionBatchDispatchCount() != dispatchesAfterValidBatch)
  {
    return false;
  }
  vtm::CudaDistortionBatchDesc outsideBatch = batch;
  outsideBatch.candidateGrid.reference = static_cast<const std::uint8_t *>(referencePlane.data)
                                         + (referencePlane.height + referencePlane.marginBottom)
                                             * referencePlane.strideBytes;
  outsideBatch.candidateGrid.columns = 1;
  outsideBatch.candidateGrid.rows = 1;
  if (context.computeDistortionBatch(outsideBatch, results.data())
      || context.distortionBatchDispatchCount() != dispatchesAfterValidBatch)
  {
    return false;
  }

  context.releasePictureMirrors(&sourceOwner);
  context.releasePictureMirrors(&referenceOwner);
  return true;
}

bool benchmarkSadBatch(vtm::CudaContext &context)
{
  TestPicture source(128, 128, 2, 10, 9);
  TestPicture reference(128, 128, 2, 10, 83);
  source.fillSamples(9);
  reference.fillSamples(83);
  int sourceOwner = 0;
  int referenceOwner = 0;
  const auto sourceMirror = context.registerPictureMirror(
    &sourceOwner, vtm::CudaPictureRole::Original, source.descriptor);
  const auto referenceMirror = context.registerPictureMirror(
    &referenceOwner, vtm::CudaPictureRole::Reconstruction, reference.descriptor);
  const auto &sourcePlane = source.descriptor.planes[0];
  const auto &referencePlane = reference.descriptor.planes[0];
  vtm::CudaDistortionBatchDesc batch{};
  batch.sourceMirror = sourceMirror;
  batch.referenceMirror = referenceMirror;
  batch.source = sourcePlane.data;
  batch.candidateGrid = { referencePlane.data, 16, 16, 1, 1 };
  batch.width = 64;
  batch.height = 64;
  batch.sourcePlane = 0;
  batch.referencePlane = 0;
  batch.elementSize = 2;
  batch.bitDepth = 10;
  batch.subShift = 0;
  batch.metric = vtm::CudaDistortionMetric::Sad;
  std::vector<std::uint64_t> results(256);
  context.computeDistortionBatch(batch, results.data());

  constexpr unsigned gpuIterations = 200;
  const auto gpuStart = std::chrono::steady_clock::now();
  for (unsigned iteration = 0; iteration < gpuIterations; ++iteration)
  {
    context.computeDistortionBatch(batch, results.data());
  }
  const auto gpuEnd = std::chrono::steady_clock::now();

  constexpr unsigned cpuIterations = 10;
  std::uint64_t checksum = 0;
  const auto cpuStart = std::chrono::steady_clock::now();
  for (unsigned iteration = 0; iteration < cpuIterations; ++iteration)
  {
    for (std::uint32_t candidate = 0; candidate < 256; ++candidate)
    {
      const void *candidateReference = static_cast<const std::uint8_t *>(referencePlane.data)
                                       + (candidate >> 4) * referencePlane.strideBytes
                                       + (candidate & 15) * 2;
      checksum += referenceSad(sourcePlane.data, sourcePlane.strideBytes, candidateReference,
                               referencePlane.strideBytes, 64, 64, 2, 10, 0) + iteration;
    }
  }
  const auto cpuEnd = std::chrono::steady_clock::now();
  const double gpuMilliseconds = std::chrono::duration<double, std::milli>(gpuEnd - gpuStart).count()
                                 / gpuIterations;
  const double cpuMilliseconds = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count()
                                 / cpuIterations;
  std::cout << "SAD microbenchmark 256x64x64 Pel16/10-bit: CPU " << cpuMilliseconds
            << " ms/batch, CUDA " << gpuMilliseconds << " ms/batch, ratio "
            << (cpuMilliseconds / gpuMilliseconds) << "x, checksum " << checksum << '\n';
  context.releasePictureMirrors(&sourceOwner);
  context.releasePictureMirrors(&referenceOwner);
  return gpuMilliseconds > 0.0;
}

#if VTM_CUDA_TESTING
bool runSadFailureCase(const int device, const unsigned allocationFailureStep, const unsigned executionFailures)
{
  vtm::CudaContext context;
  context.create(device);
  TestPicture source(32, 32, 2, 10, 5);
  TestPicture reference(32, 32, 2, 10, 17);
  source.fillSamples(5);
  reference.fillSamples(17);
  const auto sourceMirror = context.registerPictureMirror(
    &source, vtm::CudaPictureRole::Original, source.descriptor);
  const auto referenceMirror = context.registerPictureMirror(
    &reference, vtm::CudaPictureRole::Reconstruction, reference.descriptor);
  vtm::CudaDistortionBatchDesc batch{};
  batch.sourceMirror = sourceMirror;
  batch.referenceMirror = referenceMirror;
  batch.source = source.descriptor.planes[0].data;
  batch.candidateGrid = { reference.descriptor.planes[0].data, 8, 8, 1, 1 };
  batch.width = 8;
  batch.height = 8;
  batch.elementSize = 2;
  batch.bitDepth = 10;
  batch.metric = vtm::CudaDistortionMetric::Sad;
  std::array<std::uint64_t, 64> results{};
  context.injectDistortionFailuresForTesting(allocationFailureStep, executionFailures);
  if (context.computeDistortionBatch(batch, results.data())
      || context.isDistortionAccelerationAvailable()
      || context.distortionBatchFailureCount() != 1
      || context.distortionBatchDispatchCount() != 0)
  {
    return false;
  }
  // Poisoning is permanent for this context and must reject without touching CUDA again.
  if (context.computeDistortionBatch(batch, results.data()) || context.distortionBatchFailureCount() != 1)
  {
    return false;
  }
  context.shutdown();
  return true;
}
#endif

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
    vtm::CudaPinnedBuffer pinned;
    if (!throws([&pinned]() { pinned.allocate(64); }))
    {
      return fail("ENABLE_CUDA=OFF build accepted pinned allocation");
    }
    TestPicture picture(7, 5, 1, 8);
    if (!throws([&context, &picture]() {
          context.registerPictureMirror(&picture, vtm::CudaPictureRole::Reconstruction, picture.descriptor);
        }))
    {
      return fail("ENABLE_CUDA=OFF build accepted picture mirror registration");
    }
    return EXIT_SUCCESS;
  }
  const bool runBenchmark = argc == 3 && std::string(argv[1]) == "--cuda-benchmark";
  if (argc != 3 || (std::string(argv[1]) != "--cuda" && !runBenchmark))
  {
    return fail("Usage: CudaBackendTest [--cuda device | --cuda-benchmark device]");
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

    vtm::CudaPinnedBuffer pinned(4096);
    if (!pinned || pinned.size() != 4096)
    {
      return fail("Pinned allocation was not created");
    }
    std::memset(pinned.data(), 0x5a, pinned.size());
    vtm::CudaPinnedBuffer moved(std::move(pinned));
    if (pinned || !moved || moved.size() != 4096)
    {
      return fail("Pinned allocation move did not transfer ownership");
    }
    pinned.allocate(128);
    if (!pinned || pinned.size() != 128)
    {
      return fail("Moved-from pinned allocation could not be safely reused");
    }
    pinned.reset();
    moved.reset();
    if (moved || moved.size() != 0)
    {
      return fail("Pinned allocation reset did not release ownership");
    }

    TestPicture picture8(7, 5, 2, 8);
    TestPicture picture10(16, 10, 2, 10);
    TestPicture picture8Wide(12, 8, 4, 8);
    TestPicture picture10Wide(18, 12, 4, 10);
    int owner8 = 0;
    int owner10 = 0;
    int owner8Wide = 0;
    int owner10Wide = 0;
    const vtm::CudaMirrorHandle mirror8 = context.registerPictureMirror(
      &owner8, vtm::CudaPictureRole::Reconstruction, picture8.descriptor);
    const vtm::CudaMirrorHandle mirror10 = context.registerPictureMirror(
      &owner10, vtm::CudaPictureRole::Original, picture10.descriptor);
    const vtm::CudaMirrorHandle mirror8Wide = context.registerPictureMirror(
      &owner8Wide, vtm::CudaPictureRole::Reconstruction, picture8Wide.descriptor);
    const vtm::CudaMirrorHandle mirror10Wide = context.registerPictureMirror(
      &owner10Wide, vtm::CudaPictureRole::Original, picture10Wide.descriptor);
    if (mirror8 == mirror10 || mirror8Wide == mirror10Wide || context.pictureMirrorCount() != 4
        || !context.hasPictureMirror(&owner8, vtm::CudaPictureRole::Reconstruction)
        || context.pictureMirrorHandle(&owner10, vtm::CudaPictureRole::Original) != mirror10)
    {
      return fail("Multiple CUDA picture mirrors were not registered independently");
    }
    if (context.pictureMirrorState(mirror8) != vtm::CudaMirrorState::HostValid
        || !throws([&context, mirror8]() { (void) context.devicePicture(mirror8); })
        || !throws([&context, mirror8]() { context.markDeviceModified(mirror8); }))
    {
      return fail("Initial CUDA picture mirror state was not HostValid");
    }
    if (!throws([&context, &owner8, &picture8]() {
          context.registerPictureMirror(&owner8, vtm::CudaPictureRole::Reconstruction, picture8.descriptor);
        }))
    {
      return fail("Duplicate CUDA picture mirror registration was accepted");
    }

    for (const vtm::CudaMirrorHandle mirror : { mirror8, mirror10, mirror8Wide, mirror10Wide })
    {
      context.ensureDevice(mirror);
      if (context.pictureMirrorState(mirror) != vtm::CudaMirrorState::Synchronized)
      {
        return fail("CUDA picture upload did not produce Synchronized state");
      }
      const vtm::CudaDevicePictureDesc device = context.devicePicture(mirror);
      if (device.planeCount != vtm::CUDA_PICTURE_PLANE_COUNT || device.planes[0].data == nullptr
          || device.planes[0].pitchBytes == 0)
      {
        return fail("CUDA device picture descriptor is invalid");
      }
    }

    picture8.clearTransferredBytes();
    context.markDeviceModified(mirror8);
    if (context.pictureMirrorState(mirror8) != vtm::CudaMirrorState::DeviceValid
        || !throws([&context, mirror8]() { context.markHostModified(mirror8); }))
    {
      return fail("DeviceValid conflict handling is invalid");
    }
    context.ensureHost(mirror8);
    if (context.pictureMirrorState(mirror8) != vtm::CudaMirrorState::Synchronized || !picture8.matchesExpected())
    {
      return fail("8-bit 4:2:0 picture round-trip with margins and non-contiguous strides was not byte exact");
    }

    picture10.clearTransferredBytes();
    context.markDeviceModified(mirror10);
    context.ensureHost(mirror10);
    if (!picture10.matchesExpected())
    {
      return fail("10-bit 4:2:0 picture round-trip with margins and non-contiguous strides was not byte exact");
    }
    context.markHostModified(mirror10);
    context.ensureDevice(mirror10);

    for (const auto item : { std::make_pair(mirror8Wide, &picture8Wide),
                             std::make_pair(mirror10Wide, &picture10Wide) })
    {
      item.second->clearTransferredBytes();
      context.markDeviceModified(item.first);
      context.ensureHost(item.first);
      if (!item.second->matchesExpected())
      {
        return fail("Four-byte Pel 4:2:0 picture round-trip was not byte exact");
      }
    }

    TestPicture rebound8(7, 5, 2, 8, 91);
    context.rebindHostPicture(mirror8, rebound8.descriptor);
    if (context.pictureMirrorState(mirror8) != vtm::CudaMirrorState::HostValid)
    {
      return fail("CUDA host picture rebind did not invalidate stale device data");
    }
    context.ensureDevice(mirror8);
    rebound8.clearTransferredBytes();
    context.markDeviceModified(mirror8);
    context.ensureHost(mirror8);
    if (!rebound8.matchesExpected())
    {
      return fail("CUDA host picture rebind uploaded the old host storage");
    }

    TestPicture resized8(9, 7, 2, 8, 37);
    context.rebindHostPicture(&owner8, vtm::CudaPictureRole::Reconstruction, resized8.descriptor);
    context.ensureDevice(mirror8);
    resized8.clearTransferredBytes();
    context.markDeviceModified(mirror8);
    context.ensureHost(mirror8);
    if (!resized8.matchesExpected() || context.devicePicture(mirror8).planes[0].width != 9)
    {
      return fail("CUDA host picture geometry rebind did not reallocate the mirror");
    }

    context.releasePictureMirror(mirror8);
    if (context.pictureMirrorCount() != 3 || context.hasPictureMirror(&owner8, vtm::CudaPictureRole::Reconstruction)
        || !throws([&context, mirror8]() { context.ensureDevice(mirror8); }))
    {
      return fail("Released CUDA picture mirror remained usable");
    }
    context.releasePictureMirrors(&owner10);
    context.releasePictureMirrors(&owner8Wide);
    context.releasePictureMirrors(&owner10Wide);
    if (context.pictureMirrorCount() != 0)
    {
      return fail("Owner-based CUDA picture mirror release left registry entries");
    }

    if (!runSadBatchCase(context, 2, 8, 4, 4, 0)
        || !runSadBatchCase(context, 2, 10, 16, 12, 1)
        || !runSadBatchCase(context, 4, 8, 32, 24, 2)
        || !runSadBatchCase(context, 4, 10, 64, 32, 0))
    {
      return fail("CUDA SAD batch differed from the ordered scalar reference");
    }
#if VTM_CUDA_TESTING
    if (!runSadFailureCase(std::stoi(argv[2]), 1, 0)
        || !runSadFailureCase(std::stoi(argv[2]), 2, 0)
        || !runSadFailureCase(std::stoi(argv[2]), 0, 1))
    {
      return fail("CUDA SAD failure recovery or permanent poisoning is invalid");
    }
#endif
    if (runBenchmark && !benchmarkSadBatch(context))
    {
      return fail("CUDA SAD microbenchmark failed");
    }

#if VTM_CUDA_TESTING
    TestPicture fallbackPicture(10, 8, 2, 10, 17);
    int fallbackOwner = 0;
    context.registerPictureMirror(
      &fallbackOwner, vtm::CudaPictureRole::Original, fallbackPicture.descriptor);
    context.registerPictureMirror(
      &fallbackOwner, vtm::CudaPictureRole::Reconstruction, fallbackPicture.descriptor);
    context.injectReleaseFailuresForTesting(6, 0);
    if (!throws([&context, &fallbackOwner]() { context.releasePictureMirrors(&fallbackOwner); })
        || context.pictureMirrorCount() != 0)
    {
      return fail("Synchronous fallback did not release every mirror after asynchronous free failures");
    }

    const vtm::CudaMirrorHandle retryHandle = context.registerPictureMirror(
      &fallbackOwner, vtm::CudaPictureRole::Reconstruction, fallbackPicture.descriptor);
    context.injectReleaseFailuresForTesting(1, 1);
    if (!throws([&context, retryHandle]() { context.releasePictureMirror(retryHandle); })
        || context.pictureMirrorCount() != 1
        || !context.hasPictureMirror(&fallbackOwner, vtm::CudaPictureRole::Reconstruction))
    {
      return fail("Mirror ownership was lost after both asynchronous and synchronous free failed");
    }
    context.releasePictureMirror(retryHandle);
    if (context.pictureMirrorCount() != 0)
    {
      return fail("Retained CUDA allocation could not be released on retry");
    }
#endif

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
    TestPicture pendingPicture(32, 18, 2, 10);
    int pendingOwner = 0;
    const vtm::CudaMirrorHandle pendingMirror = context.registerPictureMirror(
      &pendingOwner, vtm::CudaPictureRole::Reconstruction, pendingPicture.descriptor);
    context.ensureDevice(pendingMirror);
    context.markDeviceModified(pendingMirror);
    allocation = context.allocateDevice(4096, vtm::CudaQueue::Compute);
    context.releaseDevice(allocation, vtm::CudaQueue::Compute);
    context.shutdown();   // verifies teardown with pending asynchronous work
    if (context.pictureMirrorCount() != 0)
    {
      return fail("CUDA context shutdown did not empty the picture mirror registry");
    }
    context.create(std::stoi(argv[2]));
    context.destroy();
    context.destroy();

    vtm::CudaContext first;
    vtm::CudaContext second;
    first.create(std::stoi(argv[2]));
    second.create(std::stoi(argv[2]));
    TestPicture crossContextPicture(8, 6, 2, 8);
    int firstOwner = 0;
    int secondOwner = 0;
    const vtm::CudaMirrorHandle firstHandle = first.registerPictureMirror(
      &firstOwner, vtm::CudaPictureRole::Reconstruction, crossContextPicture.descriptor);
    const vtm::CudaMirrorHandle secondHandle = second.registerPictureMirror(
      &secondOwner, vtm::CudaPictureRole::Reconstruction, crossContextPicture.descriptor);
    if (firstHandle == secondHandle || !throws([&second, firstHandle]() { second.ensureDevice(firstHandle); }))
    {
      return fail("CUDA mirror handle was accepted by a different context generation");
    }
    first.shutdown();
    second.shutdown();
  }
  catch (const std::exception &error)
  {
    return fail(error.what());
  }

  return EXIT_SUCCESS;
}
