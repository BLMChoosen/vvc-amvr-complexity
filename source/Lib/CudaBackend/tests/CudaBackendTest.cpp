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
#include "CommonLib/AdaptiveLoopFilter.h"
#include "CommonLib/DeblockingFilter.h"
#include "CommonLib/SampleAdaptiveOffset.h"

#include <cstdlib>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <new>
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

bool throwsWithText(const std::function<void()> &operation, const std::string &expected)
{
  try
  {
    operation();
  }
  catch (const std::exception &error)
  {
    return std::string(error.what()).find(expected) != std::string::npos;
  }
  return false;
}

bool throwsBadAlloc(const std::function<void()> &operation)
{
  try
  {
    operation();
  }
  catch (const std::bad_alloc &)
  {
    return true;
  }
  catch (...)
  {
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
              const std::uint8_t bitDepth, const std::uint8_t salt = 0,
              const std::uint8_t planeCount = vtm::CUDA_PICTURE_PLANE_COUNT,
              const std::uint8_t chromaScaleX = 1, const std::uint8_t chromaScaleY = 1,
              const std::uint16_t lumaHorizontalMargin = 3,
              const std::uint16_t lumaVerticalMargin = 2)
  {
    descriptor.planeCount = planeCount;
    for (std::size_t index = 0; index < descriptor.planeCount; ++index)
    {
      const std::uint32_t planeWidth = index == 0 ? width : (width + (1u << chromaScaleX) - 1) >> chromaScaleX;
      const std::uint32_t planeHeight = index == 0 ? height : (height + (1u << chromaScaleY) - 1) >> chromaScaleY;
      const std::uint16_t horizontalMargin = index == 0 ? lumaHorizontalMargin : 1;
      const std::uint16_t verticalMargin = index == 0 ? lumaVerticalMargin : 1;
      Plane &plane = planes[index];
      plane.rowBytes = (horizontalMargin + planeWidth + horizontalMargin) * elementSize;
      plane.stride = plane.rowBytes + (11 + index) * elementSize;
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
    for (std::size_t index = 0; index < descriptor.planeCount; ++index)
    {
      Plane &plane = planes[index];
      for (std::size_t row = 0; row < plane.fullHeight; ++row)
      {
        std::memset(plane.storage.data() + row * plane.stride, 0, plane.rowBytes);
      }
    }
  }

  bool matchesExpected() const
  {
    for (std::size_t index = 0; index < descriptor.planeCount; ++index)
    {
      if (planes[index].storage != planes[index].expected)
      {
        return false;
      }
    }
    return true;
  }

  std::size_t pinnedBytes(const std::size_t plane) const
  {
    return planes[plane].rowBytes * planes[plane].fullHeight;
  }

  std::size_t deviceBytes(const std::size_t plane) const
  {
    const std::size_t pitch = (planes[plane].rowBytes + 127) & ~std::size_t(127);
    return pitch * planes[plane].fullHeight;
  }


  void fillSamples(const std::uint8_t salt)
  {
    for (std::size_t planeIndex = 0; planeIndex < descriptor.planeCount; ++planeIndex)
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

bool validatePictureDescriptorFormats()
{
  TestPicture monochrome(17, 11, 2, 10, 1, 1, 0, 0);
  TestPicture yuv420(17, 11, 2, 10, 2, 3, 1, 1);
  TestPicture yuv422(17, 11, 2, 10, 3, 3, 1, 0);
  TestPicture yuv444(17, 11, 2, 10, 4, 3, 0, 0);
  if (!vtm::CudaContext::isPictureMirrorDescriptorSupported(monochrome.descriptor)
      || !vtm::CudaContext::isPictureMirrorDescriptorSupported(yuv420.descriptor)
      || !vtm::CudaContext::isPictureMirrorDescriptorSupported(yuv422.descriptor)
      || !vtm::CudaContext::isPictureMirrorDescriptorSupported(yuv444.descriptor))
  {
    return false;
  }

  vtm::CudaHostPictureDesc unsupported = yuv420.descriptor;
  unsupported.planes[1].width = unsupported.planes[0].width;
  unsupported.planes[2].width = unsupported.planes[0].width;
  if (vtm::CudaContext::isPictureMirrorDescriptorSupported(unsupported))
  {
    return false;
  }

  vtm::CudaHostPictureDesc overflowing = monochrome.descriptor;
  overflowing.planes[0].height = std::numeric_limits<std::uint32_t>::max();
  overflowing.planes[0].strideBytes = std::numeric_limits<std::ptrdiff_t>::max();
  overflowing.planes[0].marginTop = std::numeric_limits<std::uint16_t>::max();
  if (vtm::CudaContext::isPictureMirrorDescriptorSupported(overflowing))
  {
    return false;
  }

  vtm::CudaHostPictureDesc underflowing = monochrome.descriptor;
  underflowing.planes[0].data = reinterpret_cast<void *>(std::uintptr_t{ 1 });
  underflowing.planes[0].marginLeft = 2;
  if (vtm::CudaContext::isPictureMirrorDescriptorSupported(underflowing))
  {
    return false;
  }
  return true;
}

bool writeLoopFilterBenchmarkYuv(const std::string &path)
{
  constexpr int width = 1920;
  constexpr int height = 1080;
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) return false;
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      output.put(static_cast<char>(((x * 13) ^ (y * 29) ^ ((x + y) >> 2)) & 0xff));
  for (int component = 0; component < 2; ++component)
    for (int y = 0; y < height / 2; ++y)
      for (int x = 0; x < width / 2; ++x)
        output.put(static_cast<char>(128 + (((x * 3 + y * 5 + component * 17) & 15) - 8)));
  return output.good();
}

bool repeatAccessUnit(const std::string &inputPath, const std::string &outputPath, const int repetitions)
{
  if (repetitions < 1 || repetitions > 1000) return false;
  std::ifstream input(inputPath, std::ios::binary);
  if (!input) return false;
  const std::vector<char> bytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  if (bytes.empty()) return false;
  struct NalSpan { std::size_t begin; std::size_t end; std::uint8_t type; };
  std::vector<std::pair<std::size_t, std::size_t>> starts;
  for (std::size_t i = 0; i + 4 < bytes.size(); ++i)
  {
    const bool three = bytes[i] == 0 && bytes[i + 1] == 0 && bytes[i + 2] == 1;
    const bool four = bytes[i] == 0 && bytes[i + 1] == 0 && bytes[i + 2] == 0 && bytes[i + 3] == 1;
    if (three || four)
    {
      starts.emplace_back(i, four ? 4 : 3);
      i += (four ? 4 : 3) - 1;
    }
  }
  std::vector<NalSpan> nals;
  for (std::size_t index = 0; index < starts.size(); ++index)
  {
    const std::size_t begin = starts[index].first;
    const std::size_t end = index + 1 < starts.size() ? starts[index + 1].first : bytes.size();
    const std::size_t header = begin + starts[index].second;
    if (header + 1 >= end) return false;
    nals.push_back(NalSpan{ begin, end, static_cast<std::uint8_t>(
      static_cast<unsigned char>(bytes[header + 1]) >> 3) });
  }
  if (nals.empty()) return false;
  std::ofstream output(outputPath, std::ios::binary | std::ios::trunc);
  if (!output) return false;
  bool hasVcl = false;
  // Parameter sets and prefix APS are persistent. Write them once so repeated self-contained IDRs do not
  // repeatedly tear down the same sequence state; omit EOS/EOB and repeat only the VCL access unit.
  for (const NalSpan &nal : nals)
  {
    if (nal.type <= 11)
    {
      hasVcl = true;
      continue;
    }
    if (nal.type == 21 || nal.type == 22) continue;
    output.write(bytes.data() + nal.begin, static_cast<std::streamsize>(nal.end - nal.begin));
  }
  for (int repetition = 0; repetition < repetitions; ++repetition)
    for (const NalSpan &nal : nals)
      if (nal.type <= 11)
        output.write(bytes.data() + nal.begin, static_cast<std::streamsize>(nal.end - nal.begin));
  return hasVcl && output.good();
}

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
  const vtm::CudaMirrorMemoryStats memoryBefore = context.pictureMirrorMemoryStats();

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
  const vtm::CudaMirrorMemoryStats memoryAfter = context.pictureMirrorMemoryStats();
  if (context.pictureMirrorAllocatedPlanes(sourceMirror) != vtm::CUDA_PLANE_Y
      || context.pictureMirrorAllocatedPlanes(referenceMirror) != vtm::CUDA_PLANE_Y
      || memoryAfter.byRoleAndPlane[0][1].uploadedBytes != memoryBefore.byRoleAndPlane[0][1].uploadedBytes
      || memoryAfter.byRoleAndPlane[0][2].uploadedBytes != memoryBefore.byRoleAndPlane[0][2].uploadedBytes
      || memoryAfter.byRoleAndPlane[1][1].uploadedBytes != memoryBefore.byRoleAndPlane[1][1].uploadedBytes
      || memoryAfter.byRoleAndPlane[1][2].uploadedBytes != memoryBefore.byRoleAndPlane[1][2].uploadedBytes)
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
      std::cerr << "SAD mismatch candidate " << candidate << ": CUDA " << results[candidate]
                << ", CPU " << expected << '\n';
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
  if (!throwsWithText([&context, &outsideBatch, &results]() {
        (void) context.computeDistortionBatch(outsideBatch, results.data());
      }, "CUDA distortion block exceeds its registered mirror")
      || context.distortionBatchDispatchCount() != dispatchesAfterValidBatch
      || !context.isDistortionAccelerationAvailable())
  {
    return false;
  }

  context.releasePictureMirrors(&sourceOwner);
  context.releasePictureMirrors(&referenceOwner);
  return true;
}

vtm::CudaQpaResult referenceQpa(const TestPicture &picture, const vtm::CudaQpaTask &task)
{
  const vtm::CudaHostPlaneDesc &plane = picture.descriptor.planes[0];
  vtm::QpaActivitySums sums{};
  if (plane.elementSize == 2)
  {
    sums = vtm::computeQpaActivitySums(static_cast<const std::int16_t *>(plane.data),
                                       plane.strideBytes / sizeof(std::int16_t),
                                       task.filterArea, task.lumaArea);
  }
  else
  {
    sums = vtm::computeQpaActivitySums(static_cast<const std::int32_t *>(plane.data),
                                       plane.strideBytes / sizeof(std::int32_t),
                                       task.filterArea, task.lumaArea);
  }
  vtm::CudaQpaResult result{};
  result.ticket = task.ticket;
  result.ctuAddr = task.ctuAddr;
  result.highpassSum = sums.highpass;
  result.lumaSum = sums.luma;
  return result;
}

std::vector<vtm::CudaQpaTask> makeQpaTasks(const std::uint32_t width, const std::uint32_t height,
                                           const std::uint32_t ctuSize)
{
  std::vector<vtm::CudaQpaTask> tasks;
  for (std::uint32_t y = 0; y < height; y += ctuSize)
  {
    for (std::uint32_t x = 0; x < width; x += ctuSize)
    {
      vtm::CudaQpaTask task{};
      task.ticket = tasks.size();
      task.ctuAddr = static_cast<std::uint32_t>(tasks.size());
      task.lumaArea = { x, y, std::min(ctuSize, width - x), std::min(ctuSize, height - y) };
      const std::uint32_t filterX = x > 0 ? x - 1 : 0;
      const std::uint32_t filterY = y > 0 ? y - 1 : 0;
      task.filterArea = { filterX, filterY,
                          std::min(width - filterX, ctuSize + (x > 0 ? 2u : 1u)),
                          std::min(height - filterY, ctuSize + (y > 0 ? 2u : 1u)) };
      tasks.push_back(task);
    }
  }
  return tasks;
}

bool runQpaBatchCase(vtm::CudaContext &context, const std::uint8_t elementSize, const std::uint8_t bitDepth)
{
  // Both dimensions intentionally leave partial CTUs; TestPicture always provides a 4:2:0 descriptor.
  TestPicture picture(259, 195, elementSize, bitDepth, 43);
  picture.fillSamples(43);
  int owner = 0;
  const vtm::CudaMirrorHandle mirror = context.registerPictureMirror(
    &owner, vtm::CudaPictureRole::Original, picture.descriptor);
  const std::vector<vtm::CudaQpaTask> tasks = makeQpaTasks(259, 195, 64);
  std::vector<vtm::CudaQpaResult> results(tasks.size());
  const std::uint64_t batchesBefore = context.qpaBatchDispatchCount();
  const std::uint64_t tasksBefore = context.qpaTaskCount();
  const vtm::CudaMirrorMemoryStats memoryBefore = context.pictureMirrorMemoryStats();
  if (!context.computeQpaBatch(mirror, tasks.data(), static_cast<std::uint32_t>(tasks.size()), results.data())
      || context.qpaBatchDispatchCount() != batchesBefore + 1
      || context.qpaTaskCount() != tasksBefore + tasks.size())
  {
    std::cerr << "QPA dispatch failed for Pel" << unsigned(elementSize * 8) << '/' << unsigned(bitDepth)
              << "-bit\n";
    return false;
  }
  const vtm::CudaMirrorMemoryStats memoryAfter = context.pictureMirrorMemoryStats();
  if (context.pictureMirrorAllocatedPlanes(mirror) != vtm::CUDA_PLANE_Y
      || memoryAfter.byRoleAndPlane[0][1].uploadedBytes != memoryBefore.byRoleAndPlane[0][1].uploadedBytes
      || memoryAfter.byRoleAndPlane[0][2].uploadedBytes != memoryBefore.byRoleAndPlane[0][2].uploadedBytes)
  {
    return false;
  }
  for (std::size_t index = 0; index < tasks.size(); ++index)
  {
    const vtm::CudaQpaResult expected = referenceQpa(picture, tasks[index]);
    if (results[index].ticket != expected.ticket || results[index].ctuAddr != expected.ctuAddr
        || results[index].highpassSum != expected.highpassSum || results[index].lumaSum != expected.lumaSum)
    {
      std::cerr << "QPA mismatch Pel" << unsigned(elementSize * 8) << '/' << unsigned(bitDepth)
                << "-bit task " << index << ": hp " << results[index].highpassSum << '/' << expected.highpassSum
                << ", luma " << results[index].lumaSum << '/' << expected.lumaSum << '\n';
      return false;
    }
  }

  vtm::CudaQpaTask invalid = tasks.front();
  invalid.filterArea.width = 2;
  if (context.computeQpaBatch(mirror, &invalid, 1, results.data())
      || context.qpaBatchDispatchCount() != batchesBefore + 1)
  {
    return false;
  }
  context.releasePictureMirrors(&owner);
  return true;
}

bool runQpaMultiChunkCase(vtm::CudaContext &context)
{
  TestPicture picture(67, 65, 2, 10, 57);
  picture.fillSamples(57);
  int owner = 0;
  const auto mirror = context.registerPictureMirror(&owner, vtm::CudaPictureRole::Original, picture.descriptor);
  constexpr std::size_t count = vtm::CUDA_MAX_QPA_TASKS + 37;
  std::vector<vtm::CudaQpaTask> tasks(count);
  for (std::size_t index = 0; index < count; ++index)
  {
    tasks[index].ticket = 9000 + index;
    tasks[index].ctuAddr = static_cast<std::uint32_t>(index);
    tasks[index].filterArea = { 0, 0, 65, 65 };
    tasks[index].lumaArea = { 1, 1, 64, 63 };
  }
  std::vector<vtm::CudaQpaResult> results(count);
  for (std::size_t offset = 0; offset < count; offset += vtm::CUDA_MAX_QPA_TASKS)
  {
    const std::uint32_t chunk = static_cast<std::uint32_t>(
      std::min<std::size_t>(count - offset, vtm::CUDA_MAX_QPA_TASKS));
    if (!context.computeQpaBatch(mirror, tasks.data() + offset, chunk, results.data() + offset))
    {
      return false;
    }
  }
  for (std::size_t index = 0; index < count; ++index)
  {
    const vtm::CudaQpaResult expected = referenceQpa(picture, tasks[index]);
    if (results[index].ticket != tasks[index].ticket || results[index].ctuAddr != tasks[index].ctuAddr
        || results[index].highpassSum != expected.highpassSum || results[index].lumaSum != expected.lumaSum)
    {
      return false;
    }
  }
  context.releasePictureMirrors(&owner);
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

bool benchmarkQpaBatch(const int device)
{
  vtm::CudaContext context;
  context.create(device);
  TestPicture scratchWarmup(64, 64, 2, 10, 19);
  scratchWarmup.fillSamples(19);
  int warmupOwner = 0;
  const auto warmupMirror = context.registerPictureMirror(
    &warmupOwner, vtm::CudaPictureRole::Original, scratchWarmup.descriptor);
  const std::vector<vtm::CudaQpaTask> warmupTasks = makeQpaTasks(64, 64, 64);
  vtm::CudaQpaResult warmupResult{};
  if (!context.computeQpaBatch(warmupMirror, warmupTasks.data(), 1, &warmupResult))
  {
    return false;
  }
  context.releasePictureMirror(warmupMirror);

  TestPicture picture(512, 512, 2, 10, 73);
  picture.fillSamples(73);
  int owner = 0;
  const auto mirror = context.registerPictureMirror(&owner, vtm::CudaPictureRole::Original, picture.descriptor);
  const std::vector<vtm::CudaQpaTask> tasks = makeQpaTasks(512, 512, 64);
  std::vector<vtm::CudaQpaResult> results(tasks.size());
  const auto coldStart = std::chrono::steady_clock::now();
  if (!context.computeQpaBatch(mirror, tasks.data(), static_cast<std::uint32_t>(tasks.size()), results.data()))
  {
    return false;
  }
  const auto coldEnd = std::chrono::steady_clock::now();

  constexpr unsigned gpuIterations = 300;
  const auto gpuStart = std::chrono::steady_clock::now();
  for (unsigned iteration = 0; iteration < gpuIterations; ++iteration)
  {
    if (!context.computeQpaBatch(mirror, tasks.data(), static_cast<std::uint32_t>(tasks.size()), results.data()))
    {
      return false;
    }
  }
  const auto gpuEnd = std::chrono::steady_clock::now();

  constexpr unsigned cpuIterations = 100;
  std::uint64_t checksum = 0;
  const auto cpuStart = std::chrono::steady_clock::now();
  for (unsigned iteration = 0; iteration < cpuIterations; ++iteration)
  {
    for (const vtm::CudaQpaTask &task : tasks)
    {
      const vtm::CudaQpaResult result = referenceQpa(picture, task);
      checksum += result.highpassSum + static_cast<std::uint64_t>(result.lumaSum) + iteration;
    }
  }
  const auto cpuEnd = std::chrono::steady_clock::now();
  const double gpuMilliseconds = std::chrono::duration<double, std::milli>(gpuEnd - gpuStart).count()
                                 / gpuIterations;
  const double coldGpuMilliseconds = std::chrono::duration<double, std::milli>(coldEnd - coldStart).count();
  const double cpuMilliseconds = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count()
                                 / cpuIterations;
  std::cout << "QPA microbenchmark 64 CTUs x 64x64 Pel16/10-bit: new-mirror CUDA " << coldGpuMilliseconds
            << " ms (scratch warm; lazy Y allocation + upload + compute), warm CPU " << cpuMilliseconds
            << " ms/batch, warm CUDA " << gpuMilliseconds << " ms/batch, warm-compute ratio "
            << (cpuMilliseconds / gpuMilliseconds) << "x, checksum " << checksum << '\n';
  context.releasePictureMirrors(&owner);

  bool memoryExact = true;
  for (const auto dimensions : { std::make_pair(1920u, 1080u), std::make_pair(3840u, 2160u) })
  {
    TestPicture measured(dimensions.first, dimensions.second, 2, 10, 7);
    int measuredOwner = 0;
    const auto measuredMirror = context.registerPictureMirror(
      &measuredOwner, vtm::CudaPictureRole::Original, measured.descriptor);
    const vtm::CudaMirrorMemoryStats before = context.pictureMirrorMemoryStats();
    context.ensureDevicePlane(measuredMirror, 0);
    const vtm::CudaMirrorMemoryStats after = context.pictureMirrorMemoryStats();
    const std::uint64_t measuredDevice = after.total.currentDeviceBytes - before.total.currentDeviceBytes;
    const std::uint64_t measuredPinned = after.total.currentPinnedBytes - before.total.currentPinnedBytes;
    const std::uint64_t expectedDevice = measured.deviceBytes(0);
    const std::uint64_t expectedPinned = measured.pinnedBytes(0);
    memoryExact = memoryExact && measuredDevice == expectedDevice && measuredPinned == expectedPinned
                  && context.pictureMirrorAllocatedPlanes(measuredMirror) == vtm::CUDA_PLANE_Y;
    std::cout << "QPA luma-only mirror " << dimensions.first << 'x' << dimensions.second
              << ": measured/estimated device " << measuredDevice << '/' << expectedDevice
              << " bytes, pinned " << measuredPinned << '/' << expectedPinned << " bytes\n";
    context.releasePictureMirror(measuredMirror);
  }
  context.shutdown();
  return gpuMilliseconds > 0.0 && memoryExact;
}

#if VTM_CUDA_TESTING
bool runBatchOperationalPreflightExceptionCase(const int device)
{
  vtm::CudaContext context;
  context.create(device);
  TestPicture source(96, 80, 2, 10, 41);
  TestPicture reference(96, 80, 2, 10, 43);
  source.fillSamples(41);
  reference.fillSamples(43);
  const auto sourceMirror = context.registerPictureMirror(&source, vtm::CudaPictureRole::Original,
                                                          source.descriptor);
  const auto referenceMirror = context.registerPictureMirror(&reference, vtm::CudaPictureRole::Reconstruction,
                                                             reference.descriptor);

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
  std::array<std::uint64_t, 64> sadResults{};

  auto invalidMirrorBatch = batch;
  invalidMirrorBatch.sourceMirror = 0;
  if (!throwsWithText([&context, &invalidMirrorBatch, &sadResults]() {
        (void) context.computeDistortionBatch(invalidMirrorBatch, sadResults.data());
      }, "CUDA picture mirror handle is invalid")
      || !context.isDistortionAccelerationAvailable() || context.distortionBatchFailureCount() != 0)
  {
    return false;
  }

  auto invalidMappingBatch = batch;
  invalidMappingBatch.source = nullptr;
  if (!throwsWithText([&context, &invalidMappingBatch, &sadResults]() {
        (void) context.computeDistortionBatch(invalidMappingBatch, sadResults.data());
      }, "CUDA distortion block descriptor is invalid")
      || !context.isDistortionAccelerationAvailable() || context.distortionBatchFailureCount() != 0)
  {
    return false;
  }

  vtm::CudaQpaTask task{};
  task.ticket = 1;
  task.ctuAddr = 0;
  task.filterArea = { 0, 0, 65, 65 };
  task.lumaArea = { 0, 0, 64, 64 };
  vtm::CudaQpaResult qpaResult{};
  if (!throwsWithText([&context, &task, &qpaResult]() {
        (void) context.computeQpaBatch(0, &task, 1, &qpaResult);
      }, "CUDA picture mirror handle is invalid")
      || !context.isQpaAccelerationAvailable() || context.qpaBatchFailureCount() != 0)
  {
    return false;
  }

  auto notEligibleBatch = batch;
  notEligibleBatch.width = 0;
  auto notEligibleTask = task;
  notEligibleTask.filterArea.width = 2;
  if (context.computeDistortionBatch(notEligibleBatch, sadResults.data())
      || context.computeQpaBatch(sourceMirror, &notEligibleTask, 1, &qpaResult)
      || !context.isDistortionAccelerationAvailable() || !context.isQpaAccelerationAvailable()
      || context.distortionBatchFailureCount() != 0 || context.qpaBatchFailureCount() != 0)
  {
    return false;
  }

  context.releaseAllPictureMirrors();
  context.shutdown();
  return true;
}

bool runSadFailureCase(const int device, const unsigned allocationFailureStep, const unsigned executionFailures,
                       const vtm::CudaBatchTestFailurePoint failurePoint = vtm::CudaBatchTestFailurePoint::None)
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
  results.fill(UINT64_C(0xfedcba9876543210));
  const auto untouched = results;
  context.injectDistortionFailuresForTesting(allocationFailureStep, executionFailures);
  context.injectDistortionFailurePointForTesting(failurePoint);
  const auto compute = [&context, &batch, &results]() {
    (void) context.computeDistortionBatch(batch, results.data());
  };
  const bool threwExpected = failurePoint == vtm::CudaBatchTestFailurePoint::DiagnosticConstruction
                               ? throwsBadAlloc(compute)
                               : throwsWithText(compute, "CUDA SAD execution failed after selection:");
  if (!threwExpected
      || results != untouched || context.isDistortionAccelerationAvailable()
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
  context.releaseAllPictureMirrors();
  if (context.pictureMirrorCount() != 0
      || context.pictureMirrorMemoryStats().total.currentDeviceBytes != 0
      || context.pictureMirrorMemoryStats().total.currentPinnedBytes != 0)
  {
    return false;
  }
  context.shutdown();
  return true;
}

bool runQpaFailureCase(const int device, const unsigned allocationFailureStep, const unsigned executionFailures,
                       const vtm::CudaBatchTestFailurePoint failurePoint = vtm::CudaBatchTestFailurePoint::None)
{
  vtm::CudaContext context;
  context.create(device);
  TestPicture picture(65, 65, 2, 10, 61);
  picture.fillSamples(61);
  const auto mirror = context.registerPictureMirror(&picture, vtm::CudaPictureRole::Original, picture.descriptor);
  vtm::CudaQpaTask task{};
  task.ticket = 17;
  task.ctuAddr = 3;
  task.filterArea = { 0, 0, 65, 65 };
  task.lumaArea = { 0, 0, 64, 64 };
  vtm::CudaQpaResult result{ UINT64_C(0x123456789abcdef0), UINT32_C(0x76543210),
                            UINT32_C(0x89abcdef), UINT64_C(0xfedcba9876543210),
                            INT64_C(-1234567890123456) };
  const vtm::CudaQpaResult untouched = result;
  context.injectQpaFailuresForTesting(allocationFailureStep, executionFailures);
  context.injectQpaFailurePointForTesting(failurePoint);
  const auto compute = [&context, mirror, &task, &result]() {
    (void) context.computeQpaBatch(mirror, &task, 1, &result);
  };
  const bool threwExpected = failurePoint == vtm::CudaBatchTestFailurePoint::DiagnosticConstruction
                               ? throwsBadAlloc(compute)
                               : throwsWithText(compute, "CUDA QPA execution failed after selection:");
  if (!threwExpected
      || std::memcmp(&result, &untouched, sizeof(result)) != 0 || context.isQpaAccelerationAvailable()
      || context.qpaBatchFailureCount() != 1
      || context.qpaBatchDispatchCount() != 0 || context.qpaTaskCount() != 0)
  {
    return false;
  }
  if (context.computeQpaBatch(mirror, &task, 1, &result) || context.qpaBatchFailureCount() != 1)
  {
    return false;
  }
  context.releaseAllPictureMirrors();
  if (context.pictureMirrorCount() != 0
      || context.pictureMirrorMemoryStats().total.currentDeviceBytes != 0
      || context.pictureMirrorMemoryStats().total.currentPinnedBytes != 0)
  {
    return false;
  }
  context.shutdown();
  return true;
}

bool runRecoveryIsolationCase(const int device, const bool failQpaFirst)
{
  vtm::CudaContext context;
  context.create(device);
  TestPicture source(96, 80, 2, 10, 67);
  TestPicture reference(96, 80, 2, 10, 79);
  source.fillSamples(67);
  reference.fillSamples(79);
  const auto sourceMirror = context.registerPictureMirror(&source, vtm::CudaPictureRole::Original,
                                                          source.descriptor);
  const auto referenceMirror = context.registerPictureMirror(&reference, vtm::CudaPictureRole::Reconstruction,
                                                             reference.descriptor);

  vtm::CudaQpaTask qpaTask{};
  qpaTask.ticket = 29;
  qpaTask.ctuAddr = 7;
  qpaTask.filterArea = { 0, 0, 65, 65 };
  qpaTask.lumaArea = { 0, 0, 64, 64 };
  vtm::CudaQpaResult qpaResult{};

  vtm::CudaDistortionBatchDesc sadBatch{};
  sadBatch.sourceMirror = sourceMirror;
  sadBatch.referenceMirror = referenceMirror;
  sadBatch.source = source.descriptor.planes[0].data;
  sadBatch.candidateGrid = { reference.descriptor.planes[0].data, 8, 8, 1, 1 };
  sadBatch.width = 8;
  sadBatch.height = 8;
  sadBatch.elementSize = 2;
  sadBatch.bitDepth = 10;
  sadBatch.metric = vtm::CudaDistortionMetric::Sad;
  std::array<std::uint64_t, 64> sadResults{};

  if (failQpaFirst)
  {
    context.injectQpaFailuresForTesting(0, 1);
    if (!throwsWithText([&context, sourceMirror, &qpaTask, &qpaResult]() {
          (void) context.computeQpaBatch(sourceMirror, &qpaTask, 1, &qpaResult);
        }, "CUDA QPA execution failed after selection:")
        || context.isQpaAccelerationAvailable()
        || !context.computeDistortionBatch(sadBatch, sadResults.data())
        || !context.isDistortionAccelerationAvailable())
    {
      return false;
    }
  }
  else
  {
    context.injectDistortionFailuresForTesting(0, 1);
    if (!throwsWithText([&context, &sadBatch, &sadResults]() {
          (void) context.computeDistortionBatch(sadBatch, sadResults.data());
        }, "CUDA SAD execution failed after selection:")
        || context.isDistortionAccelerationAvailable()
        || !context.computeQpaBatch(sourceMirror, &qpaTask, 1, &qpaResult)
        || !context.isQpaAccelerationAvailable())
    {
      return false;
    }
  }
  context.shutdown();
  return true;
}

bool runMirrorPlaneFailureCase(vtm::CudaContext &context, const unsigned allocationFailureStep,
                               const unsigned uploadFailures, const unsigned downloadFailures)
{
  TestPicture picture(63, 47, 4, 10, 31);
  int owner = 0;
  const auto mirror = context.registerPictureMirror(&owner, vtm::CudaPictureRole::Reconstruction,
                                                     picture.descriptor);
  const vtm::CudaMirrorMemoryStats before = context.pictureMirrorMemoryStats();
  if (allocationFailureStep != 0 || uploadFailures != 0)
  {
    context.injectMirrorPlaneFailuresForTesting(1, allocationFailureStep, uploadFailures, 0);
    if (!throws([&context, mirror]() { context.ensureDevicePlane(mirror, 1); }))
    {
      return false;
    }
    const vtm::CudaMirrorMemoryStats failed = context.pictureMirrorMemoryStats();
    if (allocationFailureStep != 0)
    {
      if (context.pictureMirrorAllocatedPlanes(mirror) != 0
          || failed.total.currentDeviceBytes != before.total.currentDeviceBytes
          || failed.total.currentPinnedBytes != before.total.currentPinnedBytes)
      {
        return false;
      }
    }
    else if (context.pictureMirrorAllocatedPlanes(mirror) != vtm::CUDA_PLANE_CB
             || context.pictureMirrorPlaneState(mirror, 1) != vtm::CudaMirrorState::HostValid
             || failed.byRoleAndPlane[1][1].uploadedBytes != before.byRoleAndPlane[1][1].uploadedBytes)
    {
      return false;
    }
  }

  context.ensureDevicePlane(mirror, 1);
  if (downloadFailures != 0)
  {
    context.markDevicePlaneModified(mirror, 1);
    for (std::size_t row = 0; row < picture.planes[1].fullHeight; ++row)
    {
      std::memset(picture.planes[1].storage.data() + row * picture.planes[1].stride, 0,
                  picture.planes[1].rowBytes);
    }
    context.injectMirrorPlaneFailuresForTesting(1, 0, 0, downloadFailures);
    if (!throws([&context, mirror]() { context.ensureHostPlane(mirror, 1); })
        || context.pictureMirrorPlaneState(mirror, 1) != vtm::CudaMirrorState::DeviceValid)
    {
      return false;
    }
    context.ensureHostPlane(mirror, 1);
    if (picture.planes[1].storage != picture.planes[1].expected)
    {
      return false;
    }
  }
  context.releasePictureMirror(mirror);
  const vtm::CudaMirrorMemoryStats after = context.pictureMirrorMemoryStats();
  return after.total.currentDeviceBytes == before.total.currentDeviceBytes
         && after.total.currentPinnedBytes == before.total.currentPinnedBytes;
}

bool runMultiPlaneTransactionFailureCase(vtm::CudaContext &context, const std::uint8_t failurePlane)
{
  TestPicture picture(71, 53, 2, 10, static_cast<std::uint8_t>(41 + failurePlane));
  int owner = 0;
  const auto mirror = context.registerPictureMirror(&owner, vtm::CudaPictureRole::Reconstruction,
                                                     picture.descriptor);
  const auto beforeAllocation = context.pictureMirrorMemoryStats();
  context.injectMirrorPlaneFailuresForTesting(failurePlane, 2, 0, 0);
  if (!throws([&context, mirror]() { context.ensureDevice(mirror); }))
  {
    return false;
  }
  const auto afterAllocationFailure = context.pictureMirrorMemoryStats();
  for (std::uint8_t index = 0; index < picture.descriptor.planeCount; ++index)
  {
    if (context.pictureMirrorPlaneState(mirror, index) != vtm::CudaMirrorState::HostValid)
    {
      return false;
    }
  }
  if (afterAllocationFailure.total.uploadedBytes != beforeAllocation.total.uploadedBytes)
  {
    return false;
  }
  context.ensureDevice(mirror);

  context.markHostModified(mirror);
  context.injectMirrorPlaneFailuresForTesting(failurePlane, 0, 1, 0);
  if (!throws([&context, mirror]() { context.ensureDevice(mirror); }))
  {
    return false;
  }
  vtm::CudaPlaneMask completedUpload = 0;
  for (std::uint8_t index = 0; index < failurePlane; ++index)
  {
    completedUpload |= static_cast<vtm::CudaPlaneMask>(vtm::CudaPlaneMask{ 1 } << index);
    if (context.pictureMirrorPlaneState(mirror, index) != vtm::CudaMirrorState::Synchronized)
    {
      return false;
    }
  }
  if (completedUpload != 0)
  {
    const auto consumer = context.devicePicturePlanes(mirror, completedUpload);
    if (consumer.planes[0].data == nullptr)
    {
      return false;
    }
  }
  for (std::uint8_t index = failurePlane; index < picture.descriptor.planeCount; ++index)
  {
    if (context.pictureMirrorPlaneState(mirror, index) != vtm::CudaMirrorState::HostValid)
    {
      return false;
    }
  }
  context.ensureDevice(mirror);

  picture.clearTransferredBytes();
  context.markDeviceModified(mirror);
  context.injectMirrorPlaneFailuresForTesting(failurePlane, 0, 0, 1);
  if (!throws([&context, mirror]() { context.ensureHost(mirror); }))
  {
    return false;
  }
  for (std::uint8_t index = 0; index < failurePlane; ++index)
  {
    if (context.pictureMirrorPlaneState(mirror, index) != vtm::CudaMirrorState::Synchronized
        || picture.planes[index].storage != picture.planes[index].expected)
    {
      return false;
    }
  }
  for (std::uint8_t index = failurePlane; index < picture.descriptor.planeCount; ++index)
  {
    if (context.pictureMirrorPlaneState(mirror, index) != vtm::CudaMirrorState::DeviceValid)
    {
      return false;
    }
  }
  context.ensureHost(mirror);
  const bool matches = picture.matchesExpected();
  context.releasePictureMirror(mirror);
  return matches;
}

bool runSameLayoutPartialReleaseCase(const int device, const bool retainDevice)
{
  vtm::CudaContext context;
  context.create(device);
  TestPicture initial(37, 29, 2, 10, 53);
  TestPicture rebound(37, 29, 2, 10, 97);
  int owner = 0;
  const auto mirror = context.registerPictureMirror(&owner, vtm::CudaPictureRole::Reconstruction,
                                                     initial.descriptor);
  context.ensureDevicePlane(mirror, 0);
  const std::uint64_t expectedDevice = initial.deviceBytes(0);
  const std::uint64_t expectedPinned = initial.pinnedBytes(0);
  const vtm::CudaMirrorMemoryStats allocated = context.pictureMirrorMemoryStats();
  if (allocated.total.currentDeviceBytes != expectedDevice
      || allocated.total.currentPinnedBytes != expectedPinned
      || allocated.total.peakDeviceBytes != expectedDevice
      || allocated.total.peakPinnedBytes != expectedPinned
      || allocated.budgetBytes != vtm::CUDA_DEFAULT_MIRROR_MEMORY_BUDGET_BYTES
      || allocated.budgetRejections != 0)
  {
    return false;
  }

  if (retainDevice)
  {
    context.injectReleaseFailuresForTesting(1, 1);
  }
  else
  {
    context.injectPinnedReleaseFailuresForTesting(1);
  }
  if (!throws([&context, mirror]() { context.releasePictureMirror(mirror); }))
  {
    return false;
  }
  const vtm::CudaMirrorMemoryStats partial = context.pictureMirrorMemoryStats();
  if (context.pictureMirrorCount() != 1
      || partial.total.currentDeviceBytes != (retainDevice ? expectedDevice : 0)
      || partial.total.currentPinnedBytes != (retainDevice ? 0 : expectedPinned))
  {
    return false;
  }

  context.rebindHostPicture(mirror, rebound.descriptor);
  const vtm::CudaMirrorMemoryStats restored = context.pictureMirrorMemoryStats();
  if (restored.total.currentDeviceBytes != expectedDevice
      || restored.total.currentPinnedBytes != expectedPinned
      || restored.total.peakDeviceBytes != expectedDevice
      || restored.total.peakPinnedBytes != expectedPinned
      || restored.budgetBytes != allocated.budgetBytes || restored.budgetRejections != 0
      || context.pictureMirrorAllocatedPlanes(mirror) != vtm::CUDA_PLANE_Y)
  {
    return false;
  }

  context.ensureDevicePlane(mirror, 0);
  for (std::size_t row = 0; row < rebound.planes[0].fullHeight; ++row)
  {
    std::memset(rebound.planes[0].storage.data() + row * rebound.planes[0].stride, 0,
                rebound.planes[0].rowBytes);
  }
  context.markDevicePlaneModified(mirror, 0);
  context.ensureHostPlane(mirror, 0);
  if (rebound.planes[0].storage != rebound.planes[0].expected)
  {
    return false;
  }
  context.releasePictureMirror(mirror);
  const vtm::CudaMirrorMemoryStats released = context.pictureMirrorMemoryStats();
  const bool releasedCleanly = context.pictureMirrorCount() == 0
                               && released.total.currentDeviceBytes == 0
                               && released.total.currentPinnedBytes == 0
                               && released.total.peakDeviceBytes == expectedDevice
                               && released.total.peakPinnedBytes == expectedPinned
                               && released.budgetRejections == 0;
  context.shutdown();
  return releasedCleanly;
}

struct AlfReferenceAccess : AdaptiveLoopFilter
{
  static void copyFixedSet(const unsigned set, AlfCoeff *coefficients)
  {
    for (unsigned classIdx = 0; classIdx < MAX_NUM_ALF_CLASSES; ++classIdx)
    {
      const int fixed = m_classToFilterMapping[set % ALF_NUM_FIXED_FILTER_SETS][classIdx];
      std::copy_n(m_fixedFilterSetCoeff[fixed], MAX_NUM_ALF_LUMA_COEFF,
                  coefficients + classIdx * MAX_NUM_ALF_LUMA_COEFF);
    }
  }
};

vtm::CudaAlfLumaFrame makeAlfFrame(const std::uint32_t width, const std::uint32_t height,
                                   const std::uint8_t bitDepth)
{
  constexpr std::uint32_t ctuSize = 64;
  vtm::CudaAlfLumaFrame frame{};
  frame.width = width;
  frame.height = height;
  frame.ctuWidth = frame.ctuHeight = ctuSize;
  frame.ctusInWidth = (width + ctuSize - 1) / ctuSize;
  frame.ctusInHeight = (height + ctuSize - 1) / ctuSize;
  frame.minSample = 0;
  frame.maxSample = (1 << bitDepth) - 1;
  frame.vbCtuHeight = ctuSize;
  frame.vbPos = ctuSize - 4;
  frame.bitDepth = bitDepth;
  frame.elementSize = sizeof(Pel);
  return frame;
}

std::vector<vtm::CudaAlfCtuParam> makeAlfCtus(const vtm::CudaAlfLumaFrame &frame,
                                              const bool varied)
{
  const std::uint32_t count = frame.ctusInWidth * frame.ctusInHeight;
  std::vector<vtm::CudaAlfCtuParam> ctus(count);
  std::array<AlfCoeff, MAX_NUM_ALF_CLASSES * MAX_NUM_ALF_LUMA_COEFF> fixed{};
  for (std::uint32_t index = 0; index < count; ++index)
  {
    vtm::CudaAlfCtuParam &ctu = ctus[index];
    ctu.x = (index % frame.ctusInWidth) * frame.ctuWidth;
    ctu.y = (index / frame.ctusInWidth) * frame.ctuHeight;
    ctu.width = std::min(frame.ctuWidth, frame.width - ctu.x);
    ctu.height = std::min(frame.ctuHeight, frame.height - ctu.y);
    ctu.enabled = varied ? (index % 7 != 0) : 1;
    if (varied && (index & 1) == 0)
    {
      AlfReferenceAccess::copyFixedSet(index, fixed.data());
    }
    for (std::uint32_t classIdx = 0; classIdx < vtm::CUDA_ALF_CLASSES; ++classIdx)
    {
      for (std::uint32_t tap = 0; tap < vtm::CUDA_ALF_COEFFICIENTS; ++tap)
      {
        const std::uint32_t offset = classIdx * vtm::CUDA_ALF_COEFFICIENTS + tap;
        const int apsValue = tap == 12 ? 128 : int((index * 11 + classIdx * 7 + tap * 5) % 31) - 15;
        ctu.coefficients[offset] = tap == 12 ? std::int16_t(128)
          : (varied && (index & 1) == 0 ? fixed[offset] : static_cast<std::int16_t>(apsValue));
        constexpr int clips[6] = { 0, 1, 3, 7, 31, 1023 };
        ctu.clipValues[offset] = tap == 12 ? (1 << frame.bitDepth)
          : (varied ? std::min(frame.maxSample, clips[(index + classIdx + tap) % 6]) : frame.maxSample);
      }
    }
  }
  return ctus;
}

void fillAlfPattern(TestPicture &picture, const std::uint8_t bitDepth)
{
  const vtm::CudaHostPlaneDesc &host = picture.descriptor.planes[0];
  const std::uint32_t maximum = (1u << bitDepth) - 1;
  for (std::uint32_t y = 0; y < host.height; ++y)
  {
    auto *row = static_cast<std::uint8_t *>(host.data) + static_cast<std::size_t>(y) * host.strideBytes;
    for (std::uint32_t x = 0; x < host.width; ++x)
    {
      const std::uint32_t region = (x / 256) & 3;
      std::uint32_t directional = 0;
      if (region == 0) directional = (y * 73) ^ ((x >> 4) * 3);
      if (region == 1) directional = (x * 79) ^ ((y >> 4) * 5);
      if (region == 2) directional = ((x + y) * 83) ^ ((x >> 3) * 7);
      if (region == 3) directional = ((x + maximum - (y & maximum)) * 89) ^ ((y >> 3) * 11);
      std::uint32_t value = (directional + ((x * 17 + y * 29) & 7)) & maximum;
      if ((x + y * 3) % 97 == 0) value = 0;
      if ((x * 5 + y) % 101 == 0) value = maximum;
      writeSample(row + static_cast<std::size_t>(x) * host.elementSize, host.elementSize,
                  static_cast<std::int32_t>(value));
    }
  }
  picture.planes[0].expected = picture.planes[0].storage;
}

bool runAlfLumaCase(vtm::CudaContext &context, const std::uint8_t bitDepth,
                    double *cpuMilliseconds = nullptr, double *gpuMilliseconds = nullptr,
                    const bool collectDiagnostics = true)
{
  constexpr std::uint32_t width = 1924;
  constexpr std::uint32_t height = 1084;
  TestPicture picture(width, height, sizeof(Pel), bitDepth, bitDepth, 1, 0, 0);
  fillAlfPattern(picture, bitDepth);
  const vtm::CudaHostPlaneDesc &host = picture.descriptor.planes[0];
  auto sample = [&host](const int x, const int y) {
    return static_cast<Pel>(readSample(static_cast<const std::uint8_t *>(host.data)
      + static_cast<std::size_t>(y) * host.strideBytes + static_cast<std::size_t>(x) * host.elementSize,
      host.elementSize));
  };

  constexpr int margin = 4;
  const int extendedStride = int(width) + 2 * margin;
  std::vector<Pel> extended((height + 2 * margin) * extendedStride);
  Pel *active = extended.data() + margin * extendedStride + margin;
  for (int y = -margin; y < int(height) + margin; ++y)
    for (int x = -margin; x < int(width) + margin; ++x)
      active[y * extendedStride + x] = sample(std::max(0, std::min(x, int(width) - 1)),
                                               std::max(0, std::min(y, int(height) - 1)));

  std::vector<AlfClassifier> cpuFull(static_cast<std::size_t>(width) * height);
  std::vector<AlfClassifier *> cpuRows(height);
  for (std::uint32_t y = 0; y < height; ++y) cpuRows[y] = cpuFull.data() + static_cast<std::size_t>(y) * width;
  int lapData[NUM_DIRECTIONS][AdaptiveLoopFilter::m_CLASSIFICATION_BLK_SIZE + 5]
             [AdaptiveLoopFilter::m_CLASSIFICATION_BLK_SIZE + 5]{};
  int *lapRows[NUM_DIRECTIONS][AdaptiveLoopFilter::m_CLASSIFICATION_BLK_SIZE + 5]{};
  int **lap[NUM_DIRECTIONS]{};
  for (int direction = 0; direction < NUM_DIRECTIONS; ++direction)
  {
    for (int y = 0; y < AdaptiveLoopFilter::m_CLASSIFICATION_BLK_SIZE + 5; ++y)
      lapRows[direction][y] = lapData[direction][y];
    lap[direction] = lapRows[direction];
  }
  const CPelBuf source(active, extendedStride, width, height);
  const vtm::CudaAlfLumaFrame frame = makeAlfFrame(width, height, bitDepth);
  const auto cpuStart = std::chrono::steady_clock::now();
  for (std::uint32_t y = 0; y < height; y += AdaptiveLoopFilter::m_CLASSIFICATION_BLK_SIZE)
  {
    for (std::uint32_t x = 0; x < width; x += AdaptiveLoopFilter::m_CLASSIFICATION_BLK_SIZE)
    {
      const Area block(x, y, std::min<std::uint32_t>(AdaptiveLoopFilter::m_CLASSIFICATION_BLK_SIZE, width - x),
                       std::min<std::uint32_t>(AdaptiveLoopFilter::m_CLASSIFICATION_BLK_SIZE, height - y));
      AdaptiveLoopFilter::deriveClassificationBlk(cpuRows.data(), lap, source, block, block,
                                                   bitDepth + 4, frame.vbCtuHeight, frame.vbPos);
    }
  }

  std::vector<vtm::CudaAlfCtuParam> ctus = makeAlfCtus(frame, true);
  std::vector<Pel> expected(static_cast<std::size_t>(width) * height);
  for (std::uint32_t y = 0; y < height; ++y)
    std::copy_n(active + static_cast<std::size_t>(y) * extendedStride, width,
                expected.data() + static_cast<std::size_t>(y) * width);
  PelUnitBuf dst(ChromaFormat::_400, PelBuf(expected.data(), width, width, height));
  const CPelUnitBuf src(ChromaFormat::_400, source);
  XuPool pool;
  CodingStructure cs(pool);
  std::array<Pel, MAX_NUM_ALF_CLASSES * MAX_NUM_ALF_LUMA_COEFF> clips{};
  const ClpRng range{ 0, (1 << bitDepth) - 1, bitDepth, 0 };
  bool sawDisabled = false;
  bool sawPartial = false;
  for (const vtm::CudaAlfCtuParam &ctu : ctus)
  {
    sawPartial |= ctu.width != frame.ctuWidth || ctu.height != frame.ctuHeight;
    if (!ctu.enabled)
    {
      sawDisabled = true;
      continue;
    }
    for (std::size_t i = 0; i < clips.size(); ++i) clips[i] = static_cast<Pel>(ctu.clipValues[i]);
    const Area block(ctu.x, ctu.y, ctu.width, ctu.height);
    AdaptiveLoopFilter::filterBlk<ALF_FILTER_7>(cpuRows.data(), dst, src, block, block, COMPONENT_Y,
      reinterpret_cast<const AlfCoeff *>(ctu.coefficients), clips.data(), range, cs,
      frame.vbCtuHeight, frame.vbPos);
  }
  const auto cpuEnd = std::chrono::steady_clock::now();
  if (!sawDisabled || !sawPartial) return false;

  std::vector<vtm::CudaAlfClassifier> gpuClassifiers;
  if (collectDiagnostics)
    gpuClassifiers.resize(static_cast<std::size_t>(width >> 2) * (height >> 2));
  int owner = 0;
  const auto mirror = context.registerPictureMirror(&owner, vtm::CudaPictureRole::Reconstruction,
                                                     picture.descriptor);
  const vtm::AlfAccelerationStats statsBefore = context.alfStats();
  const auto gpuStart = std::chrono::steady_clock::now();
  if (context.filterAlfLumaFrame(mirror, frame, ctus.data(), static_cast<std::uint32_t>(ctus.size()),
                                 collectDiagnostics ? gpuClassifiers.data() : nullptr)
      != vtm::CudaAlfDispatchResult::Executed)
  {
    std::cerr << "ALF dispatch unexpectedly ineligible for " << unsigned(bitDepth) << "-bit case\n";
    return false;
  }
  const auto gpuEnd = std::chrono::steady_clock::now();
  const vtm::AlfAccelerationStats statsAfter = context.alfStats();
  const std::uint64_t runtimeSynchronizations = statsAfter.runtimeSynchronizations
                                                - statsBefore.runtimeSynchronizations;
  const std::uint64_t integrationSynchronizations = statsAfter.integrationSynchronizations
                                                    - statsBefore.integrationSynchronizations;
  if (statsAfter.dispatches != statsBefore.dispatches + 1
      || statsAfter.parameterUploadBytes <= statsBefore.parameterUploadBytes
      || statsAfter.commitBytes <= statsBefore.commitBytes
      || statsAfter.mirrorUploadBytes <= statsBefore.mirrorUploadBytes
      || statsAfter.mirrorDownloadBytes <= statsBefore.mirrorDownloadBytes
      || runtimeSynchronizations < (collectDiagnostics ? 3u : 2u)
      || integrationSynchronizations != runtimeSynchronizations + 2
      || statsAfter.uploadSubmissionNanoseconds <= statsBefore.uploadSubmissionNanoseconds
      || statsAfter.runtimeNanoseconds <= statsBefore.runtimeNanoseconds
      || statsAfter.downloadNanoseconds <= statsBefore.downloadNanoseconds
      || statsAfter.integrationNanoseconds - statsBefore.integrationNanoseconds
           < statsAfter.runtimeNanoseconds - statsBefore.runtimeNanoseconds)
  {
    std::cerr << "ALF telemetry mismatch: runtime/integration sync delta " << runtimeSynchronizations << "/"
              << integrationSynchronizations << ", mirror bytes "
              << (statsAfter.mirrorUploadBytes - statsBefore.mirrorUploadBytes) << "/"
              << (statsAfter.mirrorDownloadBytes - statsBefore.mirrorDownloadBytes) << '\n';
    return false;
  }
  if (collectDiagnostics)
  {
    unsigned transposeMask = 0;
    for (std::uint32_t y = 0; y < height; y += 4)
    {
      for (std::uint32_t x = 0; x < width; x += 4)
      {
        const std::uint32_t compact = (y >> 2) * (width >> 2) + (x >> 2);
        transposeMask |= 1u << cpuRows[y][x].transposeIdx;
        if (gpuClassifiers[compact].classIdx != cpuRows[y][x].classIdx
            || gpuClassifiers[compact].transposeIdx != cpuRows[y][x].transposeIdx)
        {
          std::cerr << "ALF classifier mismatch at " << x << "," << y << '\n';
          return false;
        }
      }
    }
    if (transposeMask != 0xf) return false;
  }
  for (std::uint32_t y = 0; y < height; ++y)
    for (std::uint32_t x = 0; x < width; ++x)
      if (sample(x, y) != expected[static_cast<std::size_t>(y) * width + x])
      {
        std::cerr << "ALF sample mismatch at " << x << "," << y << '\n';
        return false;
      }
  context.releasePictureMirror(mirror);
  if (cpuMilliseconds != nullptr)
    *cpuMilliseconds = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
  if (gpuMilliseconds != nullptr)
    *gpuMilliseconds = std::chrono::duration<double, std::milli>(gpuEnd - gpuStart).count();
  return true;
}

bool benchmarkAlfFrame(vtm::CudaContext &context)
{
  constexpr int runs = 5;
  std::array<double, runs> cpu{};
  std::array<double, runs> gpu{};
  // Warm allocations, kernels and driver state before recording exactly five diagnostic-free samples.
  if (!runAlfLumaCase(context, 10, nullptr, nullptr, false)) return false;
  const vtm::AlfAccelerationStats before = context.alfStats();
  for (int run = 0; run < runs; ++run)
  {
    if (!runAlfLumaCase(context, 10, &cpu[run], &gpu[run], false)) return false;
  }
  std::sort(cpu.begin(), cpu.end());
  std::sort(gpu.begin(), gpu.end());
  const vtm::AlfAccelerationStats after = context.alfStats();
  const double cpuMedian = cpu[runs / 2];
  const double gpuMedian = gpu[runs / 2];
  if (after.diagnosticDownloadBytes != before.diagnosticDownloadBytes) return false;
  std::cout << "ALF 1924x1084 Pel" << (sizeof(Pel) * 8)
            << "/10-bit, warm-up + 5 measured diagnostic-free median: CPU "
            << cpuMedian << " ms, CUDA integration " << gpuMedian << " ms, speedup "
            << (gpuMedian > 0.0 ? cpuMedian / gpuMedian : 0.0) << "x; telemetry delta params/diagnostic/commit "
            << (after.parameterUploadBytes - before.parameterUploadBytes) << "/"
            << (after.diagnosticDownloadBytes - before.diagnosticDownloadBytes) << "/"
            << (after.commitBytes - before.commitBytes) << " bytes, mirror upload/download "
            << (after.mirrorUploadBytes - before.mirrorUploadBytes) << "/"
            << (after.mirrorDownloadBytes - before.mirrorDownloadBytes) << " bytes, syncs runtime/integration "
            << (after.runtimeSynchronizations - before.runtimeSynchronizations) << "/"
            << (after.integrationSynchronizations - before.integrationSynchronizations)
            << ", time upload-submit/runtime/download/integration "
            << double(after.uploadSubmissionNanoseconds - before.uploadSubmissionNanoseconds) / 1000000.0 << "/"
            << double(after.runtimeNanoseconds - before.runtimeNanoseconds) / 1000000.0 << "/"
            << double(after.downloadNanoseconds - before.downloadNanoseconds) / 1000000.0 << "/"
            << double(after.integrationNanoseconds - before.integrationNanoseconds) / 1000000.0 << " ms. "
            << (gpuMedian < cpuMedian ? "Wall-time gate passed for this microbenchmark."
                                      : "Wall-time gate did not pass; GPUExperimentalALF remains off by default.")
            << '\n';
  return true;
}

std::vector<vtm::CudaDbfLumaTask> makeDbfTasks(const std::uint8_t bitDepth)
{
  std::vector<vtm::CudaDbfLumaTask> tasks;
  const auto add = [&](const std::uint32_t x, const std::uint32_t y, const std::uint8_t direction,
                       const std::uint8_t p, const std::uint8_t q, const std::uint8_t flags,
                       const int tc, const int beta) {
    const int bitDepthScale = 1 << (bitDepth - 8);
    tasks.push_back({ x, y, tc * bitDepthScale / 4, beta * bitDepthScale / 4,
                      0, (1 << bitDepth) - 1, direction, p, q, flags });
  };
  // Several edges share a lane and therefore prove serial edge ordering inside a CUDA lane.
  for (std::uint32_t y : { 0u, 4u, 64u, 1076u, 1080u })
  {
    add(8, y, 0, 1, 1, 0, 8, 80);
    add(16, y, 0, 3, 3, (y == 4 ? vtm::CUDA_DBF_PART_P_NO_FILTER : 0), 12, 96);
    add(28, y, 0, 5, 7, vtm::CUDA_DBF_SIDE_P_LARGE | vtm::CUDA_DBF_SIDE_Q_LARGE
                             | (y == 64 ? vtm::CUDA_DBF_PART_Q_NO_FILTER : 0), 16, 128);
    add(44, y, 0, 7, 5, vtm::CUDA_DBF_SIDE_P_LARGE | vtm::CUDA_DBF_SIDE_Q_LARGE, 20, 160);
    add(1916, y, 0, 3, 3, 0, 10, 112);
  }
  for (std::uint32_t x : { 0u, 4u, 64u, 1916u, 1920u })
  {
    add(x, 8, 1, 1, 1, 0, 8, 80);
    add(x, 16, 1, 3, 3, (x == 4 ? vtm::CUDA_DBF_PART_Q_NO_FILTER : 0), 12, 96);
    add(x, 28, 1, 7, 5, vtm::CUDA_DBF_SIDE_P_LARGE | vtm::CUDA_DBF_SIDE_Q_LARGE
                             | (x == 64 ? vtm::CUDA_DBF_PART_P_NO_FILTER : 0), 16, 128);
    add(x, 44, 1, 5, 7, vtm::CUDA_DBF_SIDE_P_LARGE | vtm::CUDA_DBF_SIDE_Q_LARGE, 20, 160);
    add(x, 1076, 1, 3, 3, 0, 10, 112);
  }
  return tasks;
}

std::vector<vtm::CudaDbfLumaTask> makeDenseDbfTasks(const std::uint8_t bitDepth)
{
  std::vector<vtm::CudaDbfLumaTask> tasks;
  tasks.reserve(140000);
  auto add = [&](const std::uint32_t x, const std::uint32_t y, const std::uint8_t direction,
                 const std::uint32_t ordinal) {
    const bool longFilter = ordinal % 5 == 0;
    const std::uint8_t flags = longFilter
      ? std::uint8_t(vtm::CUDA_DBF_SIDE_P_LARGE | vtm::CUDA_DBF_SIDE_Q_LARGE) : 0;
    const int bitDepthScale = 1 << (bitDepth - 8);
    tasks.push_back({ x, y, (12 + int(ordinal & 7)) * bitDepthScale / 4,
                      (96 + int(ordinal & 31)) * bitDepthScale / 4, 0,
                      (1 << bitDepth) - 1, direction,
                      std::uint8_t(longFilter ? 7 : 3), std::uint8_t(longFilter ? 5 : 3), flags });
  };
  std::uint32_t ordinal = 0;
  for (std::uint32_t y = 0; y + 4 <= 1084; y += 4)
    for (std::uint32_t x = 8; x <= 1920; x += 8) add(x, y, 0, ordinal++);
  for (std::uint32_t x = 0; x + 4 <= 1924; x += 4)
    for (std::uint32_t y = 8; y <= 1080; y += 8) add(x, y, 1, ordinal++);
  return tasks;
}

bool runDbfCollectorSerializationCase(const std::uint8_t bitDepth)
{
  const ClpRng range{ 0, (1 << bitDepth) - 1, bitDepth, 0 };
  std::vector<vtm::CudaDbfLumaTask> collected;
  collected.push_back(DeblockingFilter::makeCudaLumaTask(
    8, 12, DeblockingFilter::EdgeDir::VER, 1, 1, 0, 0, range, false, false, false, false));
  collected.push_back(DeblockingFilter::makeCudaLumaTask(
    16, 20, DeblockingFilter::EdgeDir::HOR, 3, 3, 7, 64, range, false, false, false, false));
  collected.push_back(DeblockingFilter::makeCudaLumaTask(
    24, 28, DeblockingFilter::EdgeDir::VER, 5, 7,
    vtm::cudaDbfMaximumTc(bitDepth), vtm::cudaDbfMaximumBeta(bitDepth), range,
    true, true, false, false));
  collected.push_back(DeblockingFilter::makeCudaLumaTask(
    32, 36, DeblockingFilter::EdgeDir::HOR, 7, 5, 11, 80, range,
    true, true, true, true));
  collected.push_back(DeblockingFilter::makeCudaLumaTask(
    40, 128, DeblockingFilter::EdgeDir::HOR, 7, 7, 8, 56, range,
    false, true, false, false));
  if (collected.size() != 5) return false;
  const auto validCommon = [&](const vtm::CudaDbfLumaTask &task) {
    return task.minSample == 0 && task.maxSample == (1 << bitDepth) - 1;
  };
  return validCommon(collected[0]) && collected[0].x == 8 && collected[0].y == 12
         && collected[0].direction == 0 && collected[0].maxFilterLenP == 1
         && collected[0].maxFilterLenQ == 1 && collected[0].tc == 0 && collected[0].beta == 0
         && collected[0].flags == 0
         && validCommon(collected[1]) && collected[1].x == 16 && collected[1].y == 20
         && collected[1].direction == 1 && collected[1].maxFilterLenP == 3
         && collected[1].maxFilterLenQ == 3 && collected[1].tc == 7 && collected[1].beta == 64
         && collected[1].flags == 0
         && validCommon(collected[2]) && collected[2].x == 24 && collected[2].y == 28
         && collected[2].direction == 0 && collected[2].maxFilterLenP == 5
         && collected[2].maxFilterLenQ == 7
         && collected[2].tc == vtm::cudaDbfMaximumTc(bitDepth)
         && collected[2].beta == vtm::cudaDbfMaximumBeta(bitDepth)
         && collected[2].flags == (vtm::CUDA_DBF_SIDE_P_LARGE | vtm::CUDA_DBF_SIDE_Q_LARGE)
         && validCommon(collected[3]) && collected[3].x == 32 && collected[3].y == 36
         && collected[3].direction == 1 && collected[3].maxFilterLenP == 7
         && collected[3].maxFilterLenQ == 5
         && collected[3].flags == (vtm::CUDA_DBF_SIDE_P_LARGE | vtm::CUDA_DBF_SIDE_Q_LARGE
                                    | vtm::CUDA_DBF_PART_P_NO_FILTER
                                    | vtm::CUDA_DBF_PART_Q_NO_FILTER)
         && validCommon(collected[4]) && collected[4].x == 40 && collected[4].y == 128
         && collected[4].direction == 1 && collected[4].maxFilterLenP == 3
         && collected[4].maxFilterLenQ == 7 && collected[4].tc == 8 && collected[4].beta == 56
         && collected[4].flags == vtm::CUDA_DBF_SIDE_Q_LARGE;
}

void fillDbfPattern(TestPicture &picture, const std::uint8_t bitDepth)
{
  auto &plane = picture.planes[0];
  const auto &host = picture.descriptor.planes[0];
  const std::uint32_t maximum = (1u << bitDepth) - 1;
  for (std::size_t y = 0; y < plane.fullHeight; ++y)
    for (std::size_t x = 0; x < plane.rowBytes / host.elementSize; ++x)
    {
      // Smooth ramps, shallow steps, impulses, and clipping extrema exercise weak/strong/long decisions.
      std::uint32_t value = static_cast<std::uint32_t>((x * 3 + y * 5 + ((x / 11) & 3) * 7) & maximum);
      if ((x + 3 * y) % 257 == 0) value = 0;
      if ((5 * x + y) % 263 == 0) value = maximum;
      writeSample(plane.storage.data() + y * plane.stride + x * host.elementSize,
                  host.elementSize, static_cast<std::int32_t>(value));
    }
  plane.expected = plane.storage;
}

bool runDbfLumaCase(vtm::CudaContext &context, const std::uint8_t bitDepth,
                    double *cpuMilliseconds = nullptr, double *gpuMilliseconds = nullptr,
                    const bool dense = false)
{
  constexpr std::uint32_t width = 1924, height = 1084;
  TestPicture picture(width, height, sizeof(Pel), bitDepth, 73, 1, 0, 0, 8, 8);
  fillDbfPattern(picture, bitDepth);
  const auto tasks = dense ? makeDenseDbfTasks(bitDepth) : makeDbfTasks(bitDepth);
  auto expected = picture.planes[0].storage;
  const auto *storageBase = picture.planes[0].storage.data();
  const std::size_t activeOffset = static_cast<const std::uint8_t *>(picture.descriptor.planes[0].data) - storageBase;
  Pel *expectedActive = reinterpret_cast<Pel *>(expected.data() + activeOffset);
  const ptrdiff_t stride = picture.descriptor.planes[0].strideBytes / sizeof(Pel);
  const auto cpuStart = std::chrono::steady_clock::now();
  DeblockingFilter::filterLumaTasksCpu(expectedActive, stride, tasks.data(),
                                       static_cast<std::uint32_t>(tasks.size()));
  const auto cpuEnd = std::chrono::steady_clock::now();
  if (expected == picture.planes[0].storage)
  {
    std::cerr << "DBF scalar reference did not exercise any filtering branch\n";
    return false;
  }

  int owner = 0;
  const auto mirror = context.registerPictureMirror(&owner, vtm::CudaPictureRole::Reconstruction,
                                                     picture.descriptor);
  const vtm::CudaDbfFrame frame{ width, height, bitDepth, sizeof(Pel), { 0, 0 } };
  const vtm::DbfAccelerationStats before = context.dbfStats();
  const auto gpuStart = std::chrono::steady_clock::now();
  const auto result = context.filterDbfLumaFrame(mirror, frame, tasks.data(),
                                                  static_cast<std::uint32_t>(tasks.size()));
  const auto gpuEnd = std::chrono::steady_clock::now();
  const vtm::DbfAccelerationStats after = context.dbfStats();
  const std::uint64_t runtimeSynchronizations =
    after.runtimeSynchronizations - before.runtimeSynchronizations;
  const std::uint64_t mirrorSynchronizations =
    after.mirrorSynchronizations - before.mirrorSynchronizations;
  const std::uint64_t integrationSynchronizations =
    after.integrationSynchronizations - before.integrationSynchronizations;
  if (result != vtm::CudaDbfDispatchResult::Executed || picture.planes[0].storage != expected
      || after.dispatches != before.dispatches + 1 || after.tasks != before.tasks + tasks.size()
      || after.parameterUploadBytes <= before.parameterUploadBytes
      || after.commitBytes != before.commitBytes + std::uint64_t(width) * height * sizeof(Pel)
      || after.mirrorUploadBytes <= before.mirrorUploadBytes
      || after.mirrorDownloadBytes <= before.mirrorDownloadBytes
      || runtimeSynchronizations < 3 || mirrorSynchronizations < 1
      || integrationSynchronizations != runtimeSynchronizations + mirrorSynchronizations + 1
      || after.scratchBytes == 0 || after.retiredScratchBytes != 0
      || after.peakScratchBytes < after.scratchBytes)
  {
    if (picture.planes[0].storage != expected)
    {
      for (std::size_t i = 0; i < expected.size(); ++i)
        if (picture.planes[0].storage[i] != expected[i])
        {
          std::cerr << "DBF first byte mismatch at storage offset " << i << " for " << unsigned(bitDepth)
                    << "-bit Pel" << sizeof(Pel) * 8
                    << ": actual=" << unsigned(picture.planes[0].storage[i])
                    << ", expected=" << unsigned(expected[i]) << '\n';
          break;
        }
    }
    return false;
  }
  context.releasePictureMirror(mirror);
  if (cpuMilliseconds) *cpuMilliseconds = std::chrono::duration<double, std::milli>(cpuEnd-cpuStart).count();
  if (gpuMilliseconds) *gpuMilliseconds = std::chrono::duration<double, std::milli>(gpuEnd-gpuStart).count();
  return true;
}

bool benchmarkDbfFrame(vtm::CudaContext &context)
{
  constexpr int runs = 5;
  std::array<double, runs> cpu{}, gpu{};
  if (!runDbfLumaCase(context, 10, nullptr, nullptr, true)) return false; // warm-up
  for (int i = 0; i < runs; ++i)
    if (!runDbfLumaCase(context, 10, &cpu[i], &gpu[i], true)) return false;
  std::sort(cpu.begin(), cpu.end()); std::sort(gpu.begin(), gpu.end());
  std::cout << "DBF 1924x1084 Pel" << sizeof(Pel)*8
            << "/10-bit, warm-up + 5 median: CPU descriptor apply " << cpu[runs/2]
            << " ms, CUDA integration " << gpu[runs/2] << " ms, speedup "
            << (gpu[runs/2] > 0.0 ? cpu[runs/2]/gpu[runs/2] : 0.0) << "x. "
            << (gpu[runs/2] < cpu[runs/2] ? "Wall-time gate passed."
                                             : "Wall-time gate did not pass; GPUExperimentalDBF remains off by default.")
            << '\n';
  return true;
}

bool runDbfGeometryCase(vtm::CudaContext &context, const std::uint8_t bitDepth)
{
  TestPicture picture(1924, 1084, sizeof(Pel), bitDepth, 77, 1, 0, 0, 8, 8);
  fillDbfPattern(picture, bitDepth);
  int owner = 0;
  const auto mirror = context.registerPictureMirror(&owner, vtm::CudaPictureRole::Reconstruction,
                                                     picture.descriptor);
  const vtm::CudaDbfFrame frame{ 1924, 1084, bitDepth, sizeof(Pel), {0,0} };
  auto task = makeDbfTasks(bitDepth).front();
  const auto rejected = [&](const vtm::CudaDbfFrame &f, const vtm::CudaDbfLumaTask &t) {
    return context.filterDbfLumaFrame(mirror, f, &t, 1) == vtm::CudaDbfDispatchResult::NotEligible;
  };
  auto alteredFrame = frame; alteredFrame.width = 1919;
  if (!rejected(alteredFrame, task)) return false;
  auto alteredTask = task; alteredTask.direction = 2;
  if (!rejected(frame, alteredTask)) return false;
  alteredTask = task; alteredTask.x = 2;
  if (!rejected(frame, alteredTask)) return false;
  alteredTask = task; alteredTask.maxFilterLenP = 4;
  if (!rejected(frame, alteredTask)) return false;
  alteredTask = task; alteredTask.flags = 0x80;
  if (!rejected(frame, alteredTask)) return false;
  alteredTask = task; alteredTask.tc = std::numeric_limits<std::int32_t>::max();
  if (!rejected(frame, alteredTask)) return false;
  alteredTask = task; alteredTask.beta = std::numeric_limits<std::int32_t>::max();
  if (!rejected(frame, alteredTask)) return false;
  alteredTask = task; alteredTask.tc = vtm::cudaDbfMaximumTc(frame.bitDepth) + 1;
  if (!rejected(frame, alteredTask)) return false;
  alteredTask = task; alteredTask.beta = vtm::cudaDbfMaximumBeta(frame.bitDepth) + 1;
  if (!rejected(frame, alteredTask)) return false;
  alteredTask = task; alteredTask.maxFilterLenP = 5;
  if (!rejected(frame, alteredTask)) return false;
  alteredTask = task; alteredTask.maxFilterLenQ = 7;
  if (!rejected(frame, alteredTask)) return false;
  alteredTask = task; alteredTask.flags = vtm::CUDA_DBF_SIDE_P_LARGE;
  if (!rejected(frame, alteredTask)) return false;
  alteredTask = task; alteredTask.flags = vtm::CUDA_DBF_SIDE_Q_LARGE;
  if (!rejected(frame, alteredTask)) return false;
  const auto beforeNoOp = context.dbfStats();
  if (context.filterDbfLumaFrame(mirror, frame, nullptr, 0) != vtm::CudaDbfDispatchResult::NoOp)
    return false;
  const auto afterNoOp = context.dbfStats();
  if (afterNoOp.noOpFrames != beforeNoOp.noOpFrames + 1
      || afterNoOp.dispatches != beforeNoOp.dispatches || afterNoOp.tasks != beforeNoOp.tasks
      || context.pictureMirrorPlaneState(mirror, 0) != vtm::CudaMirrorState::HostValid)
    return false;
  std::array<vtm::CudaDbfLumaTask, 2> boundaryTasks{ task, task };
  boundaryTasks[0].tc = 0;
  boundaryTasks[0].beta = 0;
  boundaryTasks[1].x = 16;
  boundaryTasks[1].tc = vtm::cudaDbfMaximumTc(frame.bitDepth);
  boundaryTasks[1].beta = vtm::cudaDbfMaximumBeta(frame.bitDepth);
  if (context.filterDbfLumaFrame(mirror, frame, boundaryTasks.data(),
                                 static_cast<std::uint32_t>(boundaryTasks.size()))
      != vtm::CudaDbfDispatchResult::Executed)
    return false;
  context.releasePictureMirror(mirror);
  return true;
}

#if VTM_CUDA_TESTING
bool runDbfFailureCase(const int device, const vtm::CudaDbfTestFailurePoint point)
{
  const std::uint64_t liveDeviceBefore = vtm::CudaContext::dbfLiveDeviceAllocationsForTesting();
  const std::uint64_t livePinnedBefore = vtm::CudaContext::dbfLivePinnedAllocationsForTesting();
  vtm::CudaContext context;
  context.create(device);
  TestPicture picture(1924, 1084, sizeof(Pel), 10, 81, 1, 0, 0, 8, 8);
  fillDbfPattern(picture, 10);
  const auto untouched = picture.planes[0].storage;
  int owner = 0;
  const auto mirror = context.registerPictureMirror(&owner, vtm::CudaPictureRole::Reconstruction,
                                                     picture.descriptor);
  const auto tasks = makeDbfTasks(10);
  const vtm::CudaDbfFrame frame{ 1924, 1084, 10, sizeof(Pel), {0,0} };
  context.injectDbfFailureForTesting(point);
  const bool threw = throwsWithText([&]() {
    (void) context.filterDbfLumaFrame(mirror, frame, tasks.data(),
                                      static_cast<std::uint32_t>(tasks.size()));
  }, "CUDA DBF execution failed after selection:");
  const auto stats = context.dbfStats();
  const bool rejected = context.filterDbfLumaFrame(mirror, frame, tasks.data(),
    static_cast<std::uint32_t>(tasks.size())) == vtm::CudaDbfDispatchResult::NotEligible;
  const bool valid = threw && rejected && picture.planes[0].storage == untouched
                     && !context.isDbfAccelerationAvailable() && stats.failures == 1 && stats.poisoned
                     && stats.scratchBytes == 0 && stats.retiredScratchBytes == 0;
  context.releasePictureMirror(mirror);
  context.shutdown();
  return valid
         && vtm::CudaContext::dbfLiveDeviceAllocationsForTesting() == liveDeviceBefore
         && vtm::CudaContext::dbfLivePinnedAllocationsForTesting() == livePinnedBefore;
}

bool runDbfScratchFailureCase(const int device, const vtm::CudaDbfTestFailurePoint point)
{
  const auto isOldRelease = [](const vtm::CudaDbfTestFailurePoint value) {
    return value >= vtm::CudaDbfTestFailurePoint::OldTasksDeviceRelease
           && value <= vtm::CudaDbfTestFailurePoint::OldLaneOffsetsPinnedRelease;
  };
  const auto isRecoveryRelease = [](const vtm::CudaDbfTestFailurePoint value) {
    return value >= vtm::CudaDbfTestFailurePoint::RecoveryTasksDeviceRelease
           && value <= vtm::CudaDbfTestFailurePoint::RecoveryLaneOffsetsPinnedRelease;
  };
  const auto isPinnedRecovery = [](const vtm::CudaDbfTestFailurePoint value) {
    return value == vtm::CudaDbfTestFailurePoint::RecoveryTasksPinnedRelease
           || value == vtm::CudaDbfTestFailurePoint::RecoveryLaneOffsetsPinnedRelease;
  };
  const std::uint64_t liveDeviceBefore = vtm::CudaContext::dbfLiveDeviceAllocationsForTesting();
  const std::uint64_t livePinnedBefore = vtm::CudaContext::dbfLivePinnedAllocationsForTesting();
  vtm::CudaContext context;
  context.create(device);
  const auto allTasks = makeDbfTasks(10);
  const vtm::CudaDbfFrame frame{ 1924, 1084, 10, sizeof(Pel), {0,0} };

  TestPicture warm(1924, 1084, sizeof(Pel), 10, 83, 1, 0, 0, 8, 8);
  fillDbfPattern(warm, 10);
  int warmOwner = 0;
  const auto warmMirror = context.registerPictureMirror(&warmOwner, vtm::CudaPictureRole::Reconstruction,
                                                         warm.descriptor);
  if (isOldRelease(point)
      && context.filterDbfLumaFrame(warmMirror, frame, allTasks.data(), 1)
           != vtm::CudaDbfDispatchResult::Executed)
    return false;

  TestPicture failing(1924, 1084, sizeof(Pel), 10, 89, 1, 0, 0, 8, 8);
  fillDbfPattern(failing, 10);
  const auto untouched = failing.planes[0].storage;
  int failingOwner = 0;
  const auto failingMirror = context.registerPictureMirror(
    &failingOwner, vtm::CudaPictureRole::Reconstruction, failing.descriptor);
  context.injectDbfFailureForTesting(point);
  const bool threw = throwsWithText([&]() {
    (void) context.filterDbfLumaFrame(failingMirror, frame, allTasks.data(),
                                      static_cast<std::uint32_t>(allTasks.size()));
  }, "CUDA DBF execution failed after selection:");
  const vtm::DbfAccelerationStats stats = context.dbfStats();
  const bool recoveryRelease = isRecoveryRelease(point);
  const bool retainedExactlyOne = recoveryRelease
    && vtm::CudaContext::dbfLiveDeviceAllocationsForTesting()
         == liveDeviceBefore + (isPinnedRecovery(point) ? 0u : 1u)
    && vtm::CudaContext::dbfLivePinnedAllocationsForTesting()
         == livePinnedBefore + (isPinnedRecovery(point) ? 1u : 0u);
  const bool recoveredAll = !recoveryRelease
    && vtm::CudaContext::dbfLiveDeviceAllocationsForTesting() == liveDeviceBefore
    && vtm::CudaContext::dbfLivePinnedAllocationsForTesting() == livePinnedBefore;
  bool valid = threw && failing.planes[0].storage == untouched
               && !context.isDbfAccelerationAvailable() && stats.failures == 1 && stats.poisoned
               && stats.dispatches == (isOldRelease(point) ? 1u : 0u)
               && stats.peakScratchBytes >= stats.scratchBytes
               && (recoveryRelease ? stats.scratchBytes > 0 : stats.scratchBytes == 0)
               && (recoveryRelease ? retainedExactlyOne : recoveredAll);
  if (!recoveryRelease) valid = valid && stats.retiredScratchBytes == 0;
  context.shutdown();
  return valid
         && vtm::CudaContext::dbfLiveDeviceAllocationsForTesting() == liveDeviceBefore
         && vtm::CudaContext::dbfLivePinnedAllocationsForTesting() == livePinnedBefore;
}
#endif

bool runAlfGeometryCase(vtm::CudaContext &context)
{
  const vtm::AlfAccelerationStats before = context.alfStats();
  constexpr std::uint32_t width = 1924;
  constexpr std::uint32_t height = 1084;
  TestPicture picture(width, height, sizeof(Pel), 10, 44, 1, 0, 0);
  picture.fillSamples(44);
  int owner = 0;
  const auto mirror = context.registerPictureMirror(&owner, vtm::CudaPictureRole::Reconstruction,
                                                     picture.descriptor);
  const vtm::CudaAlfLumaFrame frame = makeAlfFrame(width, height, 10);
  const std::vector<vtm::CudaAlfCtuParam> validCtus = makeAlfCtus(frame, false);
  auto rejected = [&](vtm::CudaAlfLumaFrame alteredFrame, std::vector<vtm::CudaAlfCtuParam> alteredCtus,
                      std::uint32_t count) {
    return context.filterAlfLumaFrame(mirror, alteredFrame, alteredCtus.data(), count)
           == vtm::CudaAlfDispatchResult::NotEligible;
  };
  vtm::CudaAlfLumaFrame altered = frame;
  altered.ctusInWidth--;
  if (!rejected(altered, validCtus, static_cast<std::uint32_t>(validCtus.size()))) return false;
  altered = frame;
  altered.ctusInHeight--;
  if (!rejected(altered, validCtus, static_cast<std::uint32_t>(validCtus.size()))) return false;
  altered = frame;
  altered.minSample = -1;
  if (!rejected(altered, validCtus, static_cast<std::uint32_t>(validCtus.size()))) return false;
  altered = frame;
  altered.maxSample = 1 << frame.bitDepth;
  if (!rejected(altered, validCtus, static_cast<std::uint32_t>(validCtus.size()))) return false;
  if (!rejected(frame, validCtus, static_cast<std::uint32_t>(validCtus.size() - 1))) return false;
  for (int field = 0; field < 5; ++field)
  {
    auto ctus = validCtus;
    vtm::CudaAlfCtuParam &last = ctus.back();
    if (field == 0) last.x = std::numeric_limits<std::uint32_t>::max();
    if (field == 1) last.y++;
    if (field == 2) last.width++;
    if (field == 3) last.height++;
    if (field == 4) last.enabled = 2;
    if (!rejected(frame, std::move(ctus), static_cast<std::uint32_t>(validCtus.size()))) return false;
  }
  for (const std::int32_t invalidClip : { std::numeric_limits<std::int32_t>::min(), -1,
                                          (1 << frame.bitDepth) + 1 })
  {
    auto ctus = validCtus;
    ctus.front().clipValues[0] = invalidClip;
    if (!rejected(frame, std::move(ctus), static_cast<std::uint32_t>(validCtus.size()))) return false;
  }
  for (const std::int16_t invalidCoefficient : { std::numeric_limits<std::int16_t>::min(),
                                                  std::int16_t(-129), std::int16_t(128) })
  {
    auto ctus = validCtus;
    ctus.front().coefficients[0] = invalidCoefficient;
    if (!rejected(frame, std::move(ctus), static_cast<std::uint32_t>(validCtus.size()))) return false;
  }
  {
    auto ctus = validCtus;
    ctus.front().coefficients[vtm::CUDA_ALF_COEFFICIENTS - 1] = 127;
    if (!rejected(frame, std::move(ctus), static_cast<std::uint32_t>(validCtus.size()))) return false;
  }
  if (context.alfStats().dispatches != before.dispatches
      || context.alfStats().failures != before.failures) return false;
  auto boundaryCtus = validCtus;
  boundaryCtus.front().clipValues[0] = 0;
  boundaryCtus.front().clipValues[1] = 1 << frame.bitDepth;
  boundaryCtus.front().coefficients[0] = -128;
  boundaryCtus.front().coefficients[1] = 127;
  if (context.filterAlfLumaFrame(mirror, frame, boundaryCtus.data(), static_cast<std::uint32_t>(boundaryCtus.size()))
      != vtm::CudaAlfDispatchResult::Executed) return false;
  context.releasePictureMirror(mirror);
  return true;
}

bool runAlfFailureCase(const int device, const vtm::CudaAlfTestFailurePoint failurePoint)
{
  const bool releaseFailure = failurePoint == vtm::CudaAlfTestFailurePoint::OldDeviceRelease
                              || failurePoint == vtm::CudaAlfTestFailurePoint::OldPinnedRelease;
  vtm::CudaContext context;
  context.create(device);
  TestPicture small(1920, 1080, sizeof(Pel), 10, 31, 1, 0, 0);
  small.fillSamples(31);
  int smallOwner = 0;
  const auto smallMirror = context.registerPictureMirror(&smallOwner, vtm::CudaPictureRole::Reconstruction,
                                                          small.descriptor);
  const auto smallFrame = makeAlfFrame(1920, 1080, 10);
  const auto smallCtus = makeAlfCtus(smallFrame, false);
  if (releaseFailure)
  {
    if (context.filterAlfLumaFrame(smallMirror, smallFrame, smallCtus.data(),
                                   static_cast<std::uint32_t>(smallCtus.size()))
        != vtm::CudaAlfDispatchResult::Executed) return false;
  }

  TestPicture large(1924, 1084, sizeof(Pel), 10, 37, 1, 0, 0);
  large.fillSamples(37);
  int largeOwner = 0;
  const auto largeMirror = context.registerPictureMirror(&largeOwner, vtm::CudaPictureRole::Reconstruction,
                                                          large.descriptor);
  const auto largeFrame = makeAlfFrame(1924, 1084, 10);
  const auto largeCtus = makeAlfCtus(largeFrame, false);
  std::vector<vtm::CudaAlfClassifier> diagnostics(static_cast<std::size_t>(largeFrame.width >> 2)
                                                   * (largeFrame.height >> 2));
  context.injectAlfFailureForTesting(failurePoint);
  bool didThrow = false;
  bool clearMessage = false;
  try
  {
    (void) context.filterAlfLumaFrame(largeMirror, largeFrame, largeCtus.data(),
                                      static_cast<std::uint32_t>(largeCtus.size()), diagnostics.data());
  }
  catch (const std::exception &error)
  {
    didThrow = true;
    clearMessage = std::string(error.what()).find("CUDA ALF execution failed:") != std::string::npos;
  }
  const vtm::AlfAccelerationStats stats = context.alfStats();
  const bool quarantined = context.pictureMirrorPlaneState(largeMirror, 0) == vtm::CudaMirrorState::HostValid;
  const bool rejectedAfterPoison = context.filterAlfLumaFrame(
    largeMirror, largeFrame, largeCtus.data(), static_cast<std::uint32_t>(largeCtus.size()))
    == vtm::CudaAlfDispatchResult::NotEligible;
  bool valid = didThrow && clearMessage && rejectedAfterPoison && quarantined && stats.failures == 1
                     && stats.poisoned && !stats.enabled
                     && stats.dispatches == (releaseFailure ? 1u : 0u)
                     && stats.scratchBytes == 0 && stats.retiredScratchBytes == 0
                     && large.matchesExpected();
  context.releasePictureMirror(largeMirror);
  context.releasePictureMirror(smallMirror);
  const vtm::CudaMirrorMemoryStats memory = context.pictureMirrorMemoryStats();
  valid = valid && context.pictureMirrorCount() == 0
          && memory.total.currentDeviceBytes == 0 && memory.total.currentPinnedBytes == 0;
  context.shutdown();
  return valid;
}
#endif

struct SaoReferenceAccess : SampleAdaptiveOffset
{
  void apply(const int bitDepth, const ClpRng &range, const vtm::CudaSaoLumaCtuParam &ctu,
             const Pel *source, Pel *destination, const ptrdiff_t stride,
             const bool left, const bool right, const bool above, const bool below)
  {
    if (m_signLineBuf1.size() < std::size_t(ctu.width) + 1)
    {
      m_signLineBuf1.resize(std::size_t(ctu.width) + 1);
      m_signLineBuf2.resize(std::size_t(ctu.width) + 1);
    }
    int horizontalBoundaries[] = { -1, -1, -1 };
    int verticalBoundaries[] = { -1, -1, -1 };
    int offsets[vtm::CUDA_SAO_NUM_OFFSETS]{};
    std::copy_n(ctu.offsets, vtm::CUDA_SAO_NUM_OFFSETS, offsets);
    offsetBlock(bitDepth, range, static_cast<SAOModeNewTypes>(ctu.type), offsets,
                source, destination, stride, stride, ctu.width, ctu.height,
                left, right, above, below, left && above, right && above,
                left && below, right && below, false, horizontalBoundaries,
                verticalBoundaries, 0, 0);
  }
};

vtm::CudaLoopFilterChainFrame makeChainFrame(const std::uint32_t width, const std::uint32_t height,
                                              const std::uint8_t bitDepth, const std::uint8_t stages)
{
  constexpr std::uint32_t ctuSize = 64;
  vtm::CudaLoopFilterChainFrame frame{};
  frame.width = width;
  frame.height = height;
  frame.ctuWidth = frame.ctuHeight = ctuSize;
  frame.ctusInWidth = (width + ctuSize - 1) / ctuSize;
  frame.ctusInHeight = (height + ctuSize - 1) / ctuSize;
  frame.ctuCount = frame.ctusInWidth * frame.ctusInHeight;
  frame.lmcsLutSize = (stages & vtm::CUDA_LOOP_FILTER_LMCS) ? (1u << bitDepth) : 0;
  frame.minSample = 0;
  frame.maxSample = (1 << bitDepth) - 1;
  frame.bitDepth = bitDepth;
  frame.elementSize = sizeof(Pel);
  frame.stages = stages;
  return frame;
}

std::vector<vtm::CudaSaoLumaCtuParam> makeSaoCtus(const vtm::CudaLoopFilterChainFrame &frame,
                                                   const bool includeDisabled)
{
  std::vector<vtm::CudaSaoLumaCtuParam> ctus(frame.ctuCount);
  const int maximumOffset = (1 << (frame.bitDepth - 5)) - 1;
  for (std::uint32_t index = 0; index < frame.ctuCount; ++index)
  {
    auto &ctu = ctus[index];
    ctu.x = (index % frame.ctusInWidth) * frame.ctuWidth;
    ctu.y = (index / frame.ctusInWidth) * frame.ctuHeight;
    ctu.width = std::min(frame.ctuWidth, frame.width - ctu.x);
    ctu.height = std::min(frame.ctuHeight, frame.height - ctu.y);
    ctu.enabled = !includeDisabled || index % 11 != 0;
    ctu.type = static_cast<std::int8_t>(index % 5);
    if (!ctu.enabled)
    {
      ctu.type = -1;
      continue;
    }
    for (std::uint32_t offset = 0; offset < vtm::CUDA_SAO_NUM_OFFSETS; ++offset)
    {
      const int magnitude = int((index * 7 + offset * 3) % (maximumOffset + 1));
      ctu.offsets[offset] = ((index + offset) & 1) == 0 ? magnitude : -magnitude;
    }
    if (ctu.type != static_cast<std::int8_t>(SAOModeNewTypes::BO))
      ctu.offsets[SAO_CLASS_EO_PLAIN] = 0;
  }
  return ctus;
}

void applySaoReference(const vtm::CudaLoopFilterChainFrame &frame,
                       const std::vector<vtm::CudaSaoLumaCtuParam> &ctus,
                       const Pel *source, Pel *destination, const ptrdiff_t stride)
{
  SaoReferenceAccess reference;
  reference.create(frame.width, frame.height, ChromaFormat::_400, frame.ctuWidth, frame.ctuHeight, 0, 0, 0);
  const ClpRng range{ frame.minSample, frame.maxSample, frame.bitDepth, 0 };
  for (const auto &ctu : ctus)
  {
    if (!ctu.enabled) continue;
    const bool left = ctu.x != 0;
    const bool above = ctu.y != 0;
    const bool right = ctu.x + ctu.width != frame.width;
    const bool below = ctu.y + ctu.height != frame.height;
    reference.apply(frame.bitDepth, range, ctu,
                    source + std::size_t(ctu.y) * stride + ctu.x,
                    destination + std::size_t(ctu.y) * stride + ctu.x,
                    stride, left, right, above, below);
  }
  reference.destroy();
}

std::vector<Pel> applyAlfReferenceFrame(const vtm::CudaAlfLumaFrame &frame,
                                        const std::vector<vtm::CudaAlfCtuParam> &ctus,
                                        const Pel *source, const ptrdiff_t sourceStride)
{
  constexpr int margin = 4;
  const int extendedStride = int(frame.width) + 2 * margin;
  std::vector<Pel> extended((frame.height + 2 * margin) * extendedStride);
  Pel *extendedActive = extended.data() + margin * extendedStride + margin;
  for (int y = -margin; y < int(frame.height) + margin; ++y)
    for (int x = -margin; x < int(frame.width) + margin; ++x)
      extendedActive[y * extendedStride + x] = source[
        std::size_t(std::max(0, std::min(y, int(frame.height) - 1))) * sourceStride
        + std::max(0, std::min(x, int(frame.width) - 1))];

  std::vector<AlfClassifier> classifiers(std::size_t(frame.width) * frame.height);
  std::vector<AlfClassifier *> classifierRows(frame.height);
  for (std::uint32_t y = 0; y < frame.height; ++y)
    classifierRows[y] = classifiers.data() + std::size_t(y) * frame.width;
  int lapData[NUM_DIRECTIONS][AdaptiveLoopFilter::m_CLASSIFICATION_BLK_SIZE + 5]
             [AdaptiveLoopFilter::m_CLASSIFICATION_BLK_SIZE + 5]{};
  int *lapRows[NUM_DIRECTIONS][AdaptiveLoopFilter::m_CLASSIFICATION_BLK_SIZE + 5]{};
  int **lap[NUM_DIRECTIONS]{};
  for (int direction = 0; direction < NUM_DIRECTIONS; ++direction)
  {
    for (int y = 0; y < AdaptiveLoopFilter::m_CLASSIFICATION_BLK_SIZE + 5; ++y)
      lapRows[direction][y] = lapData[direction][y];
    lap[direction] = lapRows[direction];
  }
  const CPelBuf sourceBuffer(extendedActive, extendedStride, frame.width, frame.height);
  for (std::uint32_t y = 0; y < frame.height; y += AdaptiveLoopFilter::m_CLASSIFICATION_BLK_SIZE)
    for (std::uint32_t x = 0; x < frame.width; x += AdaptiveLoopFilter::m_CLASSIFICATION_BLK_SIZE)
    {
      const Area block(x, y, std::min<std::uint32_t>(AdaptiveLoopFilter::m_CLASSIFICATION_BLK_SIZE,
                                                     frame.width - x),
                       std::min<std::uint32_t>(AdaptiveLoopFilter::m_CLASSIFICATION_BLK_SIZE,
                                               frame.height - y));
      AdaptiveLoopFilter::deriveClassificationBlk(classifierRows.data(), lap, sourceBuffer, block, block,
                                                   frame.bitDepth + 4, frame.vbCtuHeight, frame.vbPos);
    }

  std::vector<Pel> output(std::size_t(frame.width) * frame.height);
  for (std::uint32_t y = 0; y < frame.height; ++y)
    std::copy_n(source + std::size_t(y) * sourceStride, frame.width,
                output.data() + std::size_t(y) * frame.width);
  PelUnitBuf destination(ChromaFormat::_400, PelBuf(output.data(), frame.width, frame.width, frame.height));
  const CPelUnitBuf sourceUnit(ChromaFormat::_400, sourceBuffer);
  XuPool pool;
  CodingStructure cs(pool);
  std::array<Pel, MAX_NUM_ALF_CLASSES * MAX_NUM_ALF_LUMA_COEFF> clips{};
  const ClpRng range{ frame.minSample, frame.maxSample, frame.bitDepth, 0 };
  for (const auto &ctu : ctus)
  {
    if (!ctu.enabled) continue;
    for (std::size_t i = 0; i < clips.size(); ++i) clips[i] = static_cast<Pel>(ctu.clipValues[i]);
    const Area block(ctu.x, ctu.y, ctu.width, ctu.height);
    AdaptiveLoopFilter::filterBlk<ALF_FILTER_7>(classifierRows.data(), destination, sourceUnit,
      block, block, COMPONENT_Y, reinterpret_cast<const AlfCoeff *>(ctu.coefficients), clips.data(),
      range, cs, frame.vbCtuHeight, frame.vbPos);
  }
  return output;
}

bool activePlaneEquals(const TestPicture &picture, const Pel *expected, const ptrdiff_t expectedStride)
{
  const auto &host = picture.descriptor.planes[0];
  for (std::uint32_t y = 0; y < host.height; ++y)
    for (std::uint32_t x = 0; x < host.width; ++x)
      if (readSample(static_cast<const std::uint8_t *>(host.data) + std::size_t(y) * host.strideBytes
                       + std::size_t(x) * host.elementSize, host.elementSize)
          != expected[std::size_t(y) * expectedStride + x])
        return false;
  return true;
}

bool runSaoChainCase(vtm::CudaContext &context, const std::uint8_t bitDepth)
{
  constexpr std::uint32_t width = 1924;
  constexpr std::uint32_t height = 1084;
  TestPicture picture(width, height, sizeof(Pel), bitDepth, 43, 1, 0, 0, 8, 8);
  fillDbfPattern(picture, bitDepth);
  const auto frame = makeChainFrame(width, height, bitDepth, vtm::CUDA_LOOP_FILTER_SAO);
  const auto sao = makeSaoCtus(frame, true);
  constexpr ptrdiff_t margin = 1;
  const ptrdiff_t stride = width + 2 * margin;
  std::vector<Pel> source((height + 2 * margin) * stride);
  Pel *sourceActive = source.data() + margin * stride + margin;
  for (std::uint32_t y = 0; y < height; ++y)
    for (std::uint32_t x = 0; x < width; ++x)
      sourceActive[std::size_t(y) * stride + x] = static_cast<Pel>(readSample(
        static_cast<const std::uint8_t *>(picture.descriptor.planes[0].data)
          + std::size_t(y) * picture.descriptor.planes[0].strideBytes + std::size_t(x) * sizeof(Pel), sizeof(Pel)));
  std::vector<Pel> expected = source;
  Pel *expectedActive = expected.data() + margin * stride + margin;
  applySaoReference(frame, sao, sourceActive, expectedActive, stride);

  int owner = 0;
  const auto mirror = context.registerPictureMirror(&owner, vtm::CudaPictureRole::Reconstruction,
                                                     picture.descriptor);
  const auto before = context.loopFilterChainStats();
  if (context.filterLoopFilterChain(mirror, frame, nullptr, nullptr, 0, sao.data(),
                                    static_cast<std::uint32_t>(sao.size()), nullptr, nullptr, 0)
      != vtm::CudaLoopFilterChainDispatchResult::Executed)
    return false;
  const auto after = context.loopFilterChainStats();
  if (!activePlaneEquals(picture, expectedActive, stride) || after.dispatches != before.dispatches + 1
      || after.saoCtus != before.saoCtus + frame.ctuCount
      || after.mirrorUploadBytes <= before.mirrorUploadBytes
      || after.mirrorDownloadBytes <= before.mirrorDownloadBytes)
    return false;

  // A second execution from the exact same source proves deterministic reuse of persistent scratch.
  for (std::uint32_t y = 0; y < height; ++y)
    for (std::uint32_t x = 0; x < width; ++x)
      writeSample(static_cast<std::uint8_t *>(picture.descriptor.planes[0].data)
                    + std::size_t(y) * picture.descriptor.planes[0].strideBytes + std::size_t(x) * sizeof(Pel),
                  sizeof(Pel), sourceActive[std::size_t(y) * stride + x]);
  context.markHostPlaneModified(mirror, 0);
  if (context.filterLoopFilterChain(mirror, frame, nullptr, nullptr, 0, sao.data(),
                                    static_cast<std::uint32_t>(sao.size()), nullptr, nullptr, 0)
        != vtm::CudaLoopFilterChainDispatchResult::Executed
      || !activePlaneEquals(picture, expectedActive, stride))
    return false;
  context.releasePictureMirror(mirror);
  return true;
}

bool runFullLoopFilterChainCase(vtm::CudaContext &context, const std::uint8_t bitDepth)
{
  constexpr std::uint32_t width = 1924;
  constexpr std::uint32_t height = 1084;
  const std::uint8_t stages = vtm::CUDA_LOOP_FILTER_LMCS | vtm::CUDA_LOOP_FILTER_DBF
                              | vtm::CUDA_LOOP_FILTER_SAO | vtm::CUDA_LOOP_FILTER_ALF;
  TestPicture picture(width, height, sizeof(Pel), bitDepth, 61, 1, 0, 0, 8, 8);
  fillDbfPattern(picture, bitDepth);
  const auto frame = makeChainFrame(width, height, bitDepth, stages);
  std::vector<std::int32_t> lut(frame.lmcsLutSize);
  for (std::uint32_t i = 0; i < frame.lmcsLutSize; ++i)
    lut[i] = (i * 3 + (i >> 2) + 17) & frame.maxSample;
  const auto dbf = makeDbfTasks(bitDepth);
  const auto sao = makeSaoCtus(frame, true);
  const vtm::CudaAlfLumaFrame alfFrame = makeAlfFrame(width, height, bitDepth);
  auto alf = makeAlfCtus(alfFrame, true);

  constexpr ptrdiff_t margin = 8;
  const ptrdiff_t stride = width + 2 * margin;
  std::vector<Pel> expected((height + 2 * margin) * stride);
  Pel *active = expected.data() + margin * stride + margin;
  for (std::uint32_t y = 0; y < height; ++y)
    for (std::uint32_t x = 0; x < width; ++x)
    {
      const int input = readSample(static_cast<const std::uint8_t *>(picture.descriptor.planes[0].data)
        + std::size_t(y) * picture.descriptor.planes[0].strideBytes + std::size_t(x) * sizeof(Pel), sizeof(Pel));
      active[std::size_t(y) * stride + x] = static_cast<Pel>(lut[input]);
    }
  DeblockingFilter::filterLumaTasksCpu(active, stride, dbf.data(), static_cast<std::uint32_t>(dbf.size()));
  const std::vector<Pel> saoSource = expected;
  applySaoReference(frame, sao, saoSource.data() + margin * stride + margin, active, stride);
  const std::vector<Pel> finalExpected = applyAlfReferenceFrame(alfFrame, alf, active, stride);

  int owner = 0;
  const auto mirror = context.registerPictureMirror(&owner, vtm::CudaPictureRole::Reconstruction,
                                                     picture.descriptor);
  const auto before = context.loopFilterChainStats();
  if (context.filterLoopFilterChain(mirror, frame, lut.data(), dbf.data(), static_cast<std::uint32_t>(dbf.size()),
                                    sao.data(), static_cast<std::uint32_t>(sao.size()), &alfFrame, alf.data(),
                                    static_cast<std::uint32_t>(alf.size()))
      != vtm::CudaLoopFilterChainDispatchResult::Executed)
    return false;
  const auto after = context.loopFilterChainStats();
  if (!activePlaneEquals(picture, finalExpected.data(), width) || after.dispatches != before.dispatches + 1
      || after.dbfTasks != before.dbfTasks + dbf.size() || after.saoCtus != before.saoCtus + sao.size()
      || after.alfCtus != before.alfCtus + alf.size())
    return false;

  // CCALF is represented as an unsupported preflight feature: zero host mutation and zero dispatch.
  std::vector<std::uint8_t> committed = picture.planes[0].storage;
  auto rejected = frame;
  rejected.unsupportedFeatures = vtm::CUDA_LOOP_FILTER_UNSUPPORTED_CCALF;
  const auto beforeReject = context.loopFilterChainStats();
  if (context.filterLoopFilterChain(mirror, rejected, lut.data(), dbf.data(),
                                    static_cast<std::uint32_t>(dbf.size()), sao.data(),
                                    static_cast<std::uint32_t>(sao.size()), &alfFrame, alf.data(),
                                    static_cast<std::uint32_t>(alf.size()))
        != vtm::CudaLoopFilterChainDispatchResult::NotEligible
      || picture.planes[0].storage != committed
      || context.loopFilterChainStats().dispatches != beforeReject.dispatches)
    return false;
  context.releasePictureMirror(mirror);
  return true;
}

bool runLoopFilterChainStageSubsetCase(vtm::CudaContext &context, const std::uint8_t bitDepth,
                                       const std::uint8_t stages, const bool zeroDbfTasks)
{
  constexpr std::uint32_t width = 1924;
  constexpr std::uint32_t height = 1084;
  TestPicture picture(width, height, sizeof(Pel), bitDepth, 97 + stages, 1, 0, 0, 8, 8);
  fillDbfPattern(picture, bitDepth);
  const std::vector<std::uint8_t> originalStorage = picture.planes[0].storage;
  const auto frame = makeChainFrame(width, height, bitDepth, stages);
  std::vector<std::int32_t> lut(frame.lmcsLutSize);
  for (std::uint32_t i = 0; i < frame.lmcsLutSize; ++i)
    lut[i] = (i * 5 + (i >> 3) + 11) & frame.maxSample;
  std::vector<vtm::CudaDbfLumaTask> dbf;
  if ((stages & vtm::CUDA_LOOP_FILTER_DBF) != 0 && !zeroDbfTasks) dbf = makeDbfTasks(bitDepth);
  std::vector<vtm::CudaSaoLumaCtuParam> sao;
  if ((stages & vtm::CUDA_LOOP_FILTER_SAO) != 0) sao = makeSaoCtus(frame, true);
  vtm::CudaAlfLumaFrame alfFrame{};
  std::vector<vtm::CudaAlfCtuParam> alf;
  if ((stages & vtm::CUDA_LOOP_FILTER_ALF) != 0)
  {
    alfFrame = makeAlfFrame(width, height, bitDepth);
    alf = makeAlfCtus(alfFrame, true);
  }

  constexpr ptrdiff_t margin = 8;
  const ptrdiff_t stride = width + 2 * margin;
  std::vector<Pel> expected((height + 2 * margin) * stride);
  Pel *active = expected.data() + margin * stride + margin;
  for (std::uint32_t y = 0; y < height; ++y)
    for (std::uint32_t x = 0; x < width; ++x)
    {
      int sample = readSample(static_cast<const std::uint8_t *>(picture.descriptor.planes[0].data)
        + std::size_t(y) * picture.descriptor.planes[0].strideBytes + std::size_t(x) * sizeof(Pel), sizeof(Pel));
      if ((stages & vtm::CUDA_LOOP_FILTER_LMCS) != 0) sample = lut[sample];
      active[std::size_t(y) * stride + x] = static_cast<Pel>(sample);
    }
  if (!dbf.empty())
    DeblockingFilter::filterLumaTasksCpu(active, stride, dbf.data(), static_cast<std::uint32_t>(dbf.size()));
  if (!sao.empty())
  {
    const std::vector<Pel> source = expected;
    applySaoReference(frame, sao, source.data() + margin * stride + margin, active, stride);
  }
  std::vector<Pel> alfExpected;
  if (!alf.empty()) alfExpected = applyAlfReferenceFrame(alfFrame, alf, active, stride);
  const Pel *finalExpected = alf.empty() ? active : alfExpected.data();
  const ptrdiff_t finalStride = alf.empty() ? stride : width;

  int owner = 0;
  const auto mirror = context.registerPictureMirror(&owner, vtm::CudaPictureRole::Reconstruction,
                                                     picture.descriptor);
  const auto invoke = [&]() {
    return context.filterLoopFilterChain(
      mirror, frame, lut.empty() ? nullptr : lut.data(), dbf.empty() ? nullptr : dbf.data(),
      static_cast<std::uint32_t>(dbf.size()), sao.empty() ? nullptr : sao.data(),
      static_cast<std::uint32_t>(sao.size()), alf.empty() ? nullptr : &alfFrame,
      alf.empty() ? nullptr : alf.data(), static_cast<std::uint32_t>(alf.size()));
  };
  const auto before = context.loopFilterChainStats();
  bool valid = context.preflightLoopFilterChain(
                 mirror, frame, lut.empty() ? nullptr : lut.data(), dbf.empty() ? nullptr : dbf.data(),
                 static_cast<std::uint32_t>(dbf.size()), sao.empty() ? nullptr : sao.data(),
                 static_cast<std::uint32_t>(sao.size()), alf.empty() ? nullptr : &alfFrame,
                 alf.empty() ? nullptr : alf.data(), static_cast<std::uint32_t>(alf.size()))
                 == vtm::CudaLoopFilterChainDispatchResult::Executed
               && invoke() == vtm::CudaLoopFilterChainDispatchResult::Executed
               && activePlaneEquals(picture, finalExpected, finalStride);
  std::copy(originalStorage.begin(), originalStorage.end(), picture.planes[0].storage.begin());
  context.markHostPlaneModified(mirror, 0);
  valid = valid && invoke() == vtm::CudaLoopFilterChainDispatchResult::Executed
          && activePlaneEquals(picture, finalExpected, finalStride);
  const auto after = context.loopFilterChainStats();
  valid = valid && after.dispatches == before.dispatches + 2
          && after.integrationSynchronizations >= before.integrationSynchronizations + 2
          && after.scratchBytes >= after.retiredScratchBytes
          && after.peakScratchBytes >= after.scratchBytes;
  context.releasePictureMirror(mirror);
  return valid;
}

bool runLoopFilterChainStageSubsetMatrix(vtm::CudaContext &context, const std::uint8_t bitDepth)
{
  struct Case
  {
    std::uint8_t stages;
    bool zeroDbfTasks;
  };
  const Case cases[] = {
    { vtm::CUDA_LOOP_FILTER_LMCS, false },
    { vtm::CUDA_LOOP_FILTER_DBF, false },
    { vtm::CUDA_LOOP_FILTER_SAO, false },
    { vtm::CUDA_LOOP_FILTER_ALF, false },
    { std::uint8_t(vtm::CUDA_LOOP_FILTER_LMCS | vtm::CUDA_LOOP_FILTER_ALF), false },
    { std::uint8_t(vtm::CUDA_LOOP_FILTER_DBF | vtm::CUDA_LOOP_FILTER_ALF), true },
    { std::uint8_t(vtm::CUDA_LOOP_FILTER_DBF | vtm::CUDA_LOOP_FILTER_ALF), false },
    { std::uint8_t(vtm::CUDA_LOOP_FILTER_SAO | vtm::CUDA_LOOP_FILTER_ALF), false }
  };
  for (const Case &test : cases)
    if (!runLoopFilterChainStageSubsetCase(context, bitDepth, test.stages, test.zeroDbfTasks)) return false;
  return true;
}

bool runLoopFilterChainDisabledStageContractCase(vtm::CudaContext &context)
{
  constexpr std::uint32_t width = 1924;
  constexpr std::uint32_t height = 1084;
  TestPicture picture(width, height, sizeof(Pel), 10, 79, 1, 0, 0, 8, 8);
  fillDbfPattern(picture, 10);
  const auto original = picture.planes[0].storage;
  const vtm::CudaLoopFilterChainFrame frame =
    makeChainFrame(width, height, 10, vtm::CUDA_LOOP_FILTER_DBF);
  auto emptyFrame = frame;
  emptyFrame.stages = 0;
  const auto beforeNoOp = context.loopFilterChainStats();
  if (context.filterLoopFilterChain(0, emptyFrame, nullptr, nullptr, 0, nullptr, 0, nullptr, nullptr, 0)
        != vtm::CudaLoopFilterChainDispatchResult::NoOp
      || context.loopFilterChainStats().dispatches != beforeNoOp.dispatches)
    return false;
  int owner = 0;
  const auto mirror = context.registerPictureMirror(&owner, vtm::CudaPictureRole::Reconstruction,
                                                     picture.descriptor);

  const auto sao = makeSaoCtus(frame, false);
  const auto rejected = context.filterLoopFilterChain(mirror, frame, nullptr, nullptr, 0,
    sao.data(), static_cast<std::uint32_t>(sao.size()), nullptr, nullptr, 0);
  if (rejected != vtm::CudaLoopFilterChainDispatchResult::NotEligible
      || picture.planes[0].storage != original)
  {
    context.releasePictureMirror(mirror);
    return false;
  }

  const auto executed = context.filterLoopFilterChain(mirror, frame, nullptr, nullptr, 0,
                                                       nullptr, 0, nullptr, nullptr, 0);
  const bool valid = executed == vtm::CudaLoopFilterChainDispatchResult::NoOp
                     && picture.planes[0].storage == original;
  context.releasePictureMirror(mirror);
  return valid;
}

#if VTM_CUDA_TESTING
bool runLoopFilterChainFailureCase(const int device,
                                   const vtm::CudaLoopFilterChainTestFailurePoint point)
{
  constexpr std::uint32_t width = 1924;
  constexpr std::uint32_t height = 1084;
  const std::uint8_t stages = vtm::CUDA_LOOP_FILTER_LMCS | vtm::CUDA_LOOP_FILTER_DBF
                              | vtm::CUDA_LOOP_FILTER_SAO | vtm::CUDA_LOOP_FILTER_ALF;
  vtm::CudaContext context;
  context.create(device);
  const std::uint64_t liveDeviceBefore = vtm::CudaContext::chainLiveDeviceAllocationsForTesting();
  const std::uint64_t livePinnedBefore = vtm::CudaContext::chainLivePinnedAllocationsForTesting();
  const bool needsExistingScratch = point == vtm::CudaLoopFilterChainTestFailurePoint::OldDeviceRelease
                                    || point == vtm::CudaLoopFilterChainTestFailurePoint::OldPinnedRelease;
  if (needsExistingScratch)
  {
    constexpr std::uint32_t warmWidth = width;
    constexpr std::uint32_t warmHeight = height;
    constexpr std::uint8_t warmStages = vtm::CUDA_LOOP_FILTER_LMCS;
    TestPicture warmPicture(warmWidth, warmHeight, sizeof(Pel), 10, 73, 1, 0, 0, 8, 8);
    fillDbfPattern(warmPicture, 10);
    const auto warmFrame = makeChainFrame(warmWidth, warmHeight, 10, warmStages);
    std::vector<std::int32_t> warmLut(warmFrame.lmcsLutSize);
    for (std::uint32_t i = 0; i < warmFrame.lmcsLutSize; ++i) warmLut[i] = i;
    int warmOwner = 0;
    const auto warmMirror = context.registerPictureMirror(
      &warmOwner, vtm::CudaPictureRole::Reconstruction, warmPicture.descriptor);
    const auto warmResult = context.filterLoopFilterChain(
      warmMirror, warmFrame, warmLut.data(), nullptr, 0, nullptr, 0, nullptr, nullptr, 0);
    context.releasePictureMirror(warmMirror);
    if (warmResult != vtm::CudaLoopFilterChainDispatchResult::Executed)
    {
      std::cerr << "chain fault warm-up rejected point=" << static_cast<int>(point)
                << " result=" << static_cast<int>(warmResult) << '\n';
      context.shutdown();
      return false;
    }
  }
  TestPicture picture(width, height, sizeof(Pel), 10, 79, 1, 0, 0, 8, 8);
  fillDbfPattern(picture, 10);
  const auto original = picture.planes[0].storage;
  const auto frame = makeChainFrame(width, height, 10, stages);
  std::vector<std::int32_t> lut(frame.lmcsLutSize);
  for (std::uint32_t i = 0; i < frame.lmcsLutSize; ++i) lut[i] = i;
  const auto dbf = makeDbfTasks(10);
  const auto sao = makeSaoCtus(frame, true);
  const vtm::CudaAlfLumaFrame alfFrame = makeAlfFrame(width, height, 10);
  const auto alf = makeAlfCtus(alfFrame, true);
  int owner = 0;
  const auto mirror = context.registerPictureMirror(&owner, vtm::CudaPictureRole::Reconstruction,
                                                     picture.descriptor);
  context.injectLoopFilterChainFailureForTesting(point);
  const bool failed = throws([&]() {
    (void) context.filterLoopFilterChain(mirror, frame, lut.data(), dbf.data(),
      static_cast<std::uint32_t>(dbf.size()), sao.data(), static_cast<std::uint32_t>(sao.size()),
      &alfFrame, alf.data(), static_cast<std::uint32_t>(alf.size()));
  });
  const auto stats = context.loopFilterChainStats();
  const std::uint64_t liveDeviceDuring = vtm::CudaContext::chainLiveDeviceAllocationsForTesting();
  const std::uint64_t livePinnedDuring = vtm::CudaContext::chainLivePinnedAllocationsForTesting();
  const bool quarantined = context.pictureMirrorPlaneState(mirror, 0) == vtm::CudaMirrorState::HostValid;
  const auto retry = context.filterLoopFilterChain(mirror, frame, lut.data(), dbf.data(),
    static_cast<std::uint32_t>(dbf.size()), sao.data(), static_cast<std::uint32_t>(sao.size()),
    &alfFrame, alf.data(), static_cast<std::uint32_t>(alf.size()));
  const bool valid = failed && picture.planes[0].storage == original && quarantined
                     && stats.failures == 1 && stats.poisoned && !stats.enabled
                     && retry == vtm::CudaLoopFilterChainDispatchResult::NotEligible
                     && stats.scratchBytes >= stats.retiredScratchBytes
                     && stats.peakScratchBytes >= stats.scratchBytes;
  context.releasePictureMirror(mirror);
  const bool mirrorReleased = context.pictureMirrorMemoryStats().total.currentDeviceBytes == 0
                              && context.pictureMirrorMemoryStats().total.currentPinnedBytes == 0;
  context.shutdown();
  const bool teardownClean = vtm::CudaContext::chainLiveDeviceAllocationsForTesting() == liveDeviceBefore
                             && vtm::CudaContext::chainLivePinnedAllocationsForTesting() == livePinnedBefore;
  const bool retainedAsInjected = point == vtm::CudaLoopFilterChainTestFailurePoint::RecoveryDeviceRelease
    ? liveDeviceDuring > liveDeviceBefore && stats.scratchBytes != 0
    : point == vtm::CudaLoopFilterChainTestFailurePoint::RecoveryPinnedRelease
      ? livePinnedDuring > livePinnedBefore && stats.scratchBytes != 0 : true;
  if (!(valid && mirrorReleased && teardownClean && retainedAsInjected))
  {
    std::cerr << "chain fault diagnostics point=" << static_cast<int>(point)
              << " failed=" << failed << " unchanged=" << (picture.planes[0].storage == original)
              << " quarantined=" << quarantined << " failures=" << stats.failures
              << " poisoned=" << stats.poisoned << " enabled=" << stats.enabled
              << " retry=" << static_cast<int>(retry) << " scratch=" << stats.scratchBytes
              << " retired=" << stats.retiredScratchBytes << " peak=" << stats.peakScratchBytes
              << " liveDevice=" << liveDeviceBefore << '/' << liveDeviceDuring << '/'
              << vtm::CudaContext::chainLiveDeviceAllocationsForTesting()
              << " livePinned=" << livePinnedBefore << '/' << livePinnedDuring << '/'
              << vtm::CudaContext::chainLivePinnedAllocationsForTesting()
              << " mirrorReleased=" << mirrorReleased << " retained=" << retainedAsInjected << '\n';
  }
  return valid && mirrorReleased && teardownClean && retainedAsInjected;
}


bool runLoopFilterChainFailureMatrix(const int device)
{
  for (const auto point : {
         vtm::CudaLoopFilterChainTestFailurePoint::GrowLutDevice,
         vtm::CudaLoopFilterChainTestFailurePoint::GrowLutPinned,
         vtm::CudaLoopFilterChainTestFailurePoint::GrowSaoDevice,
         vtm::CudaLoopFilterChainTestFailurePoint::GrowSaoPinned,
         vtm::CudaLoopFilterChainTestFailurePoint::GrowSaoOutput,
         vtm::CudaLoopFilterChainTestFailurePoint::OldDeviceRelease,
         vtm::CudaLoopFilterChainTestFailurePoint::OldPinnedRelease,
         vtm::CudaLoopFilterChainTestFailurePoint::Upload,
         vtm::CudaLoopFilterChainTestFailurePoint::LmcsLaunch,
         vtm::CudaLoopFilterChainTestFailurePoint::LmcsCompletion,
         vtm::CudaLoopFilterChainTestFailurePoint::DbfStage,
         vtm::CudaLoopFilterChainTestFailurePoint::SaoSnapshot,
         vtm::CudaLoopFilterChainTestFailurePoint::SaoLaunch,
         vtm::CudaLoopFilterChainTestFailurePoint::SaoCompletion,
         vtm::CudaLoopFilterChainTestFailurePoint::AlfStage,
         vtm::CudaLoopFilterChainTestFailurePoint::Download,
         vtm::CudaLoopFilterChainTestFailurePoint::Commit,
         vtm::CudaLoopFilterChainTestFailurePoint::RecoveryDeviceRelease,
         vtm::CudaLoopFilterChainTestFailurePoint::RecoveryPinnedRelease })
    if (!runLoopFilterChainFailureCase(device, point))
    {
      std::cerr << "loop-filter chain failure point failed: " << static_cast<int>(point) << '\n';
      return false;
    }
  return true;
}
#endif

}   // namespace

int main(const int argc, char *argv[])
{
  vtm::ComputeConfig config;
  if (config.backend != vtm::ComputeBackend::CPU || config.device != 0
      || config.enableExperimentalSad || config.enableExperimentalQpa || config.enableExperimentalAlf
      || config.enableExperimentalDbf || config.enableExperimentalLoopFilterChain)
  {
    return fail("ComputeConfig defaults are invalid");
  }
  if (!runDbfCollectorSerializationCase(8) || !runDbfCollectorSerializationCase(10))
  {
    return fail("CUDA DBF production collector serializer changed fields or emission order");
  }
  if (argc == 3 && (std::string(argv[1]) == "--write-alf-yuv"
                    || std::string(argv[1]) == "--write-dbf-yuv"))
  {
    return writeLoopFilterBenchmarkYuv(argv[2]) ? EXIT_SUCCESS
                                                 : fail("Could not write loop-filter benchmark YUV");
  }
  if (argc == 5 && std::string(argv[1]) == "--repeat-access-unit")
  {
    return repeatAccessUnit(argv[2], argv[3], std::stoi(argv[4]))
             ? EXIT_SUCCESS : fail("Could not repeat ALF access unit");
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
  if (!validatePictureDescriptorFormats())
  {
    return fail("CUDA-free picture descriptor validation rejected 400/420/422/444 or accepted overflow");
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
  const bool runChainOnly = argc == 3 && std::string(argv[1]) == "--cuda-loop-chain";
  if (argc != 3 || (std::string(argv[1]) != "--cuda" && !runBenchmark && !runChainOnly))
  {
    return fail("Usage: CudaBackendTest [--cuda device | --cuda-benchmark device | --cuda-loop-chain device]");
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
    if (runChainOnly)
    {
      std::cerr << "chain-test sao-8\n";
      bool passed = runSaoChainCase(context, 8);
      if (passed) { std::cerr << "chain-test sao-10\n"; passed = runSaoChainCase(context, 10); }
      if (passed) { std::cerr << "chain-test full-8\n"; passed = runFullLoopFilterChainCase(context, 8); }
      if (passed) { std::cerr << "chain-test full-10\n"; passed = runFullLoopFilterChainCase(context, 10); }
      if (passed) { std::cerr << "chain-test subsets-8\n"; passed = runLoopFilterChainStageSubsetMatrix(context, 8); }
      if (passed) { std::cerr << "chain-test subsets-10\n"; passed = runLoopFilterChainStageSubsetMatrix(context, 10); }
      if (passed) { std::cerr << "chain-test disabled-stage-contract\n"; passed = runLoopFilterChainDisabledStageContractCase(context); }
      context.shutdown();
#if VTM_CUDA_TESTING
      if (passed) { std::cerr << "chain-test failures\n"; passed = runLoopFilterChainFailureMatrix(std::stoi(argv[2])); }
#endif
      return passed ? EXIT_SUCCESS
                    : fail("CUDA resident loop-filter chain differed from normative stage references");
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
#if VTM_CUDA_TESTING
    pinned.allocate(256);
    context.injectPinnedReleaseFailuresForTesting(1);
    if (pinned.reset() || !pinned || pinned.size() != 256 || !pinned.reset() || pinned)
    {
      return fail("CUDA pinned release failure did not retain ownership for retry");
    }
#endif

    const auto roundTripFormat = [&context](TestPicture &formatPicture) {
      const auto handle = context.registerPictureMirror(&formatPicture, vtm::CudaPictureRole::Original,
                                                        formatPicture.descriptor);
      context.ensureDevice(handle);
      if (context.pictureMirrorAllocatedPlanes(handle)
          != static_cast<vtm::CudaPlaneMask>((vtm::CudaPlaneMask{ 1 } << formatPicture.descriptor.planeCount) - 1)
          || context.devicePicture(handle).planeCount != formatPicture.descriptor.planeCount)
      {
        return false;
      }
      formatPicture.clearTransferredBytes();
      context.markDeviceModified(handle);
      context.ensureHost(handle);
      const bool matches = formatPicture.matchesExpected();
      context.releasePictureMirror(handle);
      return matches;
    };
    TestPicture runtime400(19, 13, 2, 10, 5, 1, 0, 0);
    TestPicture runtime422(19, 13, 2, 10, 6, 3, 1, 0);
    TestPicture runtime444(19, 13, 4, 10, 7, 3, 0, 0);
    if (!roundTripFormat(runtime400) || !roundTripFormat(runtime422) || !roundTripFormat(runtime444))
    {
      return fail("CUDA picture mirrors failed 400/422/444 full-picture round-trip");
    }

    TestPicture budgetPicture(64, 64, 2, 10, 8);
    const auto budgetMirror = context.registerPictureMirror(&budgetPicture, vtm::CudaPictureRole::Original,
                                                            budgetPicture.descriptor);
    const std::uint64_t requiredLumaBytes = budgetPicture.deviceBytes(0) + budgetPicture.pinnedBytes(0);
    context.setPictureMirrorMemoryBudget(requiredLumaBytes - 1);
    const std::uint64_t budgetRejections = context.pictureMirrorMemoryStats().budgetRejections;
    if (!throws([&context, budgetMirror]() { context.ensureDevicePlane(budgetMirror, 0); })
        || context.pictureMirrorAllocatedPlanes(budgetMirror) != 0
        || context.pictureMirrorMemoryStats().total.currentDeviceBytes != 0
        || context.pictureMirrorMemoryStats().total.currentPinnedBytes != 0
        || context.pictureMirrorMemoryStats().budgetRejections != budgetRejections + 1)
    {
      return fail("CUDA mirror memory budget did not reject before allocation");
    }
    context.releasePictureMirror(budgetMirror);
    context.setPictureMirrorMemoryBudget(vtm::CUDA_DEFAULT_MIRROR_MEMORY_BUDGET_BYTES);

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

    const vtm::CudaMirrorMemoryStats registeredMemory = context.pictureMirrorMemoryStats();
    if (registeredMemory.total.currentDeviceBytes != 0 || registeredMemory.total.currentPinnedBytes != 0
        || context.pictureMirrorAllocatedPlanes(mirror8) != 0)
    {
      return fail("CUDA picture registration eagerly allocated mirror planes");
    }
    context.ensureDevicePlane(mirror8, 0);
    const vtm::CudaMirrorMemoryStats lumaMemory = context.pictureMirrorMemoryStats();
    const vtm::CudaDevicePictureDesc lumaDevice = context.devicePicturePlanes(mirror8, vtm::CUDA_PLANE_Y);
    if (context.pictureMirrorAllocatedPlanes(mirror8) != vtm::CUDA_PLANE_Y
        || context.pictureMirrorState(mirror8) != vtm::CudaMirrorState::Mixed
        || context.pictureMirrorPlaneState(mirror8, 0) != vtm::CudaMirrorState::Synchronized
        || context.pictureMirrorPlaneState(mirror8, 1) != vtm::CudaMirrorState::HostValid
        || lumaDevice.planes[0].data == nullptr || lumaDevice.planes[1].data != nullptr
        || lumaDevice.planes[2].data != nullptr || !throws([&context, mirror8]() { (void) context.devicePicture(mirror8); })
        || lumaMemory.total.currentDeviceBytes != picture8.deviceBytes(0)
        || lumaMemory.total.currentPinnedBytes != picture8.pinnedBytes(0)
        || lumaMemory.byRoleAndPlane[1][1].uploadedBytes != 0
        || lumaMemory.byRoleAndPlane[1][2].uploadedBytes != 0)
    {
      return fail("Luma-only CUDA mirror allocation/state/telemetry is invalid");
    }
    context.ensureDevicePlane(mirror8, 1);
    for (std::size_t row = 0; row < picture8.planes[1].fullHeight; ++row)
    {
      std::memset(picture8.planes[1].storage.data() + row * picture8.planes[1].stride, 0,
                  picture8.planes[1].rowBytes);
    }
    context.markDevicePlaneModified(mirror8, 1);
    context.ensureHostPlane(mirror8, 1);
    if (picture8.planes[1].storage != picture8.planes[1].expected
        || context.pictureMirrorAllocatedPlanes(mirror8) != (vtm::CUDA_PLANE_Y | vtm::CUDA_PLANE_CB)
        || context.pictureMirrorPlaneState(mirror8, 2) != vtm::CudaMirrorState::HostValid)
    {
      return fail("Lazy chroma allocation or per-plane round-trip did not preserve data");
    }
    context.ensureDevicePlane(mirror8, 2);

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

    TestPicture lazyRebind16(33, 31, 2, 10, 11);
    TestPicture lazyRebind32(35, 29, 4, 10, 23);
    int lazyOwner = 0;
    const vtm::CudaMirrorHandle lazyMirror = context.registerPictureMirror(
      &lazyOwner, vtm::CudaPictureRole::Original, lazyRebind16.descriptor);
    context.ensureDevicePlane(lazyMirror, 0);
    context.rebindHostPicture(lazyMirror, lazyRebind32.descriptor);
    if (context.pictureMirrorAllocatedPlanes(lazyMirror) != vtm::CUDA_PLANE_Y
        || context.pictureMirrorPlaneState(lazyMirror, 0) != vtm::CudaMirrorState::HostValid)
    {
      return fail("CUDA lazy mirror resize/element-size rebind did not preserve the requested plane set");
    }
    context.ensureDevicePlane(lazyMirror, 0);
    if (context.devicePicturePlanes(lazyMirror, vtm::CUDA_PLANE_Y).planes[0].elementSize != 4)
    {
      return fail("CUDA lazy mirror rebind retained the old element size");
    }
    context.ensureDevicePlane(lazyMirror, 1);
    if (context.pictureMirrorAllocatedPlanes(lazyMirror) != (vtm::CUDA_PLANE_Y | vtm::CUDA_PLANE_CB))
    {
      return fail("CUDA chroma allocation after a luma-only rebind is invalid");
    }
    context.releasePictureMirror(lazyMirror);
    if (context.pictureMirrorMemoryStats().total.currentDeviceBytes != 0
        || context.pictureMirrorMemoryStats().total.currentPinnedBytes != 0)
    {
      return fail("CUDA mirror release did not return current memory telemetry to zero");
    }

    if (!runSadBatchCase(context, 2, 8, 4, 4, 0)
        || !runSadBatchCase(context, 2, 10, 16, 12, 1)
        || !runSadBatchCase(context, 4, 8, 32, 24, 2)
        || !runSadBatchCase(context, 4, 10, 64, 32, 0))
    {
      return fail("CUDA SAD batch differed from the ordered scalar reference");
    }
    if (!runQpaBatchCase(context, 2, 8) || !runQpaBatchCase(context, 2, 10)
        || !runQpaBatchCase(context, 4, 8) || !runQpaBatchCase(context, 4, 10)
        || !runQpaMultiChunkCase(context))
    {
      return fail("CUDA QPA cross-CTU batch differed from the real ordered CPU calculation");
    }
    if (!runAlfLumaCase(context, 8) || !runAlfLumaCase(context, 10)
        || !runAlfGeometryCase(context))
    {
      return fail("CUDA luma ALF classifier/filter differed from the scalar VTM reference");
    }
    if (!runDbfLumaCase(context, 8) || !runDbfLumaCase(context, 10)
        || !runDbfGeometryCase(context, 8) || !runDbfGeometryCase(context, 10))
    {
      return fail("CUDA luma DBF differed from the scalar DeblockingFilter reference");
    }
    if (!runSaoChainCase(context, 8) || !runSaoChainCase(context, 10)
        || !runFullLoopFilterChainCase(context, 8) || !runFullLoopFilterChainCase(context, 10)
        || !runLoopFilterChainStageSubsetMatrix(context, 8)
        || !runLoopFilterChainStageSubsetMatrix(context, 10)
        || !runLoopFilterChainDisabledStageContractCase(context))
    {
      return fail("CUDA resident loop-filter chain differed from normative stage references");
    }
#if VTM_CUDA_TESTING
    if (!runLoopFilterChainFailureMatrix(std::stoi(argv[2])))
    {
      return fail("CUDA loop-filter chain rollback, fail-fast, or permanent poisoning is invalid");
    }
    if (!runBatchOperationalPreflightExceptionCase(std::stoi(argv[2])))
    {
      return fail("CUDA SAD/QPA operational preflight exceptions were reported as NotEligible");
    }
    if (!runSadFailureCase(std::stoi(argv[2]), 1, 0)
        || !runSadFailureCase(std::stoi(argv[2]), 2, 0)
        || !runSadFailureCase(std::stoi(argv[2]), 0, 1)
        || !runSadFailureCase(std::stoi(argv[2]), 0, 0, vtm::CudaBatchTestFailurePoint::Upload)
        || !runSadFailureCase(std::stoi(argv[2]), 0, 0, vtm::CudaBatchTestFailurePoint::KernelLaunch)
        || !runSadFailureCase(std::stoi(argv[2]), 0, 0, vtm::CudaBatchTestFailurePoint::ResultDownload)
        || !runSadFailureCase(std::stoi(argv[2]), 0, 0, vtm::CudaBatchTestFailurePoint::Completion)
        || !runSadFailureCase(std::stoi(argv[2]), 0, 0, vtm::CudaBatchTestFailurePoint::Publication)
        || !runSadFailureCase(std::stoi(argv[2]), 0, 0,
                              vtm::CudaBatchTestFailurePoint::DiagnosticConstruction))
    {
      return fail("CUDA SAD failure recovery or permanent poisoning is invalid");
    }
    if (!runQpaFailureCase(std::stoi(argv[2]), 1, 0)
        || !runQpaFailureCase(std::stoi(argv[2]), 2, 0)
        || !runQpaFailureCase(std::stoi(argv[2]), 3, 0)
        || !runQpaFailureCase(std::stoi(argv[2]), 4, 0)
        || !runQpaFailureCase(std::stoi(argv[2]), 0, 1)
        || !runQpaFailureCase(std::stoi(argv[2]), 0, 0, vtm::CudaBatchTestFailurePoint::Upload)
        || !runQpaFailureCase(std::stoi(argv[2]), 0, 0, vtm::CudaBatchTestFailurePoint::KernelLaunch)
        || !runQpaFailureCase(std::stoi(argv[2]), 0, 0, vtm::CudaBatchTestFailurePoint::ResultDownload)
        || !runQpaFailureCase(std::stoi(argv[2]), 0, 0, vtm::CudaBatchTestFailurePoint::Completion)
        || !runQpaFailureCase(std::stoi(argv[2]), 0, 0, vtm::CudaBatchTestFailurePoint::Publication)
        || !runQpaFailureCase(std::stoi(argv[2]), 0, 0, vtm::CudaBatchTestFailurePoint::ResultCorruption)
        || !runQpaFailureCase(std::stoi(argv[2]), 0, 0,
                              vtm::CudaBatchTestFailurePoint::DiagnosticConstruction))
    {
      return fail("CUDA QPA failure recovery, discard, or permanent poisoning is invalid");
    }
    if (!runAlfFailureCase(std::stoi(argv[2]), vtm::CudaAlfTestFailurePoint::GrowCtuDevice)
        || !runAlfFailureCase(std::stoi(argv[2]), vtm::CudaAlfTestFailurePoint::GrowPinnedCtu)
        || !runAlfFailureCase(std::stoi(argv[2]), vtm::CudaAlfTestFailurePoint::GrowClassifiers)
        || !runAlfFailureCase(std::stoi(argv[2]), vtm::CudaAlfTestFailurePoint::GrowOutput)
        || !runAlfFailureCase(std::stoi(argv[2]), vtm::CudaAlfTestFailurePoint::OldDeviceRelease)
        || !runAlfFailureCase(std::stoi(argv[2]), vtm::CudaAlfTestFailurePoint::OldPinnedRelease)
        || !runAlfFailureCase(std::stoi(argv[2]), vtm::CudaAlfTestFailurePoint::ParameterUpload)
        || !runAlfFailureCase(std::stoi(argv[2]), vtm::CudaAlfTestFailurePoint::KernelLaunch)
        || !runAlfFailureCase(std::stoi(argv[2]), vtm::CudaAlfTestFailurePoint::KernelCompletion)
        || !runAlfFailureCase(std::stoi(argv[2]), vtm::CudaAlfTestFailurePoint::DiagnosticDownload)
        || !runAlfFailureCase(std::stoi(argv[2]), vtm::CudaAlfTestFailurePoint::CommitCopy)
        || !runAlfFailureCase(std::stoi(argv[2]), vtm::CudaAlfTestFailurePoint::CommitCompletion))
    {
      return fail("CUDA ALF rollback or permanent poisoning is invalid");
    }
    for (const auto point : { vtm::CudaDbfTestFailurePoint::Allocation,
                              vtm::CudaDbfTestFailurePoint::ParameterUpload,
                              vtm::CudaDbfTestFailurePoint::SnapshotCopy,
                              vtm::CudaDbfTestFailurePoint::VerticalLaunch,
                              vtm::CudaDbfTestFailurePoint::VerticalCompletion,
                              vtm::CudaDbfTestFailurePoint::HorizontalLaunch,
                              vtm::CudaDbfTestFailurePoint::HorizontalCompletion,
                              vtm::CudaDbfTestFailurePoint::CommitCopy,
                              vtm::CudaDbfTestFailurePoint::CommitCompletion })
    {
      if (!runDbfFailureCase(std::stoi(argv[2]), point))
        return fail("CUDA DBF rollback or permanent poisoning is invalid");
    }
    for (const auto point : {
           vtm::CudaDbfTestFailurePoint::GrowTasksDevice,
           vtm::CudaDbfTestFailurePoint::GrowTasksPinned,
           vtm::CudaDbfTestFailurePoint::GrowLaneOffsetsDevice,
           vtm::CudaDbfTestFailurePoint::GrowLaneOffsetsPinned,
           vtm::CudaDbfTestFailurePoint::GrowOutputDevice,
           vtm::CudaDbfTestFailurePoint::OldTasksDeviceRelease,
           vtm::CudaDbfTestFailurePoint::OldLaneOffsetsDeviceRelease,
           vtm::CudaDbfTestFailurePoint::OldOutputDeviceRelease,
           vtm::CudaDbfTestFailurePoint::OldTasksPinnedRelease,
           vtm::CudaDbfTestFailurePoint::OldLaneOffsetsPinnedRelease,
           vtm::CudaDbfTestFailurePoint::RecoveryTasksDeviceRelease,
           vtm::CudaDbfTestFailurePoint::RecoveryLaneOffsetsDeviceRelease,
           vtm::CudaDbfTestFailurePoint::RecoveryOutputDeviceRelease,
           vtm::CudaDbfTestFailurePoint::RecoveryTasksPinnedRelease,
           vtm::CudaDbfTestFailurePoint::RecoveryLaneOffsetsPinnedRelease })
    {
      if (!runDbfScratchFailureCase(std::stoi(argv[2]), point))
        return fail("CUDA DBF scratch ownership, accounting, recovery, or teardown retry is invalid");
    }
    if (!runRecoveryIsolationCase(std::stoi(argv[2]), true)
        || !runRecoveryIsolationCase(std::stoi(argv[2]), false))
    {
      return fail("CUDA QPA and SAD failure recovery are not isolated");
    }
    if (!runMirrorPlaneFailureCase(context, 1, 0, 0)
        || !runMirrorPlaneFailureCase(context, 2, 0, 0)
        || !runMirrorPlaneFailureCase(context, 3, 0, 0)
        || !runMirrorPlaneFailureCase(context, 0, 1, 0)
        || !runMirrorPlaneFailureCase(context, 0, 0, 1))
    {
      return fail("CUDA per-plane mirror failure rollback or retry is invalid");
    }
    if (!runMultiPlaneTransactionFailureCase(context, 1)
        || !runMultiPlaneTransactionFailureCase(context, 2))
    {
      return fail("CUDA multi-plane mirror transfer failure was partially published without synchronization");
    }
    if (!runSameLayoutPartialReleaseCase(std::stoi(argv[2]), true)
        || !runSameLayoutPartialReleaseCase(std::stoi(argv[2]), false))
    {
      return fail("CUDA same-layout rebind could not reconstruct independently retained mirror resources");
    }
#endif
    if (runBenchmark && (!benchmarkSadBatch(context) || !benchmarkQpaBatch(std::stoi(argv[2]))
                         || !benchmarkAlfFrame(context) || !benchmarkDbfFrame(context)))
    {
      return fail("CUDA backend microbenchmark failed");
    }

#if VTM_CUDA_TESTING
    TestPicture fallbackPicture(10, 8, 2, 10, 17);
    int fallbackOwner = 0;
    const vtm::CudaMirrorHandle fallbackOriginal = context.registerPictureMirror(
      &fallbackOwner, vtm::CudaPictureRole::Original, fallbackPicture.descriptor);
    const vtm::CudaMirrorHandle fallbackReconstruction = context.registerPictureMirror(
      &fallbackOwner, vtm::CudaPictureRole::Reconstruction, fallbackPicture.descriptor);
    context.ensureDevice(fallbackOriginal);
    context.ensureDevice(fallbackReconstruction);
    context.injectReleaseFailuresForTesting(6, 0);
    if (!throws([&context, &fallbackOwner]() { context.releasePictureMirrors(&fallbackOwner); })
        || context.pictureMirrorCount() != 0)
    {
      return fail("Synchronous fallback did not release every mirror after asynchronous free failures");
    }

    const vtm::CudaMirrorHandle retryHandle = context.registerPictureMirror(
      &fallbackOwner, vtm::CudaPictureRole::Reconstruction, fallbackPicture.descriptor);
    context.ensureDevicePlane(retryHandle, 0);
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

    const vtm::CudaMirrorMemoryStats beforePinnedFailure = context.pictureMirrorMemoryStats();
    const vtm::CudaMirrorHandle pinnedFailureHandle = context.registerPictureMirror(
      &fallbackOwner, vtm::CudaPictureRole::Reconstruction, fallbackPicture.descriptor);
    context.ensureDevicePlane(pinnedFailureHandle, 0);
    const std::uint64_t retainedPinnedBytes = context.pictureMirrorMemoryStats().total.currentPinnedBytes
                                              - beforePinnedFailure.total.currentPinnedBytes;
    context.injectPinnedReleaseFailuresForTesting(1);
    if (!throws([&context, pinnedFailureHandle]() { context.releasePictureMirror(pinnedFailureHandle); })
        || context.pictureMirrorCount() != 1
        || !context.hasPictureMirror(&fallbackOwner, vtm::CudaPictureRole::Reconstruction)
        || context.pictureMirrorMemoryStats().total.currentPinnedBytes
             != beforePinnedFailure.total.currentPinnedBytes + retainedPinnedBytes)
    {
      return fail("CUDA pinned mirror release failure lost ownership or accounting");
    }
    context.releasePictureMirror(pinnedFailureHandle);
    if (context.pictureMirrorCount() != 0
        || context.pictureMirrorMemoryStats().total.currentPinnedBytes
             != beforePinnedFailure.total.currentPinnedBytes)
    {
      return fail("CUDA retained pinned mirror allocation could not be released on retry");
    }

    TestPicture rebindFailurePicture(21, 17, 2, 10, 13);
    TestPicture rebindFailureResized(25, 19, 4, 10, 29);
    const vtm::CudaMirrorHandle rebindFailureHandle = context.registerPictureMirror(
      &fallbackOwner, vtm::CudaPictureRole::Original, rebindFailurePicture.descriptor);
    context.ensureDevicePlane(rebindFailureHandle, 0);
    context.injectReleaseFailuresForTesting(1, 1);
    if (!throws([&context, rebindFailureHandle, &rebindFailureResized]() {
          context.rebindHostPicture(rebindFailureHandle, rebindFailureResized.descriptor);
        })
        || context.pictureMirrorAllocatedPlanes(rebindFailureHandle) != vtm::CUDA_PLANE_Y)
    {
      return fail("CUDA mirror rebind lost per-plane ownership after release failures");
    }
    context.rebindHostPicture(rebindFailureHandle, rebindFailureResized.descriptor);
    context.ensureDevicePlane(rebindFailureHandle, 0);
    if (context.devicePicturePlanes(rebindFailureHandle, vtm::CUDA_PLANE_Y).planes[0].width != 25)
    {
      return fail("CUDA mirror rebind could not recover after retained-plane release retry");
    }
    context.releasePictureMirror(rebindFailureHandle);
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
