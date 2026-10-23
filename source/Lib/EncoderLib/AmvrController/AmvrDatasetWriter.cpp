/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2024, ITU/ISO/IEC
 * All rights reserved.
 */

#include "AmvrDatasetWriter.h"

AmvrDatasetWriter::AmvrDatasetWriter()
  : m_sampleCount(0)
{
}

AmvrDatasetWriter::~AmvrDatasetWriter()
{
  close();
}

bool AmvrDatasetWriter::open(const std::string& filename)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  if (m_file.is_open())
  {
    m_file.close();
  }

  m_filename = filename;
  m_sampleCount = 0;

  m_file.open(m_filename, std::ios::binary | std::ios::out | std::ios::trunc);
  if (!m_file.is_open())
  {
    std::cerr << "[AMVR Collector] Error: Failed to open dataset output file: " << m_filename << std::endl;
    return false;
  }

  // Write binary header
  AmvrDatasetHeader header;
  header.magic[0] = AMVR_DATASET_MAGIC[0];
  header.magic[1] = AMVR_DATASET_MAGIC[1];
  header.magic[2] = AMVR_DATASET_MAGIC[2];
  header.magic[3] = AMVR_DATASET_MAGIC[3];
  header.version = AMVR_DATASET_VERSION;
  header.numFeatures = static_cast<uint32_t>(AMVR_NUM_FEATURES);
  header.sampleSizeBytes = static_cast<uint32_t>(sizeof(AmvrDatasetSample));

  m_file.write(reinterpret_cast<const char*>(&header), sizeof(header));
  std::cout << "[AMVR Collector] Initialized dataset file: " << m_filename << " (sample size: " << header.sampleSizeBytes << " bytes)" << std::endl;
  return true;
}

void AmvrDatasetWriter::writeSample(const AmvrDatasetSample& sample)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  if (!m_file.is_open())
  {
    return;
  }

  m_file.write(reinterpret_cast<const char*>(&sample), sizeof(sample));
  m_sampleCount++;
}

void AmvrDatasetWriter::close()
{
  std::lock_guard<std::mutex> lock(m_mutex);
  if (m_file.is_open())
  {
    m_file.flush();
    m_file.close();
    std::cout << "[AMVR Collector] Closed dataset file: " << m_filename << " (" << m_sampleCount << " samples written)" << std::endl;
  }
}
