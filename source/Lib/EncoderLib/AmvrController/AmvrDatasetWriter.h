/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2024, ITU/ISO/IEC
 * All rights reserved.
 */

#pragma once

#include "AmvrDataTypes.h"
#include <string>
#include <fstream>
#include <mutex>
#include <iostream>

class AmvrDatasetWriter
{
public:
  AmvrDatasetWriter();
  ~AmvrDatasetWriter();

  bool open(const std::string& filename);
  void close();
  bool isOpen() const { return m_file.is_open(); }

  void writeSample(const AmvrDatasetSample& sample);

  uint64_t getSampleCount() const { return m_sampleCount; }

private:
  std::ofstream m_file;
  std::string   m_filename;
  uint64_t      m_sampleCount;
  std::mutex    m_mutex;
};
