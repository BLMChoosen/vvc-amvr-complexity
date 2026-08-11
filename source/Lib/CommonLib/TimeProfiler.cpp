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

/** \file     TimeProfiler.cpp
    \brief    hierarchical, low-overhead time profiler for the encoder/decoder pipeline
*/

#include "TimeProfiler.h"

#if ENABLE_TIME_PROFILING

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <mutex>
#include <vector>

#if TPROF_USE_TSC && !defined(_MSC_VER)
#include <cpuid.h>
#endif

//! \ingroup CommonLib
//! \{

namespace tprof
{
  // ==================================================================================================================
  // Static description tables generated from the X-macro lists
  // ==================================================================================================================

  static const char *const STAGE_NAME[NUM_STAGES] = {
#define TPROF_MAKE_STAGE_NAME(id, depth, label) #id,
    TPROF_STAGE_LIST(TPROF_MAKE_STAGE_NAME)
#undef TPROF_MAKE_STAGE_NAME
  };

  static const char *const STAGE_LABEL[NUM_STAGES] = {
#define TPROF_MAKE_STAGE_LABEL(id, depth, label) label,
    TPROF_STAGE_LIST(TPROF_MAKE_STAGE_LABEL)
#undef TPROF_MAKE_STAGE_LABEL
  };

  static const int STAGE_DEPTH[NUM_STAGES] = {
#define TPROF_MAKE_STAGE_DEPTH(id, depth, label) depth,
    TPROF_STAGE_LIST(TPROF_MAKE_STAGE_DEPTH)
#undef TPROF_MAKE_STAGE_DEPTH
  };

  static const char *const COUNTER_NAME[NUM_COUNTERS] = {
#define TPROF_MAKE_COUNTER_NAME(id, label) #id,
    TPROF_COUNTER_LIST(TPROF_MAKE_COUNTER_NAME)
#undef TPROF_MAKE_COUNTER_NAME
  };

  static const char *const COUNTER_LABEL[NUM_COUNTERS] = {
#define TPROF_MAKE_COUNTER_LABEL(id, label) label,
    TPROF_COUNTER_LIST(TPROF_MAKE_COUNTER_LABEL)
#undef TPROF_MAKE_COUNTER_LABEL
  };

  static const char *const HIST_NAME[NUM_HISTS] = {
#define TPROF_MAKE_HIST_NAME(id, label) #id,
    TPROF_HIST_LIST(TPROF_MAKE_HIST_NAME)
#undef TPROF_MAKE_HIST_NAME
  };

  static const char *const HIST_LABEL[NUM_HISTS] = {
#define TPROF_MAKE_HIST_LABEL(id, label) label,
    TPROF_HIST_LIST(TPROF_MAKE_HIST_LABEL)
#undef TPROF_MAKE_HIST_LABEL
  };

  // ==================================================================================================================
  // Global state
  // ==================================================================================================================

  thread_local ThreadState *g_state = nullptr;

  namespace
  {
    std::mutex               g_registryMutex;
    ThreadState             *g_registryHead  = nullptr;
    std::atomic<uint64_t>    g_threadCounter{ 0 };

    bool   g_initialised    = false;
    bool   g_tscInvariant   = false;
    double g_ticksPerSecond = 0.0;   ///< clock frequency, refined at report() time
    double g_scopeCostTicks = 0.0;   ///< measured cost of one begin()/end() pair

    std::chrono::steady_clock::time_point g_initWallTime;
    uint64_t                              g_initTicks = 0;

    /// One row of the per-picture time series.
    struct PictureRecord
    {
      int      poc;
      int      sliceType;
      int      qp;
      int      tLayer;
      int      trialPass;
      uint64_t wallTicks;
      uint64_t incl[NUM_STAGES];
      uint64_t self[NUM_STAGES];
      uint64_t counters[NUM_COUNTERS];
    };

    std::vector<PictureRecord> g_pictures;
    PictureRecord              g_pictureBase = {};
    int                        g_pictureDepth = 0;
    uint64_t                   g_pictureStartTicks = 0;

    const char *envOrNull(const char *name)
    {
      const char *v = getenv(name);
      return (v != nullptr && v[0] != '\0') ? v : nullptr;
    }

    // ----------------------------------------------------------------------------------------------------------------
    // Clock calibration
    // ----------------------------------------------------------------------------------------------------------------

    bool hasInvariantTsc()
    {
#if TPROF_USE_TSC
#if defined(_MSC_VER)
      int regs[4] = { 0, 0, 0, 0 };
      __cpuid(regs, 0x80000000);
      if ((unsigned int) regs[0] < 0x80000007u)
      {
        return false;
      }
      __cpuid(regs, 0x80000007);
      return ((regs[3] >> 8) & 1) != 0;
#else
      unsigned int a = 0, b = 0, c = 0, d = 0;
      if (__get_cpuid_max(0x80000000u, nullptr) < 0x80000007u)
      {
        return false;
      }
      if (!__get_cpuid(0x80000007u, &a, &b, &c, &d))
      {
        return false;
      }
      return ((d >> 8) & 1) != 0;
#endif
#else
      return false;
#endif
    }

    /// Busy-wait calibration of the tick source against steady_clock. Refined later in report() over
    /// the whole run, where the relative error of the same measurement is orders of magnitude smaller.
    double calibrateTicksPerSecond()
    {
#if TPROF_USE_TSC
      using namespace std::chrono;
      const steady_clock::time_point t0 = steady_clock::now();
      const uint64_t                 c0 = readTicks();
      steady_clock::time_point       t1 = t0;
      uint64_t                       c1 = c0;
      // 20 ms is enough for a sub-0.1% estimate and keeps the encoder start-up delay unnoticeable.
      while (duration_cast<nanoseconds>(t1 - t0).count() < 20 * 1000 * 1000)
      {
        t1 = steady_clock::now();
        c1 = readTicks();
      }
      const double seconds = duration_cast<duration<double>>(t1 - t0).count();
      return seconds > 0.0 ? (double) (c1 - c0) / seconds : 0.0;
#else
      typedef std::chrono::steady_clock::period P;
      return (double) P::den / (double) P::num;
#endif
    }

    /// Measures the cost of one instrumented scope, so the report can state how much of the measured
    /// time is the measurement itself.
    double measureScopeCostTicks()
    {
      const int         iterations = 1 << 18;
      volatile uint64_t sink       = 0;

      // Warm up the code paths and the accumulator cache lines.
      for (int i = 0; i < 4096; i++)
      {
        begin(PROBE_STAGE);
        sink = sink + (uint64_t) i;
        end(PROBE_STAGE);
      }

      const uint64_t base0 = readTicks();
      for (int i = 0; i < iterations; i++)
      {
        sink = sink + (uint64_t) i;
      }
      const uint64_t base1 = readTicks();

      const uint64_t prof0 = readTicks();
      for (int i = 0; i < iterations; i++)
      {
        begin(PROBE_STAGE);
        sink = sink + (uint64_t) i;
        end(PROBE_STAGE);
      }
      const uint64_t prof1 = readTicks();

      // Discard the probe samples again: PROBE_STAGE must not show up in the report.
      ThreadState &t = state();
      memset(&t.acc[PROBE_STAGE], 0, sizeof(StageAcc));
      t.acc[PROBE_STAGE].minTicks = UINT64_MAX;

      const double gross = (double) (prof1 - prof0) - (double) (base1 - base0);
      return gross > 0.0 ? gross / (double) iterations : 0.0;
    }

    // ----------------------------------------------------------------------------------------------------------------
    // Aggregation
    // ----------------------------------------------------------------------------------------------------------------

    struct Totals
    {
      StageAcc acc[NUM_STAGES];
      uint64_t counters[NUM_COUNTERS];
      uint64_t hists[NUM_HISTS][HIST_BINS];
      uint64_t overflows;
      uint64_t mismatches;
      int      threads;
    };

    void collect(Totals &out)
    {
      memset(&out, 0, sizeof(Totals));
      for (int s = 0; s < NUM_STAGES; s++)
      {
        out.acc[s].minTicks = UINT64_MAX;
      }

      std::lock_guard<std::mutex> guard(g_registryMutex);
      for (const ThreadState *t = g_registryHead; t != nullptr; t = t->next)
      {
        out.threads++;
        out.overflows += t->overflows;
        out.mismatches += t->mismatches;
        for (int s = 0; s < NUM_STAGES; s++)
        {
          const StageAcc &a = t->acc[s];
          StageAcc       &o = out.acc[s];
          o.calls += a.calls;
          o.outerCalls += a.outerCalls;
          o.inclTicks += a.inclTicks;
          o.selfTicks += a.selfTicks;
          o.sumSqTicks += a.sumSqTicks;
          if (a.calls > 0)
          {
            o.minTicks = std::min(o.minTicks, a.minTicks);
            o.maxTicks = std::max(o.maxTicks, a.maxTicks);
          }
          o.maxActive = std::max(o.maxActive, a.maxActive);
        }
        for (int c = 0; c < NUM_COUNTERS; c++)
        {
          out.counters[c] += t->counters[c];
        }
        for (int h = 0; h < NUM_HISTS; h++)
        {
          for (int b = 0; b < HIST_BINS; b++)
          {
            out.hists[h][b] += t->hists[h][b];
          }
        }
      }
      for (int s = 0; s < NUM_STAGES; s++)
      {
        if (out.acc[s].calls == 0)
        {
          out.acc[s].minTicks = 0;
        }
      }
    }

    inline double toSeconds(uint64_t ticks)
    {
      return g_ticksPerSecond > 0.0 ? (double) ticks / g_ticksPerSecond : 0.0;
    }

    inline double toMicroseconds(double ticks)
    {
      return g_ticksPerSecond > 0.0 ? ticks * 1.0e6 / g_ticksPerSecond : 0.0;
    }

    inline double toNanoseconds(double ticks)
    {
      return g_ticksPerSecond > 0.0 ? ticks * 1.0e9 / g_ticksPerSecond : 0.0;
    }

    /// Mean inclusive duration of one outermost invocation, in ticks.
    double meanTicks(const StageAcc &a)
    {
      return a.outerCalls > 0 ? (double) a.inclTicks / (double) a.outerCalls : 0.0;
    }

    /// Population standard deviation of the outermost invocation duration, in ticks. Recursive
    /// invocations are excluded on purpose: their durations overlap with the outer ones.
    double stdDevTicks(const StageAcc &a)
    {
      if (a.outerCalls < 2)
      {
        return 0.0;
      }
      const double n    = (double) a.outerCalls;
      const double mean = (double) a.inclTicks / n;
      const double var  = a.sumSqTicks / n - mean * mean;
      return var > 0.0 ? sqrt(var) : 0.0;
    }

    const char *sliceTypeName(int sliceType)
    {
      // Matches SliceType in CommonLib/CommonDef.h (B, P, I).
      static const char *const names[] = { "B", "P", "I" };
      return (sliceType >= 0 && sliceType < 3) ? names[sliceType] : "?";
    }

    /// Width of the console report, i.e. of the widest table row.
    static constexpr int REPORT_WIDTH = 168;

    void printRule(char c = '-')
    {
      char rule[REPORT_WIDTH + 2];
      memset(rule, c, REPORT_WIDTH);
      rule[REPORT_WIDTH] = '\0';
      printf("%s\n", rule);
    }

    /// Opens a CSV in append mode and reports whether a header still has to be written, so that a
    /// batch of runs can accumulate into a single file.
    FILE *openCsv(const char *path, bool &needHeader)
    {
      needHeader = true;
      FILE *f    = fopen(path, "a+");
      if (f == nullptr)
      {
        fprintf(stderr, "[TimeProfiler] cannot open '%s' for writing\n", path);
        return nullptr;
      }
      fseek(f, 0, SEEK_END);
      needHeader = ftell(f) <= 0;
      return f;
    }
  }   // namespace

  // ==================================================================================================================
  // Per-thread accumulators
  // ==================================================================================================================

  ThreadState *createThreadState()
  {
    ThreadState *t = new ThreadState;
    memset(t, 0, sizeof(ThreadState));
    for (int s = 0; s < NUM_STAGE_SLOTS; s++)
    {
      t->acc[s].minTicks = UINT64_MAX;
    }
    t->threadId = g_threadCounter++;

    {
      std::lock_guard<std::mutex> guard(g_registryMutex);
      t->next        = g_registryHead;
      g_registryHead = t;
    }

    g_state = t;
    return t;
  }

  // ==================================================================================================================
  // Per-picture time series
  // ==================================================================================================================

  void ScopedPicture::beginPicture(int poc, int sliceType, int qp, int tLayer, bool trialPass)
  {
    if (g_pictureDepth++ != 0)
    {
      return;   // nested slice compression: attribute everything to the outermost record
    }

    Totals snapshot;
    collect(snapshot);

    g_pictureBase.poc       = poc;
    g_pictureBase.sliceType = sliceType;
    g_pictureBase.qp        = qp;
    g_pictureBase.tLayer    = tLayer;
    g_pictureBase.trialPass = trialPass ? 1 : 0;
    for (int s = 0; s < NUM_STAGES; s++)
    {
      g_pictureBase.incl[s] = snapshot.acc[s].inclTicks;
      g_pictureBase.self[s] = snapshot.acc[s].selfTicks;
    }
    for (int c = 0; c < NUM_COUNTERS; c++)
    {
      g_pictureBase.counters[c] = snapshot.counters[c];
    }
    g_pictureStartTicks = readTicks();
  }

  void ScopedPicture::endPicture()
  {
    if (g_pictureDepth == 0)
    {
      return;
    }
    if (--g_pictureDepth != 0)
    {
      return;
    }

    const uint64_t stop = readTicks();

    Totals snapshot;
    collect(snapshot);

    PictureRecord rec = g_pictureBase;
    rec.wallTicks     = stop - g_pictureStartTicks;
    for (int s = 0; s < NUM_STAGES; s++)
    {
      rec.incl[s] = snapshot.acc[s].inclTicks - g_pictureBase.incl[s];
      rec.self[s] = snapshot.acc[s].selfTicks - g_pictureBase.self[s];
    }
    for (int c = 0; c < NUM_COUNTERS; c++)
    {
      rec.counters[c] = snapshot.counters[c] - g_pictureBase.counters[c];
    }
    g_pictures.push_back(rec);
  }
}   // namespace tprof

// ====================================================================================================================
// TimeProfiler facade
// ====================================================================================================================

using namespace tprof;

void TimeProfiler::init()
{
  if (g_initialised)
  {
    return;
  }
  g_initialised = true;

  // Make sure the calling thread owns its accumulators before anything is timed.
  (void) state();

  g_tscInvariant   = hasInvariantTsc();
  g_ticksPerSecond = calibrateTicksPerSecond();
  g_scopeCostTicks = measureScopeCostTicks();

  g_initWallTime = std::chrono::steady_clock::now();
  g_initTicks    = readTicks();

  g_pictures.reserve(1024);

  std::atexit(&TimeProfiler::destroy);

#if TPROF_USE_TSC
  if (!g_tscInvariant)
  {
    fprintf(stderr,
            "[TimeProfiler] warning: this CPU does not report an invariant TSC; timings may drift with the\n"
            "               core frequency. Rebuild with -DTPROF_USE_TSC=0 to use std::chrono::steady_clock.\n");
  }
#endif
}

void TimeProfiler::reset()
{
  std::lock_guard<std::mutex> guard(g_registryMutex);
  for (ThreadState *t = g_registryHead; t != nullptr; t = t->next)
  {
    const uint64_t     id   = t->threadId;
    ThreadState *const next = t->next;
    memset(t, 0, sizeof(ThreadState));
    for (int s = 0; s < NUM_STAGE_SLOTS; s++)
    {
      t->acc[s].minTicks = UINT64_MAX;
    }
    t->threadId = id;
    t->next     = next;
  }
  g_pictures.clear();
  g_pictureDepth = 0;
}

void TimeProfiler::destroy()
{
  std::lock_guard<std::mutex> guard(g_registryMutex);
  ThreadState *t = g_registryHead;
  while (t != nullptr)
  {
    ThreadState *const next = t->next;
    delete t;
    t = next;
  }
  g_registryHead = nullptr;
  g_state        = nullptr;
  g_pictures.clear();
  g_pictures.shrink_to_fit();
}

void TimeProfiler::report()
{
  // Refine the calibration over the whole run: the measurement is the same as at start-up but the
  // interval is many orders of magnitude longer, so the residual error becomes negligible.
#if TPROF_USE_TSC
  if (g_initTicks != 0)
  {
    const double elapsed =
      std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - g_initWallTime)
        .count();
    const uint64_t elapsedTicks = readTicks() - g_initTicks;
    if (elapsed > 1.0 && elapsedTicks > 0)
    {
      g_ticksPerSecond = (double) elapsedTicks / elapsed;
    }
  }
#endif

  Totals totals;
  collect(totals);

  uint64_t totalCalls = 0;
  for (int s = 0; s < NUM_STAGES; s++)
  {
    totalCalls += totals.acc[s].calls;
  }

  const uint64_t refTicks   = totals.acc[ENCODER].inclTicks;
  const double   refSeconds = toSeconds(refTicks);
  const double   overheadSeconds =
    g_ticksPerSecond > 0.0 ? (double) totalCalls * g_scopeCostTicks / g_ticksPerSecond : 0.0;

  const char *const tag        = envOrNull("VTM_TPROF_TAG");
  const char *const summaryCsv = envOrNull("VTM_TPROF_CSV");
  const char *const frameCsv   = envOrNull("VTM_TPROF_FRAME_CSV");
  const bool        quiet      = envOrNull("VTM_TPROF_QUIET") != nullptr;

  // ------------------------------------------------------------------------------------------------------------------
  // Console report
  // ------------------------------------------------------------------------------------------------------------------
  if (!quiet)
  {
    printf("\n");
    printRule('=');
    printf(" Time Profiler report");
    if (tag != nullptr)
    {
      printf("   [%s]", tag);
    }
    printf("\n");
    printRule();
#if TPROF_USE_TSC
    printf(" clock            : x86 TSC%s, %.4f GHz (calibrated over %.3f s)\n",
           g_tscInvariant ? " (invariant)" : " (NOT invariant - timings may drift)",
           g_ticksPerSecond / 1.0e9,
           refSeconds);
#else
    printf(" clock            : std::chrono::steady_clock, %.4f GHz nominal\n", g_ticksPerSecond / 1.0e9);
#endif
    printf(" scope cost       : %.2f ns per instrumented scope (measured at start-up)\n",
           toNanoseconds(g_scopeCostTicks));
    printf(" instrumentation  : %llu scopes -> ~%.3f s (%.3f %% of the encoder total)\n",
           (unsigned long long) totalCalls,
           overheadSeconds,
           refSeconds > 0.0 ? 100.0 * overheadSeconds / refSeconds : 0.0);
    printf(" threads profiled : %d\n", totals.threads);
    if (totals.overflows != 0 || totals.mismatches != 0)
    {
      printf(" WARNING          : %llu stack overflows, %llu unbalanced begin/end pairs\n",
             (unsigned long long) totals.overflows,
             (unsigned long long) totals.mismatches);
    }
    printRule();
    printf(" %-48s %12s %12s %12s %7s %12s %7s %12s %12s %12s %12s\n",
           "Stage",
           "Calls",
           "Outer",
           "Total[s]",
           "Tot%",
           "Self[s]",
           "Self%",
           "Avg[us]",
           "Min[us]",
           "Max[us]",
           "SD[us]");
    printRule();

    for (int s = 0; s < NUM_STAGES; s++)
    {
      const StageAcc &a = totals.acc[s];

      char label[80];
      const int indent = std::min(STAGE_DEPTH[s] * 2, 20);
      memset(label, ' ', (size_t) indent);
      snprintf(label + indent, sizeof(label) - (size_t) indent, "%s", STAGE_LABEL[s]);

      if (a.calls == 0)
      {
        printf(" %-48.48s %12s %12s %12s %7s %12s %7s %12s %12s %12s %12s\n", label, "-", "-", "-", "-", "-", "-", "-",
               "-", "-", "-");
        continue;
      }

      const double inclSec = toSeconds(a.inclTicks);
      const double selfSec = toSeconds(a.selfTicks);
      printf(" %-48.48s %12llu %12llu %12.3f %6.2f%% %12.3f %6.2f%% %12.3f %12.3f %12.3f %12.3f\n",
             label,
             (unsigned long long) a.calls,
             (unsigned long long) a.outerCalls,
             inclSec,
             refSeconds > 0.0 ? 100.0 * inclSec / refSeconds : 0.0,
             selfSec,
             refSeconds > 0.0 ? 100.0 * selfSec / refSeconds : 0.0,
             toMicroseconds(meanTicks(a)),
             toMicroseconds((double) a.minTicks),
             toMicroseconds((double) a.maxTicks),
             toMicroseconds(stdDevTicks(a)));
    }
    printRule();
    printf(" Total%% and Self%% are relative to the inclusive time of '%s'. Self time excludes nested stages.\n",
           STAGE_LABEL[ENCODER]);
    printf(" 'Outer' counts the outermost invocations only; Avg/Min/Max/SD refer to those, so that recursive stages\n"
           " are not counted twice. For a non-recursive stage Outer equals Calls.\n");

    // ---- event counters ----------------------------------------------------------------------------------------------
    bool anyCounter = false;
    for (int c = 0; c < NUM_COUNTERS; c++)
    {
      anyCounter = anyCounter || totals.counters[c] != 0;
    }
    if (anyCounter)
    {
      printf("\n Event counters\n");
      printRule();
      for (int c = 0; c < NUM_COUNTERS; c++)
      {
        printf("  %-32.32s %16llu   %s\n", COUNTER_NAME[c], (unsigned long long) totals.counters[c], COUNTER_LABEL[c]);
      }
    }

    // ---- histograms --------------------------------------------------------------------------------------------------
    for (int h = 0; h < NUM_HISTS; h++)
    {
      uint64_t sum = 0;
      for (int b = 0; b < HIST_BINS; b++)
      {
        sum += totals.hists[h][b];
      }
      if (sum == 0)
      {
        continue;
      }
      printf("\n Histogram %s : %s\n", HIST_NAME[h], HIST_LABEL[h]);
      printRule();
      for (int b = 0; b < HIST_BINS; b++)
      {
        if (totals.hists[h][b] == 0)
        {
          continue;
        }
        printf("  bin %2d %16llu  %6.2f %%\n", b, (unsigned long long) totals.hists[h][b],
               100.0 * (double) totals.hists[h][b] / (double) sum);
      }
    }

    printRule('=');
    printf("\n");
    fflush(stdout);
  }

  // ------------------------------------------------------------------------------------------------------------------
  // Summary CSV (one row per stage / counter / histogram bin)
  // ------------------------------------------------------------------------------------------------------------------
  if (summaryCsv != nullptr)
  {
    bool  needHeader = true;
    FILE *f          = openCsv(summaryCsv, needHeader);
    if (f != nullptr)
    {
      if (needHeader)
      {
        fprintf(f,
                "tag;kind;name;bin;calls;outer_calls;incl_ms;self_ms;incl_pct;self_pct;avg_us;min_us;max_us;sd_us;"
                "max_recursion\n");
      }
      const char *const t = tag != nullptr ? tag : "";
      for (int s = 0; s < NUM_STAGES; s++)
      {
        const StageAcc &a       = totals.acc[s];
        const double    inclSec = toSeconds(a.inclTicks);
        const double    selfSec = toSeconds(a.selfTicks);
        fprintf(f, "%s;stage;%s;;%llu;%llu;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f;%u\n",
                t,
                STAGE_NAME[s],
                (unsigned long long) a.calls,
                (unsigned long long) a.outerCalls,
                inclSec * 1000.0,
                selfSec * 1000.0,
                refSeconds > 0.0 ? 100.0 * inclSec / refSeconds : 0.0,
                refSeconds > 0.0 ? 100.0 * selfSec / refSeconds : 0.0,
                toMicroseconds(meanTicks(a)),
                a.outerCalls > 0 ? toMicroseconds((double) a.minTicks) : 0.0,
                a.outerCalls > 0 ? toMicroseconds((double) a.maxTicks) : 0.0,
                toMicroseconds(stdDevTicks(a)),
                a.maxActive);
      }
      for (int c = 0; c < NUM_COUNTERS; c++)
      {
        fprintf(f, "%s;counter;%s;;%llu;;;;;;;;;;\n", t, COUNTER_NAME[c], (unsigned long long) totals.counters[c]);
      }
      for (int h = 0; h < NUM_HISTS; h++)
      {
        for (int b = 0; b < HIST_BINS; b++)
        {
          if (totals.hists[h][b] != 0)
          {
            fprintf(f, "%s;hist;%s;%d;%llu;;;;;;;;;;\n", t, HIST_NAME[h], b, (unsigned long long) totals.hists[h][b]);
          }
        }
      }
      fprintf(f, "%s;meta;scope_cost_ns;;;;;;;;%.6f;;;;\n", t, toNanoseconds(g_scopeCostTicks));
      fprintf(f, "%s;meta;instrumentation_ms;;%llu;;%.6f;;;;;;;;\n", t, (unsigned long long) totalCalls,
              overheadSeconds * 1000.0);
      fclose(f);
    }
  }

  // ------------------------------------------------------------------------------------------------------------------
  // Per-picture CSV (one row per compressed slice)
  // ------------------------------------------------------------------------------------------------------------------
  if (frameCsv != nullptr && !g_pictures.empty())
  {
    bool  needHeader = true;
    FILE *f          = openCsv(frameCsv, needHeader);
    if (f != nullptr)
    {
      if (needHeader)
      {
        fprintf(f, "tag;poc;slice_type;qp;tlayer;trial_pass;wall_ms");
        for (int s = 0; s < NUM_STAGES; s++)
        {
          fprintf(f, ";%s_ms", STAGE_NAME[s]);
        }
        for (int s = 0; s < NUM_STAGES; s++)
        {
          fprintf(f, ";%s_self_ms", STAGE_NAME[s]);
        }
        for (int c = 0; c < NUM_COUNTERS; c++)
        {
          fprintf(f, ";%s", COUNTER_NAME[c]);
        }
        fprintf(f, "\n");
      }
      const char *const t = tag != nullptr ? tag : "";
      for (const PictureRecord &rec: g_pictures)
      {
        fprintf(f, "%s;%d;%s;%d;%d;%d;%.6f", t, rec.poc, sliceTypeName(rec.sliceType), rec.qp, rec.tLayer,
                rec.trialPass, toSeconds(rec.wallTicks) * 1000.0);
        for (int s = 0; s < NUM_STAGES; s++)
        {
          fprintf(f, ";%.6f", toSeconds(rec.incl[s]) * 1000.0);
        }
        for (int s = 0; s < NUM_STAGES; s++)
        {
          fprintf(f, ";%.6f", toSeconds(rec.self[s]) * 1000.0);
        }
        for (int c = 0; c < NUM_COUNTERS; c++)
        {
          fprintf(f, ";%llu", (unsigned long long) rec.counters[c]);
        }
        fprintf(f, "\n");
      }
      fclose(f);
    }
  }
}

//! \}

#else   // !ENABLE_TIME_PROFILING

// Keep the translation unit non-empty for toolchains that complain about object files without any
// public symbol.
namespace tprof
{
  extern const int timeProfilerDisabled;
  const int        timeProfilerDisabled = 0;
}

#endif   // ENABLE_TIME_PROFILING
