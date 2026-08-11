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

/** \file     TimeProfiler.h
    \brief    hierarchical, low-overhead time profiler for the encoder/decoder pipeline (header)

    The profiler accumulates, per instrumented stage, the inclusive and the exclusive (self) time,
    the call count and per-call statistics (min / max / standard deviation). Stages may be nested
    and may recurse; nesting is resolved with an explicit stack, so the self time of a stage never
    contains the time spent in its children and the inclusive time of a recursive stage is not
    counted more than once per outermost invocation.

    Everything compiles away completely when ENABLE_TIME_PROFILING is 0 (the default for a plain
    upstream build): all public macros expand to `do {} while (0)` and no profiler symbol is
    referenced.

    Instrumenting code:
    \code
      TPROF_SCOPE( ME_INTEGER );          // RAII, ends at scope exit (also on early return)
      TPROF_SCOPE_ID( someRuntimeStage ); // same, stage picked at run time
      TPROF_BEGIN( ME_INTEGER ); ... TPROF_END( ME_INTEGER );   // explicit pair
      TPROF_COUNT( AMVR_TEST_VALID );     // event counter, += 1
      TPROF_ADD  ( AMVR_TEST_VALID, n );  // event counter, += n
      TPROF_HIST ( AMVR_TEST_PRECISION, cu.imv );   // histogram bin, += 1
    \endcode

    Do not put side effects into the macro arguments: they are not evaluated when the profiler is
    disabled.
*/

#ifndef __TIMEPROFILER__
#define __TIMEPROFILER__

#include <cstddef>
#include <cstdint>

// ENABLE_TIME_PROFILING is normally injected by CMake. Default to "off" so that a compilation unit
// pulled into a build that does not go through the project CMakeLists stays unaffected.
#ifndef ENABLE_TIME_PROFILING
#define ENABLE_TIME_PROFILING 0
#endif

// ====================================================================================================================
// Instrumentation points. Adding a stage/counter/histogram is a one-line change here.
// ====================================================================================================================

// X( id, reportDepth, label )
//   reportDepth only controls the indentation of the console report, it does not influence the
//   measurement itself (nesting is tracked at run time).
#define TPROF_STAGE_LIST(X)                                                                                            \
  X(ENCODER, 0, "Encoder (total)")                                                                                     \
  X(PIC_COMPRESS, 1, "Picture: CU compression (compressSlice)")                                                        \
  X(CTU_COMPRESS, 2, "CTU/CU: xCompressCU")                                                                            \
  X(MODE_SPLIT, 3, "CU: split RD test")                                                                                \
  X(MODE_INTER_ME, 3, "CU: inter ME mode test")                                                                        \
  X(MODE_INTER_NOAMVR, 4, "CU: inter, no AMVR")                                                                        \
  X(MODE_INTER_AMVR, 4, "CU: inter, AMVR (IMV)")                                                                       \
  X(AMVR_FPEL, 5, "AMVR trial: full-pel")                                                                              \
  X(AMVR_4PEL, 5, "AMVR trial: 4-pel")                                                                                 \
  X(AMVR_HPEL, 5, "AMVR trial: half-pel (alt. IF)")                                                                    \
  X(INTER_SEARCH, 5, "predInterSearch")                                                                                \
  X(ME_INTEGER, 6, "ME: integer search")                                                                               \
  X(ME_FRAC, 6, "ME: fractional refinement")                                                                           \
  X(ME_INT_REFINE, 6, "ME: integer MVP refinement (IMV)")                                                              \
  X(AFFINE_SEARCH, 6, "Affine: xPredAffineInterSearch")                                                                \
  X(AFFINE_ME, 7, "Affine: xAffineMotionEstimation")                                                                   \
  X(AFFINE_MERGE, 4, "CU: affine merge mode test")                                                                     \
  X(MODE_GEO, 3, "CU: GPM/GEO mode test")                                                                              \
  X(MODE_CIIP, 3, "CU: CIIP mode test")                                                                                \
  X(INTER_RESIDUAL, 5, "Inter: residual coding + RD")                                                                  \
  X(MODE_MERGE, 3, "CU: merge/skip mode test")                                                                         \
  X(MODE_HASH_INTER, 3, "CU: hash inter mode test")                                                                    \
  X(MODE_INTRA, 3, "CU: intra mode test")                                                                              \
  X(INTRA_EST_LUMA, 4, "Intra: luma mode estimation (SATD fast search)")                                              \
  X(INTRA_LUMA_QT, 4, "Intra: luma RD search")                                                                         \
  X(INTRA_CHROMA_QT, 4, "Intra: chroma RD search")                                                                     \
  X(INTRA_ISP, 4, "Intra: ISP sub-partition search")                                                                  \
  X(MODE_IBC, 3, "CU: IBC mode test")                                                                                  \
  X(IBC_SEARCH, 4, "IBC: block vector search")                                                                         \
  X(MODE_PALETTE, 3, "CU: palette mode test")                                                                          \
  X(TRANSFORM_QUANT, 4, "Transform + Quantization (xTransform/xQuant)")                                              \
  X(PIC_DEBLOCK, 1, "Picture: deblocking filter")                                                                      \
  X(SAO_SEARCH, 2, "Picture: SAO search")                                                                              \
  X(PIC_SAO, 1, "Picture: SAO execution")                                                                              \
  X(ALF_SEARCH, 2, "Picture: ALF search")                                                                              \
  X(PIC_ALF, 1, "Picture: ALF execution")                                                                              \
  X(PIC_CCALF, 1, "Picture: CCALF execution")                                                                          \
  X(PIC_LMCS, 1, "Picture: LMCS (Reshaper)")                                                                           \
  X(PIC_ENTROPY, 1, "Picture: entropy coding (encodeSlice)")

// X( id, label )
#define TPROF_COUNTER_LIST(X)                                                                                          \
  X(CU_COUNT, "Total CUs evaluated")                                                                                   \
  X(CU_SPLIT_COUNT, "Total split decisions tested")                                                                    \
  X(AMVR_TEST_HPEL_SKIPPED, "AMVR half-pel trials skipped by the early-exit heuristic")                                 \
  X(AMVR_TEST_VALID, "AMVR trials that produced a usable mode")                                                         \
  X(AMVR_TEST_ZERO_MVD_EXIT, "AMVR trials abandoned because all MVDs were zero")                                        \
  X(AMVR_AFFINE_ENABLED, "AMVR trials in which affine AMVR was enabled")                                                \
  X(AMVR_HPEL_TRIALS, "AMVR half-pel trial count")                                                                     \
  X(AMVR_FPEL_TRIALS, "AMVR full-pel trial count")                                                                     \
  X(AMVR_4PEL_TRIALS, "AMVR 4-pel trial count")                                                                        \
  X(ME_UNI, "Uni-prediction motion estimations")                                                                        \
  X(ME_BI, "Bi-prediction motion estimations")                                                                          \
  X(AFFINE_ME_UNI, "Uni-prediction affine motion estimations")                                                          \
  X(AFFINE_ME_BI, "Bi-prediction affine motion estimations")                                                            \
  X(AFFINE_MERGE_COUNT, "Affine merge mode trials")                                                                    \
  X(GEO_TEST_COUNT, "GPM/GEO mode trials")                                                                             \
  X(CIIP_TEST_COUNT, "CIIP mode trials")                                                                               \
  X(INTER_SKIP_COUNT, "Inter skip mode selected")                                                                      \
  X(TRANSFORM_SKIP_COUNT, "Transform skip mode selected")                                                              \
  X(MTS_TEST_COUNT, "MTS transform candidates tested")                                                                 \
  X(LFNST_TEST_COUNT, "LFNST candidates tested")                                                                       \
  X(INTRA_MPM_COUNT, "Intra MPM candidates selected")

// X( id, label ) - each histogram has TPROF_HIST_BINS bins, out-of-range bins are clamped.
#define TPROF_HIST_LIST(X)                                                                                             \
  X(AMVR_TEST_PRECISION, "AMVR RD trials per cu.imv value (0=IMV_OFF, 1=IMV_FPEL, 2=IMV_4PEL, 3=IMV_HPEL)")             \
  X(INTER_CU_LOG2AREA, "log2(W*H) of CUs entering the inter ME mode test")                                              \
  X(AFFINE_CU_LOG2AREA, "log2(W*H) of CUs entering the affine search")                                                  \
  X(CU_DEPTH, "CU depth level distribution")                                                                           \
  X(TRANSFORM_SIZE_LOG2, "log2 size of transform blocks evaluated")                                                     \
  X(INTRA_LUMA_MODE, "Selected Intra Luma mode distribution (0..66)")                                                  \
  X(MTS_IDX, "Selected MTS transform index distribution")

// ====================================================================================================================
// Public API
// ====================================================================================================================

#if ENABLE_TIME_PROFILING

#include <chrono>

#if defined(_MSC_VER)
#define TPROF_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define TPROF_INLINE inline __attribute__((always_inline))
#else
#define TPROF_INLINE inline
#endif

// Pick the time source. The TSC path is roughly 3-4x cheaper than steady_clock and is the default
// on x86; -DTPROF_USE_TSC=0 falls back to std::chrono::steady_clock everywhere.
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#define TPROF_X86 1
#else
#define TPROF_X86 0
#endif

#ifndef TPROF_USE_TSC
#define TPROF_USE_TSC TPROF_X86
#endif

#if TPROF_USE_TSC && !TPROF_X86
// The build system asked for the TSC on a target that has none.
#undef TPROF_USE_TSC
#define TPROF_USE_TSC 0
#endif

#ifndef TPROF_CHECKS
#define TPROF_CHECKS 0
#endif

#if TPROF_USE_TSC
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

//! \ingroup CommonLib
//! \{

namespace tprof
{
  enum Stage : int
  {
#define TPROF_MAKE_STAGE_ENUM(id, depth, label) id,
    TPROF_STAGE_LIST(TPROF_MAKE_STAGE_ENUM)
#undef TPROF_MAKE_STAGE_ENUM
      NUM_STAGES,
    // internal, never reported as a pipeline stage: used to self-measure the instrumentation cost
    PROBE_STAGE = NUM_STAGES,
    NUM_STAGE_SLOTS
  };

  enum Counter : int
  {
#define TPROF_MAKE_COUNTER_ENUM(id, label) id,
    TPROF_COUNTER_LIST(TPROF_MAKE_COUNTER_ENUM)
#undef TPROF_MAKE_COUNTER_ENUM
      NUM_COUNTERS
  };

  enum Hist : int
  {
#define TPROF_MAKE_HIST_ENUM(id, label) HIST_##id,
    TPROF_HIST_LIST(TPROF_MAKE_HIST_ENUM)
#undef TPROF_MAKE_HIST_ENUM
      NUM_HISTS
  };

  static constexpr int MAX_STACK_DEPTH = 128;
  static constexpr int HIST_BINS       = 16;

  struct StageAcc
  {
    uint64_t calls;         ///< number of completed invocations, recursive ones included
    uint64_t outerCalls;    ///< number of completed outermost invocations
    uint64_t inclTicks;     ///< inclusive time, counted once per outermost invocation
    uint64_t selfTicks;     ///< inclusive time minus the time spent in nested stages
    uint64_t minTicks;      ///< shortest outermost invocation
    uint64_t maxTicks;      ///< longest outermost invocation
    double   sumSqTicks;    ///< sum of squared outermost invocation lengths (for the std deviation)
    uint32_t active;        ///< current recursion depth of this stage
    uint32_t maxActive;     ///< deepest recursion observed
  };

  /// Per-thread accumulators. Heap allocated and owned by the global registry, so the data outlives
  /// the thread it belongs to and stays valid until report()/destroy().
  struct alignas(64) ThreadState
  {
    int      depth;                            ///< current nesting depth, 0 == nothing running
    uint16_t stkStage[MAX_STACK_DEPTH];        ///< stage id per nesting level (diagnostics only)
    uint64_t stkStart[MAX_STACK_DEPTH];        ///< entry timestamp per nesting level
    uint64_t stkChild[MAX_STACK_DEPTH];        ///< time spent in children per nesting level
    StageAcc acc[NUM_STAGE_SLOTS];
    uint64_t counters[NUM_COUNTERS];
    uint64_t hists[NUM_HISTS][HIST_BINS];
    uint64_t overflows;                        ///< begin() calls dropped because the stack was full
    uint64_t mismatches;                       ///< unbalanced begin/end pairs seen (TPROF_CHECKS)
    ThreadState *next;                         ///< registry chain
    uint64_t     threadId;
  };

  /// Current thread's accumulators, nullptr until the thread touches the profiler for the first time.
  extern thread_local ThreadState *g_state;

  /// Cold path: allocates and registers the accumulators of the calling thread.
  ThreadState *createThreadState();

  static TPROF_INLINE ThreadState &state()
  {
    ThreadState *s = g_state;
#if defined(__GNUC__) || defined(__clang__)
    if (__builtin_expect(s == nullptr, 0))
#else
    if (s == nullptr)
#endif
    {
      s = createThreadState();
    }
    return *s;
  }

  static TPROF_INLINE uint64_t readTicks()
  {
#if TPROF_USE_TSC
    return (uint64_t) __rdtsc();
#else
    return (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count();
#endif
  }

  static TPROF_INLINE void begin(Stage s)
  {
    ThreadState &t = state();
    const int    d = t.depth + 1;
    t.depth        = d;
    if (d >= MAX_STACK_DEPTH)
    {
      // Cannot happen with the stage set below, but never write past the stack.
      t.overflows++;
      return;
    }
    StageAcc &a = t.acc[s];
    a.active++;
    if (a.active > a.maxActive)
    {
      a.maxActive = a.active;
    }
    t.stkStage[d] = (uint16_t) s;
    t.stkChild[d] = 0;
    t.stkStart[d] = readTicks();   // read the clock last, so the bookkeeping above is not measured
  }

  static TPROF_INLINE void end(Stage s)
  {
    const uint64_t stop = readTicks();   // read the clock first, for the same reason
    ThreadState   &t    = state();
    const int      d    = t.depth;
    t.depth             = d - 1;
    if (d >= MAX_STACK_DEPTH)
    {
      return;
    }
#if TPROF_CHECKS
    if (d <= 0 || t.stkStage[d] != (uint16_t) s)
    {
      t.mismatches++;
      t.depth = d > 0 ? d - 1 : 0;
      return;
    }
#endif
    const uint64_t elapsed = stop - t.stkStart[d];
    StageAcc      &a       = t.acc[s];
    a.calls++;
    a.selfTicks += elapsed - t.stkChild[d];
    if (--a.active == 0)
    {
      // Only the outermost invocation contributes to the inclusive time and the per-call statistics,
      // otherwise a recursive stage would count the same wall-clock interval several times.
      a.outerCalls++;
      a.inclTicks += elapsed;
      a.minTicks = elapsed < a.minTicks ? elapsed : a.minTicks;
      a.maxTicks = elapsed > a.maxTicks ? elapsed : a.maxTicks;
      a.sumSqTicks += (double) elapsed * (double) elapsed;
    }
    t.stkChild[d - 1] += elapsed;
  }

  static TPROF_INLINE void addCounter(Counter c, uint64_t n) { state().counters[c] += n; }

  static TPROF_INLINE void addHist(Hist h, int bin)
  {
    const int b = bin < 0 ? 0 : (bin >= HIST_BINS ? HIST_BINS - 1 : bin);
    state().hists[h][b]++;
  }

  /// RAII stage guard. Ends the stage on every exit path, including early returns and exceptions.
  class ScopedStage
  {
  public:
    explicit ScopedStage(Stage s) : m_stage(s) { begin(s); }
    ~ScopedStage() { end(m_stage); }
    ScopedStage(const ScopedStage &)            = delete;
    ScopedStage &operator=(const ScopedStage &) = delete;

  private:
    Stage m_stage;
  };

  /// RAII picture/slice guard. Snapshots all accumulators on entry and stores the deltas on exit,
  /// which yields the per-picture time series written to the frame CSV.
  class ScopedPicture
  {
  public:
    ScopedPicture(int poc, int sliceType, int qp, int tLayer, bool trialPass)
    {
      beginPicture(poc, sliceType, qp, tLayer, trialPass);
    }
    ~ScopedPicture() { endPicture(); }
    ScopedPicture(const ScopedPicture &)            = delete;
    ScopedPicture &operator=(const ScopedPicture &) = delete;

    static void beginPicture(int poc, int sliceType, int qp, int tLayer, bool trialPass);
    static void endPicture();
  };
}   // namespace tprof

/// Facade kept deliberately close to the plain "start/stop/report" style, for call sites that
/// cannot use the RAII macros.
class TimeProfiler
{
public:
  /// Initialises the accumulators, calibrates the clock and measures the instrumentation cost.
  /// Safe to call more than once (subsequent calls are ignored).
  static void init();
  /// Prints the console report and writes the CSV files selected through the environment.
  static void report();
  /// Drops every accumulated sample but keeps the calibration.
  static void reset();
  /// Releases the per-thread accumulators. Called implicitly at exit.
  static void destroy();

  static void start(tprof::Stage s) { tprof::begin(s); }
  static void stop(tprof::Stage s) { tprof::end(s); }
};

//! \}

#define TPROF_CONCAT_(a, b) a##b
#define TPROF_CONCAT(a, b) TPROF_CONCAT_(a, b)

#define TPROF_INIT() TimeProfiler::init()
#define TPROF_REPORT() TimeProfiler::report()
#define TPROF_SCOPE(stage) ::tprof::ScopedStage TPROF_CONCAT(tprofScope_, __COUNTER__)(::tprof::stage)
#define TPROF_SCOPE_ID(stageExpr) ::tprof::ScopedStage TPROF_CONCAT(tprofScope_, __COUNTER__)(stageExpr)
#define TPROF_BEGIN(stage) ::tprof::begin(::tprof::stage)
#define TPROF_END(stage) ::tprof::end(::tprof::stage)
#define TPROF_COUNT(counter) ::tprof::addCounter(::tprof::counter, 1)
#define TPROF_ADD(counter, n) ::tprof::addCounter(::tprof::counter, (uint64_t) (n))
#define TPROF_HIST(hist, bin) ::tprof::addHist(::tprof::HIST_##hist, (int) (bin))
#define TPROF_PICTURE_SCOPE(poc, sliceType, qp, tLayer, trialPass)                                                     \
  ::tprof::ScopedPicture TPROF_CONCAT(tprofPic_, __COUNTER__)(poc, sliceType, qp, tLayer, trialPass)

#else   // !ENABLE_TIME_PROFILING

#define TPROF_INIT()                                                                                                   \
  do                                                                                                                   \
  {                                                                                                                    \
  } while (0)
#define TPROF_REPORT()                                                                                                 \
  do                                                                                                                   \
  {                                                                                                                    \
  } while (0)
#define TPROF_SCOPE(stage)                                                                                             \
  do                                                                                                                   \
  {                                                                                                                    \
  } while (0)
#define TPROF_SCOPE_ID(stageExpr)                                                                                      \
  do                                                                                                                   \
  {                                                                                                                    \
  } while (0)
#define TPROF_BEGIN(stage)                                                                                             \
  do                                                                                                                   \
  {                                                                                                                    \
  } while (0)
#define TPROF_END(stage)                                                                                               \
  do                                                                                                                   \
  {                                                                                                                    \
  } while (0)
#define TPROF_COUNT(counter)                                                                                           \
  do                                                                                                                   \
  {                                                                                                                    \
  } while (0)
#define TPROF_ADD(counter, n)                                                                                          \
  do                                                                                                                   \
  {                                                                                                                    \
  } while (0)
#define TPROF_HIST(hist, bin)                                                                                          \
  do                                                                                                                   \
  {                                                                                                                    \
  } while (0)
#define TPROF_PICTURE_SCOPE(poc, sliceType, qp, tLayer, trialPass)                                                     \
  do                                                                                                                   \
  {                                                                                                                    \
  } while (0)

#endif   // ENABLE_TIME_PROFILING

#endif   // __TIMEPROFILER__
