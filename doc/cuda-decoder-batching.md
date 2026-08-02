# Decoder batching profiler

## Scope, build isolation, and runtime gate

This is a measurement probe for possible future decoder GPU batching. It does not add a decoder-inter CUDA kernel,
dispatch, or command-line switch, and it makes no viability decision by itself.

The complete implementation is removed at preprocessing time unless the dedicated CMake option is enabled. Normal
builds therefore contain no profiler state, hot-path guards, environment lookup, output string, or profiler symbol.
Use separate out-of-source trees:

```text
cmake -S . -B build-decoder-normal -DVTM_ENABLE_DECODER_BATCH_PROFILING=OFF
cmake --build build-decoder-normal --config Release --target DecoderApp EncoderApp --clean-first

cmake -S . -B build-decoder-profile -DVTM_ENABLE_DECODER_BATCH_PROFILING=ON
cmake --build build-decoder-profile --config Release --target DecoderApp --clean-first
```

Profiling builds place all archives, libraries, executables, and debug artifacts in output directories suffixed with
`-decoder-batch-profile`; this prevents a profiling `DecoderLib` from being linked into a normal executable. The
compile definition is public only in the profiling tree because `DecCu` is embedded by value in `DecLib` and every
consumer must see the same class layout.

Even in a profiling build, collection is disabled unless `VTM_DECODER_BATCH_PROFILE=1` is present. Output is one
stable line of JSON prefixed with `DECODER_BATCH_PROFILE`. Schema 6 identifies the process, decoder instance, actual
owner and reporting threads, hook count, thread-mismatch count, first/last POC, and picture count. Every hook locks
the per-instance state and compares the real `std::thread::id` with the first hook owner, so a handoff is both race-free
and visible. The complete JSON line is assembled in memory and emitted with one `fwrite` while holding a process-wide
output mutex, preventing two decoder instances from interleaving records. The profile owns its state with
`std::unique_ptr`. Fixed logarithmic histograms retain run counts, maxima, and p50/p90/p99 upper bounds in constant
memory independent of sequence length.

Every `DecoderApp` build also writes `DecoderApp.exe.build-manifest.json` after linking. It binds the executable
SHA-256 to the profiling option, configuration, and `CMakeCache.txt` SHA-256. This is deliberately outside the
executable: normal binaries still contain no profiler runtime strings or symbols.

## Effective motion-compensation paths

Classification occurs after `xDeriveCuMvs`. An eligible CU is not added to the modeled queue until `xReconInter` has
passed its immediate reconstructed-neighbour consumers; this keeps the current CU out of the batch that must finish
for those consumers. A task is one luma CU. Runs contain consecutive CUs that execute the same effective MC path, not
merely the same `interDir`:

- ordinary uni prediction;
- the `xCheckIdenticalMotion` nominal-bi fast path, which executes one L0 prediction;
- uni weighted prediction;
- ordinary bi average;
- bi weighted prediction;
- bi coding-weight prediction (`BCW`);
- a separate RPR variant of every path above.

The classifier mirrors the decoder conditions for identical motion, `UseWP`/`WPBiPred`, `BCW_DEFAULT`, scaled
references, DMVR, and BDOF. IBC is tested before the generic non-inter case; compile-time assertions specifically
prove that an IBC CU maps to `ibc`, not `intra_or_plt`. GPM, affine/PROF, CIIP, sub-PU MC, DMVR, BDOF, IBC, intra, and palette paths are reported
as exclusions rather than being folded into uni/bi. A CU whose PUs require different effective paths is also
reported separately.

Runs are not flushed at CTU or row boundaries. In the current decoder one `DecCu` executes CTUs sequentially and its
prediction storage is picture-backed, so those boundaries do not establish a pixel dependency. HMVP is likewise not
a GPU wait: `CU::saveMotionForHmvp` consumes the already decoded motion fields and can remain a CPU state update while
an eventual GPU MC batch is outstanding. MC runs end at an effective-path change, an incompatible tool, a true
reconstruction consumer (intra, IBC, or CIIP), a picture boundary, or stream end. A future threaded/WPP decoder must
use one profiler/queue per worker or add an explicit worker-handoff boundary; thread identity is recorded so such
mixing is detectable.

The two calls to `calculateChromaAdjVpduNei` reachable from `xReconInter` may consume already reconstructed
neighbouring luma. `Reshape::chromaAdjVpduReadsLuma` reproduces the function's exact decoder cache predicate: the
initial `(-1,-1)` sentinel misses, coordinates are quantized to 64 samples for a 128-sample CTU and otherwise to the
CTU size, a second TU/CU in the same VPDU hits, a VPDU transition misses, and encoder calls never use the decoder
cache. Only a real cache miss flushes outstanding MC and inverse/reconstruction work with reason
`lmcs_chroma_adj`. Because the current CU is queued only after `xReconInter` returns, it is not accidentally included
on the producer side of that dependency. Compile-time focal assertions cover the sentinel, multiple positions in one
VPDU, the 64-sample transition, and encoder behaviour. The other two decoder calls are in intra reconstruction; the
existing `intra_or_plt` dependency flush happens before entering those functions. CIIP and IBC remain explicit
immediate consumer/exclusion boundaries.

## Future IBC-buffer scheduler contract

`xFillIBCBuffer` currently copies every reconstructed CU into the circular IBC buffer immediately. A future GPU MC
queue cannot keep doing that: an enqueued CU's reconstructed pixels are not available until its batch completes. The
profiler therefore models one pending fill for each eligible queued CU while `SPS::getIBCFlag()` is true. Pending fills
accumulate across same-path CUs and are applied in original sequence when the MC batch is completed; they do not force
a flush for each CU. Path/tool/picture/stream completion naturally commits them. Additionally, the model flushes and
checks for zero pending fills before `xDeriveCuMvs` on an IBC CU with reason `ibc_pre_mv_consumer`, before
`resetIBCBuffer`, and before every `resetVPDUforIBC` invalidation. This pre-MV point matters because IBC motion
derivation may consume IBC state before `xReconInter`. The later prepare/classification still records the CU as
`ibc`, but first checks the invariant that pending fills are already zero and therefore does not perform a second
effective flush. The JSON records queued/applied counts, maximum pending depth, sequence numbers, boundary and prepare
checks/violations, and pending-at-report; the runner rejects any order or boundary violation.

The fused scheduler has a separate deferred-fill queue for each capability core, independent of whether the older
same-path model queued that CU. Every fused-eligible CU records exactly one post-reconstruction fill, including a CU
whose PUs use mixed supported paths. At every consumer/reset/picture/stream boundary, each core first closes its
window and replays its fills in CU order. An RPR CU is immediate for Core A because that core rejects it, while the
same fill is deferred for Core B + RPR. Schema 6 exposes queued/applied counts, sequence numbers, maximum pending
depth, immediate fills, disabled-SPS no-ops, boundary checks, violations, and pending fills separately for both cores.

When `SPS::getIBCFlag()` is false, these buffer writes cannot be consumed by IBC and are modeled as semantically
unobservable no-ops. The same exact predicate suppresses reset boundaries, avoiding false batch fragmentation. A
compile-time model assertion enqueues two fills without an intervening flush, completes both in order, checks the
consumer/reset invariant, and checks the disabled-SPS no-op case. A focused source-integration test fixes the required
order as pre-MV hook, `xDeriveCuMvs`, then prepare/classification.

This is a scheduler model, not executable GPU batching. A real integration must defer the physical
`xFillIBCBuffer` copies until batch completion and then replay them in CU order before any listed consumer/reset. The
current CPU decoder still executes its physical fill immediately after each CU.

The same probe records actual nonzero inter `invTransformNxN` component calls and ordinary inter reconstruction CUs.
Those distributions also cross CTUs and rows, but are flushed before consumers that require reconstructed samples and
at picture boundaries. IBC operations are excluded from these candidate streams.

## Fused MC + residual + reconstruction windows

Schema 6 adds a second, independent scheduler model for one future heterogeneous dispatch containing motion
compensation, dequantization/inverse transform, and reconstruction. It does not change the legacy per-path and
per-transform-key measurements. A fused CU candidate is opened after motion derivation, accumulates every logical TU
and every normative component `invTransformNxN` call during `xReconInter`, and is committed only after reconstruction.
If any PU or TU is unsupported, the previous window is closed and the entire current CU is rejected before any of its
work is counted. There is no partial-CU coverage.

The fused model deliberately does **not** close a window when uni/identical/weighted/bi-average/BCW mode changes, when
RPR status changes in Core B, or when component, transform width/height, or supported transform type changes. Regular
DCT2 and transform skip can coexist in one window. It closes only for a real reconstructed-pixel dependency, an
unsupported feature, a POC boundary, or stream end. The already-audited IBC pre-MV/deferred-fill/reset points and an
actual LMCS VPDU-cache miss remain hard boundaries. Intra/palette, IBC, GPM, affine/PROF, CIIP, sub-PU, DMVR, BDOF,
invalid DPB references, MTS/other transforms, scaling lists, LFNST, SBT, joint CbCr, ACT, and LMCS chroma-residual
scaling are explicit rejection reasons.

Two capability sets are measured concurrently:

- **Core A** accepts heterogeneous uni, identical-uni, weighted uni/bi, bi-average, and BCW plus regular DCT2 and
  transform skip, but rejects RPR and all exclusions above.
- **Core B + RPR** is identical except that scaled-reference prediction is accepted. It still does not model the
  other excluded tools.

Each completed window contributes constant-memory logarithmic histograms for logical CUs, logical TUs (one count at
the luma block of each inter TU), luma and all-component pixels, reference prediction operations, normative
component-transform tasks, descriptor bytes, quantized-coefficient bytes, dirty-boundary downloads, and total
transfer bytes. The JSON retains all 65 histogram buckets, totals, maximum, and p50/p90/p99 upper bounds, allowing the
runner to merge independent processes without inventing samples. Reference-mode totals and the number of windows
containing each mode are recorded separately, as are coverage, rejection, and flush reasons.

The transfer model assumes decoded references and the output `Picture` mirror are already resident. Recurring H2D is
therefore limited to explicit measurement-only POD descriptors and dense `TCoeff` payloads. Each prediction-operation
descriptor carries destination/reference plane offsets and strides, destination position, motion vector, refIdx,
dimensions, reference list, component/combine/interpolation flags, BCW index, and effective weights/offsets. Shared
metadata is deduplicated once by actual identity and values: output-picture layout, slice/reference mapping, weighted
prediction parameters, RPR scale and current/reference scaling windows, and BCW weights. WP/RPR/BCW therefore have
explicit nonzero costs instead of being claimed as free. Shared-once H2D also includes grouped scan elements for each
observed shape, one inverse-DCT2 matrix for each observed 1-D size, and the flat inverse-quant constants. These tables
are amortized over the observed windows, and all byte additions/multiplications are overflow-checked. When a CPU dependency or final
consumer ends a window, the model conservatively charges all dirty output components in that window at `sizeof(Pel)`
as D2H. A real range-aware mirror could download less; the report keeps this conservative charge visible instead of
hiding it in a descriptor-only estimate. The POD sizes and element sizes are versioned in each record and are not a
promised kernel ABI.

## Reproducible residual-bearing matrix

`tools/cuda/run_decoder_batch_profile.py` generates deterministic moving, textured 4:2:0 inputs at 8 and 10 bits,
encodes RA and LD streams with at least four pictures, and rejects any case that has no nonzero inter-CBF inverse
transform work. It hashes the runner, executables, CMake caches, configs, inputs, bitstreams, encoder reconstructions,
and decoder outputs; saves every command and stdout/stderr log; and requires byte-identical output from:

1. the normal build;
2. the profiling build with its runtime gate off; and
3. the profiling build with collection on.

The runner requires distinct normal/profiling executable paths and distinct CMake build roots. It verifies both the
OFF/ON cache values and the adjacent post-link manifests, including executable and cache hashes. Consequently a
profiling executable copied or swapped into an OFF build is rejected even if its path and cache appear normal.
`--dry-run` performs those checks, hashes/generates the deterministic inputs, and writes every planned command without
launching the codec. Example:

```text
python tools/cuda/run_decoder_batch_profile.py \
  --encoder <normal-EncoderApp> --decoder-normal <normal-DecoderApp> \
  --decoder-profile <profiling-DecoderApp> \
  --normal-build-dir <normal-build> --profile-build-dir <profiling-build> \
  --cfg-dir cfg --output-dir <measurement-output> \
  --width 256 --height 144 --frames 4 --qp 22
```

AI is intentionally not a matrix row because it has no inter-prediction workload. AI decoding remains covered by the
normal codec tests; it is not evidence for or against decoder inter batching.

The local reproducible run used 256x144, four pictures, and QP 22. Every row contained nonzero inter-CBF work, and
the normal, profile-off, and profile-on decoded outputs had the same SHA-256 within that row. `MC max` is the maximum
over all effective-path classes; the versioned report retains every path and histogram separately.

| Profile/depth | Eligible luma | MC max tasks / pixels | Inverse-transform runs / max tasks / pixels | Reconstruction runs / max tasks / pixels | Decoded SHA-256 |
|---|---:|---:|---:|---:|---|
| RA 8 | 98,304 / 147,456 | 54 / 5,632 | 104 / 207 / 13,584 | 104 / 81 / 9,472 | `7A485040E42A83EB78A4CAE48AAF7FCB5DBB317FCEC3E2513E751FC2A5504070` |
| RA 10 | 85,632 / 147,456 | 27 / 5,504 | 96 / 162 / 13,048 | 96 / 56 / 8,768 | `D56AE5E8C2DF5288A4D9606E2A4F9EA786809CAE2E9AF7B599DD34240318AA32` |
| LD 8 | 102,208 / 147,456 | 31 / 13,824 | 30 / 315 / 32,992 | 30 / 108 / 22,144 | `C1DF016AFADFD3D794A823461AA6D4A44F7003EBC21758EA94498434D9418359` |
| LD 10 | 101,312 / 147,456 | 23 / 6,144 | 69 / 368 / 32,096 | 69 / 137 / 22,720 | `4600247A067AC90EC52D26526FDF810AC82278379D1A8020A930303581FBD05E` |

These measurements supersede the earlier CTU-local synthetic probe. In particular, inverse-transform work is now
present and runs cross CTU/row boundaries. They are useful workload distributions, not a launch threshold or a
performance result.

The four rows above were redecoded from their existing bitstreams after the IBC/LMCS/thread-safety corrections; no
re-encode was used. Their SPSs disable IBC, so observable queued/applied fills and IBC boundary checks are zero; the
model instead counted 821, 609, 542, and 603 eligible disabled-SPS no-ops for RA8, RA10, LD8, and LD10. No
`lmcs_chroma_adj` event occurred either. Consequently the final MC, inverse-transform, and reconstruction numbers in
the table did not change from the preceding corrected report. An intermediate measurement that flushed on
unobservable disabled-SPS buffer resets produced smaller runs and was discarded. IBC pending-fill and LMCS cache
boundary coverage is therefore static/focal in this corpus, not dynamic codec coverage. The runner additionally
launched two profiling decoder processes concurrently on RA 8.
Both complete schema-6 records parsed independently, their `(process_id, decoder_instance)` identities were distinct,
their owner-thread mismatch counters were zero, and both decoded hashes matched the RA 8 hash above.

### Transform/dequant coverage preflight

Schema 6 retains profiling of each physical inter `invTransformNxN` call by component, transform size, effective transform,
dequant path, QP, coefficient count, nonzero coefficient count, and the relevant fallback features. It also records
CBF/non-CBF block counts and potential per-key batches inside each existing reconstruction-dependency window. This
instrumentation remains compile-time gated and is absent from the normal build.

Across the four streams, 7,683 inverse transforms covered 621,160 pixels and contained 357,592 nonzero coefficients.
DCT2 accounted for 3,546 tasks (46.15%) and 454,328 pixels (73.14%); transform skip accounted for 4,137 tasks
(53.85%) and 166,832 pixels (26.86%). After excluding joint CbCr, 3,528 tasks and 453,976 pixels remain in the
regular-DCT2 prototype class. All DCT2 work used dependent quantization with flat scaling; all transform-skip work
used scalar dequantization with flat scaling. These streams contained no MTS, LFNST, SBT, ACT, LMCS-residual-scaling,
or scaling-list transform calls; joint CbCr accounted for 29 calls and 600 pixels. Observed transform QPs were 38-42.

Square regular-DCT2 sizes 4, 8, 16, and 32 together cover only 718 tasks and 97,456 pixels: 20.35% and 21.47% of the
regular-DCT2 class, or 9.35% and 15.69% of all inverse-transform work. They therefore cannot satisfy a 50% coverage
target. The minimum balanced six-shape set found in this corpus is `8x8`, `4x16`, `8x16`, `16x8`, `32x8`, and
`16x16`; it covers 1,949 tasks (55.24%) and 246,912 pixels (54.39%) of the regular-DCT2 class. This is still only
25.37% of all inverse-transform tasks because transform skip is the majority task path.

The candidate groups are not naturally large at the current dependency boundaries: per component/shape/dequant key,
the observed 90th-percentile upper bound is at most three tasks and the maximum is four. A DCT2-only kernel with a
CPU transform-skip fallback would therefore flush too frequently to claim a productive batched GPU unit from this
corpus. The implementation decision must either include transform skip in the first GPU unit, widen/restructure the
dependency window with proven semantics, or obtain a different representative corpus before selecting DCT2-only
kernels.

## Decision status

No fixed task/pixel threshold is used. Kernel launch cost, descriptor preparation, transfers, residency, and overlap
must be benchmarked with a prototype before a break-even point can be justified. The corrected distributions are
reported as measurements only. No decoder-inter kernel is implemented in this change; implementation or rejection is
deferred until a prototype supplies a kernel cost model for the measured residual-bearing RA/LD 8/10 workloads.

### Schema-6 fused-window measurement

The residual-bearing 256x144, four-picture corpus was decoded again after the fused-window instrumentation. In every
row the normal executable, profiling executable with collection disabled, and profiling executable with collection
enabled produced the same decoded SHA-256 shown in the earlier table. Values below are in `p50/p90/p99/max` order:
the three percentiles are upper bounds of logarithmic buckets, while `max` is exact. Consequently a percentile upper
bound can be greater than the exact maximum. `TU` is a logical inter TU; `transform` is a nonzero normative component
transform.

| Row | Eligible / considered CUs | Windows | CU | TU | Component pixels | Prediction ops | Transform tasks | Estimated transfer/run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RA 8 | 811 / 1,372 | 98 | 7/31/63/56 | 7/31/63/56 | 1,023/8,191/16,383/10,464 | 7/31/127/88 | 15/63/255/145 | 1,049,460 B |
| RA 10 | 604 / 1,213 | 103 | 3/15/63/55 | 3/15/63/55 | 511/4,095/16,383/13,056 | 7/31/127/107 | 7/63/127/159 | 934,532 B |
| LD 8 | 540 / 1,050 | 53 | 7/31/63/52 | 7/31/63/52 | 2,047/16,383/32,767/20,928 | 15/63/127/75 | 15/127/255/141 | 1,087,744 B |
| LD 10 | 594 / 1,153 | 76 | 7/31/63/56 | 7/31/63/56 | 1,023/8,191/16,383/12,960 | 7/63/127/104 | 15/63/255/165 | 1,099,528 B |

Across the four independent runs there were 330 windows and 2,549 eligible of 4,788 considered CUs. Recurring
descriptor H2D was 686,084 bytes, qcoeff H2D was 2,194,176 bytes, per-run shared tables and metadata summed to
134,492 bytes, and the conservative dirty-boundary D2H charge was 1,156,512 bytes, for 4,171,264 estimated transfer bytes across the
four runs. Rejections were: 1,797 intra/palette CUs, 99 GPM, 4 affine/PROF, 263 CIIP, 50 BDOF, and 26 joint-CbCr.
The runner-merged histogram upper bounds/exact maxima were: CU and TU `7/31/63/56`, luma pixels
`1,023/4,095/16,383/13,952`, component pixels `1,023/8,191/16,383/20,928`, prediction operations
`7/31/127/107`, and component-transform tasks `15/63/255/165`. Recurring per-window transfer (descriptors, qcoeff,
and dirty D2H, excluding shared-once tables) was `8,191/32,767/131,071/128,244` bytes. The report's aggregate was
computed by merging the emitted 65-bucket histograms, not by averaging per-row percentiles.

The schema-6 runner recomputes rejection/coverage sums, flush/window sums, metric/transfer correspondence,
shared-byte decomposition, every histogram quantile, exact-maximum bucket membership, and per-core IBC ordering. It
rejects older schema-5 records. Before collection begins, an isolated heap-owned executable self-test exercises
uni-to-bi batching, a mixed multi-PU CU, independent fused IBC fills, RPR Core A/Core B handling including a first CU
after a POC boundary, WP/RPR incremental metadata costs, and whole-CU discard after a supported then unsupported TU.
Failure aborts profiler construction; successful tests do not change the decoder instance or any emitted counters.

The corpus contains no RPR, so Core A and Core B + RPR are numerically identical here; this run does not dynamically
measure the incremental coverage of RPR. It also contains no weighted-prediction mode, IBC consumer, LMCS cache miss,
scaling list, MTS, LFNST, SBT, or ACT candidate. Those boundaries remain source/focal coverage, not corpus evidence.
The windows are materially larger than the earlier per-transform-key groups, but the measurement alone does not show
that a fused CUDA implementation is faster: kernel time, launch cost, overlap, and the conservative dirty-download
assumption remain unmeasured. The data supports prototyping one heterogeneous fused unit before selecting thresholds;
it does not support enabling such a path or claiming an end-to-end gain.
