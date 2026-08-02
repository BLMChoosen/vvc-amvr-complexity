# Experimental CUDA resident decoder loop-filter chain

## Scope and activation

`--GPUExperimentalLoopFilterChain=1` enables the experimental decoder path when `--GPUBackend=cuda` is selected. The option is deliberately **off by default**. Current end-to-end benchmarks do not pass the speed gate, so it must not be enabled by default yet.

The implementation keeps reconstructed luma resident on the device across the normative frame-level order:

1. inverse LMCS, when active;
2. vertical and horizontal deblocking;
3. SAO;
4. ALF.

Eligible pictures perform one reconstruction-mirror upload before the first active stage and one download after the last active stage. Intermediate stage data stays on the device. Disabled stages are omitted. A stage set that becomes empty after normative SAO/ALF parameter reconstruction returns `NoOp` before mirror lookup, dispatch, or transfer. A DBF-only picture with zero serialized luma edges is likewise a no-op.

The upload event is waited on by the stream of the first actual consumer. LMCS, non-empty DBF, and SAO begin on the DBF stream; ALF-only and zero-task-DBF-plus-ALF begin on the ALF stream. This is an event dependency and does not introduce a host synchronization.

Forward LMCS remains in the CU reconstruction path. The chain handles only frame-level inverse LMCS. Chroma filtering remains normative CPU work. The decoder returns immediately after a successful composite dispatch, so standalone GPU DBF/ALF integration paths do not also execute. Their reusable runtime primitives are called inside the composite dispatch, while standalone dispatch telemetry remains independent.

## Eligibility and unsupported syntax

The current path accepts 4:2:0 pictures with matching 8- or 10-bit luma/chroma depth, `Pel` sizes of 16 or 32 bits, at least 1920x1080 pixels, no more than 4096 CTUs, and exactly one slice, tile, and subpicture. Virtual boundaries and LADF are excluded. ALF requires luma dimensions divisible by four.

CCALF is rejected before any mirror mutation or device upload. This is required because normative CCALF reads the pre-ALF luma snapshot. Multi-slice/tile/subpicture boundary semantics and virtual boundaries must be implemented and tested before those pictures become eligible.

Builds with `ENABLE_TRACING=1` are also rejected before collection or picture mutation. The existing trace contract contains intermediate luma checkpoints; supporting those checkpoints on the resident path would require explicit diagnostic downloads after individual stages. Such builds therefore retain the complete CPU filter and trace sequence.

SAO syntax can reconstruct to an all-disabled luma picture even when chroma SAO remains active. In that case the SAO luma stage is removed after CTU collection instead of submitting an empty stage. Chroma SAO processing remains on the CPU.

All frame, stage, task, CTU, coefficient, clip, LUT, margin, and pointer/count contracts are validated before selection. DBF and SAO luma PODs are first collected without modifying any component. A pure backend preflight then validates the PODs and mirror. `NotEligible` falls through to the unmodified CPU pipeline exactly once. Only an eligible picture applies deferred CPU chroma filtering and starts CUDA work. An operational error after that selection is fatal for the process, quarantines the mirror as host-valid, recovers owned scratch, and permanently poisons only the composite chain for the context.

## Telemetry

Decoder shutdown reports composite dispatches and pixels, DBF tasks, SAO/ALF CTUs, parameter bytes, internal device copies, mirror upload/download bytes, current/retired/peak scratch, runtime/integration synchronizations, collection and per-stage times, total runtime/integration time, failures, preflight rejections, enabled state, poisoned state, and whether the feature was disabled by its flag. Chain scratch is a complete retained-memory total: composite-chain, reused DBF, and reused ALF current and retired allocations. Its peak is sampled from the combined allocations actually retained at the same time, rather than the sum of unrelated per-stage historical peaks.

For a single 1920x1080 eligible picture, the measured reconstruction mirror transfer was 8,266,752 bytes in each direction: one upload and one download including the allocated pitch/margins. Internal DBF/SAO/ALF transactional copies are reported separately and are not PCIe transfers.

## Correctness coverage

The focused CUDA test compares each stage against VTM normative references at 8 and 10 bits, including partial right/bottom CTUs, adjacent SAO CTUs with different EO/BO modes and offsets, non-identity inverse LMCS LUTs, non-identity DBF tasks, ALF classification/filtering, disabled CTUs, deterministic repeats, disabled-stage pointer/count contracts, empty-stage and DBF zero-task no-ops, CCALF preflight rejection, rollback, poisoning, and injected allocation/upload/launch/completion/download/commit/release failures. The stage-subset matrix covers LMCS-only, DBF-only, SAO-only, ALF-only, LMCS+ALF, DBF+ALF with both zero and non-zero DBF work, and SAO+ALF. Live allocation counters prove recovery retains failed releases and teardown later returns them to baseline. Separate growth checks prove the combined peak includes the simultaneously live old and replacement DBF/ALF scratch, rather than only the post-release current allocation.

`tools/cuda/run_acceptance_matrix.py` schema 3 automates the complete external 1080p/4K x AI/RA/LD x 8/10 CPU/CUDA matrix. Acceptance requires a hashed manifest, exactly one warm-up plus five measured runs, and a 288-process preflight. It validates actual raw-input sizes, fixes and verifies codec bit depths, makes the first-delivery `CCALF=0` exclusion explicit, rejects nondeterminism, compares encoder bitstreams/reconstructions and decoded YUV byte-for-byte, re-hashes external inputs after execution, and records complete structured CUDA telemetry. See [External CUDA acceptance matrix](cuda-acceptance-matrix.md) for the manifest schema, commands, resume semantics, and the explicit current IMV/Affine AMVR telemetry gap.

The full external matrix has **not** been run as part of this change and is not claimed here. No corpus was downloaded.

Focused checks passed for both 16-bit and 32-bit `Pel` builds. NVIDIA Compute Sanitizer reported zero memcheck errors and zero racecheck hazards. CPU-disabled-backend tests, the CUDA chain suite, process-level CPU/CUDA parity, forced-preflight CPU fallback parity/zero-dispatch, and process-level fatal failure injection passed. The external acceptance runner remains the reproducible route for validating the complete codec/profile matrix on the intended sequence set.

## Performance gate

One local 1920x1080 AI 8-bit fixture benchmark used one warm-up plus five measured runs:

| Fixture | CPU | CUDA chain | CPU/CUDA ratio |
|---|---:|---:|---:|
| AI 8-bit | 261.543 ms | 411.622 ms | 0.635x |

These results are correctness evidence, not a portable performance claim. The chain remains opt-in until representative, repeated benchmarks show a reproducible end-to-end gain.
