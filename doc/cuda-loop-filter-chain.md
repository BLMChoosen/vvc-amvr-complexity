# Experimental CUDA resident decoder loop-filter chain

## Scope and activation

`--GPUExperimentalLoopFilterChain=1` enables the experimental decoder path when `--GPUBackend=cuda` is selected. The option is deliberately **off by default**. Current end-to-end benchmarks do not pass the speed gate, so it must not be enabled by default yet.

The implementation keeps reconstructed luma resident on the device across the normative frame-level order:

1. inverse LMCS, when active;
2. vertical and horizontal deblocking;
3. SAO;
4. ALF.

Eligible pictures perform one reconstruction-mirror upload before the first active stage and one download after the last active stage. Intermediate stage data stays on the device. Disabled stages are omitted. A DBF-only picture with zero serialized luma edges returns `NoOp` before mirror allocation or transfer.

Forward LMCS remains in the CU reconstruction path. The chain handles only frame-level inverse LMCS. Chroma filtering remains normative CPU work. The decoder returns immediately after a successful composite dispatch, so standalone GPU DBF/ALF integration paths do not also execute. Their reusable runtime primitives are called inside the composite dispatch, while standalone dispatch telemetry remains independent.

## Eligibility and unsupported syntax

The current path accepts 4:2:0 pictures with matching 8- or 10-bit luma/chroma depth, `Pel` sizes of 16 or 32 bits, at least 1920x1080 pixels, no more than 4096 CTUs, and exactly one slice, tile, and subpicture. Virtual boundaries and LADF are excluded. ALF requires luma dimensions divisible by four.

CCALF is rejected before any mirror mutation or device upload. This is required because normative CCALF reads the pre-ALF luma snapshot. Multi-slice/tile/subpicture boundary semantics and virtual boundaries must be implemented and tested before those pictures become eligible.

SAO syntax can reconstruct to an all-disabled picture. In that case the SAO luma stage is converted to an explicit no-op instead of submitting an empty, invalid CTU descriptor set. Chroma SAO processing remains on the CPU.

All frame, stage, task, CTU, coefficient, clip, LUT, margin, and pointer/count contracts are validated before selection. An operational error after selection is fatal for the process, quarantines the mirror as host-valid, recovers owned scratch, and permanently poisons only the composite chain for the context.

## Telemetry

Decoder shutdown reports composite dispatches and pixels, DBF tasks, SAO/ALF CTUs, parameter bytes, internal device copies, mirror upload/download bytes, current/retired/peak scratch, runtime/integration synchronizations, collection and per-stage times, total runtime/integration time, failures, preflight rejections, enabled state, poisoned state, and whether the feature was disabled by its flag.

For a single 1920x1080 eligible picture, the measured reconstruction mirror transfer was 8,266,752 bytes in each direction: one upload and one download including the allocated pitch/margins. Internal DBF/SAO/ALF transactional copies are reported separately and are not PCIe transfers.

## Correctness coverage

The focused CUDA test compares each stage against VTM normative references at 8 and 10 bits, including partial right/bottom CTUs, adjacent SAO CTUs with different EO/BO modes and offsets, non-identity inverse LMCS LUTs, non-identity DBF tasks, ALF classification/filtering, disabled CTUs, deterministic repeats, disabled-stage pointer/count contracts, DBF zero-task no-op, CCALF preflight rejection, rollback, poisoning, and injected allocation/upload/launch/completion/download/commit failures.

The following real 1920x1080 4:2:0 matrix was decoded once on CPU and twice through CUDA. All three output hashes matched in every row; CUDA reported zero failures and zero not-eligible pictures.

| Mode | Depth | Frames | SHA-256 | DBF tasks | SAO CTUs | ALF CTUs |
|---|---:|---:|---|---:|---:|---:|
| AI | 8 | 1 | `9407B9DD680E833CC2CC2DC5AA989834E1E5894E780B433C93BBE77E0F1A1338` | 84,426 | 135 | 135 |
| AI | 10 | 1 | `0E1434CB451B5CA0A4C92EAA3DD7CBDA5C1B862E77B3C64D703F893C4D11C8D5` | 105,512 | 135 | 135 |
| RA | 8 | 4 | `4364F6DCFD5C66E9607DBBD9D2CD8FC65CDED8BEE56F9A5C110A17FF3F6F3A2E` | 41,584 | 135 | 540 |
| RA | 10 | 4 | `F6757DF9FB27690F688B5E7140E1A5DA93D85ECCE86EAC50FCB36A288C174750` | 64,501 | 135 | 540 |
| LD | 8 | 4 | `6136A4835B80BFFF7870944230062F821EC10F4097439DABF2788175DCD6E6D4` | 17,388 | 135 | 540 |
| LD | 10 | 4 | `AA88EF636CE7947809BA2F016EA570C510AE135B1631B5A9688115274DA4A8A9` | 37,628 | 135 | 540 |

The encoders permitted LMCS in this real-stream matrix but did not choose active LMCS syntax, so real-stream LMCS time was zero. Non-identity inverse LMCS is covered by the normative 8/10-bit focused tests; this limitation must remain explicit until a real active-LMCS fixture is added.

Final focused checks passed for both 16-bit and 32-bit `Pel` builds. NVIDIA Compute Sanitizer reported zero memcheck errors and zero racecheck hazards. CPU-disabled-backend tests, the complete CUDA backend suite, process-level CPU/CUDA parity, and process-level fatal failure injection also passed.

## Performance gate

Warm-up plus five measured runs produced these median end-to-end decoder times on the local test system:

| Fixture | CPU | CUDA chain | CPU/CUDA ratio |
|---|---:|---:|---:|
| AI 8-bit | 280.479 ms | 441.342 ms | 0.636x |
| AI 10-bit | 284.467 ms | 450.288 ms | 0.632x |
| RA 10-bit | 333.321 ms | 551.780 ms | 0.604x |
| LD 10-bit | 292.842 ms | 440.567 ms | 0.665x |

Representative accumulated CUDA telemetry was:

| Fixture | Collection | DBF | SAO | ALF | Runtime | Integration | Mirror up/down | Runtime/integration syncs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AI 10-bit | 29.165 ms | 4.955 ms | 0.216 ms | 0.938 ms | 7.790 ms | 16.973 ms | 8,266,752 / 8,266,752 B | 12 / 14 |
| RA 10-bit | 40.247 ms | 7.297 ms | 0.209 ms | 13.068 ms | 22.280 ms | 52.237 ms | 33,067,008 / 33,067,008 B | 18 / 26 |
| LD 10-bit | 22.282 ms | 5.591 ms | 0.484 ms | 8.526 ms | 16.336 ms | 39.758 ms | 33,067,008 / 33,067,008 B | 18 / 26 |

These results are correctness evidence, not a portable performance claim. The chain remains opt-in until representative, repeated benchmarks show a reproducible end-to-end gain.
