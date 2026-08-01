# Experimental CUDA luma ALF

`--GPUBackend=cuda --GPUExperimentalALF=1` enables the experimental decoder luma-ALF path. It is deliberately
off by default. The CPU remains authoritative for syntax, APS reconstruction, eligibility, chroma/CCALF, tile,
slice, subpicture and explicit virtual-boundary cases.

The CUDA path is eligible only for a single tile/slice/subpicture, luma-only ALF, 8- or 10-bit pictures, exact
4-aligned CTU geometry and at least 1920x1080 pixels. The 1080p gate avoids dispatch and transfer overhead on
small pictures. Unsupported input returns `NotEligible` and runs the normative CPU path. Once an eligible CUDA
operation starts, allocation, upload, launch, synchronization, diagnostic-download or commit errors are fatal.
The ALF runtime is recovered, its reconstruction mirror is quarantined as host-authoritative, ALF CUDA remains
disabled for the context, and the exception reaches `DecoderApp`, which exits nonzero. There is no silent CPU
fallback after a partially started ALF dispatch.

The implementation filters into private scratch and commits to the reconstruction mirror only after the kernels
complete. Scratch growth swaps complete owners transactionally; both current and retired owners remain accounted
and are released by recovery/teardown. Decoder telemetry separates runtime-only transfers, synchronizations and
time from full integration measurements. Integration covers mirror-upload submission, the upload-to-ALF stream
dependency, runtime, host publication and mirror download; it reports mirror upload/download bytes and
upload-submission/runtime/download/integration time separately. `uploadSubmissionNanoseconds` measures host-side
staging/submission plus dependency enqueue without an extra host synchronization. Because the ALF stream waits on
that dependency, `runtimeNanoseconds` includes any device-side wait for the mirror upload to complete.

Build and run the deterministic CUDA tests, fault-injection suite and microbenchmark with:

```text
cmake -S . -B build-alf-cuda -DENABLE_CUDA=ON -DVTM_BUILD_CUDA_BACKEND_TESTS=ON -DVTM_CUDA_ARCHITECTURES=120
cmake --build build-alf-cuda --config Release --target CudaBackendTest DecoderApp
ctest --test-dir build-alf-cuda -C Release --output-on-failure
$suffix = ((Select-String build-alf-cuda/CMakeCache.txt '^VTM_CUDA_OUTPUT_SUFFIX:STRING=').Line -split '=')[1]
$test = Get-ChildItem bin -Recurse -Filter CudaBackendTest.exe | Where-Object DirectoryName -Like "*$suffix*" | Select-Object -First 1
& $test.FullName --cuda 0
& $test.FullName --cuda-benchmark 0
```

`--cuda 0` includes all ALF fault points and proves fail-fast poisoning, host-authoritative mirror quarantine,
resource release and context teardown. The benchmark performs one unmeasured warm-up followed by exactly five
diagnostic-free measured runs. It compares the real `AdaptiveLoopFilter::filterBlk<ALF_FILTER_7>` CPU
implementation with CUDA on a 1924x1084 partial-CTU picture and prints the median wall time plus runtime and
integration telemetry.

The measurements intentionally show the two conflicting performance results that control the rollout decision.
On the initial RTX 5060/Pel16 target, the kernel-oriented 1924x1084 microbenchmark measured 45.8721 ms CPU versus
6.0759 ms CUDA (7.55x) at the five-run median. A validated long stream containing one SPS/PPS/APS and 30 repeated
self-contained 1920x1080 IDRs produced identical 93,312,000-byte CPU/CUDA YUV output (SHA-256
`0D9E57A2361FC6B8AE1CB90494722BC0798F7C9C1180031CFA9AC2CCC9C535FD`), but decoder wall medians were
6550.394 ms CPU versus 6728.958 ms CUDA (0.9735x). The microbenchmark wins while the end-to-end decoder loses;
therefore the wall-time gate has not passed and `GPUExperimentalALF` remains experimental and off by default.

For another end-to-end measurement, use a conforming luma-only-ALF IDR access unit. Repeat it only if its
VPS/SPS/PPS and self-contained IDR semantics remain valid after concatenation. Compare decoded YUV size and
SHA-256 before interpreting wall time, run CPU and CUDA in alternating order, discard warm-up, and report at least
five measured samples for each backend.
