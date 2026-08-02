# CUDA decoder transform tombstone

An experimental heterogeneous CUDA backend for transform skip and DCT-II inverse transform was evaluated and removed. It did not demonstrate a consistent steady-state performance benefit across representative decoder batches, so it was not integrated into `DecCu`.

No decoder-transform CUDA backend, API, build option, runtime flag, or test command remains. The decoder's `TrQuant` implementation is the only active transform path.

## Revisit criteria

A new prototype should start only after the versioned [decoder batching profiler](cuda-decoder-batching.md) and [`run_decoder_batch_profile.py`](../tools/cuda/run_decoder_batch_profile.py) demonstrate sustained batches on the intended corpus. Its design should also have a credible way to:

- fuse transform/reconstruction with adjacent GPU stages;
- avoid uploading prediction samples solely for reconstruction;
- avoid downloading dirty reconstruction regions before their next consumer;
- preserve bit-exact decoder output and show a consistent end-to-end steady-state gain.

## Current verification

The following PowerShell commands verify that the removed surface is absent and that the retained CPU and CUDA backends build and pass their registered contract tests:

```powershell
rg -n 'CudaDecoderTransform|computeDecoderTransformBatch|decoderTransformBatch|cuda-transform' source CMakeLists.txt
if ($LASTEXITCODE -ne 1) { throw 'decoder-transform CUDA surface is present or rg failed' }

cmake -S . -B build-transform-check-cpu -DENABLE_CUDA=OFF -DVTM_BUILD_CUDA_BACKEND_TESTS=ON
cmake --build build-transform-check-cpu --config Release --target CudaBackendTest
ctest --test-dir build-transform-check-cpu -C Release --output-on-failure -R '^CudaBackend\.(ParseConfig|Disabled)$'

cmake -S . -B build-transform-check-cuda -DENABLE_CUDA=ON -DVTM_BUILD_CUDA_BACKEND_TESTS=ON -DVTM_CUDA_ARCHITECTURES=native
cmake --build build-transform-check-cuda --config Release --target CudaBackendTest
ctest --test-dir build-transform-check-cuda -C Release --output-on-failure -R '^CudaBackend\.(ParseConfig|Runtime)$'
```

The CUDA commands require a supported CUDA toolkit and device.
