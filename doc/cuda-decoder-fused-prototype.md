# CUDA fused decoder prototype (no-go)

Date: 2026-08-01

Decision: **no-go**. The prototype option, target, API/ABI, implementation, and
tests were removed and were never integrated into the production decoder. No
reproducible historical benchmark or correctness result was preserved in this
repository.

Reopen only if at least one assumption changes materially, such as:

- substantially larger batches can remain entirely device-resident;
- adjacent decoder stages also move to the GPU, eliminating transfer/sync costs;
- transforms receive shape-specialized, parallel kernels instead of the generic
  prototype path;
- profiling on target hardware demonstrates a credible backend-only win before
  any `DecCu` integration.

Current repository checks:

```powershell
rg -n "CudaDecoderFusedPrototype|VTM_ENABLE_DECODER_FUSED_PROTOTYPE|FUSED_BENCH|FusedTask|dispatchBatch" CMakeLists.txt source

Get-ChildItem -LiteralPath bin,lib -Recurse -Force -ErrorAction SilentlyContinue |
  Where-Object Name -Match "CudaDecoderFusedPrototype|FUSED_BENCH|FusedTask"

Test-Path -LiteralPath build-fused-prototype
Get-ChildItem -Directory -Filter "build*" |
  ForEach-Object { Get-ChildItem -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue } |
  Where-Object Name -Match "CudaDecoderFusedPrototype|FUSED_BENCH|FusedTask"

cmake -S . -B build-backend-health-cpu -DENABLE_CUDA=OFF -DVTM_BUILD_CUDA_BACKEND_TESTS=ON
cmake --build build-backend-health-cpu --config Release --target CudaBackendTest --parallel 8
ctest --test-dir build-backend-health-cpu -C Release -R "^CudaBackend\." --output-on-failure

cmake -S . -B build-backend-health-cuda -DENABLE_CUDA=ON -DVTM_BUILD_CUDA_BACKEND_TESTS=ON -DVTM_CUDA_ARCHITECTURES=native
cmake --build build-backend-health-cuda --config Release --target CudaBackendTest --parallel 8
ctest --test-dir build-backend-health-cuda -C Release -R "^CudaBackend\." --output-on-failure
```

The `rg` command and both name-based output searches must return no matches;
`Test-Path` must return `False`. These checks deliberately inspect binary/build
artifacts by filename, not by treating binaries as source text. Both
configure/build/test sequences must succeed on a host with the required
toolchains; the CUDA sequence also requires a supported NVIDIA GPU and CUDA
toolkit.
