# Experimental CUDA luma deblocking filter

`GPUExperimentalDBF` is an experimental DecoderApp switch. It is `0` by default and the CPU
`DeblockingFilter` remains the normative path whenever the complete preflight is not satisfied.

## Scope and selection contract

The initial CUDA path accepts 4:2:0 pictures with equal 8- or 10-bit luma/chroma depth, a 16- or
32-bit `Pel`, at least 1920x1080 active luma samples, one slice, one tile, one subpicture, no virtual
boundaries, LADF disabled, and picture deblocking enabled. The reconstruction mirror must provide
at least eight samples of margin on every side. An unsupported tool or layout is `NotEligible` and
uses the untouched CPU path before CUDA selection.

Once descriptor collection starts, CUDA has been selected. Allocation, upload, snapshot, either
kernel, completion, commit, or download failure is fatal and permanently poisons DBF acceleration
in that CUDA context. There is no silent CPU replay after CPU chroma has been filtered. Test builds
can inject every stage through `CudaDbfTestFailurePoint`; DecoderApp also recognizes
`VTM_CUDA_DBF_TEST_FAILURE=allocation|upload|snapshot|vertical-launch|vertical-sync|horizontal-launch|horizontal-sync|commit|commit-sync`.

A valid collection may contain zero luma tasks (for example, a high-QP picture whose normative CPU
pass changes no luma samples). This is recorded as a no-op frame: chroma and mirror ownership remain
coherent, no CUDA upload/kernel/commit is submitted, and it is not counted as a dispatch or failure.

VVC/VTM 24 has no HEVC PCM or transquant-bypass syntax. The applicable VVC lossless-side rule in
this implementation is palette (`CU::isPLT`); CPU serializes the P/Q no-filter flags and CUDA restores
the exact untouched side. BDPCM affects the CPU boundary-strength derivation and is therefore also
preserved. If PCM/transquant-bypass syntax is introduced by a downstream fork, that fork is not
eligible until its P/Q semantics are added to the CPU descriptor collector.

## Normative split and race proof

The CPU retains all syntax-dependent work: CU/TU/PU traversal, slice/tile/subpicture and virtual-
boundary decisions, boundary strength, QP, beta/tc, maximum P/Q filter lengths, affine restrictions,
horizontal CTU restrictions, clip range, and palette P/Q flags. Chroma remains entirely on CPU.
For luma it emits one POD per normative four-sample edge segment. LADF is excluded because its QP
shift depends on pixels already modified by the preceding edge.

CUDA snapshots the complete luma plane including margins into private scratch, runs the normative
vertical pass, then the horizontal pass, and commits active luma only after both complete. A vertical
lane owns one global four-row band and walks its descriptors in original CPU order. A horizontal lane
owns one global four-column band and likewise walks its descriptors in CPU order. Filtering reads and
writes only along the edge normal, so two lanes in the same pass have disjoint read/write sample sets.
Dependencies between neighboring edges remain serial inside a lane. This gives exactly two kernel
launches per frame without coloring races or reordered edge dependencies.

## Verification and performance status

`CudaBackendTest` compares CUDA directly with `DeblockingFilter::filterLumaTasksCpu` for 8/10-bit,
Pel16/Pel32, weak/strong/long-tap 3/5/7 filters, P/Q no-filter, multiple serial edges per lane, partial
CTU dimensions (1924x1084), descriptor rejection, and all transactional failure points. The CUDA test
also passed Compute Sanitizer memcheck and racecheck with zero findings.

On the initial RTX 5060 target, a dense 1924x1084 Pel16/10-bit microbenchmark (warm-up plus five
measured runs, about 129k descriptors) produced median CPU descriptor application of 11.196 ms and
CUDA integration of 9.378 ms (1.19x). A real 1920x1080 AI decode was byte-identical, with 105,512
descriptors and 8.876 ms reported DBF runtime, but end-to-end DecoderApp wall time regressed from
0.310 s to 0.441 s. That dispatch uploaded 2.96 MB of parameters, committed 4.15 MB, transferred
8.27 MB in each mirror direction, and used six synchronizations. Therefore the end-to-end gate has
not passed and `GPUExperimentalDBF` intentionally remains off by default.

Real DecoderApp CPU/CUDA output was byte-identical for AI, RA, and LD. The SHA-256 hashes were
`0E1434CB451B5CA0A4C92EAA3DD7CBDA5C1B862E77B3C64D703F893C4D11C8D5` (AI),
`F6757DF9FB27690F688B5E7140E1A5DA93D85ECCE86EAC50FCB36A288C174750` (RA), and
`AA88EF636CE7947809BA2F016EA570C510AE135B1631B5A9688115274DA4A8A9` (LD). The four-frame
RA and LD streams each contained real inter B pictures; both recorded one CUDA dispatch and three
normative luma no-op frames at the deliberately high validation QPs.
