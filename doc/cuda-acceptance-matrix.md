# External CUDA acceptance matrix

`tools/cuda/run_acceptance_matrix.py` schema 3 is the reproducible gate for the first CUDA delivery. It covers four distinct external 4:2:0 sources (1920x1080 and 3840x2160, each at 8 and 10 bits), three encoder profiles (AI, RA, LD), and CPU/CUDA encode and decode paths.

## Immutable manifest

Acceptance requires `--manifest`; legacy `--input-8`, `--input-10`, `--cfg-dir`, profile, depth, and frame overrides are rejected. Every path in the manifest is absolute. Raw inputs must be outside the repository and distinct. Profile and sequence configs may be inside or outside the repository, but their expected SHA-256 hashes are mandatory. Each input records and validates its exact byte size and SHA-256 before command generation. Inputs are hashed again after success or failure, so a mutation prevents an acceptance claim.

The manifest schema is:

```json
{
  "schema": 1,
  "encoder_overrides": { "CCALF": 0 },
  "profile_configs": {
    "AI": { "path": "C:\\vtm-corpus\\cfg\\encoder_intra_vtm.cfg", "sha256": "<64 hex digits>" },
    "RA": { "path": "C:\\vtm-corpus\\cfg\\encoder_randomaccess_vtm.cfg", "sha256": "<64 hex digits>" },
    "LD": { "path": "C:\\vtm-corpus\\cfg\\encoder_lowdelay_vtm.cfg", "sha256": "<64 hex digits>" }
  },
  "inputs": [
    {
      "name": "sequence-1080p-8",
      "resolution": "1080p",
      "width": 1920,
      "height": 1080,
      "chroma": "420",
      "source_bit_depth": 8,
      "internal_bit_depth": 8,
      "frames": 4,
      "expected_size_bytes": 12441600,
      "path": "D:\\vtm-corpus\\sequence-1080p-8.yuv",
      "sha256": "<64 hex digits>",
      "sequence_config": { "path": "D:\\vtm-corpus\\sequence-1080p-8.cfg", "sha256": "<64 hex digits>" }
    }
  ]
}
```

The `inputs` array must contain exactly the four `(resolution, source_bit_depth)` pairs `1080p/8`, `1080p/10`, `4K/8`, and `4K/10`. The abbreviated example above shows one entry; the other three have the same fields. Dimensions are fixed at 1920x1080 and 3840x2160. Source and internal bit depths must match. Ten-bit raw samples occupy two bytes. `frames` must be at least four, and `expected_size_bytes` must equal `width * height * 3/2 * bytes_per_sample * frames`.

CCALF exclusion is not implicit: the manifest must declare `encoder_overrides.CCALF=0`, every encoder command includes the fixed `--CCALF=0` contract, conflicting extra arguments are rejected, and the report records the exclusion.

## Preflight and execution

Run the complete non-executing preflight first:

```powershell
python tools/cuda/run_acceptance_matrix.py `
  --encoder C:\vtm-build\bin\EncoderApp.exe `
  --decoder C:\vtm-build\bin\DecoderApp.exe `
  --manifest D:\vtm-corpus\acceptance-manifest.json `
  --output-dir D:\vtm-results\cuda-acceptance `
  --dry-run
```

The preflight hashes every declared artifact, emits every command and output path, estimates YUV/bitstream output space, records free disk space, and must list exactly 288 subprocesses: 12 cases times four groups (encode CPU, encode CUDA, decode CPU, decode CUDA) times one warm-up plus five measured runs.

Remove `--dry-run` to execute. Acceptance always uses exactly one warm-up and five measured repeats; another `--repeats` value is accepted only with `--mode smoke`, whose report has `acceptance_claim: false`. The runner computes medians, requires deterministic repeated hashes, and compares CPU/CUDA bitstreams, encoder reconstructions, and decoded YUV byte-for-byte. Reconstruction and decoded output sizes must also match the manifest geometry/depth.

The runner writes `matrix-report.json` atomically after each complete case and records a structured failure before returning an error. `--resume` reuses only cases marked `completed`, only when the runner, manifest, executable, config, input, device, and argument contract hash is unchanged, and only after re-hashing every retained measured output against the prior report. Executables, profile configs, sequence configs, and the manifest are re-hashed at shutdown too.

## Depth and telemetry gates

Each encoder command fixes source width/height and luma/chroma `InputBitDepth`, `InternalBitDepth`, and `OutputBitDepth`; each decoder command fixes luma/chroma `OutputBitDepth`. The encoder's effective input and internal-depth lines are parsed, so an 8-bit case that remains configured internally as 10-bit fails. Both input and coded chroma formats are fixed to 4:2:0. User-supplied extra arguments cannot override these settings, CCALF, backend/device selection, output paths, or experimental feature flags.

Schema 3 retains structured telemetry rather than a lossy pass/fail summary:

- encoder SAD batches and gates, QPA batches/tasks and failure/fallback/enable/poison/disabled gates, mirror memory/transfers/budget rejections, effective depths, and IMV/Affine AMVR counters;
- decoder chain dispatches/pixels/stage work, parameter/internal/mirror bytes, scratch, runtime/integration synchronizations, collection/LMCS/DBF/SAO/ALF/runtime/integration timings, and failure/not-eligible/enable/poison/disabled gates.

Current `EncoderApp` output does not expose numeric IMV/Affine AMVR counters. The runner does not fabricate them: smoke reports `status: unavailable`, while an executed acceptance run fails with that explicit telemetry gap. When present, all five measured counter pairs must be deterministic and equal between CPU and CUDA. The full external matrix therefore remains unclaimed until the binary emits `VTM search counters IMV/Affine AMVR: <imv>/<affine-amvr>` (or the parser and documentation are updated together for an equivalent real counter source).

The runner does not download corpus files and this change does not execute the heavy 1080p/4K matrix.
