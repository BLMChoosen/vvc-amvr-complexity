#!/usr/bin/env python3
"""Generate residual-bearing inter streams and measure decoder batching distributions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from array import array
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


PROFILE_PREFIX = "DECODER_BATCH_PROFILE "
PROFILES = {
    "RA": "encoder_randomaccess_vtm.cfg",
    "LD": "encoder_lowdelay_vtm.cfg",
}


def checked_file(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return path


def checked_directory(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"directory does not exist: {path}")
    return path


def verify_build_mode(build_dir: Path, expected: str) -> Path:
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        raise RuntimeError(f"CMake cache is missing: {cache}")
    marker = f"VTM_ENABLE_DECODER_BATCH_PROFILING:BOOL={expected}"
    if marker not in cache.read_text(encoding="utf-8", errors="replace"):
        raise RuntimeError(f"{build_dir} does not declare {marker}")
    return cache


def decoder_manifest_path(executable: Path) -> Path:
    return Path(f"{executable}.build-manifest.json")


def verify_build_binding(executable: Path, build_dir: Path, expected: str) -> tuple[Path, Path, dict]:
    cache = verify_build_mode(build_dir, expected)
    manifest_path = decoder_manifest_path(executable)
    if not manifest_path.is_file():
        raise RuntimeError(f"decoder build manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid decoder build manifest {manifest_path}: {error}") from error
    if manifest.get("schema") != 1:
        raise RuntimeError(f"unsupported decoder build manifest schema: {manifest.get('schema')}")
    if manifest.get("artifact") != executable.name:
        raise RuntimeError(f"manifest artifact does not match {executable.name}: {manifest.get('artifact')}")
    if manifest.get("decoder_batch_profiling") != expected:
        raise RuntimeError(
            f"manifest mode for {executable} is {manifest.get('decoder_batch_profiling')}, expected {expected}")
    executable_hash = sha256(executable)
    if manifest.get("executable_sha256", "").upper() != executable_hash:
        raise RuntimeError(f"manifest executable SHA-256 does not match {executable}")
    cache_hash = sha256(cache)
    if manifest.get("cmake_cache_sha256", "").upper() != cache_hash:
        raise RuntimeError(f"manifest CMake cache SHA-256 does not match {cache}")
    if not manifest.get("configuration"):
        raise RuntimeError(f"manifest configuration is missing: {manifest_path}")
    return cache, manifest_path, manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def artifact(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_plane(handle, values: array, bit_depth: int) -> None:
    if bit_depth == 8:
        handle.write(bytes(values))
        return
    if sys.byteorder != "little":
        values.byteswap()
    handle.write(values.tobytes())


def generate_input(path: Path, width: int, height: int, frames: int, bit_depth: int) -> None:
    """Create deterministic moving 4:2:0 content with texture and nonzero temporal residual."""
    path.parent.mkdir(parents=True, exist_ok=True)
    maximum = (1 << bit_depth) - 1
    scale = 1 << max(0, bit_depth - 8)
    with path.open("wb") as output:
        for frame in range(frames):
            y_values = array("B" if bit_depth == 8 else "H")
            for y in range(height):
                for x in range(width):
                    noise = ((x * 37 + y * 17 + frame * 53) ^ ((x + frame * 11) * (y + 3))) & 63
                    checker = 72 if (((x + frame * 9) // 24 + (y + frame * 5) // 24) & 1) else 0
                    moving = 96 if ((x - frame * 19) % width) < width // 5 and ((y - frame * 7) % height) < height // 4 else 0
                    y_values.append(min(maximum, (32 + checker + moving + noise) * scale))
            write_plane(output, y_values, bit_depth)

            chroma_width = (width + 1) // 2
            chroma_height = (height + 1) // 2
            for component in range(2):
                values = array("B" if bit_depth == 8 else "H")
                for y in range(chroma_height):
                    for x in range(chroma_width):
                        base = 96 + component * 32
                        motion = ((x * (11 + component * 2) + y * 7 + frame * (23 + component * 6)) & 63)
                        values.append(min(maximum, (base + motion) * scale))
                write_plane(output, values, bit_depth)


def run(command: list[str], cwd: Path, log_base: Path, environment: dict[str, str] | None = None) -> dict:
    log_base.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        check=False,
    )
    stdout_log = Path(f"{log_base}.stdout.log")
    stderr_log = Path(f"{log_base}.stderr.log")
    stdout_log.write_text(completed.stdout, encoding="utf-8")
    stderr_log.write_text(completed.stderr, encoding="utf-8")
    record = {
        "command": command,
        "returncode": completed.returncode,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
    }
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {subprocess.list2cmdline(command)}; "
            f"stdout={stdout_log}; stderr={stderr_log}"
        )
    record["stdout"] = completed.stdout
    record["stderr"] = completed.stderr
    return record


def parse_profile(text: str) -> dict:
    records = []
    for line in text.splitlines():
        if line.startswith(PROFILE_PREFIX):
            records.append(json.loads(line[len(PROFILE_PREFIX):]))
    if len(records) != 1:
        raise RuntimeError(f"expected exactly one {PROFILE_PREFIX.strip()} record, found {len(records)}")
    record = records[0]
    if record.get("schema") != 4:
        raise RuntimeError(f"unsupported decoder batching schema: {record.get('schema')}")
    for field in ("process_id", "decoder_instance", "owner_thread_hash", "report_thread_hash", "hook_calls"):
        if not isinstance(record.get(field), int) or record[field] <= 0:
            raise RuntimeError(f"invalid decoder batching identity field {field}: {record.get(field)}")
    if record.get("thread_mismatch_hooks") != 0:
        raise RuntimeError(f"decoder batching hooks changed owner thread: {record.get('thread_mismatch_hooks')}")
    fills = record.get("ibc_buffer_fills")
    if not isinstance(fills, dict):
        raise RuntimeError("decoder batching record has no IBC-buffer fill model")
    if fills.get("boundary_violations") != 0 or fills.get("prepare_violations") != 0 \
            or fills.get("pending_at_report") != 0:
        raise RuntimeError(f"pending IBC-buffer fills crossed a boundary: {fills}")
    if fills.get("queued") != fills.get("applied") \
            or fills.get("last_queued_sequence") != fills.get("last_applied_sequence"):
        raise RuntimeError(f"pending IBC-buffer fills were not applied in order: {fills}")
    workload = record.get("transform_workload")
    if not isinstance(workload, dict) or not isinstance(workload.get("totals"), dict):
        raise RuntimeError("decoder batching record has no transform workload")
    return record


def decode(
    decoder: Path,
    bitstream: Path,
    output: Path,
    log_base: Path,
    profile_value: str | None,
    root: Path,
) -> tuple[dict, dict | None]:
    output.unlink(missing_ok=True)
    environment = os.environ.copy()
    if profile_value is None:
        environment.pop("VTM_DECODER_BATCH_PROFILE", None)
    else:
        environment["VTM_DECODER_BATCH_PROFILE"] = profile_value
    execution = run(
        [str(decoder), f"--BitstreamFile={bitstream}", f"--ReconFile={output}"],
        root,
        log_base,
        environment,
    )
    if not output.is_file():
        raise RuntimeError(f"decoder did not create {output}")
    profile = parse_profile(execution["stdout"] + "\n" + execution["stderr"]) if profile_value == "1" else None
    execution.pop("stdout")
    execution.pop("stderr")
    return execution, profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True, type=checked_file)
    parser.add_argument("--decoder-normal", required=True, type=checked_file)
    parser.add_argument("--decoder-profile", required=True, type=checked_file)
    parser.add_argument("--normal-build-dir", required=True, type=checked_directory)
    parser.add_argument("--profile-build-dir", required=True, type=checked_directory)
    parser.add_argument("--cfg-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--qp", type=int, default=27)
    parser.add_argument("--dry-run", action="store_true", help="validate inputs/build roots and save planned commands")
    parser.add_argument("--reuse-encoded", action="store_true",
                        help="reuse existing per-case bitstream/reconstruction artifacts")
    args = parser.parse_args()
    if args.decoder_normal == args.decoder_profile:
        parser.error("normal and profiling decoder executables must be distinct")
    if args.normal_build_dir == args.profile_build_dir:
        parser.error("normal and profiling CMake build directories must be distinct")
    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2:
        parser.error("--width and --height must be positive even values")
    if args.frames < 4:
        parser.error("--frames must be at least 4")
    if not 0 <= args.qp <= 51:
        parser.error("--qp must be in [0, 51]")

    root = Path(__file__).resolve().parents[2]
    cfg_dir = args.cfg_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    normal_cache, normal_manifest, normal_binding = verify_build_binding(
        args.decoder_normal, args.normal_build_dir, "OFF")
    profile_cache, profile_manifest, profile_binding = verify_build_binding(
        args.decoder_profile, args.profile_build_dir, "ON")
    configs = {name: (cfg_dir / filename).resolve() for name, filename in PROFILES.items()}
    missing = [str(path) for path in configs.values() if not path.is_file()]
    if missing:
        parser.error(f"missing profile config(s): {missing}")

    report = {
        "schema": 1,
        "mode": "dry-run" if args.dry_run else "measurement",
        "runner": artifact(Path(__file__).resolve()),
        "parameters": {"width": args.width, "height": args.height, "frames": args.frames, "qp": args.qp},
        "artifacts": {
            "encoder": artifact(args.encoder),
            "decoder_normal": artifact(args.decoder_normal),
            "decoder_profile": artifact(args.decoder_profile),
            "normal_cmake_cache": artifact(normal_cache),
            "profile_cmake_cache": artifact(profile_cache),
            "normal_decoder_manifest": {**artifact(normal_manifest), "binding": normal_binding},
            "profile_decoder_manifest": {**artifact(profile_manifest), "binding": profile_binding},
            "configs": {name: artifact(path) for name, path in configs.items()},
        },
        "cases": {},
    }

    inputs: dict[int, Path] = {}
    for depth in (8, 10):
        input_path = output_dir / f"moving-{args.width}x{args.height}-{args.frames}f-{depth}bit.yuv"
        generate_input(input_path, args.width, args.height, args.frames, depth)
        inputs[depth] = input_path
    report["artifacts"]["inputs"] = {str(depth): artifact(path) for depth, path in inputs.items()}

    for profile_name, config in configs.items():
        for bit_depth, input_path in inputs.items():
            case_name = f"{profile_name}-{bit_depth}bit"
            case_dir = output_dir / case_name
            case_dir.mkdir(parents=True, exist_ok=True)
            bitstream = case_dir / "stream.vvc"
            encoder_recon = case_dir / "encoder-recon.yuv"
            encode_command = [
                str(args.encoder), "-c", str(config),
                f"--InputFile={input_path}", f"--InputBitDepth={bit_depth}", "--InputChromaFormat=420",
                f"--SourceWidth={args.width}", f"--SourceHeight={args.height}", "--FrameRate=30",
                f"--FramesToBeEncoded={args.frames}", f"--QP={args.qp}", "--InternalBitDepth=10",
                f"--BitstreamFile={bitstream}", f"--ReconFile={encoder_recon}",
            ]
            if args.dry_run:
                report["cases"][case_name] = {
                    "encode": {"command": encode_command},
                    "decode_normal": {
                        "command": [str(args.decoder_normal), f"--BitstreamFile={bitstream}",
                                    f"--ReconFile={case_dir / 'decode-normal.yuv'}"]},
                    "decode_profile_off": {
                        "environment": {"VTM_DECODER_BATCH_PROFILE": "0"},
                        "command": [str(args.decoder_profile), f"--BitstreamFile={bitstream}",
                                    f"--ReconFile={case_dir / 'decode-profile-off.yuv'}"]},
                    "decode_profile_on": {
                        "environment": {"VTM_DECODER_BATCH_PROFILE": "1"},
                        "command": [str(args.decoder_profile), f"--BitstreamFile={bitstream}",
                                    f"--ReconFile={case_dir / 'decode-profile-on.yuv'}"]},
                }
                continue
            if args.reuse_encoded and bitstream.is_file() and encoder_recon.is_file():
                encode_execution = {
                    "command": encode_command,
                    "reused": True,
                    "stdout_log": str(case_dir / "encode.stdout.log"),
                    "stderr_log": str(case_dir / "encode.stderr.log"),
                }
            else:
                encode_execution = run(encode_command, root, case_dir / "encode")
                encode_execution.pop("stdout")
                encode_execution.pop("stderr")

            normal_yuv = case_dir / "decode-normal.yuv"
            profile_off_yuv = case_dir / "decode-profile-off.yuv"
            profile_on_yuv = case_dir / "decode-profile-on.yuv"
            normal_execution, _ = decode(
                args.decoder_normal, bitstream, normal_yuv, case_dir / "decode-normal", None, root)
            off_execution, _ = decode(
                args.decoder_profile, bitstream, profile_off_yuv, case_dir / "decode-profile-off", "0", root)
            on_execution, profile = decode(
                args.decoder_profile, bitstream, profile_on_yuv, case_dir / "decode-profile-on", "1", root)

            hashes = {name: sha256(path) for name, path in {
                "normal": normal_yuv, "profile_off": profile_off_yuv, "profile_on": profile_on_yuv,
            }.items()}
            if len(set(hashes.values())) != 1:
                raise RuntimeError(f"{case_name}: profiler changed decoded samples: {hashes}")
            if profile is None or profile["inverse_transform"]["runs"] == 0 \
                    or profile["inverse_transform"]["tasks_max"] == 0:
                raise RuntimeError(f"{case_name}: generated stream contains no nonzero inter-CBF transform work")
            if profile["pictures"] < args.frames:
                raise RuntimeError(f"{case_name}: decoded only {profile['pictures']} profiled pictures")

            report["cases"][case_name] = {
                "bitstream": artifact(bitstream),
                "encoder_reconstruction": artifact(encoder_recon),
                "decoded_sha256": hashes["normal"],
                "decoded_bytes": normal_yuv.stat().st_size,
                "encode": encode_execution,
                "decode_normal": normal_execution,
                "decode_profile_off": off_execution,
                "decode_profile_on": on_execution,
                "profile": profile,
            }
            (output_dir / "decoder-batch-profile-report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8")

    if not args.dry_run:
        first_name, first_case = next(iter(report["cases"].items()))
        concurrent_dir = output_dir / "concurrent-profile"
        bitstream = Path(first_case["bitstream"]["path"])

        def concurrent_decode(index: int) -> tuple[dict, dict, dict]:
            output = concurrent_dir / f"decode-{index}.yuv"
            execution, profile = decode(
                args.decoder_profile, bitstream, output, concurrent_dir / f"decode-{index}", "1", root)
            if profile is None:
                raise RuntimeError("concurrent profiling decoder emitted no profile")
            output_artifact = artifact(output)
            if output_artifact["sha256"] != first_case["decoded_sha256"]:
                raise RuntimeError(f"concurrent decoder {index} changed decoded samples")
            return execution, profile, output_artifact

        with ThreadPoolExecutor(max_workers=2) as executor:
            concurrent_results = list(executor.map(concurrent_decode, range(2)))
        identities = {
            (profile["process_id"], profile["decoder_instance"])
            for _, profile, _ in concurrent_results
        }
        if len(identities) != 2:
            raise RuntimeError(f"concurrent decoder identities are not unique: {sorted(identities)}")
        report["concurrency_check"] = {
            "source_case": first_name,
            "instances": [
                {"execution": execution, "profile_identity": {
                    "process_id": profile["process_id"],
                    "decoder_instance": profile["decoder_instance"],
                    "owner_thread_hash": profile["owner_thread_hash"],
                    "report_thread_hash": profile["report_thread_hash"],
                    "hook_calls": profile["hook_calls"],
                    "thread_mismatch_hooks": profile["thread_mismatch_hooks"],
                }, "output": output_artifact}
                for execution, profile, output_artifact in concurrent_results
            ],
        }

    (output_dir / "decoder-batch-profile-report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(output_dir / "decoder-batch-profile-report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
