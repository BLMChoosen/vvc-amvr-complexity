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
    if record.get("schema") != 7:
        raise RuntimeError(f"unsupported decoder batching schema: {record.get('schema')}")
    for field in ("process_id", "decoder_instance", "owner_thread_hash", "report_thread_hash", "hook_calls"):
        if not isinstance(record.get(field), int) or record[field] <= 0:
            raise RuntimeError(f"invalid decoder batching identity field {field}: {record.get(field)}")
    if record.get("thread_mismatch_hooks") != 0:
        raise RuntimeError(f"decoder batching hooks changed owner thread: {record.get('thread_mismatch_hooks')}")
    self_test = record.get("scheduler_self_test")
    if not isinstance(self_test, dict) or self_test.get("passed") is not True \
            or not isinstance(self_test.get("weighted_metadata_extra_bytes"), int) \
            or self_test["weighted_metadata_extra_bytes"] <= 0 \
            or not isinstance(self_test.get("rpr_metadata_extra_bytes"), int) \
            or self_test["rpr_metadata_extra_bytes"] <= 0:
        raise RuntimeError(f"decoder fused scheduler/model self-test is missing or failed: {self_test}")
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
    fused = record.get("fused_windows")
    if not isinstance(fused, dict) or fused.get("model") != "mc_dequant_inverse_transform_reconstruction":
        raise RuntimeError("decoder batching record has no fused-window model")
    assumptions = fused.get("assumptions")
    required_assumptions = (
        "references_resident", "output_mirror_resident", "heterogeneous_mc_paths",
        "heterogeneous_components", "heterogeneous_transform_shapes", "heterogeneous_ts_dct2",
        "dirty_output_downloaded_at_cpu_boundaries",
    )
    if not isinstance(assumptions, dict) or any(assumptions.get(key) is not True for key in required_assumptions):
        raise RuntimeError(f"fused-window residency/heterogeneity assumptions are incomplete: {assumptions}")
    cores = fused.get("cores")
    if not isinstance(cores, dict) or set(cores) != {"core_a", "core_b_rpr"}:
        raise RuntimeError(f"fused-window core set is invalid: {cores}")
    metric_names = (
        "cus", "tus", "luma_pixels", "component_pixels", "prediction_operations", "transform_tasks",
        "descriptor_bytes", "qcoeff_bytes", "dirty_download_bytes", "total_transfer_bytes",
    )
    descriptor_layout = fused.get("descriptor_layout_bytes")
    shared_layout = fused.get("shared_metadata_layout_bytes")
    if not isinstance(descriptor_layout, dict) or any(
            not isinstance(descriptor_layout.get(name), int) or descriptor_layout[name] <= 0
            for name in ("cu", "prediction_operation", "transform_task", "qcoeff_element", "pel_element")):
        raise RuntimeError(f"fused-window recurring descriptor layout is incomplete: {descriptor_layout}")
    if not isinstance(shared_layout, dict) or any(
            not isinstance(shared_layout.get(name), int) or shared_layout[name] <= 0
            for name in ("picture", "slice", "reference", "weighted_prediction", "rpr", "bcw")):
        raise RuntimeError(f"fused-window shared metadata layout is incomplete: {shared_layout}")
    for core_name, core in cores.items():
        if not isinstance(core, dict):
            raise RuntimeError(f"{core_name} fused-window record is invalid")
        coverage = core.get("coverage")
        if not isinstance(coverage, dict) \
                or coverage.get("considered_cus") != coverage.get("eligible_cus", 0) + coverage.get("rejected_cus", 0) \
                or coverage.get("considered_luma_pixels") != coverage.get("eligible_luma_pixels", 0) \
                + coverage.get("rejected_luma_pixels", 0):
            raise RuntimeError(f"{core_name} fused-window coverage does not balance: {coverage}")
        rejections = core.get("rejections")
        if not isinstance(rejections, dict) \
                or sum(item.get("cus", -1) for item in rejections.values()) != coverage["rejected_cus"] \
                or sum(item.get("luma_pixels", -1) for item in rejections.values()) \
                != coverage["rejected_luma_pixels"]:
            raise RuntimeError(f"{core_name} fused-window rejection totals do not match coverage: {rejections}")
        flush_reasons = core.get("flush_reasons")
        if not isinstance(flush_reasons, dict) or sum(flush_reasons.values()) != core.get("windows"):
            raise RuntimeError(f"{core_name} fused-window flush totals do not match windows: {flush_reasons}")
        metrics = core.get("metrics")
        if not isinstance(metrics, dict) or any(name not in metrics for name in metric_names):
            raise RuntimeError(f"{core_name} fused-window metrics are incomplete")
        for metric_name in metric_names:
            metric = metrics[metric_name]
            if not isinstance(metric, dict) or any(not isinstance(metric.get(key), int) or metric[key] < 0
                                                   for key in ("total", "p50_upper", "p90_upper", "p99_upper", "max")):
                raise RuntimeError(f"{core_name} fused-window metric is invalid: {metric_name}={metric}")
            buckets = metric.get("log2_buckets")
            if not isinstance(buckets, list) or len(buckets) != 65 \
                    or any(not isinstance(value, int) or value < 0 for value in buckets) \
                    or sum(buckets) != core.get("windows"):
                raise RuntimeError(f"{core_name} fused-window histogram is invalid: {metric_name}")
            if metric["p50_upper"] != histogram_quantile_upper(buckets, 50) \
                    or metric["p90_upper"] != histogram_quantile_upper(buckets, 90) \
                    or metric["p99_upper"] != histogram_quantile_upper(buckets, 99):
                raise RuntimeError(f"{core_name} fused-window quantiles do not match buckets: {metric_name}")
            maximum_bucket = 0 if metric["max"] == 0 else metric["max"].bit_length()
            if maximum_bucket >= len(buckets) or (core.get("windows", 0) != 0 and buckets[maximum_bucket] == 0):
                raise RuntimeError(f"{core_name} fused-window maximum does not match buckets: {metric_name}")
        if metrics["cus"]["total"] != coverage["eligible_cus"] \
                or metrics["luma_pixels"]["total"] != coverage["eligible_luma_pixels"]:
            raise RuntimeError(f"{core_name} fused-window metrics do not match coverage")
        if metrics["total_transfer_bytes"]["total"] != metrics["descriptor_bytes"]["total"] \
                + metrics["qcoeff_bytes"]["total"] + metrics["dirty_download_bytes"]["total"]:
            raise RuntimeError(f"{core_name} recurring transfer metric does not balance")
        transfer = core.get("transfer_model")
        if not isinstance(transfer, dict):
            raise RuntimeError(f"{core_name} fused-window transfer model is missing")
        shared = transfer.get("shared_bytes_once")
        descriptor = transfer.get("descriptor_h2d_bytes")
        qcoeff = transfer.get("qcoeff_h2d_bytes")
        dirty = transfer.get("dirty_boundary_d2h_bytes")
        if any(not isinstance(value, int) or value < 0 for value in (shared, descriptor, qcoeff, dirty)):
            raise RuntimeError(f"{core_name} fused-window transfer fields are invalid: {transfer}")
        shared_fields = (
            "shared_scan_bytes_once", "shared_matrix_bytes_once", "shared_dequant_constant_bytes_once",
            "shared_picture_metadata_bytes_once", "shared_slice_metadata_bytes_once",
            "shared_reference_metadata_bytes_once", "shared_weighted_prediction_metadata_bytes_once",
            "shared_rpr_metadata_bytes_once", "shared_bcw_metadata_bytes_once",
        )
        if any(not isinstance(transfer.get(field), int) or transfer[field] < 0 for field in shared_fields) \
                or shared != sum(transfer[field] for field in shared_fields):
            raise RuntimeError(f"{core_name} fused-window shared metadata does not balance: {transfer}")
        windows = core["windows"]
        expected_amortized = (shared + windows - 1) // windows if windows else 0
        if transfer.get("shared_bytes_amortized_per_window") != expected_amortized:
            raise RuntimeError(f"{core_name} fused-window amortized shared bytes are invalid: {transfer}")
        if descriptor != metrics["descriptor_bytes"]["total"] \
                or qcoeff != metrics["qcoeff_bytes"]["total"] \
                or dirty != metrics["dirty_download_bytes"]["total"]:
            raise RuntimeError(f"{core_name} fused-window transfer fields disagree with metrics")
        if transfer.get("estimated_h2d_bytes_run") != descriptor + qcoeff + shared \
                or transfer.get("estimated_d2h_bytes_run") != dirty \
                or transfer.get("estimated_total_transfer_bytes_run") != descriptor + qcoeff + shared + dirty:
            raise RuntimeError(f"{core_name} fused-window transfer model does not balance: {transfer}")
        feature_coverage = core.get("feature_coverage")
        if not isinstance(feature_coverage, dict) or any(
                not isinstance(feature_coverage.get(field), int) or feature_coverage[field] < 0
                for field in ("rpr_cus", "rpr_prediction_operations", "weighted_prediction_units",
                              "bcw_prediction_units")):
            raise RuntimeError(f"{core_name} fused-window feature coverage is invalid: {feature_coverage}")
        fused_fills = core.get("ibc_deferred_fills")
        if not isinstance(fused_fills, dict) or fused_fills.get("pending") != 0 \
                or fused_fills.get("boundary_violations") != 0 \
                or fused_fills.get("queued") != fused_fills.get("applied") \
                or fused_fills.get("last_queued_sequence") != fused_fills.get("last_applied_sequence"):
            raise RuntimeError(f"{core_name} deferred IBC fills crossed a boundary or reordered: {fused_fills}")
    return record


def fused_case_summary(profile: dict) -> dict:
    """Retain the decision inputs without inventing a launch threshold or speedup claim."""
    return {
        name: {
            "coverage": core["coverage"],
            "windows": core["windows"],
            "metrics": core["metrics"],
            "ref_modes": core["ref_modes"],
            "rejections": core["rejections"],
            "flush_reasons": core["flush_reasons"],
            "transfer_model": core["transfer_model"],
            "feature_coverage": core["feature_coverage"],
            "ibc_deferred_fills": core["ibc_deferred_fills"],
        }
        for name, core in profile["fused_windows"]["cores"].items()
    }


def histogram_quantile_upper(buckets: list[int], percentile: int) -> int:
    count = sum(buckets)
    if count == 0:
        return 0
    rank = (count * percentile + 99) // 100
    cumulative = 0
    for index, frequency in enumerate(buckets):
        cumulative += frequency
        if cumulative >= rank:
            if index == 0:
                return 0
            if index == 64:
                return (1 << 64) - 1
            return (1 << index) - 1
    raise RuntimeError("fused-window histogram rank was not reached")


def aggregate_fused_profiles(profiles: list[dict]) -> dict:
    """Merge constant-memory histograms and counters across independently decoded corpus rows."""
    aggregate: dict[str, dict] = {}
    for core_name in ("core_a", "core_b_rpr"):
        cores = [profile["fused_windows"]["cores"][core_name] for profile in profiles]
        windows = sum(core["windows"] for core in cores)
        coverage = {
            field: sum(core["coverage"][field] for core in cores)
            for field in cores[0]["coverage"]
        }
        metrics = {}
        for metric_name in cores[0]["metrics"]:
            buckets = [sum(core["metrics"][metric_name]["log2_buckets"][index] for core in cores)
                       for index in range(65)]
            metrics[metric_name] = {
                "total": sum(core["metrics"][metric_name]["total"] for core in cores),
                "p50_upper": histogram_quantile_upper(buckets, 50),
                "p90_upper": histogram_quantile_upper(buckets, 90),
                "p99_upper": histogram_quantile_upper(buckets, 99),
                "max": max(core["metrics"][metric_name]["max"] for core in cores),
                "log2_buckets": buckets,
            }
        ref_mode_names = set().union(*(core["ref_modes"] for core in cores))
        ref_modes = {
            name: {
                "prediction_units": sum(core["ref_modes"].get(name, {}).get("prediction_units", 0) for core in cores),
                "windows": sum(core["ref_modes"].get(name, {}).get("windows", 0) for core in cores),
            }
            for name in sorted(ref_mode_names)
        }
        rejection_names = set().union(*(core["rejections"] for core in cores))
        rejections = {
            name: {
                "cus": sum(core["rejections"].get(name, {}).get("cus", 0) for core in cores),
                "luma_pixels": sum(core["rejections"].get(name, {}).get("luma_pixels", 0) for core in cores),
            }
            for name in sorted(rejection_names)
        }
        flush_names = set().union(*(core["flush_reasons"] for core in cores))
        flush_reasons = {
            name: sum(core["flush_reasons"].get(name, 0) for core in cores)
            for name in sorted(flush_names)
        }
        transfer_fields = set().union(*(core["transfer_model"] for core in cores))
        transfer = {
            field: sum(core["transfer_model"].get(field, 0) for core in cores)
            for field in sorted(transfer_fields)
            if field != "shared_bytes_amortized_per_window"
        }
        transfer["shared_bytes_amortized_per_window"] = (
            (transfer["shared_bytes_once"] + windows - 1) // windows if windows else 0
        )
        feature_fields = set().union(*(core["feature_coverage"] for core in cores))
        feature_coverage = {
            field: sum(core["feature_coverage"].get(field, 0) for core in cores)
            for field in sorted(feature_fields)
        }
        fill_fields = set().union(*(core["ibc_deferred_fills"] for core in cores))
        ibc_fills = {
            field: (max(core["ibc_deferred_fills"].get(field, 0) for core in cores)
                    if field == "max_pending"
                    else sum(core["ibc_deferred_fills"].get(field, 0) for core in cores))
            for field in sorted(fill_fields)
            if field not in ("last_queued_sequence", "last_applied_sequence", "pending")
        }
        ibc_fills["pending"] = 0
        aggregate[core_name] = {
            "coverage": coverage, "windows": windows, "metrics": metrics, "ref_modes": ref_modes,
            "rejections": rejections, "flush_reasons": flush_reasons, "transfer_model": transfer,
            "feature_coverage": feature_coverage, "ibc_deferred_fills": ibc_fills,
        }
    return aggregate


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
        "fused_window_summaries": {},
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
            if any(core["windows"] == 0 for core in profile["fused_windows"]["cores"].values()):
                raise RuntimeError(f"{case_name}: generated stream contains no fused candidate window")
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
            report["fused_window_summaries"][case_name] = fused_case_summary(profile)
            (output_dir / "decoder-batch-profile-report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8")

    if not args.dry_run:
        report["fused_window_aggregate"] = aggregate_fused_profiles(
            [case["profile"] for case in report["cases"].values()]
        )
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
