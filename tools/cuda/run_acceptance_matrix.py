#!/usr/bin/env python3
"""Run the reproducible external VTM CPU/CUDA acceptance matrix.

Acceptance is deliberately strict: a versioned manifest must describe four
immutable 4:2:0 inputs (1080p/4K, 8/10 bit) and the three profile configs.  A
preflight contains exactly 288 commands (12 cases x 4 run groups x 6 runs).
Legacy command-line inputs remain available only for explicitly labelled smoke
runs and are never reported as acceptance evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


RUNNER_VERSION = "3.0.0"
REPORT_SCHEMA = 3
MANIFEST_SCHEMA = 1
ACCEPTANCE_REPEATS = 5
EXPECTED_ACCEPTANCE_PROCESSES = 288
PROFILES = {
    "AI": "encoder_intra_vtm.cfg",
    "RA": "encoder_randomaccess_vtm.cfg",
    "LD": "encoder_lowdelay_vtm.cfg",
}
RESOLUTIONS = {
    "1080p": (1920, 1080),
    "4K": (3840, 2160),
}
HEX_SHA256_RE = re.compile(r"^[0-9A-Fa-f]{64}$")

SAD_RE = re.compile(r"CUDA SAD batches:\s*(\d+), failures:\s*(\d+), disabled:\s*(\d+)")
QPA_RE = re.compile(
    r"CUDA QPA batches:\s*(\d+), tasks:\s*(\d+), failures:\s*(\d+), "
    r"fallbacks:\s*(\d+), enabled:\s*(\d+), poisoned:\s*(\d+), disabled-by-flag:\s*(\d+)"
)
MIRROR_MEMORY_RE = re.compile(
    r"CUDA mirror bytes device current/peak:\s*(\d+)/(\d+), pinned current/peak:\s*(\d+)/(\d+)"
)
MIRROR_TRANSFER_RE = re.compile(r"CUDA mirror transfers uploaded/downloaded:\s*(\d+)/(\d+) bytes")
MIRROR_BUDGET_RE = re.compile(r"CUDA mirror budget:\s*(\d+) bytes, rejections:\s*(\d+)")
DEPTH_RE = re.compile(
    r"(?P<kind>Input|Internal) bit depth\s*:\s*\(Y:(?P<y>\d+),\s*C:(?P<c>\d+)\)"
)
# This line is intentionally not synthesized by the runner.  Current binaries
# do not emit it, so executed acceptance fails with a precise telemetry gap.
SEARCH_COUNTERS_RE = re.compile(
    r"(?:VTM )?search counters IMV/Affine AMVR:\s*(\d+)/(\d+)", re.IGNORECASE
)
CHAIN_RE = re.compile(
    r"CUDA loop-filter chain frames/pixels/DBF tasks/SAO CTUs/ALF CTUs:\s*"
    r"(\d+)/(\d+)/(\d+)/(\d+)/(\d+),\s*"
    r"params/internal copies/mirror up/down:\s*(\d+)/(\d+)/(\d+)/(\d+) bytes,\s*"
    r"scratch current/retired/peak:\s*(\d+)/(\d+)/(\d+) bytes,\s*"
    r"syncs runtime/integration:\s*(\d+)/(\d+),\s*"
    r"time collection/LMCS/DBF/SAO/ALF/runtime/integration:\s*"
    r"([0-9]+(?:\.[0-9]+)?)/([0-9]+(?:\.[0-9]+)?)/([0-9]+(?:\.[0-9]+)?)/"
    r"([0-9]+(?:\.[0-9]+)?)/([0-9]+(?:\.[0-9]+)?)/([0-9]+(?:\.[0-9]+)?)/"
    r"([0-9]+(?:\.[0-9]+)?) ms,\s*"
    r"failures/not-eligible:\s*(\d+)/(\d+),\s*enabled:\s*(\d+),\s*"
    r"poisoned:\s*(\d+),\s*disabled-by-flag:\s*(\d+)",
    re.DOTALL,
)


class ValidationError(ValueError):
    """A reproducibility or acceptance contract was violated."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def checked_file(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return path


def file_record(path: Path) -> dict:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def expected_yuv420_size(width: int, height: int, bit_depth: int, frames: int) -> int:
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValidationError("4:2:0 dimensions must be positive and even")
    if bit_depth not in (8, 10):
        raise ValidationError("source bit depth must be 8 or 10")
    if frames <= 0:
        raise ValidationError("frames must be positive")
    bytes_per_sample = 1 if bit_depth == 8 else 2
    return width * height * 3 // 2 * bytes_per_sample * frames


def _required_object(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValidationError(f"{label} must be a JSON object")
    return value


def _required_int(mapping: dict, key: str, label: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{label}.{key} must be an integer")
    return value


def _required_hash(mapping: dict, key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not HEX_SHA256_RE.fullmatch(value):
        raise ValidationError(f"{label}.{key} must be an expected SHA-256 hash")
    return value.upper()


def _manifest_file(
    mapping: dict,
    label: str,
    hash_key: str,
    *,
    root: Path,
    must_be_external: bool,
) -> tuple[Path, str]:
    raw_path = mapping.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ValidationError(f"{label}.path must be absolute")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise ValidationError(f"{label}.path does not exist: {path}")
    if must_be_external and _is_within(path, root):
        raise ValidationError(f"{label}.path must be outside the repository: {path}")
    expected_hash = _required_hash(mapping, hash_key, label)
    actual_hash = sha256(path)
    if actual_hash != expected_hash:
        raise ValidationError(
            f"{label} SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        )
    return path, actual_hash


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    config: Path
    sha256: str

    def record(self) -> dict:
        return {"name": self.name, "config": file_record(self.config)}


@dataclass(frozen=True)
class InputSpec:
    name: str
    resolution: str
    width: int
    height: int
    chroma: str
    source_bit_depth: int
    internal_bit_depth: int
    frames: int
    path: Path
    expected_size_bytes: int
    sha256: str
    sequence_config: Path
    sequence_config_sha256: str

    @property
    def key(self) -> tuple[str, int]:
        return self.resolution, self.source_bit_depth

    def record(self) -> dict:
        return {
            "name": self.name,
            "resolution": self.resolution,
            "width": self.width,
            "height": self.height,
            "chroma": self.chroma,
            "source_bit_depth": self.source_bit_depth,
            "internal_bit_depth": self.internal_bit_depth,
            "output_bit_depth": self.internal_bit_depth,
            "frames": self.frames,
            "expected_size_bytes": self.expected_size_bytes,
            "input": file_record(self.path),
            "sequence_config": file_record(self.sequence_config),
        }


@dataclass
class Run:
    milliseconds: float
    command: list[str]
    stdout: str
    stderr: str
    stdout_log: Path
    stderr_log: Path


def load_acceptance_manifest(path: Path, root: Path) -> tuple[list[ProfileSpec], list[InputSpec], dict]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read manifest {path}: {error}") from error
    manifest = _required_object(raw, "manifest")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValidationError(f"manifest.schema must be {MANIFEST_SCHEMA}")
    overrides = _required_object(manifest.get("encoder_overrides"), "encoder_overrides")
    if overrides.get("CCALF") != 0:
        raise ValidationError("manifest.encoder_overrides.CCALF must explicitly be 0")

    raw_profiles = _required_object(manifest.get("profile_configs"), "profile_configs")
    if set(raw_profiles) != set(PROFILES):
        raise ValidationError("profile_configs must contain exactly AI, RA, and LD")
    profiles: list[ProfileSpec] = []
    for name in PROFILES:
        entry = _required_object(raw_profiles[name], f"profile_configs.{name}")
        config, actual_hash = _manifest_file(
            entry, f"profile_configs.{name}", "sha256", root=root, must_be_external=False
        )
        profiles.append(ProfileSpec(name, config, actual_hash))

    raw_inputs = manifest.get("inputs")
    if not isinstance(raw_inputs, list) or len(raw_inputs) != 4:
        raise ValidationError("manifest.inputs must contain exactly four entries")
    inputs: list[InputSpec] = []
    seen_names: set[str] = set()
    seen_paths: set[Path] = set()
    for index, item in enumerate(raw_inputs):
        label = f"inputs[{index}]"
        entry = _required_object(item, label)
        name = entry.get("name")
        resolution = entry.get("resolution")
        chroma = str(entry.get("chroma"))
        if not isinstance(name, str) or not name:
            raise ValidationError(f"{label}.name must be a non-empty string")
        if name in seen_names:
            raise ValidationError(f"duplicate input name: {name}")
        if resolution not in RESOLUTIONS:
            raise ValidationError(f"{label}.resolution must be 1080p or 4K")
        width = _required_int(entry, "width", label)
        height = _required_int(entry, "height", label)
        if (width, height) != RESOLUTIONS[resolution]:
            raise ValidationError(
                f"{label} dimensions {width}x{height} do not match {resolution}"
            )
        if chroma != "420":
            raise ValidationError(f"{label}.chroma must be \"420\"")
        source_depth = _required_int(entry, "source_bit_depth", label)
        internal_depth = _required_int(entry, "internal_bit_depth", label)
        if source_depth not in (8, 10) or internal_depth != source_depth:
            raise ValidationError(
                f"{label} source/internal bit depths must be the same and either 8 or 10"
            )
        frames = _required_int(entry, "frames", label)
        if frames < 4:
            raise ValidationError(f"{label}.frames must be at least 4 for RA/LD acceptance")
        expected_size = _required_int(entry, "expected_size_bytes", label)
        calculated_size = expected_yuv420_size(width, height, source_depth, frames)
        if expected_size != calculated_size:
            raise ValidationError(
                f"{label}.expected_size_bytes is {expected_size}, expected {calculated_size}"
            )
        input_path, input_hash = _manifest_file(
            entry, label, "sha256", root=root, must_be_external=True
        )
        if input_path in seen_paths:
            raise ValidationError(f"input files must be distinct: {input_path}")
        if input_path.stat().st_size != expected_size:
            raise ValidationError(
                f"{label} size mismatch: expected {expected_size}, got {input_path.stat().st_size}"
            )
        sequence = _required_object(entry.get("sequence_config"), f"{label}.sequence_config")
        sequence_path, sequence_hash = _manifest_file(
            sequence,
            f"{label}.sequence_config",
            "sha256",
            root=root,
            must_be_external=False,
        )
        inputs.append(
            InputSpec(
                name,
                resolution,
                width,
                height,
                chroma,
                source_depth,
                internal_depth,
                frames,
                input_path,
                expected_size,
                input_hash,
                sequence_path,
                sequence_hash,
            )
        )
        seen_names.add(name)
        seen_paths.add(input_path)
    expected_keys = {(resolution, depth) for resolution in RESOLUTIONS for depth in (8, 10)}
    actual_keys = {item.key for item in inputs}
    if actual_keys != expected_keys:
        raise ValidationError(
            "inputs must contain exactly 1080p8, 1080p10, 4K8, and 4K10"
        )
    return profiles, sorted(inputs, key=lambda item: (list(RESOLUTIONS).index(item.resolution), item.source_bit_depth)), manifest


def execute(command: list[str], cwd: Path, log_base: Path) -> Run:
    log_base.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        check=False,
    )
    elapsed = (time.perf_counter() - start) * 1000.0
    stdout_log = Path(f"{log_base}.stdout.log")
    stderr_log = Path(f"{log_base}.stderr.log")
    stdout_log.write_text(process.stdout, encoding="utf-8")
    stderr_log.write_text(process.stderr, encoding="utf-8")
    run = Run(elapsed, command, process.stdout, process.stderr, stdout_log, stderr_log)
    if process.returncode != 0:
        rendered = subprocess.list2cmdline(command)
        raise RuntimeError(
            f"command failed ({process.returncode}): {rendered}\n"
            f"stdout: {stdout_log}\nstderr: {stderr_log}"
        )
    return run


def parse_effective_depths(text: str, expected: int) -> dict:
    parsed: dict[str, dict[str, int]] = {}
    for match in DEPTH_RE.finditer(text):
        parsed[match.group("kind").lower()] = {
            "luma": int(match.group("y")),
            "chroma": int(match.group("c")),
        }
    if set(parsed) != {"input", "internal"}:
        raise RuntimeError("encoder effective input/internal bit-depth telemetry is missing")
    if any(value != expected for pair in parsed.values() for value in pair.values()):
        raise RuntimeError(
            f"encoder effective bit depth differs from the {expected}-bit case: {parsed}"
        )
    parsed["output"] = {"luma": expected, "chroma": expected, "source": "fixed-command"}
    return parsed


def parse_search_counters(text: str, required: bool) -> dict:
    counter_matches = SEARCH_COUNTERS_RE.findall(text)
    if counter_matches:
        return {
            "status": "available",
            "imv": int(counter_matches[-1][0]),
            "affine_amvr": int(counter_matches[-1][1]),
        }
    unavailable = {
        "status": "unavailable",
        "imv": None,
        "affine_amvr": None,
        "reason": "EncoderApp does not emit numeric IMV/Affine AMVR counters",
    }
    if required:
        raise RuntimeError(
            "required numeric IMV/Affine AMVR telemetry is unavailable; acceptance cannot be claimed"
        )
    return unavailable


def parse_cpu_encoder_telemetry(text: str, expected_bit_depth: int, require_search_counters: bool) -> dict:
    return {
        "effective_bit_depths": parse_effective_depths(text, expected_bit_depth),
        "search_counters": parse_search_counters(text, require_search_counters),
    }


def parse_encoder_telemetry(
    text: str,
    expect_sad: bool,
    expected_bit_depth: int | None = None,
    require_search_counters: bool = False,
) -> dict:
    sad_matches = SAD_RE.findall(text)
    qpa_matches = QPA_RE.findall(text)
    mirror_memory_matches = MIRROR_MEMORY_RE.findall(text)
    mirror_transfer_matches = MIRROR_TRANSFER_RE.findall(text)
    mirror_budget_matches = MIRROR_BUDGET_RE.findall(text)
    if not sad_matches or not qpa_matches:
        raise RuntimeError("CUDA encoder telemetry is missing; the run may have fallen back to CPU")
    if not mirror_memory_matches or not mirror_transfer_matches or not mirror_budget_matches:
        raise RuntimeError("CUDA encoder mirror telemetry is missing")
    sad_values = [int(value) for value in sad_matches[-1]]
    qpa_values = [int(value) for value in qpa_matches[-1]]
    sad = dict(zip(("batches", "failures", "disabled"), sad_values))
    qpa = dict(
        zip(
            ("batches", "tasks", "failures", "fallbacks", "enabled", "poisoned", "disabled_by_flag"),
            qpa_values,
        )
    )
    if sad["failures"] != 0 or sad["disabled"] != 0:
        raise RuntimeError(f"CUDA SAD reported failure or disablement: {sad}")
    if expect_sad and sad["batches"] == 0:
        raise RuntimeError("CUDA SAD was expected for an inter profile but dispatched zero batches")
    if (
        qpa["batches"] == 0
        or qpa["tasks"] == 0
        or qpa["failures"] != 0
        or qpa["fallbacks"] != 0
        or qpa["enabled"] != 1
        or qpa["poisoned"] != 0
        or qpa["disabled_by_flag"] != 0
    ):
        raise RuntimeError(f"CUDA QPA did not execute cleanly: {qpa}")
    memory_values = [int(value) for value in mirror_memory_matches[-1]]
    transfer_values = [int(value) for value in mirror_transfer_matches[-1]]
    budget_values = [int(value) for value in mirror_budget_matches[-1]]
    mirror = {
        "device_current_bytes": memory_values[0],
        "device_peak_bytes": memory_values[1],
        "pinned_current_bytes": memory_values[2],
        "pinned_peak_bytes": memory_values[3],
        "uploaded_bytes": transfer_values[0],
        "downloaded_bytes": transfer_values[1],
        "budget_bytes": budget_values[0],
        "budget_rejections": budget_values[1],
    }
    if mirror["budget_rejections"] != 0:
        raise RuntimeError(f"CUDA mirror budget rejected allocations: {mirror}")
    search_counters = parse_search_counters(text, require_search_counters)
    telemetry = {"sad": sad, "qpa": qpa, "mirror": mirror, "search_counters": search_counters}
    if expected_bit_depth is not None:
        telemetry["effective_bit_depths"] = parse_effective_depths(text, expected_bit_depth)
    return telemetry


def parse_decoder_telemetry(text: str) -> dict:
    matches = CHAIN_RE.findall(text)
    if not matches:
        raise RuntimeError("CUDA loop-filter telemetry is missing; the run may have fallen back to CPU")
    raw = matches[-1]
    integers = [int(value) for value in raw[:14]]
    timings = [float(value) for value in raw[14:21]]
    gates = [int(value) for value in raw[21:]]
    telemetry = {
        "batches": {
            "dispatches": integers[0],
            "pixels": integers[1],
            "dbf_tasks": integers[2],
            "sao_ctus": integers[3],
            "alf_ctus": integers[4],
        },
        "transfers_bytes": {
            "parameters": integers[5],
            "internal_copies": integers[6],
            "mirror_upload": integers[7],
            "mirror_download": integers[8],
        },
        "scratch_bytes": {"current": integers[9], "retired": integers[10], "peak": integers[11]},
        "synchronizations": {"runtime": integers[12], "integration": integers[13]},
        "timings_ms": dict(
            zip(("collection", "lmcs", "dbf", "sao", "alf", "runtime", "integration"), timings)
        ),
        "gates": dict(
            zip(("failures", "not_eligible", "enabled", "poisoned", "disabled_by_flag"), gates)
        ),
    }
    gate = telemetry["gates"]
    if (
        telemetry["batches"]["dispatches"] == 0
        or gate["failures"] != 0
        or gate["not_eligible"] != 0
        or gate["enabled"] != 1
        or gate["poisoned"] != 0
        or gate["disabled_by_flag"] != 0
    ):
        raise RuntimeError(f"CUDA loop-filter chain did not execute cleanly: {telemetry}")
    return telemetry


def deterministic_hashes(paths: Iterable[Path]) -> list[str]:
    hashes = [sha256(path) for path in paths]
    if len(set(hashes)) != 1:
        raise RuntimeError(f"repeated outputs are not deterministic: {hashes}")
    return hashes


def validate_search_counter_parity(cpu_result: dict, cuda_result: dict, label: str) -> None:
    def series(result: dict) -> list[tuple[int, int]]:
        values = []
        for run in result.get("runs", []):
            counters = (run.get("telemetry") or {}).get("search_counters") or {}
            if counters.get("status") != "available":
                raise RuntimeError(f"{label}: numeric IMV/Affine AMVR counters are unavailable")
            values.append((counters["imv"], counters["affine_amvr"]))
        return values

    cpu_values = series(cpu_result)
    cuda_values = series(cuda_result)
    if not cpu_values or len(cpu_values) != len(cuda_values):
        raise RuntimeError(f"{label}: IMV/Affine AMVR counter run counts differ")
    if len(set(cpu_values)) != 1 or len(set(cuda_values)) != 1:
        raise RuntimeError(
            f"{label}: repeated IMV/Affine AMVR counters are nondeterministic: "
            f"CPU={cpu_values}, CUDA={cuda_values}"
        )
    if cpu_values != cuda_values:
        raise RuntimeError(
            f"{label}: CPU/CUDA IMV/Affine AMVR counters differ: "
            f"CPU={cpu_values}, CUDA={cuda_values}"
        )


def prepare_outputs(outputs: Iterable[Path]) -> None:
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.unlink(missing_ok=True)


def ensure_outputs(outputs: Iterable[Path], label: str) -> None:
    missing = [str(output) for output in outputs if not output.is_file()]
    if missing:
        raise RuntimeError(f"{label} did not create expected artifacts: {missing}")


def ensure_output_sizes(outputs: Sequence[Path], expected_sizes: Sequence[int | None], label: str) -> None:
    if len(outputs) != len(expected_sizes):
        raise RuntimeError(f"{label} output-size contract has the wrong arity")
    mismatches = [
        f"{path}: expected {expected}, got {path.stat().st_size}"
        for path, expected in zip(outputs, expected_sizes)
        if expected is not None and path.stat().st_size != expected
    ]
    if mismatches:
        raise RuntimeError(f"{label} output size mismatch: {mismatches}")


def run_repeated(
    command_factory: Callable[[str], list[str]],
    outputs_factory: Callable[[str], list[Path]],
    cwd: Path,
    repeats: int,
    log_prefix: Path,
    validator: Callable[[str], dict] | None,
    dry_run: bool,
    expected_output_sizes: Sequence[int | None] | None = None,
) -> dict:
    tags = ["warmup"] + [f"run-{index + 1}" for index in range(repeats)]
    if dry_run:
        return {
            "status": "planned",
            "commands": [command_factory(tag) for tag in tags],
            "outputs": [[str(path) for path in outputs_factory(tag)] for tag in tags],
        }

    completed: list[tuple[str, Run, list[Path], dict | None]] = []
    for tag in tags:
        outputs = outputs_factory(tag)
        prepare_outputs(outputs)
        run = execute(command_factory(tag), cwd, Path(f"{log_prefix}-{tag}"))
        ensure_outputs(outputs, tag)
        if expected_output_sizes is not None:
            ensure_output_sizes(outputs, expected_output_sizes, tag)
        try:
            telemetry = validator(run.stdout + "\n" + run.stderr) if validator else None
        except RuntimeError as error:
            raise RuntimeError(
                f"{error}; stdout: {run.stdout_log}; stderr: {run.stderr_log}"
            ) from error
        completed.append((tag, run, outputs, telemetry))

    warm_tag, warm_run, warm_outputs, warm_telemetry = completed[0]
    for output in warm_outputs:
        output.unlink(missing_ok=True)
    measured = completed[1:]
    per_artifact = list(zip(*(outputs for _, _, outputs, _ in measured)))
    return {
        "status": "completed",
        "warmup": {
            "tag": warm_tag,
            "milliseconds": warm_run.milliseconds,
            "command": warm_run.command,
            "stdout_log": str(warm_run.stdout_log),
            "stderr_log": str(warm_run.stderr_log),
            "telemetry": warm_telemetry,
        },
        "median_ms": statistics.median(run.milliseconds for _, run, _, _ in measured),
        "runs": [
            {
                "tag": tag,
                "milliseconds": run.milliseconds,
                "command": run.command,
                "stdout_log": str(run.stdout_log),
                "stderr_log": str(run.stderr_log),
                "telemetry": telemetry,
            }
            for tag, run, _, telemetry in measured
        ],
        "hashes": [deterministic_hashes(paths) for paths in per_artifact],
        "outputs": [[str(path) for path in paths] for paths in per_artifact],
    }


def _option_name(argument: str) -> str:
    return argument.split("=", 1)[0].lstrip("-").lower()


def validate_extra_args(encoder_args: Sequence[str], decoder_args: Sequence[str]) -> None:
    reserved_encoder = {
        "c", "i", "b", "o", "f", "cf", "inputfile", "bitstreamfile", "reconfile", "framestobeencoded",
        "sourcewidth", "sourceheight", "inputbitdepth",
        "inputbitdepthc", "internalbitdepth", "outputbitdepth", "outputbitdepthc",
        "inputchromaformat", "chromaformat", "ccalf", "gpubackend", "gpudevice",
        "gpuexperimentalsad", "gpuexperimentalqpa",
    }
    reserved_decoder = {
        "b", "o", "d", "bitstreamfile", "reconfile", "outputbitdepth", "outputbitdepthc", "gpubackend",
        "gpudevice", "gpuexperimentalloopfilterchain",
    }
    collisions = sorted({_option_name(arg) for arg in encoder_args} & reserved_encoder)
    if collisions:
        raise ValidationError(f"--encoder-arg cannot override fixed acceptance options: {collisions}")
    collisions = sorted({_option_name(arg) for arg in decoder_args} & reserved_decoder)
    if collisions:
        raise ValidationError(f"--decoder-arg cannot override fixed acceptance options: {collisions}")


def resumable_case_is_valid(case: object, expected_repeats: int | None = None) -> bool:
    if not isinstance(case, dict) or case.get("status") != "completed":
        return False
    expected_artifacts = {"encode_cpu": 2, "encode_cuda": 2, "decode_cpu": 1, "decode_cuda": 1}
    for group_name in ("encode_cpu", "encode_cuda", "decode_cpu", "decode_cuda"):
        group = case.get(group_name)
        if not isinstance(group, dict) or group.get("status") != "completed":
            return False
        hashes = group.get("hashes")
        outputs = group.get("outputs")
        if not isinstance(hashes, list) or not isinstance(outputs, list) or len(hashes) != len(outputs):
            return False
        if len(hashes) != expected_artifacts[group_name]:
            return False
        for artifact_hashes, artifact_outputs in zip(hashes, outputs):
            if (
                not isinstance(artifact_hashes, list)
                or not isinstance(artifact_outputs, list)
                or len(artifact_hashes) != len(artifact_outputs)
            ):
                return False
            if expected_repeats is not None and len(artifact_hashes) != expected_repeats:
                return False
            for expected_hash, raw_path in zip(artifact_hashes, artifact_outputs):
                path = Path(raw_path)
                if (
                    not isinstance(expected_hash, str)
                    or not HEX_SHA256_RE.fullmatch(expected_hash)
                    or not path.is_file()
                    or sha256(path) != expected_hash.upper()
                ):
                    return False
    return True


def verify_contract_artifacts(
    runner_path: Path,
    args: argparse.Namespace,
    manifest_path: Path | None,
    profiles: Sequence[ProfileSpec],
    inputs: Sequence[InputSpec],
    expected_by_path: dict[str, str] | None = None,
) -> list[dict]:
    def expected(path: Path) -> str:
        resolved = str(path.resolve())
        if expected_by_path is not None and resolved in expected_by_path:
            return expected_by_path[resolved]
        return sha256(path)

    declared: list[tuple[str, Path, str]] = [
        ("runner", runner_path, expected(runner_path)),
        ("encoder", args.encoder, expected(args.encoder)),
        ("decoder", args.decoder, expected(args.decoder)),
    ]
    if manifest_path:
        declared.append(("manifest", manifest_path, expected(manifest_path)))
    declared.extend((f"profile:{item.name}", item.config, item.sha256) for item in profiles)
    declared.extend((f"sequence-config:{item.name}", item.sequence_config, item.sequence_config_sha256) for item in inputs)
    unique: dict[Path, tuple[str, str]] = {}
    for label, path, expected_hash in declared:
        resolved = path.resolve()
        if resolved not in unique:
            unique[resolved] = (label, expected_hash)
    records = []
    for path, (label, expected_hash) in unique.items():
        if expected_by_path is not None:
            expected_hash = expected_by_path.get(str(path), expected_hash)
        actual_hash = sha256(path) if path.is_file() else None
        records.append({
            "label": label,
            "path": str(path),
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "unchanged": actual_hash == expected_hash,
        })
    return records


def estimate_case_output_bytes(item: InputSpec, run_count: int) -> dict:
    yuv_artifacts = 4 * run_count  # two encoder recon groups and two decoder groups
    bitstream_artifacts = 2 * run_count
    yuv_bytes = yuv_artifacts * item.expected_size_bytes
    # Raw-input size is a conservative planning estimate, not a codec bound.
    estimated_bitstream_bytes = bitstream_artifacts * item.expected_size_bytes
    return {
        "yuv_artifacts": yuv_artifacts,
        "yuv_bytes_exact": yuv_bytes,
        "bitstream_artifacts": bitstream_artifacts,
        "bitstream_bytes_estimate": estimated_bitstream_bytes,
        "total_bytes_estimate": yuv_bytes + estimated_bitstream_bytes,
        "bitstream_estimate_method": "one raw input size per bitstream; planning estimate only",
    }


def count_planned_processes(cases: dict) -> int:
    count = 0
    for case in cases.values():
        for group in ("encode_cpu", "encode_cuda", "decode_cpu", "decode_cuda"):
            count += len(case[group].get("commands", []))
    return count


def verify_immutable_inputs(inputs: Sequence[InputSpec]) -> list[dict]:
    records = []
    for item in inputs:
        current_size = item.path.stat().st_size if item.path.is_file() else None
        current_hash = sha256(item.path) if item.path.is_file() else None
        record = {
            "name": item.name,
            "path": str(item.path),
            "expected_size_bytes": item.expected_size_bytes,
            "actual_size_bytes": current_size,
            "expected_sha256": item.sha256,
            "actual_sha256": current_hash,
            "unchanged": current_size == item.expected_size_bytes and current_hash == item.sha256,
        }
        records.append(record)
    return records


def _legacy_specs(args: argparse.Namespace, root: Path) -> tuple[list[ProfileSpec], list[InputSpec]]:
    if args.cfg_dir is None:
        raise ValidationError("legacy smoke mode requires --cfg-dir")
    cfg_dir = args.cfg_dir.resolve()
    if not cfg_dir.is_dir():
        raise ValidationError(f"config directory does not exist: {cfg_dir}")
    profiles = []
    for name in dict.fromkeys(args.profiles):
        path = (cfg_dir / PROFILES[name]).resolve()
        if not path.is_file():
            raise ValidationError(f"missing VTM profile config: {path}")
        profiles.append(ProfileSpec(name, path, sha256(path)))
    specs = []
    frames = args.frames or 4
    for depth in dict.fromkeys(args.bit_depths):
        input_path = getattr(args, f"input_{depth}")
        sequence = getattr(args, f"sequence_config_{depth}")
        if input_path is None or sequence is None:
            raise ValidationError(
                f"--input-{depth} and --sequence-config-{depth} are required for legacy smoke"
            )
        specs.append(
            InputSpec(
                f"legacy-{depth}", "legacy", 0, 0, "420", depth, depth, frames,
                input_path, input_path.stat().st_size, sha256(input_path), sequence, sha256(sequence)
            )
        )
    return profiles, specs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True, type=checked_file)
    parser.add_argument("--decoder", required=True, type=checked_file)
    parser.add_argument("--manifest", type=checked_file, help="schema-1 external acceptance manifest")
    parser.add_argument("--mode", choices=("acceptance", "smoke"), default="acceptance")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=ACCEPTANCE_REPEATS)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--encoder-arg", action="append", default=[])
    parser.add_argument("--decoder-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true", help="preflight, hash, and list commands without execution")
    parser.add_argument("--resume", action="store_true", help="reuse only complete cases from a matching report")
    # Legacy schema-2 CLI, intentionally smoke-only.
    parser.add_argument("--cfg-dir", type=Path)
    parser.add_argument("--input-8", type=checked_file)
    parser.add_argument("--input-10", type=checked_file)
    parser.add_argument("--sequence-config-8", type=checked_file)
    parser.add_argument("--sequence-config-10", type=checked_file)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--bit-depths", nargs="+", type=int, choices=(8, 10), default=[8, 10])
    parser.add_argument("--frames", type=int)
    return parser


def _report_contract(args: argparse.Namespace, manifest_path: Path | None, profiles: Sequence[ProfileSpec], inputs: Sequence[InputSpec]) -> str:
    value = {
        "runner": sha256(Path(__file__).resolve()),
        "encoder": sha256(args.encoder),
        "decoder": sha256(args.decoder),
        "manifest": sha256(manifest_path) if manifest_path else None,
        "mode": args.mode,
        "repeats": args.repeats,
        "device": args.device,
        "profiles": [(item.name, item.sha256) for item in profiles],
        "inputs": [(item.name, item.sha256, item.sequence_config_sha256) for item in inputs],
        "encoder_args": args.encoder_arg,
        "decoder_args": args.decoder_arg,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest().upper()


def run_matrix(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.frames is not None and args.frames <= 0:
        parser.error("--frames must be positive")
    if args.mode == "acceptance" and args.repeats != ACCEPTANCE_REPEATS:
        parser.error("acceptance requires exactly --repeats 5; use --mode smoke for other counts")
    legacy_used = any(
        value is not None
        for value in (
            args.cfg_dir, args.input_8, args.input_10, args.sequence_config_8,
            args.sequence_config_10, args.frames,
        )
    ) or args.profiles != list(PROFILES) or args.bit_depths != [8, 10]
    if args.mode == "acceptance" and legacy_used:
        parser.error("legacy input/config/profile/depth/frame flags are forbidden in acceptance mode; use --manifest")
    if args.mode == "acceptance" and args.manifest is None:
        parser.error("acceptance requires --manifest")
    if args.manifest is not None and legacy_used:
        parser.error("--manifest cannot be combined with legacy matrix flags")
    try:
        validate_extra_args(args.encoder_arg, args.decoder_arg)
    except ValidationError as error:
        parser.error(str(error))

    root = Path(__file__).resolve().parents[2]
    try:
        if args.manifest:
            profiles, inputs, manifest = load_acceptance_manifest(args.manifest, root)
        else:
            profiles, inputs = _legacy_specs(args, root)
            manifest = None
    except ValidationError as error:
        parser.error(str(error))

    if args.mode == "acceptance":
        if len(profiles) != 3 or len(inputs) != 4:
            parser.error("acceptance requires the complete 12-case resolution/profile/depth matrix")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "matrix-report.json"
    contract_hash = _report_contract(args, args.manifest, profiles, inputs)
    old_cases: dict = {}
    if args.resume and report_path.is_file():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("contract_sha256") != contract_hash:
            parser.error("--resume report contract does not match the current inputs/tools/options")
        old_cases = previous.get("cases", {})

    report = {
        "schema": REPORT_SCHEMA,
        "runner_version": RUNNER_VERSION,
        "contract_sha256": contract_hash,
        "execution_mode": args.mode,
        "dry_run": args.dry_run,
        "acceptance_claim": False,
        "status": "preflight" if args.dry_run else "running",
        "warmup_runs": 1,
        "measured_runs": args.repeats,
        "device": args.device,
        "matrix": {
            "resolutions": list(RESOLUTIONS) if args.mode == "acceptance" else sorted({item.resolution for item in inputs}),
            "profiles": [item.name for item in profiles],
            "source_bit_depths": sorted({item.source_bit_depth for item in inputs}),
            "internal_bit_depth_policy": "equal-to-source",
            "output_bit_depth_policy": "equal-to-internal",
            "chroma": "420",
            "case_count": len(profiles) * len(inputs),
        },
        "first_delivery_exclusions": {
            "CCALF": {
                "value": 0,
                "encoder_argument": "--CCALF=0",
                "manifest_value": manifest.get("encoder_overrides", {}).get("CCALF") if manifest else None,
                "reason": "explicitly excluded from the first CUDA loop-filter delivery",
            }
        },
        "telemetry_contract": {
            "encoder": ["SAD batches/gates", "QPA batches/tasks/gates", "mirror memory/transfers/rejections", "effective depths", "IMV/Affine AMVR counters"],
            "decoder": ["chain batches", "transfers", "scratch", "synchronizations", "per-stage timings", "fail-fast/fallback eligibility gates"],
            "known_binary_gap": "current EncoderApp does not emit numeric IMV/Affine AMVR counters; executed acceptance fails until it does",
        },
        "artifacts": {
            "runner": file_record(Path(__file__).resolve()),
            "encoder": file_record(args.encoder),
            "decoder": file_record(args.decoder),
            "manifest": file_record(args.manifest) if args.manifest else None,
            "profile_configs": {item.name: item.record() for item in profiles},
            "inputs": {item.name: item.record() for item in inputs},
        },
        "input_immutability": {"before": verify_immutable_inputs(inputs), "after": None},
        "artifact_immutability": {
            "before": verify_contract_artifacts(Path(__file__).resolve(), args, args.manifest, profiles, inputs),
            "after": None,
        },
        "preflight": {"expected_process_count": EXPECTED_ACCEPTANCE_PROCESSES if args.mode == "acceptance" else None},
        "cases": {},
        "failure": None,
    }
    total_estimated = 0

    try:
        for profile in profiles:
            for item in inputs:
                case_name = f"{item.resolution}-{profile.name}-{item.source_bit_depth}bit"
                reusable = old_cases.get(case_name)
                if reusable and not args.dry_run and resumable_case_is_valid(reusable, args.repeats):
                    reused = json.loads(json.dumps(reusable))
                    reused["reused_from_report"] = True
                    report["cases"][case_name] = reused
                    continue
                case_dir = output_dir / case_name
                case_dir.mkdir(parents=True, exist_ok=True)
                case: dict = {
                    "status": "running",
                    "resolution": item.resolution,
                    "width": item.width,
                    "height": item.height,
                    "chroma": item.chroma,
                    "source_bit_depth": item.source_bit_depth,
                    "internal_bit_depth": item.internal_bit_depth,
                    "output_bit_depth": item.internal_bit_depth,
                    "frames": item.frames,
                    "expected_input_size_bytes": item.expected_size_bytes,
                    "expected_input_sha256": item.sha256,
                    "profile": profile.name,
                    "expected_cuda_operations": ["QPA", "loop-filter-chain"] + ([] if profile.name == "AI" else ["SAD"]),
                    "output_space": estimate_case_output_bytes(item, args.repeats + 1),
                }
                total_estimated += case["output_space"]["total_bytes_estimate"]
                report["cases"][case_name] = case

                for backend in ("cpu", "cuda"):
                    def encode_outputs(tag: str, *, selected=backend, directory=case_dir) -> list[Path]:
                        return [directory / f"encode-{selected}-{tag}.vvc", directory / f"encode-{selected}-{tag}-rec.yuv"]

                    def encode_command(tag: str, *, selected=backend) -> list[str]:
                        bitstream, reconstruction = encode_outputs(tag)
                        command = [
                            str(args.encoder), "-c", str(profile.config), "-c", str(item.sequence_config),
                            f"--InputFile={item.path}", f"--BitstreamFile={bitstream}",
                            f"--ReconFile={reconstruction}", f"--FramesToBeEncoded={item.frames}",
                            f"--InputBitDepth={item.source_bit_depth}", f"--InputBitDepthC={item.source_bit_depth}",
                            f"--InternalBitDepth={item.internal_bit_depth}",
                            f"--OutputBitDepth={item.internal_bit_depth}", f"--OutputBitDepthC={item.internal_bit_depth}",
                            "--InputChromaFormat=420", "--ChromaFormat=420", "--CCALF=0",
                            "--PerceptQPA=1", "--SliceChromaQPOffsetPeriodicity=1",
                            f"--GPUBackend={selected}", f"--GPUDevice={args.device}",
                        ]
                        if item.width > 0 and item.height > 0:
                            command += [f"--SourceWidth={item.width}", f"--SourceHeight={item.height}"]
                        if selected == "cuda":
                            command += ["--GPUExperimentalSAD=1", "--GPUExperimentalQPA=1"]
                        return command + args.encoder_arg

                    if backend == "cuda":
                        validator = lambda text, inter=profile.name != "AI", depth=item.source_bit_depth: parse_encoder_telemetry(
                            text, inter, depth, require_search_counters=args.mode == "acceptance"
                        )
                    else:
                        validator = lambda text, depth=item.source_bit_depth: parse_cpu_encoder_telemetry(
                            text, depth, require_search_counters=args.mode == "acceptance"
                        )
                    case[f"encode_{backend}"] = run_repeated(
                        encode_command, encode_outputs, root, args.repeats,
                        case_dir / f"encode-{backend}", validator, args.dry_run,
                        (None, item.expected_size_bytes),
                    )

                cpu_bitstream = case_dir / "encode-cpu-run-1.vvc"
                if not args.dry_run:
                    if case["encode_cpu"]["hashes"][0][0] != case["encode_cuda"]["hashes"][0][0]:
                        raise RuntimeError(f"{case_name}: CPU/CUDA bitstreams differ")
                    if case["encode_cpu"]["hashes"][1][0] != case["encode_cuda"]["hashes"][1][0]:
                        raise RuntimeError(f"{case_name}: CPU/CUDA encoder reconstructions differ")
                    if args.mode == "acceptance":
                        validate_search_counter_parity(
                            case["encode_cpu"], case["encode_cuda"], case_name
                        )

                for backend in ("cpu", "cuda"):
                    def decode_outputs(tag: str, *, selected=backend, directory=case_dir) -> list[Path]:
                        return [directory / f"decode-{selected}-{tag}.yuv"]

                    def decode_command(tag: str, *, selected=backend) -> list[str]:
                        output = decode_outputs(tag)[0]
                        command = [
                            str(args.decoder), f"--BitstreamFile={cpu_bitstream}", f"--ReconFile={output}",
                            f"--OutputBitDepth={item.internal_bit_depth}", f"--OutputBitDepthC={item.internal_bit_depth}",
                            f"--GPUBackend={selected}", f"--GPUDevice={args.device}",
                        ]
                        if selected == "cuda":
                            command += ["--GPUExperimentalLoopFilterChain=1"]
                        return command + args.decoder_arg

                    validator = parse_decoder_telemetry if backend == "cuda" else None
                    case[f"decode_{backend}"] = run_repeated(
                        decode_command, decode_outputs, root, args.repeats,
                        case_dir / f"decode-{backend}", validator, args.dry_run,
                        (item.expected_size_bytes,),
                    )

                if not args.dry_run:
                    if case["decode_cpu"]["hashes"][0][0] != case["decode_cuda"]["hashes"][0][0]:
                        raise RuntimeError(f"{case_name}: CPU/CUDA decoded YUV differs")
                    case["status"] = "completed"
                else:
                    case["status"] = "planned"
                atomic_write_json(report_path, report)

        if args.dry_run:
            process_count = count_planned_processes(report["cases"])
            report["preflight"]["process_count"] = process_count
            report["preflight"]["commands_per_case"] = 4 * (args.repeats + 1)
            report["preflight"]["total_output_bytes_estimate"] = total_estimated
            report["preflight"]["space_available_bytes"] = shutil.disk_usage(output_dir).free
            report["preflight"]["estimated_space_fits"] = (
                total_estimated <= report["preflight"]["space_available_bytes"]
            )
            if args.mode == "acceptance" and process_count != EXPECTED_ACCEPTANCE_PROCESSES:
                raise RuntimeError(
                    f"acceptance preflight planned {process_count} processes, expected {EXPECTED_ACCEPTANCE_PROCESSES}"
                )
            report["status"] = "preflight-complete"
        else:
            report["status"] = "completed"
            report["acceptance_claim"] = args.mode == "acceptance"
    except Exception as error:
        report["status"] = "failed"
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        after = verify_immutable_inputs(inputs)
        report["input_immutability"]["after"] = after
        mutated = [item for item in after if not item["unchanged"]]
        initial_artifact_hashes = {
            item["path"]: item["expected_sha256"]
            for item in report["artifact_immutability"]["before"]
        }
        artifact_after = verify_contract_artifacts(
            Path(__file__).resolve(), args, args.manifest, profiles, inputs, initial_artifact_hashes
        )
        report["artifact_immutability"]["after"] = artifact_after
        mutated_artifacts = [item for item in artifact_after if not item["unchanged"]]
        if mutated or mutated_artifacts:
            report["status"] = "failed"
            report["acceptance_claim"] = False
            report["failure"] = {
                "type": "ArtifactMutationError",
                "message": (
                    f"declared artifacts changed during the run; inputs="
                    f"{[item['name'] for item in mutated]}, artifacts="
                    f"{[item['label'] for item in mutated_artifacts]}"
                ),
            }
        atomic_write_json(report_path, report)
    if any(not item["unchanged"] for item in report["input_immutability"]["after"]):
        raise RuntimeError("one or more external inputs changed during the run")
    if any(not item["unchanged"] for item in report["artifact_immutability"]["after"]):
        raise RuntimeError("one or more declared executables/configs changed during the run")
    print(report_path)
    return 0


def main() -> int:
    parser = build_parser()
    return run_matrix(parser.parse_args(), parser)


if __name__ == "__main__":
    raise SystemExit(main())
