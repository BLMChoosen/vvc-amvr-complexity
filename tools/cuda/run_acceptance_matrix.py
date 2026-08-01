#!/usr/bin/env python3
"""Run the reproducible VTM CPU/CUDA AI/RA/LD x 8/10 acceptance matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


RUNNER_VERSION = "2.0.0"
PROFILES = {
    "AI": "encoder_intra_vtm.cfg",
    "RA": "encoder_randomaccess_vtm.cfg",
    "LD": "encoder_lowdelay_vtm.cfg",
}
SAD_RE = re.compile(r"CUDA SAD batches:\s*(\d+), failures:\s*(\d+), disabled:\s*(\d+)")
QPA_RE = re.compile(
    r"CUDA QPA batches:\s*(\d+), tasks:\s*(\d+), failures:\s*(\d+), "
    r"fallbacks:\s*(\d+), enabled:\s*(\d+), poisoned:\s*(\d+), disabled-by-flag:\s*(\d+)"
)
CHAIN_RE = re.compile(
    r"CUDA loop-filter chain frames/pixels/DBF tasks/SAO CTUs/ALF CTUs:\s*"
    r"(\d+)/(\d+)/(\d+)/(\d+)/(\d+),.*?failures/not-eligible:\s*(\d+)/(\d+),\s*"
    r"enabled:\s*(\d+),\s*poisoned:\s*(\d+),\s*disabled-by-flag:\s*(\d+)",
    re.DOTALL,
)


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
    return {"path": str(path), "sha256": sha256(path)}


@dataclass
class Run:
    milliseconds: float
    command: list[str]
    stdout: str
    stderr: str
    stdout_log: Path
    stderr_log: Path


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


def parse_encoder_telemetry(text: str, expect_sad: bool) -> dict:
    sad_matches = SAD_RE.findall(text)
    qpa_matches = QPA_RE.findall(text)
    if not sad_matches or not qpa_matches:
        raise RuntimeError("CUDA encoder telemetry is missing; the run may have fallen back to CPU")
    sad_values = [int(value) for value in sad_matches[-1]]
    qpa_values = [int(value) for value in qpa_matches[-1]]
    sad = dict(zip(("batches", "failures", "disabled"), sad_values))
    qpa = dict(zip(("batches", "tasks", "failures", "fallbacks", "enabled", "poisoned",
                    "disabled_by_flag"), qpa_values))
    if sad["failures"] != 0 or sad["disabled"] != 0:
        raise RuntimeError(f"CUDA SAD reported failure or disablement: {sad}")
    if expect_sad and sad["batches"] == 0:
        raise RuntimeError("CUDA SAD was expected for an inter profile but dispatched zero batches")
    if (qpa["batches"] == 0 or qpa["tasks"] == 0 or qpa["failures"] != 0
            or qpa["fallbacks"] != 0 or qpa["enabled"] != 1 or qpa["poisoned"] != 0
            or qpa["disabled_by_flag"] != 0):
        raise RuntimeError(f"CUDA QPA did not execute cleanly: {qpa}")
    return {"sad": sad, "qpa": qpa}


def parse_decoder_telemetry(text: str) -> dict:
    matches = CHAIN_RE.findall(text)
    if not matches:
        raise RuntimeError("CUDA loop-filter telemetry is missing; the run may have fallen back to CPU")
    values = [int(value) for value in matches[-1]]
    telemetry = dict(zip(
        ("dispatches", "pixels", "dbf_tasks", "sao_ctus", "alf_ctus", "failures",
         "not_eligible", "enabled", "poisoned", "disabled_by_flag"), values
    ))
    if (telemetry["dispatches"] == 0 or telemetry["failures"] != 0
            or telemetry["not_eligible"] != 0 or telemetry["enabled"] != 1
            or telemetry["poisoned"] != 0 or telemetry["disabled_by_flag"] != 0):
        raise RuntimeError(f"CUDA loop-filter chain did not execute cleanly: {telemetry}")
    return telemetry


def deterministic_hashes(paths: Iterable[Path]) -> list[str]:
    hashes = [sha256(path) for path in paths]
    if len(set(hashes)) != 1:
        raise RuntimeError(f"repeated outputs are not deterministic: {hashes}")
    return hashes


def prepare_outputs(outputs: Iterable[Path]) -> None:
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.unlink(missing_ok=True)


def ensure_outputs(outputs: Iterable[Path], label: str) -> None:
    missing = [str(output) for output in outputs if not output.is_file()]
    if missing:
        raise RuntimeError(f"{label} did not create expected artifacts: {missing}")


def run_repeated(
    command_factory: Callable[[str], list[str]],
    outputs_factory: Callable[[str], list[Path]],
    cwd: Path,
    repeats: int,
    log_prefix: Path,
    validator: Callable[[str], dict] | None,
    dry_run: bool,
) -> dict:
    tags = ["warmup"] + [f"run-{index + 1}" for index in range(repeats)]
    if dry_run:
        return {
            "dry_run": True,
            "commands": [command_factory(tag) for tag in tags],
            "outputs": [[str(path) for path in outputs_factory(tag)] for tag in tags],
        }

    completed: list[tuple[str, Run, list[Path], dict | None]] = []
    for tag in tags:
        outputs = outputs_factory(tag)
        prepare_outputs(outputs)
        run = execute(command_factory(tag), cwd, Path(f"{log_prefix}-{tag}"))
        ensure_outputs(outputs, tag)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True, type=checked_file)
    parser.add_argument("--decoder", required=True, type=checked_file)
    parser.add_argument("--cfg-dir", required=True, type=Path)
    parser.add_argument("--input-8", type=checked_file)
    parser.add_argument("--input-10", type=checked_file)
    parser.add_argument("--sequence-config-8", type=checked_file)
    parser.add_argument("--sequence-config-10", type=checked_file)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--bit-depths", nargs="+", type=int, choices=(8, 10), default=[8, 10])
    parser.add_argument("--frames", type=int, help="override profile defaults (AI=1, RA/LD=4 minimum)")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--encoder-arg", action="append", default=[])
    parser.add_argument("--decoder-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true", help="validate and hash inputs, then write commands only")
    args = parser.parse_args()
    if args.frames is not None and args.frames <= 0:
        parser.error("--frames must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.frames is not None and any(profile != "AI" for profile in args.profiles) and args.frames < 4:
        parser.error("RA/LD acceptance requires --frames >= 4")

    selected_inputs: dict[int, tuple[Path, Path]] = {}
    for depth in dict.fromkeys(args.bit_depths):
        input_path = getattr(args, f"input_{depth}")
        sequence = getattr(args, f"sequence_config_{depth}")
        if input_path is None or sequence is None:
            parser.error(f"--input-{depth} and --sequence-config-{depth} are required for {depth}-bit cases")
        selected_inputs[depth] = (input_path, sequence)

    root = Path(__file__).resolve().parents[2]
    cfg_dir = args.cfg_dir.resolve()
    if not cfg_dir.is_dir():
        parser.error(f"config directory does not exist: {cfg_dir}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_configs: dict[str, Path] = {}
    for profile in dict.fromkeys(args.profiles):
        config = (cfg_dir / PROFILES[profile]).resolve()
        if not config.is_file():
            parser.error(f"missing VTM profile config: {config}")
        profile_configs[profile] = config

    report_path = output_dir / "matrix-report.json"
    report = {
        "schema": 2,
        "runner_version": RUNNER_VERSION,
        "runner": file_record(Path(__file__).resolve()),
        "mode": "dry-run" if args.dry_run else "acceptance",
        "warmup_runs": 1,
        "measured_runs": args.repeats,
        "frame_policy": {"AI_default": 1, "RA_LD_default_and_minimum": 4,
                         "override": args.frames},
        "device": args.device,
        "artifacts": {
            "encoder": file_record(args.encoder),
            "decoder": file_record(args.decoder),
            "profile_configs": {name: file_record(path) for name, path in profile_configs.items()},
            "inputs": {
                str(depth): {"input": file_record(path), "sequence_config": file_record(sequence)}
                for depth, (path, sequence) in selected_inputs.items()
            },
        },
        "cases": {},
    }

    for profile, profile_config in profile_configs.items():
        frames = args.frames if args.frames is not None else (1 if profile == "AI" else 4)
        for bit_depth, (input_path, sequence_config) in selected_inputs.items():
            case_name = f"{profile}-{bit_depth}bit"
            case_dir = output_dir / case_name
            case_dir.mkdir(parents=True, exist_ok=True)
            case: dict = {"frames": frames, "expected_cuda_operations": ["QPA", "loop-filter-chain"]}
            if profile != "AI":
                case["expected_cuda_operations"].append("SAD")
            for backend in ("cpu", "cuda"):
                def encode_outputs(tag: str, *, directory=case_dir, selected=backend) -> list[Path]:
                    return [directory / f"{selected}-{tag}.vvc", directory / f"{selected}-{tag}-rec.yuv"]

                def encode_command(tag: str, *, selected=backend) -> list[str]:
                    bitstream, reconstruction = encode_outputs(tag)
                    command = [
                        str(args.encoder), "-c", str(profile_config), "-c", str(sequence_config),
                        f"--InputFile={input_path}", f"--BitstreamFile={bitstream}",
                        f"--ReconFile={reconstruction}", f"--FramesToBeEncoded={frames}",
                        "--PerceptQPA=1", "--SliceChromaQPOffsetPeriodicity=1",
                        f"--GPUBackend={selected}", f"--GPUDevice={args.device}",
                    ]
                    if selected == "cuda":
                        command += ["--GPUExperimentalSAD=1", "--GPUExperimentalQPA=1"]
                    return command + args.encoder_arg

                validator = ((lambda text, expect=profile != "AI": parse_encoder_telemetry(text, expect))
                             if backend == "cuda" else None)
                case[f"encode_{backend}"] = run_repeated(
                    encode_command, encode_outputs, root, args.repeats,
                    case_dir / f"encode-{backend}", validator, args.dry_run)

            cpu_bitstream = Path(case["encode_cpu"]["outputs"][0][0])
            if not args.dry_run:
                if case["encode_cpu"]["hashes"][0][0] != case["encode_cuda"]["hashes"][0][0]:
                    raise RuntimeError(f"{case_name}: CPU/CUDA bitstreams differ")
                if case["encode_cpu"]["hashes"][1][0] != case["encode_cuda"]["hashes"][1][0]:
                    raise RuntimeError(f"{case_name}: CPU/CUDA encoder reconstructions differ")

            for backend in ("cpu", "cuda"):
                def decode_outputs(tag: str, *, directory=case_dir, selected=backend) -> list[Path]:
                    return [directory / f"decode-{selected}-{tag}.yuv"]

                def decode_command(tag: str, *, selected=backend) -> list[str]:
                    output = decode_outputs(tag)[0]
                    command = [
                        str(args.decoder), f"--BitstreamFile={cpu_bitstream}", f"--ReconFile={output}",
                        f"--GPUBackend={selected}", f"--GPUDevice={args.device}",
                    ]
                    if selected == "cuda":
                        command += ["--GPUExperimentalLoopFilterChain=1"]
                    return command + args.decoder_arg

                validator = parse_decoder_telemetry if backend == "cuda" else None
                case[f"decode_{backend}"] = run_repeated(
                    decode_command, decode_outputs, root, args.repeats,
                    case_dir / f"decode-{backend}", validator, args.dry_run)

            if not args.dry_run and case["decode_cpu"]["hashes"][0][0] != case["decode_cuda"]["hashes"][0][0]:
                raise RuntimeError(f"{case_name}: CPU/CUDA decoded YUV differs")
            report["cases"][case_name] = case
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
