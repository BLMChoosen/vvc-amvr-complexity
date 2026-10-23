#!/usr/bin/env python3
"""
AMVR Batch Dataset Collector
Automates parallel dataset collection across multiple sequences, QPs, and coding configurations (RA / LD).
"""

import sys
import os
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

CONFIG_MAP = {
    "ra": "cfg/encoder_randomaccess_vtm.cfg",
    "randomaccess": "cfg/encoder_randomaccess_vtm.cfg",
    "ldp": "cfg/encoder_lowdelay_P_vtm.cfg",
    "lowdelay_p": "cfg/encoder_lowdelay_P_vtm.cfg",
    "ldb": "cfg/encoder_lowdelay_vtm.cfg",
    "lowdelay": "cfg/encoder_lowdelay_vtm.cfg",
}

def run_job(cmd, log_path):
    with open(log_path, "w") as log_f:
        proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    return proc.returncode

def main():
    parser = argparse.ArgumentParser(description="Batch collection of AMVR dataset across multiple configurations")
    parser.add_argument("--encoder", type=str, default="bin/vs18/msvc-19.51/x86_64/release/EncoderApp.exe", help="Path to EncoderApp executable")
    parser.add_argument("--seq-cfgs", nargs="+", required=True, help="List of per-sequence cfg files (e.g., cfg/per-sequence/BasketballPass.cfg)")
    parser.add_argument("--qps", type=int, nargs="+", default=[22, 27, 32, 37], help="List of QPs to test (default: 22 27 32 37)")
    parser.add_argument("--configs", nargs="+", default=["ra", "ldp"], choices=list(CONFIG_MAP.keys()), help="Encoder configurations: ra, ldp, ldb (default: ra ldp)")
    parser.add_argument("--frames", type=int, default=32, help="Number of frames per run (default: 32)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel processes (default: 4)")
    parser.add_argument("--outdir", type=str, default="dataset_bins", help="Output directory for generated .bin files")
    args = parser.parse_args()

    encoder_exe = os.path.abspath(args.encoder)
    if not os.path.exists(encoder_exe):
        print(f"Error: Encoder executable not found at: {encoder_exe}")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    logs_dir = os.path.join(args.outdir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    null_device = "NUL" if os.name == "nt" else "/dev/null"

    tasks = []
    for seq_cfg in args.seq_cfgs:
        seq_name = os.path.splitext(os.path.basename(seq_cfg))[0]
        for cfg_key in args.configs:
            cfg_file = CONFIG_MAP[cfg_key.lower()]
            for qp in args.qps:
                dump_filename = f"amvr_{seq_name}_{cfg_key}_qp{qp}.bin"
                dump_path = os.path.join(args.outdir, dump_filename)
                log_path = os.path.join(logs_dir, f"{seq_name}_{cfg_key}_qp{qp}.log")

                cmd = [
                    encoder_exe,
                    "-c", cfg_file,
                    "-c", seq_cfg,
                    "-q", str(qp),
                    "-f", str(args.frames),
                    "-b", null_device,  # discard bitstream to save disk space
                    "-o", null_device,  # discard reconstructed YUV
                    "--AmvrNeuralMode=1",
                    f"--AmvrDumpFile={dump_path}"
                ]
                job_desc = f"[{seq_name} | {cfg_key.upper()} | QP {qp}]"
                tasks.append((job_desc, cmd, log_path, dump_path))

    print(f"==================================================")
    print(f" AMVR Dataset Batch Collector")
    print(f" Total jobs planned: {len(tasks)}")
    print(f" Parallel workers:   {args.workers}")
    print(f" Output folder:      {args.outdir}")
    print(f"==================================================\n")

    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {
            executor.submit(run_job, cmd, log_path): (desc, dump_path)
            for desc, cmd, log_path, dump_path in tasks
        }

        for future in as_completed(future_to_job):
            desc, dump_path = future_to_job[future]
            try:
                ret = future.result()
                completed += 1
                if ret == 0 and os.path.exists(dump_path):
                    size_kb = os.path.getsize(dump_path) / 1024.0
                    print(f"[{completed}/{len(tasks)}] SUCCESS: {desc} -> {os.path.basename(dump_path)} ({size_kb:.1f} KB)")
                else:
                    print(f"[{completed}/{len(tasks)}] FAILED (code {ret}): {desc}")
            except Exception as e:
                print(f"[{completed}/{len(tasks)}] ERROR: {desc} -> {e}")

    print("\nBatch execution finished.")
    print(f"To merge all generated files into a single master dataset, run:")
    print(f"  python tools/amvr/merge_datasets.py {args.outdir}/amvr_master.bin {args.outdir}/amvr_*.bin")

if __name__ == '__main__':
    main()
