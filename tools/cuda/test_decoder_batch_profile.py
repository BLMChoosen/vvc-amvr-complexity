#!/usr/bin/env python3
"""Focused tests for the decoder batching measurement runner."""

import json
import tempfile
import unittest
from pathlib import Path

from run_decoder_batch_profile import (
    PROFILE_PREFIX,
    decoder_manifest_path,
    generate_input,
    parse_profile,
    sha256,
    verify_build_binding,
    verify_build_mode,
)


class DecoderBatchProfileRunnerTest(unittest.TestCase):
    def test_generated_inputs_are_deterministic_and_sized(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.yuv"
            second = Path(directory) / "second.yuv"
            generate_input(first, 16, 8, 4, 10)
            generate_input(second, 16, 8, 4, 10)
            self.assertEqual(first.stat().st_size, 16 * 8 * 3 // 2 * 2 * 4)
            self.assertEqual(sha256(first), sha256(second))

    def test_profile_json_contract(self):
        payload = json.dumps({
            "schema": 4,
            "process_id": 10,
            "decoder_instance": 1,
            "owner_thread_hash": 20,
            "report_thread_hash": 20,
            "hook_calls": 3,
            "thread_mismatch_hooks": 0,
            "ibc_buffer_fills": {
                "queued": 4, "applied": 4, "last_queued_sequence": 4,
                "last_applied_sequence": 4, "boundary_violations": 0,
                "prepare_checks": 1, "prepare_violations": 0, "pending_at_report": 0,
            },
            "inverse_transform": {"runs": 1, "tasks_max": 2},
            "transform_workload": {"totals": {"tasks": 2, "pixels": 32}},
        })
        parsed = parse_profile(f"decoder output\n{PROFILE_PREFIX}{payload}\n")
        self.assertEqual(parsed["inverse_transform"]["tasks_max"], 2)

    def test_two_concurrent_instance_records_are_independently_parseable(self):
        def record(process_id):
            return PROFILE_PREFIX + json.dumps({
                "schema": 4, "process_id": process_id, "decoder_instance": 1,
                "owner_thread_hash": process_id + 10, "report_thread_hash": process_id + 10,
                "hook_calls": 4, "thread_mismatch_hooks": 0,
                "ibc_buffer_fills": {
                    "queued": 2, "applied": 2, "last_queued_sequence": 2,
                    "last_applied_sequence": 2, "boundary_violations": 0,
                    "prepare_checks": 1, "prepare_violations": 0, "pending_at_report": 0,
                },
                "transform_workload": {"totals": {"tasks": 1, "pixels": 16}},
            })
        first = parse_profile(record(100))
        second = parse_profile(record(101))
        self.assertNotEqual(
            (first["process_id"], first["decoder_instance"]),
            (second["process_id"], second["decoder_instance"]),
        )

    def test_pending_ibc_fill_cannot_cross_boundary(self):
        payload = {
            "schema": 4, "process_id": 1, "decoder_instance": 1,
            "owner_thread_hash": 2, "report_thread_hash": 2,
            "hook_calls": 1, "thread_mismatch_hooks": 0,
            "ibc_buffer_fills": {
                "queued": 2, "applied": 1, "last_queued_sequence": 2,
                "last_applied_sequence": 1, "boundary_violations": 1,
                "prepare_checks": 1, "prepare_violations": 1, "pending_at_report": 1,
            },
            "transform_workload": {"totals": {"tasks": 1, "pixels": 16}},
        }
        with self.assertRaisesRegex(RuntimeError, "crossed a boundary"):
            parse_profile(PROFILE_PREFIX + json.dumps(payload))

    def test_ibc_pre_mv_hook_precedes_motion_derivation_and_prepare(self):
        source = (Path(__file__).resolve().parents[2] / "source/Lib/DecoderLib/DecCu.cpp").read_text(
            encoding="utf-8")
        hook = source.index("if (CU::isIBC(currCU)) xProfileIbcPreMvConsumer();")
        derive = source.index("xDeriveCuMvs(currCU);", hook)
        prepare = source.index("const int mcProfilePath = xProfilePrepareMcCu(currCU);", derive)
        self.assertLess(hook, derive)
        self.assertLess(derive, prepare)

    def test_missing_or_duplicate_profile_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            parse_profile("no profile")
        record = PROFILE_PREFIX + '{"schema":4}'
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            parse_profile(record + "\n" + record)

    def test_build_mode_requires_matching_cmake_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            cache = build / "CMakeCache.txt"
            cache.write_text("VTM_ENABLE_DECODER_BATCH_PROFILING:BOOL=ON\n", encoding="utf-8")
            self.assertEqual(verify_build_mode(build, "ON"), cache)
            with self.assertRaisesRegex(RuntimeError, "does not declare"):
                verify_build_mode(build, "OFF")
            cache.unlink()
            with self.assertRaisesRegex(RuntimeError, "cache is missing"):
                verify_build_mode(build, "ON")

    def test_manifest_binds_mode_executable_and_cache_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "normal-build"
            build.mkdir()
            cache = build / "CMakeCache.txt"
            cache.write_text("VTM_ENABLE_DECODER_BATCH_PROFILING:BOOL=OFF\n", encoding="utf-8")
            decoder = root / "DecoderApp.exe"
            decoder.write_bytes(b"normal decoder")
            manifest = decoder_manifest_path(decoder)

            def write_manifest(mode="OFF", executable_hash=None):
                manifest.write_text(json.dumps({
                    "schema": 1,
                    "artifact": decoder.name,
                    "executable_sha256": executable_hash or sha256(decoder),
                    "decoder_batch_profiling": mode,
                    "configuration": "Release",
                    "cmake_cache_sha256": sha256(cache),
                }), encoding="utf-8")

            write_manifest()
            self.assertEqual(verify_build_binding(decoder, build, "OFF")[1], manifest)
            decoder.write_bytes(b"profiling decoder copied over normal")
            with self.assertRaisesRegex(RuntimeError, "executable SHA-256"):
                verify_build_binding(decoder, build, "OFF")
            write_manifest(mode="ON")
            with self.assertRaisesRegex(RuntimeError, "manifest mode"):
                verify_build_binding(decoder, build, "OFF")


if __name__ == "__main__":
    unittest.main()
