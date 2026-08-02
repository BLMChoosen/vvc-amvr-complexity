#!/usr/bin/env python3
"""Focused tests for the decoder batching measurement runner."""

import json
import tempfile
import unittest
from pathlib import Path

from run_decoder_batch_profile import (
    PROFILE_PREFIX,
    aggregate_fused_profiles,
    decoder_manifest_path,
    generate_input,
    fused_case_summary,
    parse_profile,
    sha256,
    verify_build_binding,
    verify_build_mode,
)


def fused_windows() -> dict:
    unit_metric = {
        "total": 1, "p50_upper": 1, "p90_upper": 1, "p99_upper": 1, "max": 1,
        "log2_buckets": [0, 1] + [0] * 63,
    }
    metrics = {
        name: dict(unit_metric)
        for name in (
            "cus", "tus", "component_pixels", "prediction_operations", "transform_tasks",
            "descriptor_bytes", "qcoeff_bytes", "dirty_download_bytes",
        )
    }
    metrics["luma_pixels"] = {
        "total": 16, "p50_upper": 31, "p90_upper": 31, "p99_upper": 31, "max": 16,
        "log2_buckets": [0] * 5 + [1] + [0] * 59,
    }
    metrics["total_transfer_bytes"] = {
        "total": 3, "p50_upper": 3, "p90_upper": 3, "p99_upper": 3, "max": 3,
        "log2_buckets": [0, 0, 1] + [0] * 62,
    }
    transfer = {
        "descriptor_h2d_bytes": 1, "qcoeff_h2d_bytes": 1, "shared_bytes_once": 1,
        "shared_scan_bytes_once": 0, "shared_matrix_bytes_once": 0,
        "shared_dequant_constant_bytes_once": 0, "shared_picture_metadata_bytes_once": 1,
        "shared_slice_metadata_bytes_once": 0, "shared_reference_metadata_bytes_once": 0,
        "shared_weighted_prediction_metadata_bytes_once": 0, "shared_rpr_metadata_bytes_once": 0,
        "shared_bcw_metadata_bytes_once": 0, "shared_bytes_amortized_per_window": 1,
        "dirty_boundary_d2h_bytes": 1, "estimated_h2d_bytes_run": 3,
        "estimated_d2h_bytes_run": 1, "estimated_total_transfer_bytes_run": 4,
    }
    core = {
        "coverage": {
            "considered_cus": 2, "considered_luma_pixels": 32,
            "eligible_cus": 1, "eligible_luma_pixels": 16,
            "rejected_cus": 1, "rejected_luma_pixels": 16,
        },
        "windows": 1, "metrics": metrics, "ref_modes": {"uni": {"prediction_units": 1, "windows": 1}},
        "rejections": {"intra_or_plt": {"cus": 1, "luma_pixels": 16}},
        "flush_reasons": {"stream_end": 1}, "transfer_model": transfer,
        "feature_coverage": {
            "rpr_cus": 0, "rpr_prediction_operations": 0,
            "weighted_prediction_units": 0, "bcw_prediction_units": 0,
        },
        "ibc_deferred_fills": {
            "queued": 0, "applied": 0, "max_pending": 0, "pending": 0, "immediate": 0,
            "disabled_sps_noops": 0, "last_queued_sequence": 0, "last_applied_sequence": 0,
            "boundary_checks": 1, "boundary_violations": 0,
        },
    }
    return {
        "model": "mc_dequant_inverse_transform_reconstruction",
        "assumptions": {
            "references_resident": True, "output_mirror_resident": True,
            "heterogeneous_mc_paths": True, "heterogeneous_components": True,
            "heterogeneous_transform_shapes": True, "heterogeneous_ts_dct2": True,
            "dirty_output_downloaded_at_cpu_boundaries": True,
        },
        "descriptor_layout_bytes": {
            "cu": 64, "prediction_operation": 104, "transform_task": 20,
            "qcoeff_element": 4, "pel_element": 2,
        },
        "shared_metadata_layout_bytes": {
            "picture": 72, "slice": 40, "reference": 80,
            "weighted_prediction": 88, "rpr": 44, "bcw": 4,
        },
        "cores": {"core_a": json.loads(json.dumps(core)), "core_b_rpr": json.loads(json.dumps(core))},
    }


def self_test_marker() -> dict:
    return {"passed": True, "weighted_metadata_extra_bytes": 88, "rpr_metadata_extra_bytes": 44}


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
            "schema": 6,
            "process_id": 10,
            "decoder_instance": 1,
            "owner_thread_hash": 20,
            "report_thread_hash": 20,
            "hook_calls": 3,
            "thread_mismatch_hooks": 0,
            "scheduler_self_test": self_test_marker(),
            "ibc_buffer_fills": {
                "queued": 4, "applied": 4, "last_queued_sequence": 4,
                "last_applied_sequence": 4, "boundary_violations": 0,
                "prepare_checks": 1, "prepare_violations": 0, "pending_at_report": 0,
            },
            "inverse_transform": {"runs": 1, "tasks_max": 2},
            "transform_workload": {"totals": {"tasks": 2, "pixels": 32}},
            "fused_windows": fused_windows(),
        })
        parsed = parse_profile(f"decoder output\n{PROFILE_PREFIX}{payload}\n")
        self.assertEqual(parsed["inverse_transform"]["tasks_max"], 2)

    def test_two_concurrent_instance_records_are_independently_parseable(self):
        def record(process_id):
            return PROFILE_PREFIX + json.dumps({
                "schema": 6, "process_id": process_id, "decoder_instance": 1,
                "owner_thread_hash": process_id + 10, "report_thread_hash": process_id + 10,
                "hook_calls": 4, "thread_mismatch_hooks": 0,
                "scheduler_self_test": self_test_marker(),
                "ibc_buffer_fills": {
                    "queued": 2, "applied": 2, "last_queued_sequence": 2,
                    "last_applied_sequence": 2, "boundary_violations": 0,
                    "prepare_checks": 1, "prepare_violations": 0, "pending_at_report": 0,
                },
                "transform_workload": {"totals": {"tasks": 1, "pixels": 16}},
                "fused_windows": fused_windows(),
            })
        first = parse_profile(record(100))
        second = parse_profile(record(101))
        self.assertNotEqual(
            (first["process_id"], first["decoder_instance"]),
            (second["process_id"], second["decoder_instance"]),
        )

    def test_pending_ibc_fill_cannot_cross_boundary(self):
        payload = {
            "schema": 6, "process_id": 1, "decoder_instance": 1,
            "owner_thread_hash": 2, "report_thread_hash": 2,
            "hook_calls": 1, "thread_mismatch_hooks": 0,
            "scheduler_self_test": self_test_marker(),
            "ibc_buffer_fills": {
                "queued": 2, "applied": 1, "last_queued_sequence": 2,
                "last_applied_sequence": 1, "boundary_violations": 1,
                "prepare_checks": 1, "prepare_violations": 1, "pending_at_report": 1,
            },
            "transform_workload": {"totals": {"tasks": 1, "pixels": 16}},
            "fused_windows": fused_windows(),
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

    def test_fused_candidate_finishes_after_reconstruction_before_ibc_fill(self):
        source = (Path(__file__).resolve().parents[2] / "source/Lib/DecoderLib/DecCu.cpp").read_text(
            encoding="utf-8")
        switch = source.index("switch( currCU.predMode )")
        recon = source.index("xReconInter( currCU );", switch)
        finish = source.index("xProfileFinishFusedCu();", recon)
        fill = source.index("m_pcInterPred->xFillIBCBuffer(currCU);", finish)
        self.assertLess(recon, finish)
        self.assertLess(finish, fill)

    def test_fused_windows_do_not_use_legacy_path_change_as_a_boundary(self):
        source = (Path(__file__).resolve().parents[2] / "source/Lib/DecoderLib/DecCu.cpp").read_text(
            encoding="utf-8")
        fused_reason = source[source.index("enum class FusedReason"):source.index("const char *const fusedCoreNames")]
        self.assertNotIn("PATH_CHANGE", fused_reason)
        self.assertIn("else if (fusedCandidate.rpr) rejectFusedCandidate(FusedReason::RPR, true);", source)

    def test_schema_five_and_failed_self_test_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "unsupported decoder batching schema: 5"):
            parse_profile(PROFILE_PREFIX + json.dumps({"schema": 5}))
        payload = {
            "schema": 6, "process_id": 1, "decoder_instance": 1,
            "owner_thread_hash": 2, "report_thread_hash": 2, "hook_calls": 1,
            "thread_mismatch_hooks": 0,
            "scheduler_self_test": {"passed": False},
        }
        with self.assertRaisesRegex(RuntimeError, "self-test is missing or failed"):
            parse_profile(PROFILE_PREFIX + json.dumps(payload))

    def test_fused_rejection_and_quantile_mismatches_are_rejected(self):
        payload = {
            "schema": 6, "process_id": 1, "decoder_instance": 1,
            "owner_thread_hash": 2, "report_thread_hash": 2, "hook_calls": 1,
            "thread_mismatch_hooks": 0, "scheduler_self_test": self_test_marker(),
            "ibc_buffer_fills": {
                "queued": 0, "applied": 0, "last_queued_sequence": 0, "last_applied_sequence": 0,
                "boundary_violations": 0, "prepare_checks": 0, "prepare_violations": 0,
                "pending_at_report": 0,
            },
            "transform_workload": {"totals": {"tasks": 1, "pixels": 16}},
            "fused_windows": fused_windows(),
        }
        payload["fused_windows"]["cores"]["core_a"]["rejections"]["intra_or_plt"]["cus"] = 0
        with self.assertRaisesRegex(RuntimeError, "rejection totals"):
            parse_profile(PROFILE_PREFIX + json.dumps(payload))
        payload["fused_windows"] = fused_windows()
        payload["fused_windows"]["cores"]["core_a"]["metrics"]["cus"]["p50_upper"] = 3
        with self.assertRaisesRegex(RuntimeError, "quantiles"):
            parse_profile(PROFILE_PREFIX + json.dumps(payload))

    def test_missing_or_duplicate_profile_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            parse_profile("no profile")
        record = PROFILE_PREFIX + '{"schema":6}'
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

    def test_fused_summary_retains_both_cores_and_transfer_model(self):
        profile = {"fused_windows": fused_windows()}
        summary = fused_case_summary(profile)
        self.assertEqual(set(summary), {"core_a", "core_b_rpr"})
        self.assertEqual(summary["core_a"]["transfer_model"]["estimated_total_transfer_bytes_run"], 4)

    def test_fused_histograms_aggregate_across_corpus_rows(self):
        first = {"fused_windows": fused_windows()}
        second = {"fused_windows": fused_windows()}
        aggregate = aggregate_fused_profiles([first, second])
        self.assertEqual(aggregate["core_a"]["windows"], 2)
        self.assertEqual(aggregate["core_a"]["metrics"]["cus"]["total"], 2)
        self.assertEqual(aggregate["core_a"]["metrics"]["cus"]["log2_buckets"][1], 2)
        self.assertEqual(aggregate["core_a"]["transfer_model"]["estimated_total_transfer_bytes_run"], 8)

    def test_fused_transfer_mismatch_is_rejected(self):
        payload = {
            "schema": 6, "process_id": 1, "decoder_instance": 1,
            "owner_thread_hash": 2, "report_thread_hash": 2, "hook_calls": 1,
            "thread_mismatch_hooks": 0,
            "scheduler_self_test": self_test_marker(),
            "ibc_buffer_fills": {
                "queued": 0, "applied": 0, "last_queued_sequence": 0, "last_applied_sequence": 0,
                "boundary_violations": 0, "prepare_checks": 0, "prepare_violations": 0,
                "pending_at_report": 0,
            },
            "transform_workload": {"totals": {"tasks": 1, "pixels": 16}},
            "fused_windows": fused_windows(),
        }
        payload["fused_windows"]["cores"]["core_a"]["transfer_model"]["estimated_h2d_bytes_run"] = 99
        with self.assertRaisesRegex(RuntimeError, "does not balance"):
            parse_profile(PROFILE_PREFIX + json.dumps(payload))

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
