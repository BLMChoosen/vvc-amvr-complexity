#!/usr/bin/env python3
"""Focused tests for the schema-3 external acceptance runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_acceptance_matrix as runner


ENCODER_OK = """
Input bit depth                        : (Y:8, C:8)
Internal bit depth                     : (Y:8, C:8)
CUDA SAD batches: 7, failures: 0, disabled: 0
CUDA QPA batches: 2, tasks: 510, failures: 0, fallbacks: 0, enabled: 1, poisoned: 0, disabled-by-flag: 0
CUDA mirror bytes device current/peak: 0/1000, pinned current/peak: 0/500
CUDA mirror transfers uploaded/downloaded: 2048/1024 bytes
CUDA mirror budget: 100000 bytes, rejections: 0
VTM search counters IMV/Affine AMVR: 91/17
"""

DECODER_OK = """
CUDA loop-filter chain frames/pixels/DBF tasks/SAO CTUs/ALF CTUs: 4/8294400/983/2040/2040,
params/internal copies/mirror up/down: 1/2/3/4 bytes,
scratch current/retired/peak: 5/6/7 bytes, syncs runtime/integration: 8/9,
time collection/LMCS/DBF/SAO/ALF/runtime/integration: 1.0/2.0/3.0/4.0/5.0/6.0/7.0 ms,
failures/not-eligible: 0/0, enabled: 1, poisoned: 0, disabled-by-flag: 0
"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


class TelemetryTest(unittest.TestCase):
    def test_encoder_complete_structured_telemetry(self):
        parsed = runner.parse_encoder_telemetry(ENCODER_OK, True, 8, True)
        self.assertEqual(parsed["sad"]["batches"], 7)
        self.assertEqual(parsed["qpa"]["tasks"], 510)
        self.assertEqual(parsed["mirror"]["uploaded_bytes"], 2048)
        self.assertEqual(parsed["search_counters"]["affine_amvr"], 17)
        self.assertEqual(parsed["effective_bit_depths"]["internal"]["luma"], 8)

    def test_encoder_gates_and_search_counter_gap(self):
        with self.assertRaisesRegex(RuntimeError, "QPA did not execute"):
            runner.parse_encoder_telemetry(ENCODER_OK.replace("fallbacks: 0", "fallbacks: 1"), False)
        with self.assertRaisesRegex(RuntimeError, "mirror budget"):
            runner.parse_encoder_telemetry(ENCODER_OK.replace("rejections: 0", "rejections: 1"), False)
        without_counter = ENCODER_OK.replace("VTM search counters IMV/Affine AMVR: 91/17", "")
        parsed = runner.parse_encoder_telemetry(without_counter, False)
        self.assertEqual(parsed["search_counters"]["status"], "unavailable")
        self.assertIsNone(parsed["search_counters"]["imv"])
        with self.assertRaisesRegex(RuntimeError, "acceptance cannot be claimed"):
            runner.parse_encoder_telemetry(without_counter, False, 8, True)

    def test_inter_sad_and_effective_depth_are_enforced(self):
        with self.assertRaisesRegex(RuntimeError, "zero batches"):
            runner.parse_encoder_telemetry(ENCODER_OK.replace("batches: 7", "batches: 0", 1), True)
        with self.assertRaisesRegex(RuntimeError, "differs from the 8-bit case"):
            runner.parse_encoder_telemetry(
                ENCODER_OK.replace("Internal bit depth                     : (Y:8, C:8)",
                                   "Internal bit depth                     : (Y:10, C:10)"),
                False,
                8,
            )

    def test_cpu_cuda_search_counter_parity_and_repeat_determinism(self):
        def result(values):
            return {
                "runs": [
                    {"telemetry": {"search_counters": {"status": "available", "imv": imv, "affine_amvr": affine}}}
                    for imv, affine in values
                ]
            }

        runner.validate_search_counter_parity(result([(9, 2)] * 5), result([(9, 2)] * 5), "case")
        with self.assertRaisesRegex(RuntimeError, "CPU/CUDA.*differ"):
            runner.validate_search_counter_parity(result([(9, 2)] * 5), result([(10, 2)] * 5), "case")
        with self.assertRaisesRegex(RuntimeError, "nondeterministic"):
            runner.validate_search_counter_parity(
                result([(9, 2), (10, 2), (9, 2), (9, 2), (9, 2)]), result([(9, 2)] * 5), "case"
            )

    def test_decoder_complete_structured_telemetry_and_gates(self):
        parsed = runner.parse_decoder_telemetry(DECODER_OK)
        self.assertEqual(parsed["batches"]["dispatches"], 4)
        self.assertEqual(parsed["transfers_bytes"]["mirror_download"], 4)
        self.assertEqual(parsed["synchronizations"]["runtime"], 8)
        self.assertEqual(parsed["timings_ms"]["alf"], 5.0)
        with self.assertRaisesRegex(RuntimeError, "did not execute cleanly"):
            runner.parse_decoder_telemetry(DECODER_OK.replace("failures/not-eligible: 0/0", "failures/not-eligible: 0/1"))
        with self.assertRaisesRegex(RuntimeError, "fallen back to CPU"):
            runner.parse_decoder_telemetry("VTM decoder completed")


class ManifestTest(unittest.TestCase):
    def setUp(self):
        self.repo_tmp = tempfile.TemporaryDirectory()
        self.corpus_tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.repo_tmp.name).resolve()
        self.corpus = Path(self.corpus_tmp.name).resolve()
        self.profile_entries = {}
        for name in runner.PROFILES:
            path = self.root / f"{name}.cfg"
            path.write_text(f"profile={name}\n", encoding="utf-8")
            self.profile_entries[name] = {"path": str(path), "sha256": digest(path)}

    def tearDown(self):
        self.repo_tmp.cleanup()
        self.corpus_tmp.cleanup()

    def make_manifest(self) -> Path:
        inputs = []
        for index, (resolution, (width, height)) in enumerate(runner.RESOLUTIONS.items()):
            for depth in (8, 10):
                data = bytes([index * 10 + depth]) * (depth // 2)
                path = self.corpus / f"{resolution}-{depth}.yuv"
                path.write_bytes(data)
                sequence = self.root / f"{resolution}-{depth}.cfg"
                sequence.write_text(f"SourceWidth: {width}\nSourceHeight: {height}\n", encoding="utf-8")
                inputs.append({
                    "name": f"{resolution}-{depth}",
                    "resolution": resolution,
                    "width": width,
                    "height": height,
                    "chroma": "420",
                    "source_bit_depth": depth,
                    "internal_bit_depth": depth,
                    "frames": 4,
                    "expected_size_bytes": len(data),
                    "path": str(path),
                    "sha256": digest(path),
                    "sequence_config": {"path": str(sequence), "sha256": digest(sequence)},
                })
        manifest = {
            "schema": 1,
            "encoder_overrides": {"CCALF": 0},
            "profile_configs": self.profile_entries,
            "inputs": inputs,
        }
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def load_small(self, manifest: Path):
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        sizes = {
            (entry["width"], entry["height"], entry["source_bit_depth"], entry["frames"]): entry["expected_size_bytes"]
            for entry in raw["inputs"]
        }
        with mock.patch.object(runner, "expected_yuv420_size", side_effect=lambda w, h, d, f: sizes[(w, h, d, f)]):
            return runner.load_acceptance_manifest(manifest, self.root)

    def test_manifest_validates_complete_resolution_depth_matrix(self):
        profiles, inputs, raw = self.load_small(self.make_manifest())
        self.assertEqual([profile.name for profile in profiles], ["AI", "RA", "LD"])
        self.assertEqual({item.key for item in inputs}, {("1080p", 8), ("1080p", 10), ("4K", 8), ("4K", 10)})
        self.assertEqual(raw["encoder_overrides"]["CCALF"], 0)

    def test_rejects_internal_input_missing_hash_bad_size_and_mutation(self):
        manifest = self.make_manifest()
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        entry = raw["inputs"][0]
        internal = self.root / "inside.yuv"
        internal.write_bytes(Path(entry["path"]).read_bytes())
        entry["path"] = str(internal)
        entry["sha256"] = digest(internal)
        manifest.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(runner.ValidationError, "outside the repository"):
            self.load_small(manifest)

        manifest = self.make_manifest()
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        del raw["inputs"][0]["sha256"]
        manifest.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(runner.ValidationError, "expected SHA-256"):
            self.load_small(manifest)

        manifest = self.make_manifest()
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        raw["inputs"][0]["expected_size_bytes"] += 1
        manifest.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(runner.ValidationError, "size mismatch|expected_size_bytes"):
            self.load_small(manifest)

        manifest = self.make_manifest()
        _, inputs, _ = self.load_small(manifest)
        inputs[0].path.write_bytes(b"mutated")
        after = runner.verify_immutable_inputs(inputs)
        self.assertFalse(after[0]["unchanged"])

    def test_rejects_hash_depth_chroma_resolution_and_ccalf_mismatch(self):
        mutations = [
            (lambda raw: raw["inputs"][0].update(sha256="0" * 64), "SHA-256 mismatch"),
            (lambda raw: raw["inputs"][0].update(internal_bit_depth=10), "source/internal bit depths"),
            (lambda raw: raw["inputs"][0].update(chroma="444"), "chroma must"),
            (lambda raw: raw["inputs"][0].update(width=1280), "do not match"),
            (lambda raw: raw["encoder_overrides"].update(CCALF=1), "CCALF must explicitly be 0"),
        ]
        for mutate, message in mutations:
            with self.subTest(message=message):
                manifest = self.make_manifest()
                raw = json.loads(manifest.read_text(encoding="utf-8"))
                mutate(raw)
                manifest.write_text(json.dumps(raw), encoding="utf-8")
                with self.assertRaisesRegex(runner.ValidationError, message):
                    self.load_small(manifest)

    def test_profile_and_sequence_config_hashes_are_required_and_verified(self):
        manifest = self.make_manifest()
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        del raw["profile_configs"]["AI"]["sha256"]
        manifest.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(runner.ValidationError, "expected SHA-256"):
            self.load_small(manifest)

        manifest = self.make_manifest()
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        raw["inputs"][0]["sequence_config"]["sha256"] = "F" * 64
        manifest.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(runner.ValidationError, "SHA-256 mismatch"):
            self.load_small(manifest)


class ContractAndPreflightTest(unittest.TestCase):
    def test_acceptance_requires_five_repeats_and_forbids_legacy_flags(self):
        parser = runner.build_parser()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoder = root / "EncoderApp.exe"
            decoder = root / "DecoderApp.exe"
            encoder.write_bytes(b"encoder")
            decoder.write_bytes(b"decoder")
            base = ["--encoder", str(encoder), "--decoder", str(decoder), "--output-dir", str(root / "out")]
            args = parser.parse_args(base + ["--repeats", "4"])
            with self.assertRaises(SystemExit):
                runner.run_matrix(args, parser)
            args = parser.parse_args(base + ["--cfg-dir", str(root)])
            with self.assertRaises(SystemExit):
                runner.run_matrix(args, parser)

    def test_smoke_allows_nonstandard_repeat_and_marks_no_acceptance(self):
        parser = runner.build_parser()
        args = parser.parse_args(["--encoder", __file__, "--decoder", __file__, "--output-dir", ".", "--mode", "smoke", "--repeats", "2"])
        self.assertEqual(args.mode, "smoke")
        self.assertEqual(args.repeats, 2)

    def test_reserved_depth_and_backend_overrides_are_rejected(self):
        with self.assertRaisesRegex(runner.ValidationError, "fixed acceptance options"):
            runner.validate_extra_args(["--InternalBitDepth=10"], [])
        with self.assertRaisesRegex(runner.ValidationError, "fixed acceptance options"):
            runner.validate_extra_args([], ["--GPUBackend=cpu"])
        with self.assertRaisesRegex(runner.ValidationError, "fixed acceptance options"):
            runner.validate_extra_args(["-c"], [])

    def test_resume_rehashes_every_measured_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output.bin"
            output.write_bytes(b"stable")
            value_hash = digest(output)
            decode_group = {
                "status": "completed",
                "hashes": [[value_hash]],
                "outputs": [[str(output)]],
            }
            encode_group = {
                "status": "completed",
                "hashes": [[value_hash], [value_hash]],
                "outputs": [[str(output)], [str(output)]],
            }
            case = {
                "status": "completed",
                "encode_cpu": encode_group,
                "encode_cuda": encode_group,
                "decode_cpu": decode_group,
                "decode_cuda": decode_group,
            }
            self.assertTrue(runner.resumable_case_is_valid(case, 1))
            self.assertFalse(runner.resumable_case_is_valid(case, 5))
            output.write_bytes(b"corrupt")
            self.assertFalse(runner.resumable_case_is_valid(case, 1))

    def test_contract_artifact_final_rehash_uses_initial_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner_file = root / "runner.py"
            encoder = root / "encoder.exe"
            decoder = root / "decoder.exe"
            for path, data in ((runner_file, b"runner"), (encoder, b"encoder"), (decoder, b"decoder")):
                path.write_bytes(data)
            args = argparse.Namespace(encoder=encoder, decoder=decoder)
            before = runner.verify_contract_artifacts(runner_file, args, None, [], [])
            encoder.write_bytes(b"mutated")
            expected = {item["path"]: item["expected_sha256"] for item in before}
            after = runner.verify_contract_artifacts(runner_file, args, None, [], [], expected)
            encoder_record = next(item for item in after if item["label"] == "encoder")
            self.assertFalse(encoder_record["unchanged"])

    def test_preflight_has_12_cases_24_commands_each_and_288_total(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            encoder = base / "EncoderApp.exe"
            decoder = base / "DecoderApp.exe"
            manifest = base / "manifest.json"
            output = base / "out"
            encoder.write_bytes(b"encoder")
            decoder.write_bytes(b"decoder")
            manifest.write_text("{}", encoding="utf-8")
            profile_cfg = base / "profile.cfg"
            sequence_cfg = base / "sequence.cfg"
            profile_cfg.write_bytes(b"p")
            sequence_cfg.write_bytes(b"s")
            profiles = [runner.ProfileSpec(name, profile_cfg, digest(profile_cfg)) for name in runner.PROFILES]
            inputs = []
            for resolution, (width, height) in runner.RESOLUTIONS.items():
                for depth in (8, 10):
                    source = base / f"external-{resolution}-{depth}.yuv"
                    source.write_bytes(bytes([depth]))
                    inputs.append(runner.InputSpec(
                        f"{resolution}-{depth}", resolution, width, height, "420", depth, depth, 4,
                        source, 1, digest(source), sequence_cfg, digest(sequence_cfg)
                    ))
            parser = runner.build_parser()
            args = parser.parse_args([
                "--encoder", str(encoder), "--decoder", str(decoder), "--manifest", str(manifest),
                "--output-dir", str(output), "--dry-run",
            ])
            raw_manifest = {"encoder_overrides": {"CCALF": 0}}
            with mock.patch.object(runner, "load_acceptance_manifest", return_value=(profiles, inputs, raw_manifest)):
                self.assertEqual(runner.run_matrix(args, parser), 0)
            report = json.loads((output / "matrix-report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["schema"], 3)
            self.assertEqual(report["matrix"]["case_count"], 12)
            self.assertEqual(report["preflight"]["process_count"], 288)
            self.assertEqual(report["preflight"]["commands_per_case"], 24)
            self.assertFalse(report["acceptance_claim"])
            self.assertEqual(report["status"], "preflight-complete")
            for case in report["cases"].values():
                commands = sum((case[group]["commands"] for group in ("encode_cpu", "encode_cuda", "decode_cpu", "decode_cuda")), [])
                self.assertEqual(len(commands), 24)
                rendered = "\n".join(" ".join(command) for command in commands)
                self.assertIn(f"--InputBitDepth={case['source_bit_depth']}", rendered)
                self.assertIn(f"--InternalBitDepth={case['internal_bit_depth']}", rendered)
                self.assertIn(f"--OutputBitDepth={case['output_bit_depth']}", rendered)
                self.assertIn(f"--SourceWidth={case['width']}", rendered)
                self.assertIn(f"--SourceHeight={case['height']}", rendered)
                self.assertIn("--CCALF=0", rendered)


if __name__ == "__main__":
    unittest.main()
