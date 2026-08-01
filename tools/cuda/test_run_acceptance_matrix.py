#!/usr/bin/env python3
"""Focused tests for acceptance-runner CUDA telemetry contracts."""

import unittest

from run_acceptance_matrix import parse_decoder_telemetry, parse_encoder_telemetry


ENCODER_OK = """
CUDA SAD batches: 7, failures: 0, disabled: 0
CUDA QPA batches: 2, tasks: 510, failures: 0, fallbacks: 0, enabled: 1, poisoned: 0, disabled-by-flag: 0
"""

DECODER_OK = """
CUDA loop-filter chain frames/pixels/DBF tasks/SAO CTUs/ALF CTUs: 4/8294400/983/2040/2040,
params/internal copies/mirror up/down: 1/2/3/4 bytes, failures/not-eligible: 0/0,
enabled: 1, poisoned: 0, disabled-by-flag: 0
"""


class TelemetryTest(unittest.TestCase):
    def test_encoder_expected_operations(self):
        parsed = parse_encoder_telemetry(ENCODER_OK, expect_sad=True)
        self.assertEqual(parsed["sad"]["batches"], 7)
        self.assertEqual(parsed["qpa"]["tasks"], 510)

    def test_ai_allows_zero_sad_but_requires_qpa(self):
        parse_encoder_telemetry(ENCODER_OK.replace("batches: 7", "batches: 0", 1), expect_sad=False)
        with self.assertRaisesRegex(RuntimeError, "QPA did not execute"):
            parse_encoder_telemetry(ENCODER_OK.replace("fallbacks: 0", "fallbacks: 1"), False)

    def test_inter_rejects_zero_sad(self):
        with self.assertRaisesRegex(RuntimeError, "zero batches"):
            parse_encoder_telemetry(ENCODER_OK.replace("batches: 7", "batches: 0", 1), True)

    def test_decoder_requires_dispatch_and_zero_rejections(self):
        parsed = parse_decoder_telemetry(DECODER_OK)
        self.assertEqual(parsed["dispatches"], 4)
        with self.assertRaisesRegex(RuntimeError, "did not execute cleanly"):
            parse_decoder_telemetry(DECODER_OK.replace("not-eligible: 0/0", "not-eligible: 0/1"))

    def test_missing_telemetry_reports_cpu_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "fallen back to CPU"):
            parse_decoder_telemetry("VTM decoder completed")


if __name__ == "__main__":
    unittest.main()
