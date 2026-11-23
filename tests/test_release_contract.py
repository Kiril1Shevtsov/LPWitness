"""Offline release gates independent of a solver's status or any tolerance."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from lpwitness import Problem, certificate, verify


class ReleaseContractTests(unittest.TestCase):
    def test_frozen_known_numerical_success_is_not_a_certificate(self):
        root = Path(__file__).resolve().parents[1] / "examples/original_model_gate/results/final"
        summary = json.loads((root / "summary.json").read_text())
        self.assertEqual(summary["native_layout"]["status"], 0)
        self.assertEqual(summary["native_layout"]["reported_equality_residual_l2"], 0)
        self.assertGreater(summary["native_layout"]["recomputed_equality_residual_l2"], 0)
        for record in summary["contracts"]:
            name = record["contract"]
            with self.subTest(contract=name):
                package = root / name
                problem = Problem.from_dict(json.loads((package / "model.json").read_text()))
                raw = json.loads((package / "raw.certificate.json").read_text())
                new = json.loads((package / "reconstructed.certificate.json").read_text())
                raw_result = verify(problem, raw)
                self.assertFalse(raw_result["verified"])
                self.assertFalse(raw_result["checks"]["primal_equalities"])
                self.assertTrue(verify(problem, new)["verified"])
                self.assertNotEqual(raw["x"], new["x"])

    def test_mislabeled_mathematical_status_is_never_accepted(self):
        # Exactly min x with x >= 2. The unique optimal point is 2.
        problem = Problem([1], [[1]], [2], [], [])
        claims = [
            ("optimal", {"x": [3], "y": [1], "z": []}, False),
            ("infeasible", {"y": [1], "z": []}, False),
            ("unbounded", {"x": [2], "ray": [1]}, False),
            ("optimal", {"x": [2], "y": [1], "z": []}, True),
            ("bounds", {"x": [3], "y": [1], "z": []}, True),
        ]
        for claim, proof, expected in claims:
            with self.subTest(claim=claim, expected=expected):
                report = verify(problem, certificate(problem, claim, **proof))
                self.assertIs(report["verified"], expected)

    def test_invalid_parse_is_input_error_not_a_mathematical_verdict(self):
        bad_models = {
            "bad rational": '{"format":"lpwitness/1","c":["1/0"]}',
            "nonfinite JSON": '{"format":"lpwitness/1","c":[NaN]}',
            "duplicate coefficient field": '{"format":"lpwitness/1","c":[1],"c":[2]}',
            "out-of-scope MIP": '{"format":"lpwitness/1","c":[1],"integrality":[1]}',
            "width mismatch": '{"format":"lpwitness/1","c":[1],"G":[[1,2]],"h":[1]}',
        }
        with tempfile.TemporaryDirectory() as temp:
            model, proof = (Path(temp) / name for name in ("model.json", "proof.json"))
            verified_model = Problem([1], [[1]], [2], [], [])
            proof.write_text(
                json.dumps(certificate(verified_model, "optimal", x=[2], y=[1], z=[])),
                encoding="utf-8",
            )
            for label, payload in bad_models.items():
                with self.subTest(label=label):
                    model.write_text(payload, encoding="utf-8")
                    run = subprocess.run(
                        [sys.executable, "-S", "-m", "lpwitness", "verify", str(model), str(proof)],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(run.returncode, 2, run.stderr + run.stdout)
                    self.assertIs(json.loads(run.stdout)["verified"], False)
