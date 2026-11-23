"""Numerical-copy failures are input errors, never certificate verdicts."""
import contextlib
from fractions import Fraction
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lpwitness import InputError, Problem, certificate, verify
from lpwitness.__main__ import main
from lpwitness.scipy_adapter import propose_from_scipy, scipy_solve


def result(x=1, inequality=0, equality=1):
    return SimpleNamespace(x=[x],
                           ineqlin=SimpleNamespace(marginals=[inequality]),
                           eqlin=SimpleNamespace(marginals=[equality]))


class WitnessInputTests(unittest.TestCase):
    def test_bad_numerical_values_have_context(self):
        p = Problem([1], [[1]], [1], [[1]], [1])
        for field, context in [('x', 'solver x'),
                               ('inequality', 'solver ineqlin.marginals'),
                               ('equality', 'solver eqlin.marginals')]:
            for value in ['oops', 10**1000, float('inf'), float('nan'), None, 1j]:
                with self.subTest(field=field, value=repr(value)[:40]):
                    with self.assertRaises(InputError) as caught:
                        propose_from_scipy(p, result(**{field: value}))
                    self.assertIn(context + '[0]', str(caught.exception))

    def test_bad_result_structure_is_input_error(self):
        p = Problem([1], [[1]], [1], [[1]], [1])
        for candidate in [None, SimpleNamespace(x=None), SimpleNamespace(x=1),
                          SimpleNamespace(x=[1])]:
            with self.subTest(candidate=candidate), self.assertRaises(InputError):
                propose_from_scipy(p, candidate)

    def test_large_fallback_residual_does_not_overflow(self):
        p = Problem([1], [['1e308']], [1], [], [])
        candidate = SimpleNamespace(x=[1e308],
                                    ineqlin=SimpleNamespace(marginals=[0]),
                                    eqlin=SimpleNamespace(marginals=[]))
        proof, report = propose_from_scipy(p, candidate)
        self.assertFalse(report['verified'])
        self.assertEqual(report, verify(p, proof))

    def test_exact_core_still_accepts_tiny_exact_model(self):
        p = Problem(['1e-1000'], [[1]], [0], [], [])
        proof = certificate(p, 'optimal', x=[0], y=[Fraction('1e-1000')], z=[])
        self.assertTrue(verify(p, proof)['verified'])

    def test_cli_malformed_solver_value_returns_json_error(self):
        p = Problem([1], [[1]], [1], [[1]], [1])
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'model.json'
            output = Path(directory) / 'proof.json'
            model.write_text(json.dumps(p.to_dict()))
            stdout = io.StringIO()
            with patch('lpwitness.scipy_adapter.scipy_solve', return_value=result(x='oops')):
                with contextlib.redirect_stdout(stdout):
                    code = main(['solve', str(model), '--certificate-out', str(output)])
            self.assertEqual(code, 2)
            report = json.loads(stdout.getvalue())
            self.assertFalse(report['verified'])
            self.assertEqual(report['error'], 'InputError')
            self.assertIn('solver x[0]', report['message'])
            self.assertFalse(output.exists())


@unittest.skipUnless(importlib.util.find_spec('scipy'), 'optional scipy unavailable')
class ModelFloatCopyTests(unittest.TestCase):
    def test_overflow_and_underflow_rejected_before_solver(self):
        for field in ['c', 'G', 'h', 'E', 'd']:
            for value in ['1e1000', '-1e1000', '1e-1000', '-1e-1000']:
                data = dict(c=[1], G=[[1]], h=[0], E=[[1]], d=[0])
                data[field] = [[value]] if field in ['G', 'E'] else [value]
                p = Problem(**data)
                before = p.sha256
                with self.subTest(field=field, value=value):
                    with patch('scipy.optimize.linprog') as solver:
                        with self.assertRaises(InputError) as caught:
                            scipy_solve(p)
                        solver.assert_not_called()
                    self.assertIn('model ' + field + '[0]', str(caught.exception))
                    if 'e-1000' in value:
                        self.assertIn('underflows to zero', str(caught.exception))
                    self.assertEqual(p.sha256, before)

    def test_zero_subnormal_and_ordinary_rounding_allowed(self):
        for value in ['0', '5e-324', '-5e-324', '1/3', '1e308']:
            p = Problem([value], [], [], [], [])
            with self.subTest(value=value), patch('scipy.optimize.linprog') as solver:
                scipy_solve(p)
                self.assertEqual(solver.call_args.args[0], [float(Fraction(value))])

    def test_cli_model_conversion_errors_are_json_and_no_artifact(self):
        for value in ['1e1000', '-1e-1000']:
            p = Problem([value], [], [], [], [])
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                model = Path(directory) / 'model.json'
                output = Path(directory) / 'proof.json'
                model.write_text(json.dumps(p.to_dict()))
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = main(['solve', str(model), '--certificate-out', str(output)])
                self.assertEqual(code, 2)
                report = json.loads(stdout.getvalue())
                self.assertFalse(report['verified'])
                self.assertEqual(report['error'], 'InputError')
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
