"""Optional native HiGHS API integration plus fail-closed proposal tests."""
from fractions import Fraction as F
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from lpwitness import InputError, Problem, verify
from lpwitness.highs_adapter import _equations, highs_solve, propose_from_highs
from lpwitness.scipy_adapter import from_linprog

HAVE_HIGHS = importlib.util.find_spec('highspy') is not None
if HAVE_HIGHS:
    import highspy


class EquationTests(unittest.TestCase):
    def test_unique_overdetermined_rank_deficient_inconsistent(self):
        for matrix, rhs, width, expected in [
            ([[2, 0], [0, 3], [4, 3]], [1, 1, 3], 2, [F(1,2), F(1,3)]),
            ([[1, 2], [2, 4]], [3, 6], 2, [3, 0]),
            ([[1], [1]], [0, 1], 1, None),
            ([[], []], [0, 1], 0, None),
            ([], [], 2, [0, 0]),
        ]:
            self.assertEqual(_equations(matrix, rhs, width), expected)

    def test_bounded_and_generated(self):
        self.assertIsNone(_equations([], [], 130))
        self.assertIsNone(_equations([[1]] * 4097, [0] * 4097, 1))
        rng = random.Random(53181)
        for _ in range(100):
            n = rng.randrange(1, 8)
            # Independent triangular construction with known unique solution.
            x = [F(rng.randrange(-5, 6), rng.randrange(1, 8)) for _ in range(n)]
            a = [[rng.randrange(1, 8) if j == i else
                  (rng.randrange(-5, 6) if j > i else 0)
                  for j in range(n)] for i in range(n)]
            b = [sum(u*v for u,v in zip(row,x)) for row in a]
            self.assertEqual(_equations(a, b, n), x)


@unittest.skipUnless(HAVE_HIGHS, 'optional highspy unavailable')
class HighsTests(unittest.TestCase):
    def checked(self, p, kind, **kwargs):
        before = p.sha256
        proof, report = propose_from_highs(p, highs_solve(p), **kwargs)
        self.assertTrue(report['verified'], report)
        self.assertEqual(proof['kind'], kind)
        self.assertEqual(report, verify(p, proof))
        self.assertEqual(p.sha256, before)
        return proof, report

    def test_optimal_and_equality_sign(self):
        p = from_linprog([1, 2], A_ub=[[-1, -1]], b_ub=[-3])
        self.assertEqual(self.checked(p, 'optimal')[1]['values']['primal_objective'], '3')
        p = from_linprog([-1], A_eq=[[3]], b_eq=[1], bounds=[(None,None)])
        proof, _ = self.checked(p, 'optimal', max_denominator=2)
        self.assertEqual(proof['x'], ['1/3'])
        self.assertEqual(proof['z'], ['-1/3'])

    def test_infeasible_bounds_and_equalities(self):
        self.checked(from_linprog([0], bounds=[(1, 0)]), 'infeasible')
        p = from_linprog([0], A_eq=[[1], [1]], b_eq=[0, 1], bounds=[(None,None)])
        proof, _ = self.checked(p, 'infeasible')
        self.assertLess(F(proof['z'][0]), 0)
        # Some trivial infeasible models have no exposed HiGHS ray.
        p = Problem([1], [[0]], [1], [], [])
        h = highs_solve(p)
        try:
            _, report = propose_from_highs(p, h)
        except InputError as exc:
            self.assertIn('no dual ray', str(exc))
        else:
            self.assertTrue(report['verified'])

    def test_infeasible_exact_support_repair(self):
        p = Problem([0,0], [[3,1],[-7,'-7/3']], [1,0], [], [])
        proof, report = self.checked(p, 'infeasible', max_denominator=1)
        self.assertEqual(report['values']['contradiction'], '1')
        self.assertEqual(proof['y'], ['1', '3/7'])

    def test_unbounded_anchor_and_ray_repair(self):
        self.checked(from_linprog([-1]), 'unbounded')
        p = from_linprog([-1,0], A_eq=[[1,-3]], b_eq=[0],
                         bounds=[(1,None),(None,None)])
        proof, _ = self.checked(p, 'unbounded', max_denominator=2)
        self.assertEqual(proof['x'], ['1','1/3'])
        self.assertEqual(proof['ray'], ['1','1/3'])
        p = from_linprog([0,-1], A_eq=[[7,0]], b_eq=[1])
        proof, _ = self.checked(p, 'unbounded', max_denominator=2)
        self.assertEqual(proof['x'], ['1/7','0'])

    def test_absent_ray_is_not_a_verdict(self):
        p = from_linprog([-1], bounds=[(None,None)])
        # Native HiGHS currently supplies no ray for a row-free unbounded LP.
        h = highs_solve(p)
        if not h.getPrimalRay()[1]:
            with self.assertRaisesRegex(InputError, 'no primal ray'):
                propose_from_highs(p, h)
        for status, getter in [(highspy.HighsModelStatus.kInfeasible, 'getDualRay'),
                               (highspy.HighsModelStatus.kUnbounded, 'getPrimalRay')]:
            h = Mock(); h.getModelStatus.return_value = status
            getattr(h, getter).return_value = (highspy.HighsStatus.kOk, False, [])
            with self.assertRaisesRegex(InputError, 'no .* ray'):
                propose_from_highs(p, h)

    def test_unsupported_statuses_and_invalid_validity(self):
        p = from_linprog([1])
        for status in [highspy.HighsModelStatus.kTimeLimit,
                       highspy.HighsModelStatus.kUnboundedOrInfeasible,
                       highspy.HighsModelStatus.kNotset]:
            h=Mock(); h.getModelStatus.return_value=status
            with self.assertRaisesRegex(InputError, 'unsupported HiGHS model status'):
                propose_from_highs(p, h)
        h=Mock();h.getModelStatus.return_value=highspy.HighsModelStatus.kOptimal
        h.getSolution.return_value=NS(value_valid=True,dual_valid=False)
        with self.assertRaisesRegex(InputError,'valid primal/dual'):
            propose_from_highs(p,h)

    def test_invalid_witness_vectors(self):
        p = from_linprog([1])
        for value in ['oops', float('inf'), float('nan'), None, 10**1000]:
            for field in ['x','row_dual','dual_ray','primal_ray']:
                h=Mock()
                h.getModelStatus.return_value = highspy.HighsModelStatus.kOptimal
                h.getSolution.return_value = NS(value_valid=True, dual_valid=True,
                                                col_value=[value if field=='x' else 0],
                                                row_dual=[value if field=='row_dual' else 1])
                if field in ['dual_ray','primal_ray']:
                    h.getModelStatus.return_value = (highspy.HighsModelStatus.kInfeasible
                        if field=='dual_ray' else highspy.HighsModelStatus.kUnbounded)
                    getattr(h,'getDualRay' if field=='dual_ray' else 'getPrimalRay').return_value=(highspy.HighsStatus.kOk,True,[value])
                with self.subTest(field=field,value=repr(value)[:20]):
                    with self.assertRaisesRegex(InputError,'solver '+field):
                        propose_from_highs(p,h)
        for vector in [[],[1,2],None,3]:
            h=Mock();h.getModelStatus.return_value=highspy.HighsModelStatus.kInfeasible
            h.getDualRay.return_value=(highspy.HighsStatus.kOk,True,vector)
            with self.assertRaises(InputError):propose_from_highs(p,h)
        for denominator in [0,True,1.5,10**10]:
            with self.assertRaisesRegex(InputError,'max_denominator'):
                propose_from_highs(p,h,max_denominator=denominator)

    def test_wrong_status_or_ray_does_not_pass(self):
        # This model has a minimum; neither an unbounded ray nor Farkas
        # contradiction can exist, regardless of reported solver status.
        p=from_linprog([1],bounds=[(0,1)])
        for status in [highspy.HighsModelStatus.kInfeasible,
                       highspy.HighsModelStatus.kUnbounded]:
            h=Mock();h.getModelStatus.return_value=status
            h.getDualRay.return_value=(highspy.HighsStatus.kOk,True,[-1,1])
            h.getPrimalRay.return_value=(highspy.HighsStatus.kOk,True,[-1])
            h.getSolution.return_value=NS(value_valid=True,col_value=[0])
            proof,report=propose_from_highs(p,h)
            self.assertFalse(report['verified']);self.assertEqual(report,verify(p,proof))

    def test_bad_options_and_numerical_model_copy(self):
        for options in [{'nonexistent_option':1},{'presolve':3}, {'time_limit':object()}]:
            with self.assertRaisesRegex(InputError,'option'):
                highs_solve(from_linprog([1]),**options)
        for field in ['c','G','h','E','d']:
            for value in ['1e1000','-1e1000','1e-1000','-1e-1000']:
                data=dict(c=[1],G=[[1]],h=[0],E=[[1]],d=[0])
                data[field]=[[value]] if field in ['G','E'] else [value]
                p=Problem(**data)
                with self.subTest(field=field,value=value):
                    with patch('highspy.Highs') as native:
                        with self.assertRaisesRegex(InputError,'model '+field):highs_solve(p)
                        native.assert_not_called()
        # HiGHS warns when it drops a small matrix entry; refuse this load.
        with self.assertRaisesRegex(InputError,'load.*warnings'):
            highs_solve(Problem([1],[['1e-12']],[1],[],[]))

    def test_cli_json_and_offline_roundtrip(self):
        p=from_linprog([0],bounds=[(1,0)])
        with tempfile.TemporaryDirectory() as directory:
            model=Path(directory)/'model.json';proof=Path(directory)/'proof.json'
            model.write_text(json.dumps(p.to_dict()))
            command=[sys.executable,'-m','lpwitness','solve',str(model),'--solver','highs',
                     '--certificate-out',str(proof)]
            run=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr+run.stdout)
            self.assertTrue(json.loads(run.stdout)['verified'])
            run=subprocess.run([sys.executable,'-S','-m','lpwitness','verify',str(model),str(proof)],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr+run.stdout)
            self.assertTrue(json.loads(run.stdout)['verified'])
            run=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(run.returncode,2)
            self.assertEqual(json.loads(run.stdout)['error'],'FileExistsError')
            # Failure inside the native-output context must restore JSON stdout
            # and must not create a certificate artifact.
            for value in ['1e1000', '1e-1000']:
                model.write_text(json.dumps(Problem([value],[],[],[],[]).to_dict()))
                failed=Path(directory)/('failed-'+value+'.json')
                run=subprocess.run(command[:-1]+[str(failed)],capture_output=True,text=True)
                self.assertEqual(run.returncode,2,run.stderr+run.stdout)
                self.assertEqual(json.loads(run.stdout)['error'],'InputError')
                self.assertFalse(failed.exists())

    def test_independent_one_dimensional_oracle(self):
        rng=random.Random(201113)
        counts=dict(optimal=0,infeasible=0,unbounded=0)
        for case in range(150):
            lo=F(rng.randrange(-20,21),rng.randrange(1,10))
            upper=None if case%3==0 else F(rng.randrange(-20,21),rng.randrange(1,10))
            c=rng.choice([-3,-1,1,5])
            p=from_linprog([c],bounds=[(lo,upper)])
            kind=('infeasible' if upper is not None and upper<lo else
                  'unbounded' if upper is None and c<0 else 'optimal')
            proof,report=self.checked(p,kind,max_denominator=2)
            counts[kind]+=1
            if kind=='optimal':
                expected=c*(lo if c>0 else upper)
                self.assertEqual(F(report['values']['primal_objective']),expected)
        self.assertTrue(all(counts.values()),counts)


if __name__=='__main__':unittest.main()
