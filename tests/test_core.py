import copy
from fractions import Fraction as F
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from lpwitness import InputError, ResourceLimit, Problem, certificate, load_json, rational, verify
from lpwitness.core import dot
from lpwitness.scipy_adapter import from_linprog, from_active_basis


def model(c=(1,), G=((1,),), h=(2,), E=(), d=()):
    return Problem(c, G, h, E, d)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.p = model()
        self.proof = certificate(self.p, 'optimal', x=[2], y=[1], z=[])

    def test_optimum(self):
        r = verify(self.p, self.proof)
        self.assertTrue(r['verified']); self.assertEqual(r['values']['gap'], '0')

    def test_invalid_primal(self):
        self.proof['x'] = [1]
        self.assertFalse(verify(self.p, self.proof)['verified'])

    def test_negative_dual(self):
        p = model(c=(-1,)); proof = certificate(p, 'optimal', x=[2], y=[-1], z=[])
        r = verify(p, proof)
        self.assertTrue(r['checks']['zero_gap']); self.assertFalse(r['verified'])

    def test_nonstationary(self):
        self.proof['y'] = [0]
        self.assertFalse(verify(self.p, self.proof)['checks']['dual_stationarity'])

    def test_nonoptimal_feasible(self):
        self.proof['x'] = [3]
        self.assertFalse(verify(self.p, self.proof)['checks']['zero_gap'])

    def test_bounds(self):
        proof = certificate(self.p, 'bounds', x=[3], y=[1], z=[])
        r = verify(self.p, proof)
        self.assertTrue(r['verified']); self.assertEqual(r['values']['gap'], '1')

    def test_negative_gap_rejected(self):
        proof = certificate(self.p, 'bounds', x=[1], y=[1], z=[])
        self.assertFalse(verify(self.p, proof)['verified'])

    def test_equality_dual_can_be_negative(self):
        p = model(c=(-1,), G=(), h=(), E=((1,),), d=(2,))
        proof = certificate(p, 'optimal', x=[2], y=[], z=[-1])
        self.assertTrue(verify(p, proof)['verified'])
        proof['x'] = [3]
        self.assertFalse(verify(p, proof)['checks']['primal_equalities'])

    def test_infeasible(self):
        p = model(G=((1,), (-1,)), h=(1, 0))
        proof = certificate(p, 'infeasible', y=[1, 1], z=[])
        self.assertTrue(verify(p, proof)['verified'])
        proof['y'] = [0, 0]
        self.assertFalse(verify(p, proof)['verified'])

    def test_infeasible_equality(self):
        p = model(G=(), h=(), E=((0,),), d=(1,))
        self.assertTrue(verify(p, certificate(p, 'infeasible', y=[], z=[1]))['verified'])

    def test_fake_infeasibility_negative_multiplier(self):
        p = model(G=((1,), (-1,)), h=(0, -1))
        self.assertFalse(verify(p, certificate(p, 'infeasible', y=[-1, -1], z=[]))['verified'])

    def test_unbounded(self):
        p = model(c=(-1,), h=(0,))
        proof = certificate(p, 'unbounded', x=[0], ray=[1])
        self.assertTrue(verify(p, proof)['verified'])
        for bad in [0, -1]:
            proof['ray'] = [bad]
            self.assertFalse(verify(p, proof)['verified'])

    def test_unbounded_requires_feasible_basepoint(self):
        p = model(c=(-1,), G=((0,),), h=(1,))
        proof = certificate(p, 'unbounded', x=[0], ray=[1])
        self.assertFalse(verify(p, proof)['verified'])

    def test_unbounded_ray_equality(self):
        p = model(c=(-1,), G=(), h=(), E=((1,),), d=(0,))
        self.assertFalse(verify(p, certificate(p, 'unbounded', x=[0], ray=[1]))['verified'])

    def test_binding(self):
        p2 = model(h=(3,))
        self.assertFalse(verify(p2, self.proof)['checks']['model_binding'])

    def test_normalized_binding(self):
        p2 = model(c=('1.000',), G=(('2/2',),), h=('2e0',))
        self.assertEqual(p2.sha256, self.p.sha256)

    def test_row_permutation_changes_binding(self):
        a = model(G=((1,),(-1,)),h=(0,-2))
        b = model(G=((-1,),(1,)),h=(-2,0))
        self.assertNotEqual(a.sha256,b.sha256)

    def test_rationals(self):
        for raw in ['0.1', '1e-1', '1/10', '.1', '+0.100']:
            self.assertEqual(rational(raw), F(1,10))

    def test_float_rejected(self):
        for value in [1.0, float('nan'), float('inf'), True, None]:
            with self.subTest(value=value), self.assertRaises(InputError):
                rational(value)
        with self.assertRaises(InputError):
            model(c=(1.0,))

    def test_no_tolerance(self):
        p = model(h=('1e-100',))
        proof = certificate(p, 'optimal', x=[0], y=[1], z=[])
        self.assertFalse(verify(p, proof)['verified'])

    def test_json_exact_decimal(self):
        raw = load_json('{"format":"lpwitness/1","c":[0.1],"G":[],"h":[]}')
        self.assertEqual(Problem.from_dict(raw).c[0], F(1,10))

    def test_invalid_numbers(self):
        for raw in ['NaN','Infinity','1/0','1e','1_000',' 1','1/2/3','１２','1/-2']:
            with self.subTest(raw=raw), self.assertRaises(InputError):
                rational(raw)

    def test_resource_limits(self):
        for raw in ['1e999999999','1e1001','1'*1025, 2**8193]:
            with self.subTest(raw=str(raw)[:30]), self.assertRaises(ResourceLimit):
                rational(raw)

    def test_duplicate_keys(self):
        with self.assertRaises(InputError):
            load_json('{"c":[1],"c":[2]}')

    def test_json_nonfinite(self):
        for value in ['NaN','Infinity','-Infinity']:
            with self.assertRaises(InputError):
                load_json('{"c":[' + value + ']}')

    def test_unknown_model_keys_and_shapes(self):
        for extra in [{'integrality':[1]}, {'sense':'max'}, {'A': [[1]]}, {'G':[[1,2]]}, {'h': []}]:
            raw = self.p.to_dict(); raw.update(extra)
            with self.subTest(extra=extra), self.assertRaises(InputError):
                Problem.from_dict(raw)

    def test_unknown_certificate_fields(self):
        for extra in [{'optimal':True}, {'kind':'MIP'}, {'x':[]}, {'ray':[1]}]:
            proof=copy.deepcopy(self.proof);proof.update(extra)
            with self.subTest(extra=extra), self.assertRaises(InputError):
                verify(self.p,proof)

    def test_dot_length_not_silently_truncated(self):
        with self.assertRaises(InputError):
            dot([F(1)], [])

    def test_from_linprog_bounds(self):
        p = from_linprog([1, -1], A_ub=[[2,3]], b_ub=[20],
                         A_eq=[[1,0]], b_eq=[2], bounds=[(None,3),(0,4)])
        proof, report = from_active_basis(p,[('E',0),('G',3)])
        self.assertTrue(report['verified']);self.assertEqual(proof['x'],['2','4'])

    def test_from_linprog_default_nonnegative(self):
        p = from_linprog([1,1])
        self.assertEqual(p.G, ((F(1),F(0)),(F(0),F(1))))
        self.assertEqual(p.h,(0,0))

    def test_inconsistent_bounds_preserved(self):
        p = from_linprog([1], bounds=[(2,1)])
        self.assertTrue(verify(p,certificate(p,'infeasible',y=[1,1],z=[]))['verified'])

    def test_incorrect_basis_not_certificate(self):
        p=from_linprog([-1],bounds=[(0,1)])
        proof, report=from_active_basis(p,[('G',0)])
        self.assertFalse(report['verified'])

    def test_cli_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);(p/'model.json').write_text(json.dumps(self.p.to_dict()))
            for expected, proof in [(0,self.proof),(1,{**self.proof,'x':[0]}),(2,{})]:
                (p/'proof.json').write_text(json.dumps(proof))
                r=subprocess.run([sys.executable,'-m','lpwitness','verify',str(p/'model.json'),str(p/'proof.json')],capture_output=True,text=True)
                self.assertEqual(r.returncode,expected,r.stderr+r.stdout)
                self.assertEqual(json.loads(r.stdout)['verified'],expected==0)

    def test_checked_in_examples(self):
        root=Path(__file__).resolve().parents[1]/'examples'
        cases=sorted(root.glob('*.model.json'))
        self.assertEqual(len(cases),8)
        for path in cases:
            stem=str(path).removesuffix('.model.json')
            p=Problem.from_dict(load_json(path.read_text()))
            proof=load_json(Path(stem+'.certificate.json').read_text())
            report=verify(p,proof)
            self.assertEqual(report,json.loads(Path(stem+'.report.json').read_text()))
            self.assertEqual(report['verified'],'tolerance_candidate' not in path.name)

    def test_malformed_basis(self):
        for rows in [None, [0], [['G']], [[[],0]], [('G',True)], [('G',-1)]]:
            with self.subTest(rows=rows), self.assertRaises(InputError):
                from_active_basis(self.p,rows)

    def test_generated_kkt_1000(self):
        rng=random.Random(812771)
        for case in range(1000):
            n=rng.randint(1,6);m=rng.randint(0,8);q=rng.randint(0,3)
            x=[F(rng.randint(-5,5),rng.randint(1,7)) for _ in range(n)]
            G=[[F(rng.randint(-5,5)) for _ in range(n)] for _ in range(m)]
            E=[[F(rng.randint(-5,5)) for _ in range(n)] for _ in range(q)]
            y=[F(rng.randint(0,3)) for _ in range(m)]
            z=[F(rng.randint(-3,3)) for _ in range(q)]
            h=[dot(row,x)-(F(0) if v else F(rng.randint(0,3))) for row,v in zip(G,y)]
            d=[dot(row,x) for row in E]
            c=[sum((row[j]*v for row,v in zip(G,y)),F(0))+sum((row[j]*v for row,v in zip(E,z)),F(0)) for j in range(n)]
            p=Problem(c,G,h,E,d);proof=certificate(p,'optimal',x=x,y=y,z=z)
            self.assertTrue(verify(p,proof)['verified'],case)
            # A different objective with the same witness must fail stationarity.
            c[0]+=1;p2=Problem(c,G,h,E,d)
            proof['model_sha256']=p2.sha256
            self.assertFalse(verify(p2,proof)['verified'],case)


if __name__=='__main__': unittest.main()
