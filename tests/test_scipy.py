from fractions import Fraction as F
import importlib.util
from itertools import combinations
import random
import unittest

from lpwitness import Problem, certificate, verify
from lpwitness.scipy_adapter import from_linprog, scipy_solve, propose_from_scipy


@unittest.skipUnless(importlib.util.find_spec('scipy'), 'optional scipy unavailable')
class ScipyTests(unittest.TestCase):
    def test_simple(self):
        p=from_linprog([1,2],A_ub=[[-1,-1]],b_ub=[-3])
        proof,report=propose_from_scipy(p,scipy_solve(p))
        self.assertTrue(report['verified']);self.assertEqual(report['values']['primal_objective'],'3')

    def test_rational_basis_repair(self):
        p=from_linprog([1,1], A_ub=[[-3,0],[0,-7]],b_ub=[-1,-1])
        proof,report=propose_from_scipy(p,scipy_solve(p),max_denominator=2)
        self.assertTrue(report['verified']);self.assertEqual(proof['x'],['1/3','1/7'])

    def test_high_denominator_basis_repair(self):
        p=from_linprog([1],A_ub=[[-10000019]],b_ub=[-1])
        proof,report=propose_from_scipy(p,scipy_solve(p))
        self.assertTrue(report['verified']);self.assertEqual(proof['x'],['1/10000019'])

    def test_tolerance_is_not_exact_feasibility(self):
        # Recreate the documented tolerance behavior, not a claim that HiGHS is
        # wrong to honor its requested numerical feasibility contract.
        # Use the intended positive lower bound x >= 0.0002.
        p=from_linprog([1],A_ub=[[-1]],b_ub=['-0.0002'])
        from scipy.optimize import linprog
        result=linprog([1.0],A_ub=[[-1.0]],b_ub=[-0.0002],bounds=[(0,None)],
                       method='highs',options={'presolve':False,'primal_feasibility_tolerance':1e-3})
        if result.success and abs(float(result.x[0]))<1e-12:
            proof=certificate(p,'optimal',x=[0],y=[1,0],z=[])
            self.assertFalse(verify(p,proof)['verified'])
        else:
            # Future solver versions may choose a different feasible point.
            self.assertIsNotNone(result.x)

    def test_independent_vertex_oracle_250(self):
        rng=random.Random(42031)
        for case in range(250):
            c=[rng.randint(-8,8),rng.randint(-8,8)]
            G=[[1,0],[0,1],[-1,0],[0,-1]]
            h=[0,0,-rng.randint(1,10),-rng.randint(1,10)]
            for _ in range(4):
                G.append([-rng.randint(0,9),-rng.randint(0,9)])
                h.append(-rng.randint(1,30))
            p=Problem(c,G,h,[],[])
            # Independent 2x2 determinant formula, not the proposal elimination.
            vertices=[]
            for i,j in combinations(range(len(G)),2):
                a,b=G[i];d,e=G[j];det=a*e-b*d
                if det==0:continue
                x=(F(h[i]*e-b*h[j],det),F(a*h[j]-h[i]*d,det))
                if all(row[0]*x[0]+row[1]*x[1]>=rhs for row,rhs in zip(G,h)):
                    vertices.append(c[0]*x[0]+c[1]*x[1])
            expected=min(vertices)
            proof,report=propose_from_scipy(p,scipy_solve(p))
            self.assertTrue(report['verified'],(case,report))
            self.assertEqual(F(report['values']['primal_objective']),expected,case)


if __name__=='__main__':unittest.main()
