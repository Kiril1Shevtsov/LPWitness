"""Create original synthetic, reproducible example models and witness bundles."""
from pathlib import Path
import json
import time

from lpwitness import Problem, certificate, verify
from lpwitness.scipy_adapter import from_linprog, scipy_solve, propose_from_scipy

ROOT=Path(__file__).parent

def save(name, model, proof):
    report=verify(model,proof)
    for suffix,obj in [('model',model.to_dict()),('certificate',proof),('report',report)]:
        (ROOT/f'{name}.{suffix}.json').write_text(json.dumps(obj,indent=2)+'\n')
    return report

# A synthetic two-component blending LP, not a validated process model.
# Fractions x0,x1 sum to 1. Impurity 0.02*x0 + 0.08*x1 <= 0.05.
# Price 7*x0+4*x1 -> min at x0=x1=1/2, objective 11/2.
p=from_linprog([7,4],A_ub=[['0.02','0.08']],b_ub=['0.05'],
               A_eq=[[1,1]],b_eq=[1],bounds=[(0,1),(0,1)])
proof,report=propose_from_scipy(p,scipy_solve(p))
assert report['verified'] and report['values']['primal_objective']=='11/2'
save('blending',p,proof)

# A synthetic resource allocation LP. This is not financial advice or a risk model.
p=from_linprog([-3,-2],A_ub=[[1,1],[2,1]],b_ub=[10,14])
proof,report=propose_from_scipy(p,scipy_solve(p));assert report['verified']
save('allocation',p,proof)

# Toy steady-state metabolic flux v1-v2=0, 0<=v1<=10, maximize v2.
p=from_linprog([0,-1],A_eq=[[1,-1]],b_eq=[0],bounds=[(0,10),(0,None)])
proof,report=propose_from_scipy(p,scipy_solve(p));assert report['verified']
save('flux_balance',p,proof)

# Inconsistent precision-sensitive limits: x >= 1e-10 and x <= 0.
p=Problem([0],[[1],[-1]],['1e-10',0],[],[])
save('infeasible',p,certificate(p,'infeasible',y=[1,1],z=[]))

# A feasible base point plus a recession direction with strictly lower objective.
p=Problem([-1],[[1]],[0],[],[])
save('unbounded',p,certificate(p,'unbounded',x=[0],ray=[1]))

# Conservative primal-dual objective interval, without asserting exact optimum.
p=Problem([1],[[1]],[2],[],[])
save('bounds',p,certificate(p,'bounds',x=[3],y=[1],z=[]))

# Numerical feasibility is a different contract from exact rational feasibility.
# An original minimal reproduction inspired by SciPy issue #26276; no issue
# attachment or application-owned dataset is distributed with this project.
p=from_linprog([1],A_ub=[[-1]],b_ub=['-0.0002'])
from scipy.optimize import linprog
result=linprog([1.0],A_ub=[[-1.0]],b_ub=[-0.0002],bounds=[(0,None)],
               method='highs',options={'presolve':False,'primal_feasibility_tolerance':1e-3})
raw=certificate(p,'optimal',x=[str(float(v)) for v in result.x],
                y=[str(-float(result.ineqlin.marginals[0])),str(float(result.lower.marginals[0]))],z=[])
raw_report=save('tolerance_candidate',p,raw)
exact=certificate(p,'optimal',x=['1/5000'],y=[1,0],z=[])
exact_report=save('tolerance_repaired',p,exact)
import scipy,sys
summary={'python':sys.version.split()[0],'scipy':scipy.__version__,
         'numerical_success':bool(result.success),'numerical_x':result.x.tolist(),
         'raw_witness_verified':raw_report['verified'],
         'exact_witness_verified':exact_report['verified'],
         'interpretation':'Numeric feasibility tolerance is not exact feasibility. This is not an allegation that honoring the requested tolerance is a solver bug.'}
(ROOT/'tolerance_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
