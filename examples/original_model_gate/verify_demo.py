"""Verify artifacts without SciPy, highspy, NumPy, or network; works with python -S."""
import argparse
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
from lpwitness import Problem, verify


# Frozen demonstration identities, not signatures/authentication.
CONTRACT_HASHES = {
    'source_decimal': '06d79034280574cf211a941e5e0712f1feb5e3d290e5e4ba94b938e7405ca49f',
    'solver_binary64': '9ad2a3f3edc199bb23f00795c5dd304b5d97538af7b5e1e4bfc44596a4703a4c',
}

def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(root):
    results = []
    for contract in ['source_decimal', 'solver_binary64']:
        directory = Path(root)/contract
        problem = Problem.from_dict(json.loads((directory/'model.json').read_text()))
        require(problem.sha256 == CONTRACT_HASHES[contract], 'Wrong frozen model for contract '+contract)
        raw = json.loads((directory/'raw.certificate.json').read_text())
        repaired = json.loads((directory/'reconstructed.certificate.json').read_text())
        raw_report = verify(problem, raw)
        verified = verify(problem, repaired)
        require(not raw_report['verified'], 'This frozen demo expects the original candidate to be rejected')
        require(verified['verified'], str(verified))
        # Separate direct rational residual calculation, without trusting reports.
        x = [Fraction(v) for v in repaired['x']]
        require(all(sum(a*b for a,b in zip(row,x)) == rhs for row,rhs in zip(problem.E,problem.d)), 'Reconstructed equalities fail')
        require(all(sum(a*b for a,b in zip(row,x)) >= rhs for row,rhs in zip(problem.G,problem.h)), 'Reconstructed inequalities fail')
        # An independent problem-specific optimum argument: only c[3] is
        # nonzero and it is negative. The explicit upper bound on x[3]
        # gives c*x >= c[3]*upper; the reconstructed feasible x attains it.
        require([j for j,c in enumerate(problem.c) if c] == [3], 'Unexpected objective support')
        require(problem.c[3] < 0, 'Unexpected objective direction')
        upper_rows = [rhs for row,rhs in zip(problem.G,problem.h)
                      if all(a == (-1 if j == 3 else 0) for j,a in enumerate(row))]
        require(len(upper_rows) == 1, 'Expected one explicit x[3] upper bound')
        upper = -upper_rows[0]
        require(x[3] == upper, 'Feasible candidate does not attain scalar objective bound')
        require(sum(a*b for a,b in zip(problem.c,x)) == problem.c[3]*upper,
                'Independent scalar objective bound differs')
        corrupted = deepcopy(repaired)
        corrupted['x'][0] = str(Fraction(corrupted['x'][0]) + 1)
        require(not verify(problem, corrupted)['verified'], 'Corrupted candidate was accepted')
        modified = problem.to_dict()
        modified['c'][0] = str(problem.c[0] + 1)
        require(not verify(Problem.from_dict(modified), repaired)['verified'], 'Changed model was accepted')
        results.append({'contract': contract, 'model_sha256': problem.sha256,
                        'raw_rejected': True, 'reconstructed_verified': True,
                        'independent_scalar_optimum_bound_attained': True,
                        'tampered_candidate_rejected': True, 'different_model_rejected': True})
    return results


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    print(json.dumps(check(parser.parse_args().directory), indent=2))
