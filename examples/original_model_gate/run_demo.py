"""A reproducible original-model gate for SciPy issue #20336.

Requires SciPy and lpwitness; use --highs to also exercise the optional loader.
The issued optimum certificate concerns the reconstructed rational solution,
NOT the original floating-point solver vector. Never execute upstream source.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import warnings

from lpwitness import Problem, certificate, verify
from lpwitness.core import dot
from lpwitness.scipy_adapter import scipy_solve, propose_from_scipy, from_linprog

ROOT = Path(__file__).resolve().parent


def binary64_model(problem):
    """Exact model of the binary64 numerical copy, not decimal re-rationalization."""
    def row(values):
        return [Fraction.from_float(float(v)) for v in values]
    return Problem(row(problem.c), [row(r) for r in problem.G], row(problem.h),
                   [row(r) for r in problem.E], row(problem.d))


def float_witness(problem, result):
    """Preserve the actual binary64 candidate, rather than rounding it to fit."""
    q = lambda values: [Fraction.from_float(float(v)) for v in values]
    return certificate(problem, 'optimal', x=q(result.x),
                       y=[-v for v in q(result.ineqlin.marginals)],
                       z=q(result.eqlin.marginals))


def max_equality_residual(problem, x):
    return max((abs(dot(row, x) - rhs) for row, rhs in zip(problem.E, problem.d)),
               default=Fraction(0))


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def validate_inputs(original, model_input):
    # Keep the public-layout reproduction and canonical exact model aligned.
    if from_linprog(**model_input).sha256 != original.sha256:
        raise ValueError('linprog_input does not represent the original exact model')


def run(out, check_highs=False):
    import numpy as np
    import scipy
    from scipy.optimize import linprog

    # Refuse to overwrite existing user output. The demo performs no network I/O.
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    original = Problem.from_dict(json.loads((ROOT/'models/original_decimal.json').read_text()))
    model_input = json.loads((ROOT/'models/linprog_input.json').read_text())
    validate_inputs(original, model_input)
    f = lambda values: [float(Fraction(v)) for v in values]
    # Reproduce the issue's public linprog layout, including native bounds.
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always')
        native = linprog(f(model_input['c']),
                         A_eq=[f(r) for r in model_input['A_eq']],
                         b_eq=f(model_input['b_eq']),
                         bounds=[tuple(None if v is None else float(Fraction(v)) for v in pair)
                                 for pair in model_input['bounds']], method='highs')
    if native.x is None:
        raise RuntimeError('No numerical candidate: this run does not reproduce the workflow')
    direct_residual = (np.array([f(r) for r in model_input['A_eq']]) @ native.x
                       - np.array(f(model_input['b_eq'])))
    summary = {'python': platform.python_version(), 'scipy': scipy.__version__,
               'numpy': np.__version__, 'source_issue': 'https://github.com/scipy/scipy/issues/20336',
               'native_layout': {'status': int(native.status), 'success': bool(native.success),
                                 'reported_equality_residual_l2': float(np.linalg.norm(native.eqlin.residual)),
                                 'recomputed_equality_residual_l2': float(np.linalg.norm(direct_residual)),
                                 'python_warnings': [str(w.message) for w in observed]}, 'contracts': []}
    for name, problem in [('source_decimal', original), ('solver_binary64', binary64_model(original))]:
        folder = out/name
        folder.mkdir()
        start = time.perf_counter()
        numerical = scipy_solve(problem)
        solve_seconds = time.perf_counter() - start
        if numerical.x is None:
            raise RuntimeError('No candidate from canonical solve')
        raw = float_witness(problem, numerical)
        raw_report = verify(problem, raw)
        start = time.perf_counter()
        repaired, report = propose_from_scipy(problem, numerical)
        proposal_seconds = time.perf_counter() - start
        start = time.perf_counter()
        replay = verify(problem, repaired)
        replay_seconds = time.perf_counter() - start
        if replay != report:
            raise AssertionError('Inconsistent exact replay')
        raw_x = [Fraction(v) for v in raw['x']]
        repaired_x = [Fraction(v) for v in repaired['x']]
        record = {'contract': name, 'model_sha256': problem.sha256,
                  'solver_status': int(numerical.status), 'raw_verified': raw_report['verified'],
                  'raw_equality_residual_linf_exact': str(max_equality_residual(problem, raw_x)),
                  'raw_equality_residual_linf_approx': float(max_equality_residual(problem, raw_x)),
                  'reconstructed_verified': report['verified'],
                  'reconstructed_equality_residual_linf_exact': str(max_equality_residual(problem, repaired_x)),
                  'reconstructed_objective': report['values'].get('primal_objective'),
                  'max_coordinate_change_approx': float(max(abs(a-b) for a,b in zip(raw_x,repaired_x))),
                  'seconds': {'solve': solve_seconds, 'proposal_and_gate': proposal_seconds, 'offline_gate': replay_seconds}}
        for filename, data in [('model.json', problem.to_dict()), ('raw.certificate.json', raw),
                               ('raw.report.json', raw_report), ('reconstructed.certificate.json', repaired),
                               ('reconstructed.report.json', report)]:
            save(folder/filename, data)
        summary['contracts'].append(record)
    if check_highs:
        from lpwitness.highs_adapter import highs_solve
        from lpwitness.core import InputError
        try:
            highs_solve(original)
            summary['highs_loader'] = {'loaded_without_warning': True}
        except InputError as exc:
            summary['highs_loader'] = {'loaded_without_warning': False, 'error': str(exc)}
        summary['highspy'] = importlib.metadata.version('highspy')
    save(out/'summary.json', summary)
    print(f"SciPy {scipy.__version__}: reported residual norm "
          f"{summary['native_layout']['reported_equality_residual_l2']:.9g}; "
          f"recomputed on input {summary['native_layout']['recomputed_equality_residual_l2']:.9g}")
    for record in summary['contracts']:
        print(f"{record['contract']}: raw accepted={record['raw_verified']}; "
              f"reconstructed accepted={record['reconstructed_verified']}; "
              f"exact reconstructed equality residual={record['reconstructed_equality_residual_linf_exact']}")
    print('Reconstruction changes the candidate. Its certificate does not certify the raw solver vector.')
    print(f'Offline replay from the LPWitness checkout: python -S -m examples.original_model_gate.verify_demo {out}')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='new directory for model/certificate/report artifacts')
    parser.add_argument('--highs', action='store_true', help='also check optional highspy loader')
    args = parser.parse_args()
    try:
        run(args.out, args.highs)
    except FileExistsError:
        parser.error('output directory already exists; choose a new --out path')
