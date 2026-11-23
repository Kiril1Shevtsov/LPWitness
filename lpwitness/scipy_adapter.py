"""Optional SciPy bridge. Numerical output only proposes witnesses.

Model coefficients are never rationalized. Witness guesses may be rationalized;
acceptance still requires exact verification against the original rational model.
"""
from __future__ import annotations

from fractions import Fraction
import math

from .core import InputError, Problem, certificate, dot, limited, rational, verify


def _finite_float(value, context):
    """Normalize numerical-copy errors without implying a certificate verdict."""
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputError(f'{context} cannot be converted to a finite solver float') from exc
    if not math.isfinite(converted):
        raise InputError(f'{context} is not a finite solver float')
    return converted


def from_linprog(c, *, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    """Convert SciPy's minimization convention to the explicit exact model.

    Unlike scipy.optimize.linprog, this API requires one bounds pair per variable
    (or None for the default x>=0). None endpoints mean no bound. Integers and
    rational/decimal strings are accepted; float model coefficients are rejected.
    """
    if not isinstance(c, (list, tuple)) or not c:
        raise InputError('c must be a nonempty list')
    n = len(c)
    # Validate supplied rows before iterating; do not silently truncate zip().
    aub = [] if A_ub is None else A_ub
    bub = [] if b_ub is None else b_ub
    aeq = [] if A_eq is None else A_eq
    beq = [] if b_eq is None else b_eq
    p = Problem.from_dict({'format': 'lpwitness/1', 'c': c, 'G': aub,
                           'h': bub, 'E': aeq, 'd': beq})
    G, h = [tuple(-v for v in row) for row in p.G], [-v for v in p.h]
    pairs = [(0, None)] * n if bounds is None else bounds
    if not isinstance(pairs, (list, tuple)) or len(pairs) != n:
        raise InputError('bounds must contain one pair per variable')
    for j, pair in enumerate(pairs):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise InputError('each bound must be a (lower, upper) pair')
        lo, hi = (None if v is None else rational(v) for v in pair)
        # lo>hi is a valid infeasible model, not a malformed model.
        for sign, endpoint in [(1, lo), (-1, hi)]:
            if endpoint is not None:
                G.append(tuple(Fraction(sign if k == j else 0) for k in range(n)))
                h.append(sign * endpoint)
    return Problem.from_dict({'format': 'lpwitness/1', 'c': list(p.c),
                              'G': G, 'h': h, 'E': list(p.E), 'd': list(p.d)})


def scipy_solve(problem, **options):
    """Solve a float copy; reject overflow and nonzero-to-zero underflow.

    Ordinary finite rounding remains allowed for proposal generation. The exact
    model is unchanged, and a successful numerical result is not a proof.
    """
    from scipy.optimize import linprog
    def floats(values, name):
        out = []
        for i, value in enumerate(values):
            context = f'model {name}[{i}]'
            converted = _finite_float(value, context)
            if value != 0 and converted == 0:
                raise InputError(f'{context} underflows to zero in the solver float copy')
            out.append(converted)
        return out
    return linprog(floats(problem.c, 'c'),
                   A_ub=[floats([-v for v in r], f'G[{i}]')
                         for i, r in enumerate(problem.G)] or None,
                   b_ub=floats([-v for v in problem.h], 'h') or None,
                   A_eq=[floats(r, f'E[{i}]') for i, r in enumerate(problem.E)] or None,
                   b_eq=floats(problem.d, 'd') or None,
                   bounds=[(None, None)] * len(problem.c), method='highs', options=options)


def solve_exact(matrix, rhs):
    """Small square rational elimination, used only for witness generation."""
    n = len(rhs)
    if len(matrix) != n or any(len(r) != n for r in matrix):
        raise InputError('basis must be square')
    a = [[rational(v) for v in row] + [rational(b)] for row, b in zip(matrix, rhs)]
    for j in range(n):
        pivot = next((i for i in range(j, n) if a[i][j]), None)
        if pivot is None:
            raise InputError('singular proposed basis')
        a[j], a[pivot] = a[pivot], a[j]
        scale = a[j][j]
        a[j] = [limited(v / scale) for v in a[j]]
        for i in range(n):
            if i != j and a[i][j]:
                scale = a[i][j]
                a[i] = [limited(u - limited(scale * v)) for u, v in zip(a[i], a[j])]
    return tuple(row[-1] for row in a)


def from_active_basis(problem, rows):
    """Reconstruct primal and dual from n explicit ('G'|'E', row_index) rows.

    Returns the certificate and verifier report, including rejection when a
    proposed basis is feasible but not optimal. This does not search for a basis.
    """
    n = len(problem.c)
    if n > 128:
        raise InputError('basis reconstruction is limited to 128 variables')
    if not isinstance(rows, (list, tuple)) or any(
            not isinstance(r, (list, tuple)) or len(r) != 2 for r in rows):
        raise InputError('basis rows must be pairs')
    rows = [tuple(r) for r in rows]
    for kind, i in rows:
        if not isinstance(kind, str) or kind not in {'G', 'E'} or isinstance(i, bool) or not isinstance(i, int):
            raise InputError('invalid basis row')
    if len(rows) != n or len(set(rows)) != n:
        raise InputError('supply exactly n distinct active rows')
    basis, rhs = [], []
    for kind, i in rows:
        if not isinstance(kind, str) or kind not in {'G', 'E'} or isinstance(i, bool) or not isinstance(i, int):
            raise InputError('invalid basis row')
        matrix, values = (problem.G, problem.h) if kind == 'G' else (problem.E, problem.d)
        if not 0 <= i < len(matrix):
            raise InputError('basis row index out of range')
        basis.append(matrix[i]); rhs.append(values[i])
    x = solve_exact(basis, rhs)
    multipliers = solve_exact(list(zip(*basis)), problem.c)
    y, z = [Fraction(0)] * len(problem.G), [Fraction(0)] * len(problem.E)
    for (kind, i), value in zip(rows, multipliers):
        (y if kind == 'G' else z)[i] = value
    proof = certificate(problem, 'optimal', x=x, y=y, z=z)
    return proof, verify(problem, proof)


def propose_from_scipy(problem, result, *, max_denominator=1_000_000):
    """Propose a certificate from scipy_solve's result (not arbitrary linprog).

    If rationalized numerical primal/dual vectors do not verify, try a single
    exact active-basis reconstruction. Degenerate or vertex-free LPs may remain
    unverified. This limitation never changes the verifier's acceptance rule.
    """
    if not isinstance(max_denominator, int) or isinstance(max_denominator, bool) or not 1 <= max_denominator <= 10**9:
        raise InputError('max_denominator must be in [1, 10**9]')
    if getattr(result, 'x', None) is None:
        raise InputError('solver returned no candidate; provide a ray witness separately')
    def numeric(values, name):
        return [Fraction(str(_finite_float(v, f'solver {name}[{i}]')))
                for i, v in enumerate(values)]
    try:
        raw_x = numeric(result.x, 'x')
        x = [v.limit_denominator(max_denominator) for v in raw_x]
        y = [-v.limit_denominator(max_denominator)
             for v in numeric(result.ineqlin.marginals, 'ineqlin.marginals')]
        z = [v.limit_denominator(max_denominator)
             for v in numeric(result.eqlin.marginals, 'eqlin.marginals')]
    except (AttributeError, TypeError) as exc:
        raise InputError('expected a scipy_solve result with primal and dual values') from exc
    proof = certificate(problem, 'optimal', x=x, y=y, z=z)
    report = verify(problem, proof)
    if report['verified'] or len(problem.c) > 128:
        return proof, report
    # Prefer constraints carrying nonzero dual weights; numerical closeness is
    # only a basis-selection heuristic. Never use it for a certificate verdict.
    candidates = [('E', i) for i in range(len(problem.E))]
    for i, (row, h) in enumerate(zip(problem.G, problem.h)):
        # Keep this heuristic comparison rational too: even finite inputs can
        # produce products or residuals outside the floating-point range.
        residual = abs(limited(dot(row, raw_x) - h))
        scale = max(Fraction(1), abs(h), dot([abs(a) for a in row],
                                           [abs(b) for b in raw_x]))
        if residual <= limited(Fraction(1, 10**7) * scale):
            candidates.append(('G', i))
    candidates.sort(key=lambda r: (r[0] != 'E', 0 if r[0] == 'E' else y[r[1]] == 0))
    # Incrementally select independent rows using exact elimination.
    selected, echelon = [], []
    for kind, i in candidates:
        row = list((problem.G if kind == 'G' else problem.E)[i])
        for pivot, old in echelon:
            coeff = row[pivot]
            row = [limited(u - limited(coeff * v)) for u, v in zip(row, old)]
        pivot = next((j for j, value in enumerate(row) if value), None)
        if pivot is not None:
            scale = row[pivot]
            echelon.append((pivot, [limited(v / scale) for v in row]))
            selected.append((kind, i))
        if len(selected) == len(problem.c):
            return from_active_basis(problem, selected)
    return proof, report
