"""Optional highspy proposals, including Farkas and recession rays.

HiGHS status and floating-point tolerances are never certificate verdicts.
Only core.verify against the original exact model can accept a proposal.
"""
from fractions import Fraction
from types import SimpleNamespace

from .core import InputError, Problem, certificate, dot, limited, verify
from .scipy_adapter import _finite_float, propose_from_scipy


def highs_solve(problem, **options):
    """Solve a canonical float copy with the public highspy API.

    Bounds remain explicit G rows; columns are free. Optional numerical settings
    are passed to HiGHS. A warning/error while loading is refused. Finite rounding
    can still change the numerical model; the original exact model is untouched.
    """
    import highspy
    problem = Problem.from_dict(problem.to_dict())

    def floats(values, name):
        out = []
        for i, value in enumerate(values):
            context = f'model {name}[{i}]'
            converted = _finite_float(value, context)
            if value != 0 and converted == 0:
                raise InputError(f'{context} underflows to zero in the solver float copy')
            out.append(converted)
        return out

    lp = highspy.HighsLp()
    lp.num_col_ = len(problem.c)
    lp.num_row_ = len(problem.G) + len(problem.E)
    lp.col_cost_ = floats(problem.c, 'c')
    lp.col_lower_ = [-highspy.kHighsInf] * len(problem.c)
    lp.col_upper_ = [highspy.kHighsInf] * len(problem.c)
    lp.row_lower_ = floats(problem.h, 'h') + floats(problem.d, 'd')
    lp.row_upper_ = [highspy.kHighsInf] * len(problem.G) + floats(problem.d, 'd')
    starts, indices, values = [0], [], []
    for kind, rows in [('G', problem.G), ('E', problem.E)]:
        for i, row in enumerate(rows):
            numeric = floats(row, f'{kind}[{i}]')
            for j, value in enumerate(numeric):
                if value:
                    indices.append(j)
                    values.append(value)
            starts.append(len(values))
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = starts
    lp.a_matrix_.index_ = indices
    lp.a_matrix_.value_ = values
    solver = highspy.Highs()
    for key, value in {'output_flag': False, **options}.items():
        try:
            status = solver.setOptionValue(key, value)
        except TypeError as exc:
            raise InputError(f'invalid HiGHS option {key!r}') from exc
        if status != highspy.HighsStatus.kOk:
            raise InputError(f'HiGHS rejected option {key!r}')
    if solver.passModel(lp) != highspy.HighsStatus.kOk:
        raise InputError('HiGHS could not load the numerical model without warnings')
    # A limit produces kWarning; preserve it for the proposal's explicit status
    # rejection rather than confusing it with model infeasibility.
    if solver.run() == highspy.HighsStatus.kError:
        raise InputError('HiGHS failed while solving the numerical model')
    return solver


def _numeric(values, size, name, denominator):
    try:
        raw = [Fraction(str(_finite_float(v, f'solver {name}[{i}]')))
               for i, v in enumerate(values)]
    except TypeError as exc:
        raise InputError(f'solver {name} must be a vector') from exc
    if len(raw) != size:
        raise InputError(f'solver {name} has the wrong length')
    return raw, [v.limit_denominator(denominator) for v in raw]


def _equations(matrix, rhs, width):
    """One bounded rational solution, free variables zero; never a verdict.

    Return None for an inconsistent system or the reconstruction size limit.
    Elimination is deliberately independent of the numerical solver's basis.
    All full LP conditions are checked by core.verify after reconstruction.
    """
    if width > 129 or len(matrix) > 4096:
        return None
    a = [[Fraction(v) for v in row] + [Fraction(b)] for row, b in zip(matrix, rhs)]
    pivots, rank = [], 0
    for j in range(width):
        pivot = next((i for i in range(rank, len(a)) if a[i][j]), None)
        if pivot is None:
            continue
        a[rank], a[pivot] = a[pivot], a[rank]
        scale = a[rank][j]
        a[rank] = [limited(v / scale) for v in a[rank]]
        for i in range(len(a)):
            if i != rank and a[i][j]:
                scale = a[i][j]
                a[i] = [limited(u - limited(scale * v)) for u, v in zip(a[i], a[rank])]
        pivots.append(j)
        rank += 1
    if any(not any(row[:width]) and row[-1] for row in a):
        return None
    solution = [Fraction(0)] * width
    for i, j in enumerate(pivots):
        solution[j] = a[i][-1]
    return solution


def _shim(x, y, z):
    return SimpleNamespace(x=x, ineqlin=SimpleNamespace(marginals=[-v for v in y]),
                           eqlin=SimpleNamespace(marginals=z))


def propose_from_highs(problem, solver, *, max_denominator=1_000_000):
    """Read a highs_solve result and return (certificate, exact report).

    Ray getters can run an auxiliary numerical solve, as documented by HiGHS.
    Missing rays, unsupported/limited statuses, and invalid solutions raise
    InputError. A present but inexact witness may return verified=False. There is
    no completeness promise. Ray reconstruction is limited to 128 variables and
    129 dual-support entries; unboundedness also needs an exact feasible anchor.
    """
    import highspy
    if (not isinstance(max_denominator, int) or isinstance(max_denominator, bool)
            or not 1 <= max_denominator <= 10**9):
        raise InputError('max_denominator must be in [1, 10**9]')
    n, m, e = len(problem.c), len(problem.G), len(problem.E)
    status = solver.getModelStatus()
    if status == highspy.HighsModelStatus.kOptimal:
        solution = solver.getSolution()
        if not solution.value_valid or not solution.dual_valid:
            raise InputError('HiGHS did not return a valid primal/dual solution')
        _numeric(solution.col_value, n, 'x', max_denominator)
        _numeric(solution.row_dual, m + e, 'row_dual', max_denominator)
        # Use raw values for the existing closeness-based active-set heuristic.
        return propose_from_scipy(problem, _shim(solution.col_value,
                                  solution.row_dual[:m], solution.row_dual[m:]),
                                  max_denominator=max_denominator)
    if status == highspy.HighsModelStatus.kInfeasible:
        ray_status, exists, vector = solver.getDualRay()
        if ray_status != highspy.HighsStatus.kOk or not exists:
            raise InputError('HiGHS returned no dual ray; infeasibility is unverified')
        raw, weights = _numeric(vector, m + e, 'dual_ray', max_denominator)
        proof = certificate(problem, 'infeasible', y=weights[:m], z=weights[m:])
        report = verify(problem, proof)
        if report['verified'] or n > 128:
            return proof, report
        # The numerical ray selects support only. Reconstruct exact stationarity
        # and normalize the strict Farkas contradiction to 1.
        support = [i for i, value in enumerate(raw) if value]
        if len(support) > 129:
            return proof, report
        rows, bounds = problem.G + problem.E, problem.h + problem.d
        matrix = [[rows[i][j] for i in support] for j in range(n)]
        matrix.append([bounds[i] for i in support])
        repaired = _equations(matrix, [0] * n + [1], len(support))
        if repaired is not None:
            weights = [Fraction(0)] * (m + e)
            for i, value in zip(support, repaired):
                weights[i] = value
            candidate = certificate(problem, 'infeasible', y=weights[:m], z=weights[m:])
            candidate_report = verify(problem, candidate)
            if candidate_report['verified']:
                return candidate, candidate_report
        return proof, report
    if status == highspy.HighsModelStatus.kUnbounded:
        ray_status, exists, vector = solver.getPrimalRay()
        if ray_status != highspy.HighsStatus.kOk or not exists:
            raise InputError('HiGHS returned no primal ray; unboundedness is unverified')
        # Fetch the anchor after getPrimalRay: an auxiliary solve may update it.
        solution = solver.getSolution()
        if not solution.value_valid:
            raise InputError('HiGHS returned no valid feasible-anchor candidate')
        raw_ray, ray = _numeric(vector, n, 'primal_ray', max_denominator)
        _, x = _numeric(solution.col_value, n, 'x', max_denominator)
        proof = certificate(problem, 'unbounded', x=x, ray=ray)
        report = verify(problem, proof)
        if report['verified'] or n > 128:
            return proof, report
        # Feasibility is zero-objective optimality. Reuse the exact active-basis
        # heuristic; this may fail on vertex-free feasible sets.
        feasibility = Problem.from_dict({**problem.to_dict(), 'c': [0] * n})
        anchor, anchor_report = propose_from_scipy(
            feasibility, _shim(solution.col_value, [0] * m, [0] * e),
            max_denominator=max_denominator)
        if not anchor_report['verified']:
            return proof, report
        # Select nearly tight homogeneous rows, never use tolerance to accept.
        matrix, rhs = list(problem.E), [0] * e
        for row in problem.G:
            residual = abs(dot(row, raw_ray))
            scale = max(Fraction(1), dot([abs(v) for v in row], [abs(v) for v in raw_ray]))
            if residual <= limited(Fraction(1, 10**7) * scale):
                matrix.append(row)
                rhs.append(0)
        # Preserve exactly zero numerical coordinates to choose a sparse ray.
        for j, value in enumerate(raw_ray):
            if not value:
                matrix.append([int(i == j) for i in range(n)])
                rhs.append(0)
        matrix.append(problem.c)
        rhs.append(-1)
        repaired = _equations(matrix, rhs, n)
        if repaired is not None:
            candidate = certificate(problem, 'unbounded', x=anchor['x'], ray=repaired)
            candidate_report = verify(problem, candidate)
            if candidate_report['verified']:
                return candidate, candidate_report
        return proof, report
    raise InputError(f'unsupported HiGHS model status {status}; no certificate conclusion')
