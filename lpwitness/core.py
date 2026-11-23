"""Small exact-arithmetic verifier for continuous linear-program witnesses.

The verifier imports no solver. Model: minimize c*x with G*x >= h, E*x = d.
All variables are free unless explicitly constrained. No tolerances are used.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import re

MAX_DIMENSION = 4096
MAX_CELLS = 200_000
MAX_BITS = 8192
MAX_INPUT_BYTES = 8_000_000
MAX_NUMBER_CHARS = 1024
MAX_EXPONENT = 1000
_NUMBER = re.compile(r"[+-]?(?:\d+(?:/\d+)?|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\Z", re.ASCII)


class InputError(ValueError):
    """Malformed or unsupported input. This is never a certificate verdict."""


class ResourceLimit(InputError):
    """Configured arithmetic/input budget exceeded. No verdict was produced."""


def limited(q: Fraction) -> Fraction:
    if q.numerator.bit_length() > MAX_BITS or q.denominator.bit_length() > MAX_BITS:
        raise ResourceLimit(f"rational exceeds {MAX_BITS}-bit budget")
    return q


def rational(value) -> Fraction:
    """Parse exact numbers; reject floats rather than quietly changing the model."""
    if isinstance(value, bool):
        raise InputError("boolean is not a rational coefficient")
    if isinstance(value, Fraction):
        return limited(value)
    if isinstance(value, int):
        return limited(Fraction(value))
    if not isinstance(value, str):
        raise InputError("use integers or exact decimal/fraction strings, not floats")
    if len(value) > MAX_NUMBER_CHARS:
        raise ResourceLimit("numeric token too long")
    if not _NUMBER.fullmatch(value):
        raise InputError(f"invalid rational token: {value[:60]!r}")
    if 'e' in value.lower():
        exponent = value.lower().split('e')[1]
        if len(exponent.lstrip('+-')) > 4 or abs(int(exponent)) > MAX_EXPONENT:
            raise ResourceLimit("decimal exponent exceeds budget")
    try:
        return limited(Fraction(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise InputError("invalid rational coefficient") from exc


def dot(a, b) -> Fraction:
    if len(a) != len(b):
        raise InputError("internal dot-product shape mismatch")
    out = Fraction(0)
    for u, v in zip(a, b):
        out = limited(out + limited(u * v))
    return out


def vector(raw, length, name):
    if not isinstance(raw, (list, tuple)) or len(raw) != length:
        raise InputError(f"{name} must contain {length} entries")
    return tuple(rational(v) for v in raw)


def matrix(raw, width, name):
    if not isinstance(raw, (list, tuple)) or len(raw) > MAX_DIMENSION:
        raise InputError(f"{name} must be a list of at most {MAX_DIMENSION} rows")
    if len(raw) * width > MAX_CELLS:
        raise ResourceLimit("matrix cell budget exceeded")
    return tuple(vector(row, width, name) for row in raw)


def keys(obj, allowed, required=()):
    if not isinstance(obj, dict):
        raise InputError("expected an object")
    extra = set(obj) - set(allowed)
    missing = set(required) - set(obj)
    if extra or missing:
        raise InputError(f"unexpected keys {sorted(extra)}; missing keys {sorted(missing)}")


@dataclass(frozen=True)
class Problem:
    c: tuple
    G: tuple
    h: tuple
    E: tuple
    d: tuple

    def __post_init__(self):
        if not isinstance(self.c, (list, tuple)) or not 1 <= len(self.c) <= MAX_DIMENSION:
            raise InputError('c must be a nonempty bounded vector')
        n = len(self.c)
        G, E = matrix(self.G, n, 'G'), matrix(self.E, n, 'E')
        if (len(G) + len(E)) * n > MAX_CELLS:
            raise ResourceLimit('combined matrix cell budget exceeded')
        for name, value in [('c', vector(self.c, n, 'c')), ('G', G),
                            ('h', vector(self.h, len(G), 'h')), ('E', E),
                            ('d', vector(self.d, len(E), 'd'))]:
            object.__setattr__(self, name, value)

    @classmethod
    def from_dict(cls, raw):
        keys(raw, {'format', 'c', 'G', 'h', 'E', 'd'}, {'format', 'c'})
        if raw['format'] != 'lpwitness/1':
            raise InputError('unsupported model format')
        if not isinstance(raw['c'], (list, tuple)) or not 1 <= len(raw['c']) <= MAX_DIMENSION:
            raise InputError('c must be a nonempty bounded vector')
        n = len(raw['c'])
        G, E = matrix(raw.get('G', []), n, 'G'), matrix(raw.get('E', []), n, 'E')
        if (len(G) + len(E)) * n > MAX_CELLS:
            raise ResourceLimit('combined matrix cell budget exceeded')
        return cls(vector(raw['c'], n, 'c'), G, vector(raw.get('h', []), len(G), 'h'),
                   E, vector(raw.get('d', []), len(E), 'd'))

    def to_dict(self):
        return {'format': 'lpwitness/1', 'c': list(map(str, self.c)),
                'G': [list(map(str, r)) for r in self.G], 'h': list(map(str, self.h)),
                'E': [list(map(str, r)) for r in self.E], 'd': list(map(str, self.d))}

    @property
    def sha256(self):
        # Bind the witness to the original normalized rational model, not solver output.
        data = json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':')).encode()
        return hashlib.sha256(data).hexdigest()


def load_json(text):
    """Retain JSON decimal lexemes, reject duplicate keys and non-JSON numbers."""
    if not isinstance(text, str) or len(text.encode('utf-8')) > MAX_INPUT_BYTES:
        raise ResourceLimit('input size budget exceeded')
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise InputError(f'duplicate JSON key: {key}')
            obj[key] = value
        return obj
    def reject(value):
        raise InputError(f'non-finite JSON number: {value}')
    try:
        return json.loads(text, parse_float=str, parse_int=str, parse_constant=reject,
                          object_pairs_hook=pairs)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise InputError('invalid JSON') from exc


def certificate(problem, kind, **witness):
    out = {'format': 'lpwitness-certificate/1', 'model_sha256': problem.sha256, 'kind': kind}
    out.update({k: [str(q) for q in v] for k, v in witness.items()})
    return out


def verify(problem: Problem, proof: dict) -> dict:
    """Return verified true only if every defining exact condition holds.

    A rejected witness does NOT prove the opposite model status. Malformed inputs
    raise InputError; callers must not convert exceptions into a positive verdict.
    """
    # Validate even directly constructed Problem objects at the trust boundary.
    problem = Problem.from_dict(problem.to_dict())
    keys(proof, {'format', 'model_sha256', 'kind', 'x', 'y', 'z', 'ray'},
         {'format', 'model_sha256', 'kind'})
    if proof['format'] != 'lpwitness-certificate/1':
        raise InputError('unsupported certificate format')
    kind = proof['kind']
    fields = {'optimal': {'x', 'y', 'z'}, 'bounds': {'x', 'y', 'z'},
              'infeasible': {'y', 'z'}, 'unbounded': {'x', 'ray'}}
    if not isinstance(kind, str) or kind not in fields:
        raise InputError('unsupported certificate kind (continuous LP only)')
    if set(proof) - {'format', 'model_sha256', 'kind'} != fields[kind]:
        raise InputError(f'incorrect witness fields for {kind}')
    report = {'format': 'lpwitness-report/1', 'verified': False, 'claim': kind,
              'model_sha256': problem.sha256, 'checks': {}, 'violations': [], 'values': {}}
    def check(label, condition, detail=None):
        report['checks'][label] = bool(condition)
        if not condition:
            report['violations'].append({'check': label, **({'detail': detail} if detail else {})})
    check('model_binding', proof['model_sha256'] == problem.sha256)
    if not report['checks']['model_binding']:
        return report
    n, p, q = len(problem.c), len(problem.G), len(problem.E)
    x = vector(proof['x'], n, 'x') if 'x' in proof else None
    if x is not None:
        gs = [limited(dot(row, x) - h) for row, h in zip(problem.G, problem.h)]
        es = [limited(dot(row, x) - d) for row, d in zip(problem.E, problem.d)]
        check('primal_inequalities', all(s >= 0 for s in gs),
              [{'row': i, 'slack': str(s)} for i, s in enumerate(gs) if s < 0])
        check('primal_equalities', all(s == 0 for s in es),
              [{'row': i, 'residual': str(s)} for i, s in enumerate(es) if s != 0])
        report['values']['primal_objective'] = str(dot(problem.c, x))
    if kind in {'optimal', 'bounds', 'infeasible'}:
        y, z = vector(proof['y'], p, 'y'), vector(proof['z'], q, 'z')
        check('dual_nonnegative', all(v >= 0 for v in y))
        lhs = [limited(dot(tuple(r[j] for r in problem.G), y) +
                       dot(tuple(r[j] for r in problem.E), z)) for j in range(n)]
        target = problem.c if kind != 'infeasible' else [Fraction(0)] * n
        check('dual_stationarity', lhs == list(target),
              [{'column': j, 'residual': str(limited(a - b))}
               for j, (a, b) in enumerate(zip(lhs, target)) if a != b])
        lower = limited(dot(problem.h, y) + dot(problem.d, z))
        if kind == 'infeasible':
            report['values']['contradiction'] = str(lower)
            check('strict_contradiction', lower > 0)
        else:
            gap = limited(dot(problem.c, x) - lower)
            report['values'].update(dual_objective=str(lower), gap=str(gap))
            check('zero_gap' if kind == 'optimal' else 'nonnegative_gap',
                  gap == 0 if kind == 'optimal' else gap >= 0)
    if kind == 'unbounded':
        ray = vector(proof['ray'], n, 'ray')
        check('ray_inequalities', all(dot(r, ray) >= 0 for r in problem.G))
        check('ray_equalities', all(dot(r, ray) == 0 for r in problem.E))
        slope = dot(problem.c, ray)
        report['values']['ray_objective'] = str(slope)
        check('strict_descent', slope < 0)
    report['verified'] = all(report['checks'].values())
    return report
