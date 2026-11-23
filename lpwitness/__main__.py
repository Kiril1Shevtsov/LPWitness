"""CLI: python -m lpwitness verify MODEL.json CERTIFICATE.json."""
import argparse
from contextlib import contextmanager
import os
import json
from pathlib import Path
import sys

from .core import InputError, MAX_INPUT_BYTES, Problem, load_json, verify


def read(path):
    with Path(path).open('rb') as f:
        raw = f.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise InputError('file exceeds input budget')
    return load_json(raw.decode('utf-8'))


@contextmanager
def _native_output_to_stderr():
    """CLI-only: keep native solver banners out of the JSON stdout stream.

    FD redirection is process-wide; this is not a threaded-library logging API.
    The solver's Python API does not change the caller's output descriptors.
    """
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Verify exact rational continuous-LP witnesses offline.')
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('verify', help='exit 0 verified, 1 rejected, 2 malformed/unsupported')
    check.add_argument('model'); check.add_argument('certificate')
    solve = sub.add_parser('solve', help='optional numerical proposal; exact verifier decides')
    solve.add_argument('model')
    solve.add_argument('--solver', choices=['scipy', 'highs'], default='scipy')
    solve.add_argument('--certificate-out', required=True)
    args = parser.parse_args(argv)
    try:
        problem = Problem.from_dict(read(args.model))
        if args.command == 'verify':
            report = verify(problem, read(args.certificate))
        else:
            if args.solver == 'highs':
                from .highs_adapter import highs_solve, propose_from_highs
                with _native_output_to_stderr():
                    proof, report = propose_from_highs(problem, highs_solve(problem))
            else:
                from .scipy_adapter import scipy_solve, propose_from_scipy
                proof, report = propose_from_scipy(problem, scipy_solve(problem))
            # Write rejected candidates too, explicitly marked as rejected in stdout.
            with Path(args.certificate_out).open('x', encoding='utf-8') as f:
                json.dump(proof, f, indent=2); f.write('\n')
        print(json.dumps(report, indent=2))
        return 0 if report['verified'] else 1
    except (InputError, OSError, UnicodeError, ImportError, OverflowError) as exc:
        print(json.dumps({'format': 'lpwitness-report/1', 'verified': False,
                          'error': type(exc).__name__, 'message': str(exc)}))
        return 2


if __name__ == '__main__':
    sys.exit(main())
