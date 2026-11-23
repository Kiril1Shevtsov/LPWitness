# Optional native HiGHS proposals

Unlike the SciPy public result, native `highspy` can expose artificial dual
(Farkas) rays and primal recession directions. When available, the adapter
uses these proposals, with bounded rational support reconstruction if needed.
Nothing bypasses the exact model check. The feasible starting point and
direction of an unboundedness witness are verified separately.

    python -m pip install '.[highs]'
    python -m unittest discover -s tests -p 'test_highs.py' -v

The native suite covers exact optimal proposals, inconsistent bound/equality
models, recession rays with a feasible anchor, missing rays, unsupported
or ambiguous solver statuses, and incorrectly labeled claimed statuses.
It was run with highspy 1.16.0. The actual installed solver is optional;
stdlib-only verification does not import it.

Some trivial LPs have no exposed ray. The adapter refuses input conversions
that overflow or lose a nonzero coefficient to zero and rejects a model-load
warning such as small coefficients dropped by HiGHS. A solver warning,
undetermined status, or unavailable proof is not an exact contradiction or
an opposite status. Asking for rays can run an auxiliary solve.

Use `lpwitness solve MODEL --solver highs --certificate-out NEW_FILE` to
request a proposal. Retain the input and recheck the output separately with
`lpwitness verify MODEL NEW_FILE`. File creation does not certify anything.
The CLI reserves stdout for JSON; direct library solver calls can emit native
logging. Dense rational reconstruction is limited and intentionally incomplete.
