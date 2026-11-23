# Exact certificate contract

For rational data, let P be `min c^T x` subject to `Gx >= h`, `Ex=d`, with x free.
A certificate must name the SHA-256 of the normalized original model. The hash
contains the exact objective, all coefficients and right-hand sides, with order
preserved. Rationally equal spellings normalize equally.

## Optimal or bounded objective

Supply x, y, z with:

- `Gx >= h`, `Ex=d`;
- `y >= 0`, with z unrestricted;
- `G^T y + E^T z = c`.

For every feasible v,
`c^T v = y^T Gv + z^T Ev >= y^T h + z^T d`.
Therefore `[h^T y+d^T z, c^T x]` is a rigorous interval containing the optimum.
The kind `bounds` accepts such an interval with a nonnegative gap. The kind
`optimal` additionally requires exact equality of the endpoints. Complementary
slackness need not be checked separately because zero gap already implies it.

## Infeasibility (Farkas witness)

Supply `y>=0`, z unrestricted, with
`G^T y+E^T z=0` and `h^T y+d^T z>0`.
Any feasible v would imply `0 >= h^T y+d^T z > 0`, a contradiction.
This certifies infeasibility without trusting a solver or a presolve log.

## Unboundedness

Supply a **feasible base point** x and a ray r such that
`Gr>=0`, `Er=0`, and `c^T r<0`.
Then `x+t*r` is feasible for every rational `t>=0`, and its objective tends to
minus infinity. A recession direction alone is insufficient: an empty feasible
set also admits directions. The verifier deliberately checks the base point.

## Failure semantics

Every required condition must pass. A rejected/malformed/oversized witness is not
proof of infeasibility, nonoptimality or boundedness. In particular, a failed
exact check does not establish a bug in a solver that promises only feasibility
up to its documented tolerance. No rounding of the original model is performed.
The proof is conditional on the rational model and correctness of this small
verifier, not on external physical/biomedical/economic assumptions.
