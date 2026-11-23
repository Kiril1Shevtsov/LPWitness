# LPWitness

An offline, exact-rational checker for certificates of continuous linear
programs. A solver can propose a result; LPWitness verifies the witness against
the original model you supply. It does not treat a solver's status as a proof.

**Version 0.1.0.** Tested with SciPy/HiGHS and with selected
SCIP/VIPR artifacts. No upstream integration, endorsement, or adoption is
claimed. Maintained by [Kirill Shevtsov](https://github.com/Kiril1Shevtsov).
See [citation information](CITATION.cff) and [provenance](docs/PROVENANCE.md).

## Try it in 30 seconds

In this source checkout with Python 3.10+ (no dependencies or network):

```sh
python -S -m lpwitness verify examples/blending.model.json examples/blending.certificate.json
```

This returns `"verified": true` and an exact objective of `"11/2"`. The same
command against `examples/tolerance_candidate.model.json` and its matching
certificate returns `"verified": false`.

The `verify` command emits JSON. Exit **0** means verified; **1** means the
witness does not establish its claim; **2** means input/resource/unsupported
error. A rejected witness does not imply the opposite LP status.

To install from this checkout into your own environment (build tooling may
be needed), run `python -m pip install .`. This adds the `lpwitness` command.
The verifier has no runtime dependencies. Optional source installs:
`python -m pip install '.[scipy]'` or `python -m pip install '.[highs]'`.

## Verified outcomes, and where they stop

The model is `minimize c·x` subject to `Gx ≥ h` and `Ex = d`, with otherwise
free variables. Enter integers or exact decimal/fraction strings. A normalized
model digest prevents accidental mix-ups; it is not a signature or a trust
decision about your model.

| Witness | What a verified result establishes |
|---|---|
| Optimality | Primal feasibility, dual feasibility, and exactly zero gap |
| Bounds | A provable lower and upper bound on the objective optimum |
| Infeasibility | An exact Farkas contradiction |
| Unboundedness | A feasible anchor and a direction of strictly decreasing objective |

The optional numerical adapters produce *proposals*. A repair may return a
different rational point, and may also fail on valid models. The SciPy public
result cannot export the infeasibility/ray witnesses; the optional HiGHS adapter
uses solver-exposed rays when available. See the [exact contract](docs/CONTRACT.md)
and [HiGHS adapter notes](docs/HIGHS.md).

This is not an LP solver, a sparse large-scale pipeline, a MIP proof checker,
or a formal verification of the implementation. Keep the original model
separate from solver output. Use the limits documented in the contract; they
are not a sandbox. No speed advantage over established exact tools is claimed.

## A real SciPy regression, replayed offline

[SciPy #20336](https://github.com/scipy/scipy/issues/20336) shows why a residual
reported by a numerical solver and feasibility for the original equations can
differ. LPWitness rejects the raw solution and verifies a *different* exact
optimum. The demo retains both the original decimal model and the distinct
binary64 coefficient contract, preserving the original case's
[SciPy BSD notice](examples/original_model_gate/provenance/SCIPY_LICENSE.txt).

```sh
python -S -m examples.original_model_gate.verify_demo examples/original_model_gate/results/final
python -S -m unittest discover -s tests -v
```

[Reproduce the numerical workflow](examples/original_model_gate/README.md) with
an optional local SciPy installation. [Interoperability notes](docs/INTEROPERABILITY.md) describe the tested
SCIP/VIPR subset and explicitly separate it from upstream use.

## More examples

`python -S -m examples.certificate_workflow` exercises all four certificate
kinds, a verified interval only `1e-20` wide, and a point that rounds to the
same binary64 data but is infeasible for one original exact model. With
the optional SciPy dependency, `python -m examples.certificate_workflow --scipy`
compares these checks with numerical return statuses.

`python -S -m examples.benchmark_gate` times verification of the frozen SciPy
case. Inputs and repeat order are fixed; wall-clock results depend on hardware
and load. It does not measure parsing, solving, or reconstruction.
