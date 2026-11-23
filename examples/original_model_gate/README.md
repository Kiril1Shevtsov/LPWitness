# Original-model gate: a real SciPy report, not a tolerance toy

A small, reproducible LPWitness workflow for [SciPy issue #20336](https://github.com/scipy/scipy/issues/20336). Its author reported a zero solver residual despite a nonzero residual in the submitted equations, and later switched to OR-Tools. This establishes a concrete user problem, **not demand for LPWitness or evidence of adoption**. The upstream warning fix has an existing author/branch; this is not a competing SciPy PR.

## Try the frozen artifacts without a solver

From the LPWitness checkout, with Python 3.10+:

```sh
python -S -m examples.original_model_gate.verify_demo examples/original_model_gate/results/final
```

It verifies two reconstructed optimum certificates, independently recomputes exact feasibility and a scalar objective bound (the sole objective variable attains its upper bound), rejects the two original numerical candidates, and rejects candidate/model corruption controls. It imports no SciPy, NumPy or HiGHS and uses no network. SHA binding is not authentication: keep your trusted model separately from untrusted solver output.

To check one artifact with the ordinary CLI:

```sh
# Exit 1: this raw candidate does not prove an optimum.
python -m lpwitness verify examples/original_model_gate/results/final/source_decimal/model.json examples/original_model_gate/results/final/source_decimal/raw.certificate.json
# Exit 0: this different, reconstructed candidate proves an optimum.
python -m lpwitness verify examples/original_model_gate/results/final/source_decimal/model.json examples/original_model_gate/results/final/source_decimal/reconstructed.certificate.json
```

## Repeat the numerical workflow

Install the optional SciPy dependency, then run from the checkout:

```sh
python -m pip install '.[scipy]'
python -m examples.original_model_gate.run_demo --out /tmp/lpwitness-original-model-run
python -S -m examples.original_model_gate.verify_demo /tmp/lpwitness-original-model-run
```

The output directory must not exist; the runner never overwrites an existing report. Add `--highs` only if the optional highspy dependency is installed. Results can change with numerical-solver versions; the frozen evidence below records the observed versions, not a prediction about all releases.

## What actually happened

SciPy 1.17.0 / NumPy 2.5.5 on Python 3.12.14, 33 variables, 32 equalities and 15 bound rows:

- Public `linprog` with the original bounds layout returned status 0 and reported equality-residual norm 0. Recomputing on the original input gave **0.021185053416350418**. No Python warning was observed.
- The LPWitness canonical-row solver layout reproduced the same discrepancy. Its original binary64 candidate failed the exact gate. Maximum equation residual was approximately **0.0067189211**.
- The existing SciPy adapter reconstructed a **different** rational primal/dual solution from a candidate active basis. The new solution passed every exact optimum check. Maximum coordinate change was about **66.9929** in this problem's unspecified units. This is not a certificate for the old vector and must not silently replace an application's accepted solution.
- Native exact SoPlex 9.0.0 solved both coefficient contracts below. Its rational primal outputs are exactly feasible and have the same exact objectives as the verified LPWitness certificates. This is a correctness comparison, not a speed or novelty claim.
- Optional highspy 1.16.0 loading returned a warning because small matrix coefficients would be dropped. The LPWitness HiGHS adapter refused that model copy, as designed. It did not claim infeasibility.

The source includes 20 matrix coefficients around `3.73378e-10`, below HiGHS' default small-matrix threshold. HiGHS maintainers explained this behavior in the issue and requested propagation of the load warning. LPWitness does **not** repair SciPy's logging or change HiGHS' options.

## Two models, explicitly separated

1. **source_decimal:** exact rational values of the decimal coefficient lexemes in the BSD-licensed upstream regression.
2. **solver_binary64:** exact rational values of the IEEE-754 binary64 coefficients actually sent in the numerical copy (`Fraction.from_float`), including objective, bounds and right-hand sides.

These are different mathematical models with different hashes and slightly different optimum fractions. Both raw vectors fail; both reconstructed solutions pass. This prevents confusing a decimal-vs-binary input change with the much larger omitted-coefficient effect. No pre-rounding physical data or domain interpretation is inferred.

The public-layout numerical result and the canonical-row numerical result are recorded separately. Raw-certificate artifacts come from the canonical `scipy_solve` output, whose multiplier layout matches the exact model. The adapter is not promised to accept arbitrary `linprog` result objects.

## Provenance, limits and use

Coefficient data are extracted without executing upstream source from `scipy/optimize/tests/test_linprog.py`, commit `5eb0ad93344c10feaf8d4119862dc5c3c4d71caa`, function `test_bug_20336`. See `provenance/SOURCE.json` and the retained BSD-3-Clause license. SciPy currently marks this test xfail. The private evidence bundle also retains the pinned full source and extraction script.

This is a useful pattern for small reproducible scientific LP workflows: freeze the original model, keep the solver's raw candidate distinct, and accept only an independently replayable certificate. A cheap floating residual check would already catch this particular discrepancy and is often enough. Exact certification adds a stronger, portable contract, at additional cost and with stricter input requirements.

It is not a general solver, large-sparse interface, production deployment, physical calibration or claim that every residual matters. The basis heuristic can fail on valid optima; rejection is not an opposite-status proof. The demonstration requires no core or adapter changes. Author-run exact-SoPlex comparison is complete. A separate internal model reviewer independently checked all source-decimal and binary64 coefficients, reran the numerical workflow and optimized offline replay, and exercised primal/dual/cross-contract corruption. Two provenance-consistency improvements were added after review and independently rechecked: frozen model-hash labels and agreement of the two input layouts. The final six focused tests passed. This is a narrow demo review, not a human, security or full-core audit.
