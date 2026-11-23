# Interoperability is not upstream use

- **SciPy / HiGHS:** optional adapters use numerical results to propose
  certificates. The standalone verifier does not import a numerical solver.
  The [SciPy #20336 replay](../examples/original_model_gate/README.md) checks the
  original matrix separately. SciPy's own regression for this issue does not
  invoke LPWitness. No SciPy code or change to the status API is proposed here.
- **SCIP / VIPR:** nine models were covered by a private reproducible comparison
  (eight optimal/infeasible and one unbounded). Eight LPWitness exports were
  accepted by pinned serial VIPR; 55 negative controls were not accepted.
  Sixteen SCIP 10.0.2/10.1.0 proofs on the eight supported models produced
  continuous LP witnesses verified against the original models. In this
  experiment neither SCIP version emitted an accepted direct unbounded proof.
  Native VIPR was faster on these small examples; solve-plus-proof and
  in-process verification are not equal-work timings.
- The prototype comparison used continuous linear-combination proofs, with exact
  model binding, and disabled presolve where needed. The comparison scripts and
  third-party datasets are **not** shipped here. This checkout does not offer
  a public generic VIPR parser, a mixed-integer proof checker, or a SCIP plugin.
- No project is known to reference LPWitness, run it in upstream tests, carry it
  as a dependency, or use it in production.
