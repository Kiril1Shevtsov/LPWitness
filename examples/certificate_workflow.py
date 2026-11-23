"""Compare four proof types with a numerical solver's narrower status interface.

Run python -S -m examples.certificate_workflow, or omit -S and add --scipy
to also run SciPy/HiGHS. No network access, service, or package is needed for
the proof-only workflow.
"""

import argparse
from fractions import Fraction
import json

from lpwitness import certificate, verify
from lpwitness.scipy_adapter import from_linprog, propose_from_scipy, scipy_solve


def checked(problem, kind, **witness):
    report = verify(problem, certificate(problem, kind, **witness))
    if not report["verified"]:
        raise RuntimeError(f"expected a verified {kind} witness: {report}")
    return report


def run(use_scipy=False):
    models = {
        "optimal": from_linprog([1], bounds=[(2, None)]),
        "infeasible": from_linprog([1], A_ub=[[-1], [1]], b_ub=[-1, 0], bounds=[(None, None)]),
        "unbounded": from_linprog([-1], bounds=[(0, None)]),
    }
    proofs = {
        "optimal": checked(models["optimal"], "optimal", x=[2], y=[1], z=[]),
        "infeasible": checked(models["infeasible"], "infeasible", y=[1, 1], z=[]),
        "unbounded": checked(models["unbounded"], "unbounded", x=[0], ray=[1]),
    }
    lo = "1.00000000000000000001"
    hi = "1.00000000000000000002"
    original = from_linprog([1], bounds=[(lo, None)])
    other_original = from_linprog([1], bounds=[(hi, None)])
    rounded = from_linprog([1], bounds=[(str(float(lo)), None)])
    if not float(lo) == float(hi) == 1.0:
        raise RuntimeError("unexpected binary64 rounding")
    if len({original.sha256, other_original.sha256, rounded.sha256}) != 3:
        raise RuntimeError("coefficient contracts must bind differently")

    exact_interval = checked(original, "bounds", x=[hi], y=[1], z=[])
    values = exact_interval["values"]
    if Fraction(values["dual_objective"]) != Fraction(lo):
        raise RuntimeError("incorrect lower bound")
    if Fraction(values["primal_objective"]) != Fraction(hi):
        raise RuntimeError("incorrect upper bound")

    rounded_proof = certificate(rounded, "optimal", x=[1], y=[1], z=[])
    fake_original_proof = certificate(original, "optimal", x=[1], y=[1], z=[])
    incumbent = verify(original, fake_original_proof)
    if incumbent["verified"] or incumbent["checks"]["primal_inequalities"]:
        raise RuntimeError("rounded point passed the original model")
    if verify(original, rounded_proof)["verified"]:
        raise RuntimeError("different model digest passed")
    if not verify(rounded, rounded_proof)["verified"]:
        raise RuntimeError("rounded point must be optimal for the rounded model")

    bounded = models["optimal"]
    claims_not_supported = [
        not verify(bounded, certificate(bounded, "optimal", x=[3], y=[1], z=[]))["verified"],
        not verify(bounded, certificate(bounded, "infeasible", y=[1], z=[]))["verified"],
        not verify(bounded, certificate(bounded, "unbounded", x=[2], ray=[1]))["verified"],
    ]
    if claims_not_supported != [True, True, True]:
        raise RuntimeError("incorrect status accepted")
    report = {
        "exact_claims": {key: result["verified"] for key, result in proofs.items()},
        "wrong_claims_rejected": ["feasible_but_nonoptimal", "false_infeasibility", "false_unboundedness"],
        "rounding": {
            "exact_lower": values["dual_objective"],
            "exact_upper": values["primal_objective"],
            "exact_interval_verified": True,
            "rounded_point_verified_only_for_rounded_model": True,
            "original_model_sha256": original.sha256,
            "another_original_model_sha256": other_original.sha256,
            "rounded_model_sha256": rounded.sha256,
        },
    }
    if use_scipy:
        import scipy
        from lpwitness import InputError

        imported = {}
        for key, model in {**models, "near_tie_original": original}.items():
            result = scipy_solve(model)
            record = {"status": int(result.status), "success": bool(result.success)}
            try:
                proof, checked_report = propose_from_scipy(model, result)
            except InputError:
                if result.status == 0:
                    raise
                # Nonoptimal solver statuses are not proof witnesses.
                record["scipy_candidate_exactly_verified"] = False
                record["scipy_adapter_proposal"] = "unavailable"
            else:
                record["scipy_candidate_exactly_verified"] = checked_report["verified"]
                record["scipy_adapter_proposal"] = "returned"
                record["reconstructed_or_rationalized_point"] = proof.get("x")
            if key == "near_tie_original" and result.x is not None:
                record["raw_point"] = [str(Fraction.from_float(float(v))) for v in result.x]
                raw = certificate(model, "optimal", x=record["raw_point"], y=[1], z=[])
                record["raw_point_exactly_optimal"] = verify(model, raw)["verified"]
            imported[key] = record
        report["optional_scipy"] = {"version": scipy.__version__, "cases": imported}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scipy", action="store_true", help="also compare numerical statuses")
    print(json.dumps(run(parser.parse_args().scipy), indent=2, sort_keys=True))
