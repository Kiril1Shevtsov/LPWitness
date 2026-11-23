"""Measure verification only, with deterministic SciPy #20336 witnesses."""

from collections import defaultdict
from pathlib import Path
from statistics import median
from time import perf_counter_ns
import json
import platform

from lpwitness import Problem, verify


def run():
    root = Path(__file__).resolve().parent / "original_model_gate/results/final"
    cases = {}
    for contract in ("source_decimal", "solver_binary64"):
        base = root / contract
        model = Problem.from_dict(json.loads((base / "model.json").read_text()))
        for kind in ("raw", "reconstructed"):
            proof = json.loads((base / f"{kind}.certificate.json").read_text())
            if verify(model, proof)["verified"] != (kind == "reconstructed"):
                raise RuntimeError(f"incorrect verdict for {contract}/{kind}")
            cases[f"{contract}/{kind}"] = model, proof

    calls, repeats = 8, 11
    for model, proof in cases.values():
        for _ in range(8):
            verify(model, proof)

    samples = defaultdict(list)
    names = list(cases)
    for repeat in range(repeats):
        # Rotate the order to avoid consistently giving one case an early slot.
        shift = repeat % len(names)
        for name in names[shift:] + names[:shift]:
            model, proof = cases[name]
            start = perf_counter_ns()
            for _ in range(calls):
                verify(model, proof)
            samples[name].append((perf_counter_ns() - start) / calls / 1e6)

    return {
        "python": platform.python_version(),
        "machine_arch": platform.machine(),
        "work": "verify only, warm process, model/proof objects already loaded",
        "warmup_per_case": 8,
        "batches_per_case": repeats,
        "calls_per_batch": calls,
        "ms_per_call": {
            name: {"median": median(data), "min": min(data), "max": max(data), "raw_batch_means": data}
            for name, data in sorted(samples.items())
        },
        "limit": "Deterministic inputs/order, not deterministic machine timings. No competitor or speedup claim.",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
