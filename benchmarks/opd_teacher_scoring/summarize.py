"""Validate retained OPD measurement records and print a table without a GPU."""

import argparse
import json
import math
import statistics
from pathlib import Path

MODES = ("prompt-logprobs", "actual-only", "per-position")


def summarize(path):
    data = json.loads(path.read_text())
    config = data["config"]
    lengths = config["lengths"]
    concurrency = config["concurrency"]
    repeats = config["repeats"]
    if concurrency < 1 or repeats < 1 or len(lengths) != len(set(lengths)):
        raise ValueError(f"Invalid experiment configuration: {path}")
    expected = {
        (length, mode, repeat, index)
        for length in lengths
        for mode in MODES
        for repeat in range(repeats)
        for index in range(concurrency)
    }
    observed = set()
    grouped = {(length, mode): [] for length in lengths for mode in MODES}
    batch_deltas = {mode: [] for mode in MODES}
    for row in data["requests"]:
        key = (row["lengths"], row["mode"], row["round"], row["index"])
        if key not in expected or key in observed or row["concurrency"] != concurrency:
            raise ValueError(f"Unexpected or duplicate request: {path}: {key}")
        latency = row["latency_ms"]
        delta = row["max_abs_vs_single"]
        if not math.isfinite(latency) or latency <= 0 or not math.isfinite(delta) or delta < 0:
            raise ValueError(f"Invalid measurement: {path}: {key}")
        observed.add(key)
        grouped[row["lengths"], row["mode"]].append(latency)
        batch_deltas[row["mode"]].append(delta)
    if observed != expected:
        raise ValueError(f"Incomplete run: {path}: {len(observed)}/{len(expected)} requests")
    expected_parity = {(length, mode) for length in lengths for mode in (*MODES[1:], "same-request")}
    observed_parity = set()
    for row in data["parity"]:
        key = (row["lengths"], row["mode"])
        if key not in expected_parity or key in observed_parity:
            raise ValueError(f"Unexpected or duplicate parity check: {path}: {key}")
        if not math.isfinite(row["max_abs"]) or row["max_abs"] != 0:
            raise ValueError(f"Archived exact-parity claim does not hold: {path}: {key}")
        observed_parity.add(key)
    if observed_parity != expected_parity:
        raise ValueError(f"Missing parity checks: {path}")
    rows = []
    for length in lengths:
        p, r = map(int, length.split(":"))
        medians = [statistics.median(grouped[length, mode]) for mode in MODES]
        rows.append((p, r, concurrency, *medians))
    return rows, len(observed), {mode: max(values) for mode, values in batch_deltas.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    rows = []
    total = 0
    deltas = {}
    for concurrency in (1, 4):
        file = args.results / f"concurrency-{concurrency}.json"
        run_rows, count, run_deltas = summarize(file)
        rows.extend(run_rows)
        total += count
        deltas[concurrency] = run_deltas
    print("| Prompt | Response | Concurrency | Current top-1 (ms) | Actual-only (ms) | Per-position (ms) |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for p, r, concurrency, old, actual, new in rows:
        print(f"| {p} | {r} | {concurrency} | {old:.2f} | {actual:.2f} | {new:.2f} |")
    print(f"\nValidated {total} measured requests. All recorded parity checks have zero error.")
    print("Maximum differences from each single-request reference (also occur in the old path):")
    print(json.dumps(deltas, indent=2))


if __name__ == "__main__":
    main()
