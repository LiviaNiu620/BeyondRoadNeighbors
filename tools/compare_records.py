"""Compare a fresh run record with an archived one, field by field.

usage: python tools/compare_records.py NEW ARCHIVED [--key K,protocol] [--rtol 1e-9]

Numeric fields present in both records are compared with a relative tolerance; timing, host
and provenance fields are ignored. Exits with status 1 on a mismatch.
"""
import argparse
import json
import math
import sys

IGNORE = {"seconds", "wall_seconds", "elapsed", "time", "timestamp", "started", "finished",
          "upstream", "host", "python", "numpy", "scipy", "git", "sec"}


def load(path):
    text = open(path).read().strip()
    if text.startswith("[") or (text.startswith("{") and "\n{" not in text):
        obj = json.loads(text)
        return obj if isinstance(obj, list) else [obj]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def walk(a, b, path, bad, rtol, counter):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a.keys() & b.keys():
            if k not in IGNORE and not k.endswith("_seconds"):
                walk(a[k], b[k], f"{path}.{k}", bad, rtol, counter)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", bad, rtol, counter)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        counter[0] += 1
        if not (math.isclose(a, b, rel_tol=rtol, abs_tol=1e-12) or (math.isnan(a) and math.isnan(b))):
            bad.append((path, a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new"); ap.add_argument("archived")
    ap.add_argument("--key", default="config"); ap.add_argument("--rtol", type=float, default=1e-9)
    args = ap.parse_args()
    keys = args.key.split(",")
    ident = lambda r: tuple(r.get(k) for k in keys)
    old = {ident(r): r for r in load(args.archived)}
    bad, counter, matched = [], [0], 0
    for r in load(args.new):
        if ident(r) in old:
            matched += 1
            walk(r, old[ident(r)], str(ident(r)), bad, args.rtol, counter)
    if matched == 0:
        sys.exit(f"no matching records by {keys}")
    for p, a, b in bad[:20]:
        print(f"MISMATCH {p}: new {a!r} archived {b!r}")
    status = "OK" if not bad else f"{len(bad)} mismatches"
    print(f"{args.new}: {matched} records, {counter[0]} numeric fields compared, {status}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
