"""Copy a reward cache with every EMPTY candidate (n_words == 0) set to -1e9, so the argmax over the
first n serves a non-empty draw whenever one exists (and an empty one only if all n are empty).

Why: the committed pointwise reward rates an empty completion ABOVE a typical anchor completion
(median log-odds -12 against -26 on the correctly recorded hybrid pool), a preference the recording
defect of caution (bc) masked on every pool on record, whose "empty" draws carried the prompt's tail.
Used only for readings registered as secondary; every registered primary uses the cache as scored.

Usage: .venv/bin/python analysis/nonempty_rewards.py <in.csv> <out.csv>
"""
import csv
import sys


def main(src, dst):
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    n = 0
    with open(dst, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for r in rows:
            if int(r["n_words"]) == 0:
                r["reward"], n = "-1000000000.0", n + 1
            w.writerow(r)
    print(f"{n} of {len(rows)} candidates empty; wrote {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
