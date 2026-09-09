"""Every number in the manuscript should be findable in a committed CSV.

Two number errors reached a compiled version of this paper: a relative spread of "170%" that
reproduces under no definition (the value is 115%), and a truncation ratio from a run whose target
did not contain the protected passage. Both were found by reading, which does not scale. This
script does the mechanical half: it pulls every numeric literal out of the LaTeX sources and reports
the ones that appear in no CSV under results/, so a human only has to look at those.

It is a net, not a proof. A number can be wrong and still appear somewhere; a number can be right
and appear nowhere, because it was derived (a ratio, a difference, a page count) or is a citation
year, a model size or a budget grid point. The output is a worklist ordered by how surprising the
absence is, not a pass/fail.

Usage:
  .venv/bin/python analysis/audit_numbers.py --tex ~/sub/satml --results results
"""
import argparse, csv, glob, os, re, sys

# things that are numbers but never live in a CSV
IGNORE = re.compile(r"^(19|20)\d\d$")          # years
SMALL = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "0.5", "1.5", "100", "50"}


def literals(path):
    """(number, line number) for every numeric literal inside math mode in one .tex file."""
    out = []
    for i, line in enumerate(open(path, encoding="utf-8"), 1):
        if line.lstrip().startswith("%"):
            continue
        for m in re.finditer(r"\$([^$]*)\$", line):
            # LaTeX writes a thousands separator as 32{,}011; scanning before stripping it would
            # report the fragments 32 and 011 as unsourced numbers, which is how this script first
            # produced ten false positives out of ten.
            # strip only the LaTeX thousands separator 32{,}011; a bare comma is a real separator
            # ("[0,1]"), and stripping it would manufacture the number 01.
            math = m.group(1).replace("{,}", "")
            for n in re.finditer(r"-?\d+\.?\d*", math):
                out.append((n.group(0), i))
    return out


def haystack(results_dir):
    """Every value in every CSV, as strings, plus the same values rounded to 1-4 decimals."""
    seen = set()
    for path in glob.glob(os.path.join(results_dir, "**", "*.csv"), recursive=True):
        try:
            for row in csv.reader(open(path, encoding="utf-8", errors="ignore")):
                for cell in row:
                    cell = cell.strip()
                    if not cell:
                        continue
                    seen.add(cell)
                    try:
                        v = float(cell)
                    except ValueError:
                        continue
                    for d in range(5):
                        seen.add(f"{v:.{d}f}".rstrip("."))
                        seen.add(f"{100 * v:.{d}f}".rstrip("."))
        except (OSError, csv.Error):
            continue
    return seen


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # The manuscript lives outside the repo, so this is a path the caller supplies. It must NOT
    # default to an absolute one: scripts/build_artifact.sh refuses to release an artifact
    # containing a home directory, and this default carried an author name into it.
    ap.add_argument("--tex", default=os.environ.get("SATML_DIR", "../sub/satml"),
                    help="directory holding the manuscript; $SATML_DIR, else ../sub/satml")
    ap.add_argument("--main", default="iclr_2027.tex")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()

    main_tex = os.path.join(a.tex, a.main)
    inputs = [main_tex] + [os.path.join(a.tex, m.group(1) + ".tex")
                           for m in re.finditer(r"\\input\{([^}]+)\}", open(main_tex).read())]
    hay = haystack(a.results)
    print(f"{len(hay):,} distinct values across {a.results}/**.csv\n")

    missing, total = [], 0
    for path in inputs:
        if not os.path.exists(path):
            print(f"[audit] missing input {path}", file=sys.stderr)
            continue
        for num, line in literals(path):
            total += 1
            if num in SMALL or IGNORE.match(num) or num.lstrip("-") in hay:
                continue
            missing.append((os.path.relpath(path, a.tex), line, num))

    print(f"{total} numeric literals in math mode; {len(missing)} appear in no CSV\n")
    by_file = {}
    for f, line, num in missing:
        by_file.setdefault(f, []).append((line, num))
    for f in sorted(by_file):
        nums = by_file[f]
        print(f"  {f}  ({len(nums)})")
        print("    " + ", ".join(f"{n}@L{l}" for l, n in nums[:24])
              + (" ..." if len(nums) > 24 else ""))
    print("\nDerived quantities, page counts, model sizes and grid points are expected here.")
    print("A number that should have come from a run and is not in this list is not thereby correct;")
    print("a number in this list that nobody can source is the one to chase.")


if __name__ == "__main__":
    main()
