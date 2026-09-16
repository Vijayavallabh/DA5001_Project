"""feat-120: the third protected corpus, and the three things that make it a third READING.

The arm's whole value is that it changes one variable. These pin the ones that are easy to break
silently:

(1) The 100-passage sweep corpus is a PREFIX of the 600 the memorisers train on, and both are
    stratified over every book. Caution (w) in a new costume: bookmia100_attack_train.jsonl is
    grouped by book, so a plain --limit 100 takes a hundred passages of *1984* and reports them as
    a hundred-passage corpus. Caution (h) is the other half: a passage the memoriser never saw is
    not a memorisation probe.
(2) The band-3 yardsticks quoted in the pre-registration round from results/onset_ci.csv, once --
    caution (j), which put six of seventy-two appendix cells one off in the last digit.
(3) The corpus switch on analysis/onset_gutenberg.py leaves the Gutenberg path alone.
"""
import collections
import csv
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "data/bench/bookmia100_onset100.jsonl")
BIG = os.path.join(ROOT, "data/bench/bookmia100_onset600.jsonl")
PREREG = os.path.join(ROOT, "results/onset_prediction_bookmia.md")


def _rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


needs_corpus = pytest.mark.skipif(
    not os.path.exists(SUB),
    reason="run analysis/build_bookmia_onset_subset.py first (data/bench is gitignored)")


@needs_corpus
def test_the_sweep_corpus_is_a_prefix_of_the_training_corpus():
    """A passage the memoriser never saw is not a memorisation probe -- caution (h)."""
    sub, big = _rows(SUB), _rows(BIG)
    assert len(sub) == 100 and len(big) == 600
    ids = [r["prompt_id"] for r in sub]
    assert ids == [r["prompt_id"] for r in big[:100]], "the 100 is not a prefix of the 600"


@needs_corpus
def test_both_corpora_cover_every_book_and_no_book_dominates():
    """Caution (w): --limit 100 on a book-grouped file takes one book and calls it a corpus."""
    src = os.path.join(ROOT, "data/bench/bookmia100_attack_train.jsonl")
    all_books = {r["source_novel"] for r in _rows(src)}
    for path, n in ((BIG, 600), (SUB, 100)):
        rows = _rows(path)
        per = collections.Counter(r["source_novel"] for r in rows)
        assert set(per) == all_books, (path, sorted(all_books - set(per))[:3])
        assert max(per.values()) - min(per.values()) <= 1, (path, per.most_common(3))
        assert max(per.values()) <= -(-n // len(all_books)), (path, per.most_common(1))


@needs_corpus
def test_the_corpus_reader_can_actually_read_these_records():
    """load_corpus_file wants raw_text + reference_text; a BookMIA record must carry both."""
    from analysis.corpus_file import load_corpus_file
    got = load_corpus_file(SUB)
    assert len(got) == 100
    assert all(r.get("raw_text") and r.get("reference_text") for r in _rows(SUB))
    # the reader falls back to a `prompt_text` field if raw_text is missing. BookMIA records carry
    # both, and the one it must read is raw_text -- the prefix the memoriser was trained on. The
    # header is added by the reader, identically for Gutenberg, which is why the two are comparable
    # (caution (t): the header is what a base model must not be handed, and these are LoRAs).
    raw = _rows(SUB)[0]["raw_text"]
    assert got[0].prompt_text == "Complete the prefix:\n" + raw
    assert got[0].novel_source == _rows(SUB)[0]["source_novel"]


def test_the_band_three_yardsticks_round_from_the_csv_once():
    """Caution (j): a paper number rounds from the CSV, once, and is checked mechanically."""
    ci = {r["pair"]: r for r in csv.DictReader(open(os.path.join(ROOT, "results/onset_ci.csv")))
          if r["mode"] == "single"}
    head = open(PREREG, encoding="utf-8").read().partition("\n## Scoring log")[0]
    pairs = {"KL3M-520M": "KL3M-520M + mem. KL3M-520M",
             "Pleias-1.2B": "Pleias-1.2B + mem. Pleias-1.2B",
             "Phi-3.5-mini": "Phi-3.5-mini + mem. Phi-3.5-mini"}
    seen = 0
    for short, full in pairs.items():
        m = re.search(rf"^{re.escape(short)}\s+([\d.]+)\s+\[([\d.]+), ([\d.]+)\]\s+([\d.]+)\s+"
                      r"([\d.]+)\s+([\d.]+)$", head, re.M)
        assert m, f"band-3 row for {short} is not in the committed table"
        seen += 1
        r = ci[full]
        assert float(m.group(1)) == round(float(r["ratio_point"]), 4), short
        assert float(m.group(2)) == round(float(r["ratio_lo95"]), 4), short
        assert float(m.group(3)) == round(float(r["ratio_hi95"]), 4), short
        width = float(r["ratio_hi95"]) - float(r["ratio_lo95"])
        assert abs(float(m.group(4)) - width) < 5e-5, (short, m.group(4), width)
        gut = ci[f"{short} (Gutenberg)"]
        assert float(m.group(5)) == round(float(gut["ratio_point"]), 4), short
        rng = abs(float(gut["ratio_point"]) - float(r["ratio_point"]))
        assert abs(float(m.group(6)) - rng) < 5e-5, (short, m.group(6), rng)
    assert seen == 3


def test_the_corpus_switch_leaves_the_gutenberg_path_alone():
    from analysis.onset_gutenberg import CORPORA
    assert set(CORPORA) == {"gutenberg", "bookmia"}
    for key, (pretty, pairs) in CORPORA.items():
        assert len(pairs) == 3
        assert all(lab.endswith(f"({pretty})") for lab, _, _ in pairs), key
        # every pair points at its own sweep directory, and the two corpora never share one
        assert len({run for _, run, _ in pairs}) == 3, key
    gut = {run for _, run, _ in CORPORA["gutenberg"][1]}
    book = {run for _, run, _ in CORPORA["bookmia"][1]}
    assert not (gut & book)
    # the CopyBench twins are the SAME three pairs -- that is what makes this a third reading
    assert ({t for _, _, t in CORPORA["gutenberg"][1]} == {t for _, _, t in CORPORA["bookmia"][1]})


def test_the_committed_grid_is_the_gutenberg_grid_verbatim():
    head = open(PREREG, encoding="utf-8").read().partition("\n## Scoring log")[0]
    gut = [r for r in csv.DictReader(open(os.path.join(ROOT, "results/onset_ci.csv")))
           if r["pair"].endswith("(Gutenberg)")]
    assert gut
    grid = gut[0]["k_grid"].split()
    assert all(g == grid for g in (r["k_grid"].split() for r in gut)), "Gutenberg pairs disagree"
    m = re.search(r"`-1 0 ([0-9. ]+)`", head)
    assert m, "the committed grid is no longer quoted in the pre-registration"
    assert m.group(1).split() == grid, (m.group(1).split(), grid)


def test_the_sweep_script_agrees_with_the_scorer_and_the_theory_manifest():
    """A mistyped out-tag is SILENT: onset_gutenberg.py prints 'no sweep at ..., skipping' to
    stderr and scores the other two, so the arm reads as a two-pair result rather than an error.
    Same for a mistyped memoriser dir, which the theory manifest and the sweep must agree on."""
    from analysis.onset_gutenberg import CORPORA
    sh = open(os.path.join(ROOT, "scripts/run_bookmia_sweeps.sh"), encoding="utf-8").read()
    rows = re.findall(r'^\s*"([^"]*\|[^"]*)"\s*$', sh, re.M)
    assert len(rows) == 3, rows
    table = {}
    for row in rows:
        key, tag, safe, risky = (c.strip() for c in row.split("|"))
        assert key == tag, (key, tag)
        table[tag] = (safe, risky)

    # the scorer may point at <tag>_full where appendix_seed.tex's no-crossing rule licensed a
    # grid extension (Pleias-1.2B: 43.1% against 0.0% and 0.0%); both grids stay reported.
    scorer = {r for _, r, _ in CORPORA["bookmia"][1]}
    assert {d.replace("_full", "") for d in scorer} == {f"output/phase5/fineb_{t}" for t in table}
    for d in scorer:
        if d.endswith("_full"):
            assert os.path.isdir(os.path.join(ROOT, d.replace("_full", "_ext"))), \
                f"{d} claims a merged grid but its _ext arm does not exist"

    manifest = [l.split("\t") for l in
                open(os.path.join(ROOT, "results/onset_theory_pairs_bookmia.tsv"),
                     encoding="utf-8").read().splitlines() if l.strip()]
    assert len(manifest) == 3
    # the manifest is label, memoriser, ANCHOR -- the same two models the sweep is handed
    assert {(m[2], m[1]) for m in manifest} == set(table.values())
    assert {m[0] for m in manifest} == {lab for lab, _, _ in CORPORA["bookmia"][1]}

    # the grid in the script is the one the pre-registration committed, character for character
    head = open(PREREG, encoding="utf-8").read().partition("\n## Scoring log")[0]
    g_sh = re.search(r'^GRID="([^"]+)"', sh, re.M)
    g_md = re.search(r"`(-1 0 [0-9. ]+)`", head)
    assert g_sh and g_md and g_sh.group(1).split() == g_md.group(1).split()
    # --queries-out is what recheck_violations.py reads; losing it loses the invariant recheck
    assert "--queries-out" in sh and "--modes single --limit 100" in sh


def test_the_sweep_gate_refuses_an_uncommitted_p1_and_passes_a_committed_one():
    """The out-of-sample claim IS the ordering: P1 committed, then the sweep. 'I remembered to
    commit first' is not evidence of that; git is. Caution (p) is the other half -- a gate that
    fails every configuration is not a gate, so this asserts the PASSING case too.

    Exercised in a throwaway git repo so the real one is never touched."""
    import shutil
    import subprocess
    import tempfile

    sh = os.path.join(ROOT, "scripts/run_bookmia_sweeps.sh")
    body = open(sh, encoding="utf-8").read()
    gate = body.partition("# GATE-BEGIN")[2].partition("# GATE-END")[0]
    assert gate.strip() and "GATE-BEGIN" not in gate, "the gate sentinels moved"
    assert "git ls-files --error-unmatch" in gate and "git diff --quiet HEAD" in gate

    with tempfile.TemporaryDirectory() as d:
        run = lambda *c, **kw: subprocess.run(c, cwd=d, capture_output=True, text=True, **kw)
        run("git", "init", "-q")
        run("git", "config", "user.email", "t@t"); run("git", "config", "user.name", "t")
        os.makedirs(os.path.join(d, "results"))
        script = os.path.join(d, "gate.sh")
        with open(script, "w") as fh:
            fh.write("set -u\n" + gate + "\necho GATE_PASSED\n")

        def attempt():
            return subprocess.run(["bash", script], cwd=d, capture_output=True, text=True)

        theory = os.path.join(d, "results/onset_theory_bookmia.csv")
        prereg = os.path.join(d, "results/onset_prediction_bookmia.md")
        # the real file with its P1 block stripped out -- i.e. the state it was in between 14:27
        # and 17:41, which is the state the gate has to refuse
        real = open(PREREG, encoding="utf-8").read()
        pre_p1 = real.partition("## The P1 predictions")[0] + real.partition("\n## Scoring log")[1]
        assert "pred onset" not in pre_p1.partition("\n## Scoring log")[0]
        open(prereg, "w").write(pre_p1)
        run("git", "add", "-A"); run("git", "commit", "-qm", "base")

        assert attempt().returncode == 2, "missing theory CSV must refuse"

        open(theory, "w").write("pair,pred_onset_median\nx,1.0\n")
        r = attempt(); assert r.returncode == 2 and "untracked" in r.stdout, r.stdout

        run("git", "add", "results/onset_theory_bookmia.csv")
        r = attempt(); assert r.returncode == 2 and "uncommitted" in r.stdout, r.stdout

        run("git", "commit", "-qm", "p1")
        r = attempt(); assert r.returncode == 2 and "does not quote" in r.stdout, r.stdout

        # now the pre-registration quotes the predictions, as it must before any sweep
        open(prereg, "w").write(real)          # the genuine committed file, P1 block included
        run("git", "add", "-A"); run("git", "commit", "-qm", "predictions")
        r = attempt()
        assert r.returncode == 0 and "GATE_PASSED" in r.stdout, (r.returncode, r.stdout, r.stderr)
        assert "gate passed: P1 committed at" in r.stdout


def test_the_extended_grid_is_the_committed_grid_plus_exactly_the_registered_points():
    """The extension is licensed by a rule, and its grid was committed before it ran. A merge that
    silently dropped or duplicated a budget would move the onset without moving anything visible."""
    import collections as _c
    full = os.path.join(ROOT, "output/phase5/fineb_pleias12b_full/composition_summary.csv")
    if not os.path.exists(full):
        pytest.skip("run the Pleias grid extension first")
    ks = [r["k"] for r in csv.DictReader(open(full)) if r["mode"] == "single" and r["L"] == "0"]
    assert len(ks) == len(set(ks)), _c.Counter(ks).most_common(3)   # no double-counted budget
    got = sorted(float(k) for k in ks)
    head = open(PREREG, encoding="utf-8").read().partition("\n## Scoring log")[0]
    committed = [float(x) for x in re.search(r"`(-1 0 [0-9. ]+)`", head).group(1).split()]
    registered_ext = [4.6, 5.3, 6.6]           # committed in the scoring log before the run
    assert got == sorted(committed + registered_ext), (got, sorted(committed + registered_ext))


# ---------------------------------------------------------------------------
# The GPU launchers must strip the stale in-repo driver from LD_LIBRARY_PATH.
#
# Not a style rule. LD_LIBRARY_PATH here leads with NVIDIA-Linux-x86_64-580.173.02, an extracted
# runfile in the repo root whose libnvidia-ml.so.1 shadows the system's and does not match the
# loaded kernel module (580.178.04), so every NVML call fails while CUDA compute succeeds. On
# 2026-09-16 that killed four fine-tunes AFTER they had written their merged model, in the
# post-training generate(), and each queue shell then skipped the sweep behind it. glibc reads
# LD_LIBRARY_PATH once at exec, so the repair only works from the shell -- which is why it is a
# sourced line in the launcher and not a line of python, and why a test has to guard the launcher.
LAUNCHERS = (
    "scripts/run_bookmia_memorisers.sh",
    "scripts/run_bookmia_p1.sh",
    "scripts/run_bookmia_sweeps.sh",
    "scripts/run_copybench_seeds.sh",
)


@pytest.mark.parametrize("name", LAUNCHERS)
def test_every_gpu_launcher_strips_the_shadowing_driver(name):
    path = os.path.join(ROOT, name)
    body = open(path).read()
    src = [ln for ln in body.splitlines() if "gpu_env.sh" in ln and not ln.lstrip().startswith("#")]
    assert len(src) == 1, f"{name} must source scripts/gpu_env.sh exactly once, found {len(src)}"
    # It has to run before the first GPU job, and after the cd that makes the path resolve.
    cd_at = body.index('cd "$(dirname "$0")/.."')
    assert cd_at < body.index(src[0]), f"{name} sources gpu_env.sh before cd'ing to the repo root"
    first_job = body.find(".venv/bin/python")
    assert first_job == -1 or body.index(src[0]) < first_job, f"{name} launches before the strip"


def test_the_strip_actually_removes_the_shadowing_directory():
    """Run the snippet in a shell with a poisoned path and check what comes out.

    Pinning the grep pattern by spelling would pass on a snippet that strips nothing, so this
    executes it. The directory name carries a version, and the next driver will have a different
    one, so the pattern must match the family and not one release.
    """
    import subprocess

    poisoned = "/repo/NVIDIA-Linux-x86_64-580.173.02:/usr/local/cuda/lib64:/repo/NVIDIA-Linux-x86_64-999.9:/lib"
    out = subprocess.run(
        ["bash", "-c", 'set -u; . scripts/gpu_env.sh; printf %s "$LD_LIBRARY_PATH"'],
        cwd=ROOT, env={**os.environ, "LD_LIBRARY_PATH": poisoned},
        capture_output=True, text=True, check=True,
    ).stdout
    assert "NVIDIA-Linux-x86_64-" not in out, f"stale driver survived the strip: {out}"
    assert out.split(":") == ["/usr/local/cuda/lib64", "/lib"], f"strip damaged the path: {out}"


def test_the_strip_survives_an_unset_and_an_empty_path():
    """`set -u` is on in every launcher, so an unset LD_LIBRARY_PATH must not abort the queue."""
    import subprocess

    for env in ({}, {"LD_LIBRARY_PATH": ""}):
        base = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
        res = subprocess.run(
            ["bash", "-c", 'set -u; . scripts/gpu_env.sh; printf %s "$LD_LIBRARY_PATH"'],
            cwd=ROOT, env={**base, **env}, capture_output=True, text=True,
        )
        assert res.returncode == 0, f"gpu_env.sh failed under set -u with {env}: {res.stderr}"
        assert res.stdout == "", f"expected an empty path, got {res.stdout!r}"
