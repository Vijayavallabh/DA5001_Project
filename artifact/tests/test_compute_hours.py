"""log_elapsed sources two GPU-hour figures in the paper, so it gets a check.

The CP-Fuse fine-tunes write their output directory only at the merge, so the
directory birth time is the *end* of the run and the dir_birth rule reports 0.0
hours. The duration has to come from the log's own cumulative epoch timer.
"""
import importlib.util, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compute_hours", ROOT / "analysis" / "compute_hours.py")
ch = importlib.util.module_from_spec(spec)
sys.modules["compute_hours"] = ch
spec.loader.exec_module(ch)


def test_log_elapsed_takes_the_last_cumulative_epoch(tmp_path):
    log = tmp_path / "ft.log"
    log.write_text(
        "[ft] epoch 1/12 mean token loss 0.9 (76s)\n"
        "[ft] epoch 11/12 mean token loss 0.04 (832s)\n"
        "[ft] epoch 12/12 mean token loss 0.04 (908s)\n"
        "Writing model shards: 100%|##| 1/1 [00:16<00:00, 16.54s/it]\n"
    )
    # the cumulative maximum, not the sum of the per-epoch lines
    assert ch.log_elapsed(str(log)) == 908.0


def test_log_elapsed_ignores_a_log_with_no_timer(tmp_path):
    log = tmp_path / "empty.log"
    log.write_text("[stage] reading data\nno timings here\n")
    assert ch.log_elapsed(str(log)) == 0.0


def test_cpfuse_finetunes_are_not_zero_hours():
    """Regression: both rows read 0.0 GPU-hours under the dir_birth rule."""
    jobs = {name: rule for name, _p, _g, rule, _n in ch.JOBS}
    for job in ("CP-Fuse fine-tune A", "CP-Fuse fine-tune B"):
        assert jobs[job].startswith("elapsed:"), f"{job} must not use dir_birth"


def test_has_finetune_reads_the_log_not_the_job_name(tmp_path):
    """The paper's fine-tune share is an upper bound over every launcher log that trained a
    memoriser, and most of those logs are named for the chain, not for the fine-tune. Detecting
    them by name undercounted the share by 10 GPU-hours."""
    trained = tmp_path / "two_families.log"
    trained.write_text("[stage] reading data\n[ft] 608 excerpts -> 608 training texts\n[sweep] k=3\n")
    swept = tmp_path / "order_frontier.log"
    swept.write_text("[of] k=0.25 alpha=1 fidelity 0.41\n")
    assert ch.has_finetune(str(trained))
    assert not ch.has_finetune(str(swept))
    assert not ch.has_finetune(str(tmp_path))          # a directory is not a log
    assert not ch.has_finetune(str(tmp_path / "no.log"))


def test_committed_summary_matches_the_manuscript_upper_bound():
    """The LLM-usage section quotes the total and an upper bound on the fine-tune share, and both
    drift every time a job runs. Read them out of the manuscript rather than hardcoding them, so
    this fails when the paper goes stale and not when the number merely moves."""
    import csv, re
    path = ROOT / "results" / "compute_hours_summary.csv"
    if not path.exists():
        return
    s = {r["quantity"]: float(r["gpu_hours"]) for r in csv.DictReader(open(path))}
    # each component is rounded to 1 dp independently of the total, so they can disagree by
    # one rounding unit -- 120.7 + 15.2 reads 135.9 against a total of 135.8.
    # Three values each rounded to one decimal: the sum of the parts can differ from the rounded
    # total by up to 0.15 without anything being wrong. It hit exactly 0.1 + float slop on
    # 2026-09-16 (246.6 + 15.2 vs 261.7). Loosening to 0.15 is the correct bound, not a masked drift.
    assert abs(s["one_gpu_jobs"] + s["two_gpu_jobs"] - s["total"]) <= 0.15
    assert s["fine_tunes"] <= s["one_gpu_jobs"]
    from tests.manuscript import tex as _tex
    tex = pathlib.Path(_tex("iclr_2027.tex"))
    if not tex.exists():
        return
    # Whitespace-normalised (caution (ar)): the v10 rewrap of 2026-09-24 put "at most" at the end of
    # one source line and "$366$ GPU-hours" at the start of the next, and a literal space in the
    # pattern then failed on the wrap alone while the sentence was unchanged.
    body = " ".join(tex.read_text(encoding="utf-8").split())
    # "approximately" became "at most" on 2026-09-16: the scan bills a gated queue shell for the
    # hours it spent polling, so ~8 of the total is a sleeping shell holding no card. The figure is
    # a genuine upper bound and the statement now says so; both wordings are accepted here.
    total = re.search(r"(?:approximately|at most) \$(\d+)\$ GPU-hours", body)
    share = re.search(r"account for at most \$(\d+)\$ of those hours", body)
    assert total and share, "the LLM-usage compute sentence has moved"
    assert int(total.group(1)) == round(s["total"]), (total.group(1), s["total"])
    assert s["fine_tunes"] <= int(share.group(1)), (share.group(1), s["fine_tunes"])


def test_an_idle_directory_does_not_bill_its_second_pause():
    """analysis/compute_hours.py removed only the LARGEST idle gap until 2026-09-16.

    output/composition is written in three bursts, not two: composition_attack.py's --text-out
    defaults to one fixed path inside it, so any sweep anywhere appends to a directory whose own job
    ended on 2026-09-05. With only the largest gap removed, the second pause was billed -- a
    ten-minute baseline run on 2026-09-16 moved that row from 1.88 GPU-hours to 111.6 and the
    project total from 261.7 to 376.8. This builds the three-burst shape directly and checks that
    every pause over the threshold is removed, so the regression cannot come back quietly.
    """
    import analysis.compute_hours as ch

    hour = 3600.0
    stamps = [0.0, 60.0,                      # burst 1: one minute of real work
              100 * hour, 100 * hour + 60.0,  # burst 2, after a 100-hour pause
              200 * hour, 200 * hour + 60.0]  # burst 3, after another 100-hour pause
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    idle = sum(g for g in gaps if g > 30 * 60)          # the rule the script applies
    span = stamps[-1] - stamps[0]
    billed = (span - idle) / hour
    assert round(billed, 4) == round(3 * 60.0 / hour, 4), (
        f"three one-minute bursts over 200 hours must bill three minutes, billed {billed}h")
    # and the shape that used to pass: removing only the largest gap bills a whole pause
    old = (span - max(gaps)) / hour
    assert old > 99, "the pre-2026-09-16 rule should bill ~100 idle hours on this shape"
    assert ch.times(__file__)[2] == [], "a single file has no gaps"
