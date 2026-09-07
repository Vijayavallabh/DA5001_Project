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
