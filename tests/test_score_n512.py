"""feat-181's scorer on synthetic files, written while the extension draws were still generating.
G1 must refuse one perturbed rank-0..255 reward; G2' a prompt missing one rank; each band must read
its registered word, and a directional reading under 1.7 half-widths must carry MARGINAL."""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import score_n512 as S  # noqa: E402

PIDS = [f"p{i:03d}" for i in range(40)]


def _w(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _setup(tmp_path, perturb=None, drop=None):
    res = tmp_path / "results"
    res.mkdir()
    for A in S.ARMS.values():
        old = [dict(prompt_id=p, rank=j, reward=round(0.001 * ((hash(p) + j) % 997), 5))
               for p in PIDS for j in range(256)]
        new = [dict(r) for r in old] + [dict(prompt_id=p, rank=j, reward=0.5)
                                        for p in PIDS for j in range(256, 512)]
        if perturb is not None:
            new[perturb]["reward"] = float(new[perturb]["reward"]) + 1e-5
        if drop is not None:
            new.pop(drop)
        _w(res / A["old"], old)
        _w(res / f"{A['tag']}_rewards512.csv", new)
    return res


def _ladder(res, tag, d256, d64, spread=0.01):
    _w(res / f"selection_scaling_per_prompt_{tag}.csv",
       [dict(judge="microsoft/Phi-3.5-mini-instruct", prompt_id=p, u_n64=0.5, u_n256=0.5 + d64 - d256,
             u_n512=0.5 + d64 + (spread if i % 2 else -spread)) for i, p in enumerate(PIDS)])


def _run(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    sys.argv = ["score_n512.py", "--results", "results", "--out", "results"]
    S.main()
    return capsys.readouterr().out


def test_clean_gates_and_each_band_reads_its_word(tmp_path, monkeypatch, capsys):
    res = _setup(tmp_path)
    _ladder(res, "n512a", d256=0.0, d64=0.10)          # flat last doubling, clear climb from 64
    _ladder(res, "n512b", d256=0.10, d64=0.10, spread=0.30)
    _w(res / "order_averaged_h2h__n512a_n512.csv",
       [dict(quantity="D3 difference of gains, paired", arm="x", value=-0.02, lo95=-0.04,
             hi95=-0.001, n=805, single_order=0, reading="x")])
    _w(res / "offsupport_ladder.csv", [dict(arm="A", quantity="certificate log 256", value=5.5452,
                                            lo95="", hi95="", n="", reading=145.1048)])
    out = _run(tmp_path, monkeypatch, capsys)
    assert out.count("G1 Arm") == 2 and "FAIL" not in out, out
    assert "C2 Arm A" in out and "**SATURATED**" in out and "**CLIMBS PAST 64**" in out, out
    assert "**STILL CLIMBING (MARGINAL)**" in out, out    # B: +0.10 against a wide spread
    assert "C1 Arm A: D3 at n=512 = -0.0200" in out and "**STILL BEHIND**" in out
    assert "a ratio of 23.3" in out


def test_one_reward_off_by_1e5_is_inapplicable_and_reads_nothing(tmp_path, monkeypatch, capsys):
    res = _setup(tmp_path, perturb=3)
    _ladder(res, "n512a", 0.0, 0.1)
    out = _run(tmp_path, monkeypatch, capsys)
    assert "FAIL -> INAPPLICABLE" in out and "**" not in out, out


def test_a_prompt_missing_one_rank_fails_g2(tmp_path, monkeypatch, capsys):
    res = _setup(tmp_path, drop=40 * 256 + 5)            # a rank >= 256: G1 still passes
    _ladder(res, "n512a", 0.0, 0.1)
    out = _run(tmp_path, monkeypatch, capsys)
    assert "G2' FAIL on 1 prompts" in out and "**" not in out, out
