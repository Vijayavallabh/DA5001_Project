"""feat-172's gate and bands, mutation-tested on synthetic files while the reward passes still ran.

The gate must refuse a single perturbed reward and a single missing rank-0 row, and each band must
read its registered verdict -- including the two outcomes the registration leaves unnamed, which
must be printed as unregistered rather than rounded into a named band.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import score_offsupport as S  # noqa: E402

PIDS = [f"p{i:03d}" for i in range(40)]


def _w(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _caches(res, top_new=256, perturb=None, drop=None):
    for arm in S.ARMS.values():
        old = [dict(prompt_id=p, rank=j, prompt_class="factual", n_words=10,
                    reward=round(0.001 * (hash((p, j)) % 997), 5)) for p in PIDS for j in range(64)]
        new = [dict(r) for r in old] + [dict(prompt_id=p, rank=j, prompt_class="factual",
                                             n_words=10, reward=0.5)
                                        for p in PIDS for j in range(64, top_new)]
        if perturb is not None:
            new[perturb]["reward"] = float(new[perturb]["reward"]) + 1e-5
        if drop is not None:
            new.pop(drop)
        _w(os.path.join(res, arm["old"]), old)
        _w(os.path.join(res, arm["new"]), new)


def _shape(res, tag, d128, d256):
    _w(os.path.join(res, f"selection_scaling_per_prompt{tag}.csv"),
       [dict(judge="microsoft/Phi-3.5-mini-instruct", prompt_id=p, u_n64=0.5,
             u_n128=0.5 + d128 + (0.01 if i % 2 else -0.01),
             u_n256=0.5 + d128 + d256 + (0.02 if i % 2 else -0.02))
        for i, p in enumerate(PIDS)])


def _d3(res, sfx, v):
    _w(os.path.join(res, f"order_averaged_h2h{sfx}.csv"),
       [dict(quantity="D3 selection minus metered, order-averaged", arm="x", value=v,
             lo95=v - 0.02, hi95=v + 0.02, n=805, single_order=v, reading="x")])


def _run(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    sys.argv = ["score_offsupport.py", "--results", "results", "--out", "results"]
    S.main()
    return capsys.readouterr().out


def _setup(tmp_path, **kw):
    res = tmp_path / "results"
    res.mkdir()
    _caches(str(res), **kw)
    return str(res)


def test_a_clean_ladder_passes_the_gate_and_reads_every_band(tmp_path, monkeypatch, capsys):
    res = _setup(tmp_path)
    _shape(res, "_offsup", 0.03, 0.10)
    _shape(res, "_onsup", 0.0, 0.0)
    _d3(res, "__mixpowk_judgeB", -0.0339)
    _d3(res, "__offsup_n128", -0.0250)
    _d3(res, "__offsup_n256", -0.0100)
    out = _run(tmp_path, monkeypatch, capsys)
    assert out.count(": PASS") == 2, out
    assert "B2 Arm A: **STILL CLIMBING**" in out and "B5 Arm B: **SATURATED**" in out
    assert "B1 Arm A: **CLOSES**" in out, out
    assert "B6: off-support STILL CLIMBING, on-support SATURATED" in out


def test_one_reward_off_by_1e5_is_inapplicable_and_reads_nothing(tmp_path, monkeypatch, capsys):
    res = _setup(tmp_path, perturb=7)
    _shape(res, "_offsup", 0.03, 0.10)
    _d3(res, "__mixpowk_judgeB", -0.0339)
    _d3(res, "__offsup_n128", 0.01)
    _d3(res, "__offsup_n256", 0.02)
    out = _run(tmp_path, monkeypatch, capsys)
    assert "FAIL -> INAPPLICABLE" in out and "**" not in out, out


def test_a_missing_rank_zero_row_fails_the_gate(tmp_path, monkeypatch, capsys):
    _setup(tmp_path, drop=0)
    assert "FAIL -> INAPPLICABLE" in _run(tmp_path, monkeypatch, capsys)


def test_b1_catches_ceiling_and_the_unnamed_gap(tmp_path, monkeypatch, capsys):
    for v128, v256, want in ((-0.02, 0.001, "CATCHES"), (-0.03, -0.030, "CEILING"),
                             (-0.03, -0.020, "BETWEEN (unregistered gap)")):
        sub = tmp_path / want.split()[0]
        sub.mkdir()
        res = _setup(sub)
        _d3(res, "__mixpowk_judgeB", -0.0339)
        _d3(res, "__offsup_n128", v128)
        _d3(res, "__offsup_n256", v256)
        assert f"B1 Arm A: **{want}**" in _run(sub, monkeypatch, capsys), want


def test_a_falling_ladder_is_labelled_unregistered_rather_than_saturated(tmp_path, monkeypatch,
                                                                          capsys):
    res = _setup(tmp_path)
    _shape(res, "_offsup", 0.0, -0.10)
    assert "B2 Arm A: **TURNS OVER (unregistered)**" in _run(tmp_path, monkeypatch, capsys)


def _oa(res, sfx, n, u1=0.25):
    _w(os.path.join(res, f"order_averaged_h2h_per_prompt__{sfx}.csv"),
       [{"prompt_id": p, f"u_sel_n{n}": 0.5, "u_sel_n1": u1, "gain_sel": 0.25 + 0.001 * n}
        for p in PIDS])


def test_post_hoc_rows_are_labelled_and_refuse_passes_that_do_not_pair(tmp_path, monkeypatch, capsys):
    res = _setup(tmp_path)
    _shape(res, "_offsup", 0.03, 0.10)
    _shape(res, "_onsup", 0.0, 0.0)
    for sfx, n in (("mixpowk_judgeB", 64), ("offsup_n128", 128), ("offsup_n256", 256)):
        _oa(res, sfx, n)
    _run(tmp_path, monkeypatch, capsys)
    got = list(csv.DictReader(open(os.path.join(res, "offsupport_ladder.csv"), encoding="utf-8")))
    post = [r for r in got if r["reading"] == "POST HOC"]
    assert {r["quantity"] for r in post} == {"g(256)-g(64)", "g(128)-g(64) order-averaged",
                                             "g(256)-g(128) order-averaged",
                                             "g(256)-g(64) order-averaged"}, post
    assert all("POST HOC" != r["reading"] for r in got if r["quantity"] == "g(256)-g(128)")
    _oa(res, "offsup_n256", 256, u1=0.5)            # a pass whose n=1 arm is not the others'
    try:
        _run(tmp_path, monkeypatch, capsys)
    except AssertionError as e:
        assert "do not pair" in str(e)
    else:
        raise AssertionError("order-averaged passes with different n=1 arms were paired")
