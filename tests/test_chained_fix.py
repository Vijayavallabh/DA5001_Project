"""feat-215: the merge that puts re-run chained rows back must leave every other line byte-identical."""
import json

import pytest

from analysis.chained_fix import g1, merge_csv, merge_queries, splice

HEAD = "k,mode,L,prompt_id,nv_recall\r\n"


def write(p, text):
    p.write_text(text, newline="")
    return str(p)


def test_chained_rows_are_replaced_in_place_and_nothing_else_moves(tmp_path):
    old = write(tmp_path / "old.csv", HEAD + "3.0,single,0,a,0.5\r\n3.0,chained,20,a,0.1\r\n3.0,chained,20,b,0.2\r\n"
                                             "5.0,oracle,20,a,0.7\r\n")
    new = write(tmp_path / "new.csv", HEAD + "3.0,chained,20,b,0.9\r\n3.0,chained,20,a,0.8\r\n")
    out = str(tmp_path / "out.csv")
    merge_csv(old, new, ("k", "mode", "L", "prompt_id"), out)
    assert open(out, newline="").read() == (HEAD + "3.0,single,0,a,0.5\r\n3.0,chained,20,a,0.8\r\n"
                                            "3.0,chained,20,b,0.9\r\n5.0,oracle,20,a,0.7\r\n")
    bad = write(tmp_path / "bad.csv", HEAD + "3.0,chained,20,a,0.8\r\n")  # a chained passage missing from the re-run
    with pytest.raises(AssertionError):
        merge_csv(old, bad, ("k", "mode", "L", "prompt_id"), out)


def test_g1_refuses_a_changed_non_chained_row(tmp_path):
    old = write(tmp_path / "old.csv", HEAD + "3.0,single,0,a,0.5\r\n3.0,chained,20,a,0.1\r\n")
    ok = write(tmp_path / "ok.csv", HEAD + "3.0,single,0,a,0.5\r\n3.0,chained,20,a,0.9\r\n")
    g1(old, ok)
    moved = write(tmp_path / "moved.csv", HEAD + "3.0,single,0,a,0.50\r\n3.0,chained,20,a,0.1\r\n")
    with pytest.raises(AssertionError):
        g1(old, moved)


def test_queries_keep_their_order_with_each_chained_block_where_the_old_one_stood(tmp_path):
    q = lambda k, mode, t: json.dumps(dict(k=k, mode=mode, text=t)) + "\n"  # noqa: E731
    old = write(tmp_path / "old.jsonl", q(3, "single", "s3") + q(3, "chained", "x") + q(3, "chained", "y")
                + q(5, "single", "s5") + q(5, "chained", "z"))
    new = write(tmp_path / "new.jsonl", q(3, "chained", "X") + q(5, "chained", "Z1") + q(5, "chained", "Z2"))
    out = tmp_path / "out.jsonl"
    merge_queries(old, new, str(out))
    assert [json.loads(line)["text"] for line in open(out)] == ["s3", "X", "s5", "Z1", "Z2"]


def test_a_spliced_row_must_have_been_a_copy_of_its_old_source(tmp_path):
    summ = "k,mode,L,n_passages,nv_recall_mean\r\n"
    old_src = write(tmp_path / "old_src.csv", summ + "10.0,oracle,50,100,0.4\r\n10.0,chained,50,100,0.066\r\n")
    new_src = write(tmp_path / "new_src.csv", summ + "10.0,oracle,50,100,0.4\r\n10.0,chained,50,100,0.25\r\n")
    target = write(tmp_path / "t.csv", "k,cap,mode,L,n_passages,nv_recall_mean,run\r\n"
                   "10.0,10,chained,50,100,0.066,r\r\n10.0,10,oracle,50,100,0.4,r\r\n")
    got = splice(target, lambda r: (old_src, new_src))
    assert got == ("k,cap,mode,L,n_passages,nv_recall_mean,run\r\n10.0,10,chained,50,100,0.25,r\r\n"
                   "10.0,10,oracle,50,100,0.4,r\r\n")
    wrong = write(tmp_path / "wrong.csv", summ + "10.0,oracle,50,100,0.4\r\n10.0,chained,50,100,0.07\r\n")
    with pytest.raises(AssertionError):
        splice(target, lambda r: (wrong, new_src))


def test_a_wider_rerun_is_projected_onto_the_committed_header_and_pathwise_cells_count_kl(tmp_path):
    from analysis.chained_fix import as_committed
    old, new, out = tmp_path / "old", tmp_path / "new", tmp_path / "out"
    for d in (old, new, out):
        d.mkdir()
    write(old / "composition.csv", "k,mode,L,prompt_id,invariant_violations,nv_recall\r\n")
    write(old / "composition_summary.csv", "k,mode,L,invariant_violations,nv_recall_mean\r\n")
    write(new / "composition.csv", "k,mode,L,constraint,prompt_id,R_total,invariant_violations,nv_recall\r\n"
                                   "3.0,chained,20,pathwise,a,1.5,0,0.25\r\n3.0,chained,20,pathwise,b,2.5,0,0.5\r\n")
    write(new / "composition_summary.csv", "k,mode,L,constraint,R_total_mean,invariant_violations,nv_recall_mean\r\n"
                                           "3.0,chained,20,pathwise,2.0,0,0.375\r\n")
    q = lambda pid, Z, B: json.dumps(dict(k=3.0, L=20, prompt_id=pid, Z=Z, B=B, R=0.0, mode="chained")) + "\n"  # noqa: E731
    (new / "queries.jsonl").write_text(q("a", 5.0, 4.0) + q("a", 1.0, 4.0) + q("b", 9.0, 4.0) + q("b", 9.0, 4.0))
    as_committed(str(new), str(old), str(out), pathwise=True)
    assert open(out / "composition.csv", newline="").read() == (
        "k,mode,L,prompt_id,invariant_violations,nv_recall\r\n3.0,chained,20,a,1,0.25\r\n3.0,chained,20,b,2,0.5\r\n")
    assert open(out / "composition_summary.csv", newline="").read() == (
        "k,mode,L,invariant_violations,nv_recall_mean\r\n3.0,chained,20,3,0.375\r\n")
