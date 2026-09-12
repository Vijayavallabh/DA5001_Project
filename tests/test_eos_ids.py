"""A model may declare no EOS token, and the runner must still finish.

PleIAs/Pleias-1.2b-Preview declares no `eos_token_id`. `[*None]` raised TypeError inside
_records_from_batch, so the run died after its first batch with 0 trajectories written and the
scoring step then failed on an empty directory -- two failures away from the actual cause."""
from dap.e1 import _as_id_list


def test_an_absent_eos_contributes_no_stop_id():
    assert _as_id_list(None) == []


def test_an_int_and_a_list_both_normalise():
    assert _as_id_list(2) == [2]
    assert _as_id_list([2, 3]) == [2, 3]
    assert _as_id_list((2, 3)) == [2, 3]


def test_the_runner_filters_none_before_building_the_stop_list():
    import inspect
    from dap import e1
    src = inspect.getsource(e1.H1AuditRunner._records_from_batch)
    assert "_as_id_list(self.eos_ids)" in src, "the normaliser is no longer used"
    assert "if t is not None" in src, "a None pad id would be passed through as a stop id"
