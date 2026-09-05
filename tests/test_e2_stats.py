"""feat-007: per-trajectory budget statistics replace the empirical-Bernstein proxy in E1/E2."""
from dap.e2.evaluator import AnchoredEvaluator, safe_rho
from dap.e2.types import E2Config, EvalResult
from dap.stats import budget_check


def test_budget_check():
    ms, rho, ok = budget_check([10.0, 5.0, 0.0], [20.0, 5.0, -3.0])
    assert ms == 10.0 and abs(rho - 1.0) < 1e-9 and ok  # 5/5 = 1 is allowed; B = -3 with Z = 0 is allowed
    assert budget_check([1.0], [-3.0]) == (1.0, None, False)  # spend with no accrued budget is a violation
    assert budget_check([20.01], [20.0])[2] is False and budget_check([20.0005], [20.0])[2] is True
    assert safe_rho([0.0], [-1.0]) == (None, "no_trajectory_with_positive_budget")


def test_finalize_eval_result_without_models():
    ev = AnchoredEvaluator.__new__(AnchoredEvaluator)
    ev.cfg = E2Config(k=1.0, max_new_tokens=200)
    ev.R_token = ev.cfg.K
    acc = ev._init_accumulator(dict(candidate_id="c", lineage_id="l", generation=0, source="s", domain="d", split="t", prompt_text="p"))
    acc["spends"], acc["final_budgets"], acc["delta_inits"] = [150.0, 190.0], [194.0, 194.0], [6.0, 6.0]
    acc["utilisations"], acc["activity"] = [150 / 194, 190 / 194], [[5, 10, 185], [5, 20, 175]]
    r = ev._finalize_eval_result(acc, n=2, delta=0.05)
    assert isinstance(r, EvalResult) and r.certified and r.max_spend == 190.0 and abs(r.rho - 190 / 194) < 1e-9
    assert r.U_EBB == r.max_spend  # compatibility alias for the surrogate
