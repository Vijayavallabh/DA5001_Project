"""feat-003: stage-1 and top-up trajectory seeds never collide (the released logs had one duplicate per N=20 row)."""
from dap.stats import build_trajectory_seeds


def test_no_collision_across_4_plus_16_split():
    for i in range(1000):
        pid = f"bookmia.{i // 20:02d}.{i % 20:02d}"
        stage1 = build_trajectory_seeds(pid, (42, 43, 44), 4)
        stage2 = build_trajectory_seeds(pid, (42, 43, 44), 16, start=4)
        assert len(set(stage1 + stage2)) == 20, pid
        assert all(0 <= s < 2**32 for s in stage1 + stage2)


def test_prefix_consistency_and_determinism():
    full = build_trajectory_seeds("p", (42, 43, 44), 20)
    assert full[:4] == build_trajectory_seeds("p", (42, 43, 44), 4)
    assert full[4:] == build_trajectory_seeds("p", (42, 43, 44), 16, start=4)
    assert full == build_trajectory_seeds("p", (42, 43, 44), 20)
    assert full != build_trajectory_seeds("p", (1, 2, 3), 20)  # base seeds select a replicate
    assert full == build_trajectory_seeds("other prompt", (42, 43, 44), 20)  # shared across prompts so E1/E2 can batch


def test_a_tail_run_builds_exactly_the_tail_of_a_full_run_and_the_same_batches():
    """feat-181: --trajectory-start S generates indices [S, S+n) with the seeds, ids and per-seed
    batches a full run would give them, which is what lets an n=512 pool extend a committed n=256
    one. The generator seeds every batch with its own trajectory seed (dap/e1.py:_run_seed_group)."""
    from types import SimpleNamespace
    from dap.e1 import AuditConfig, H1AuditRunner as E1Runner

    prompts = [SimpleNamespace(prompt_id=f"p{i}", prompt_text="w " * (3 + 40 * i)) for i in range(7)]

    def build(start, n):
        stub = SimpleNamespace(config=AuditConfig(trajectories_per_prompt=n, trajectory_start=start),
                               _prompt_token_length=lambda t: len(t.split()))
        return E1Runner._build_jobs(stub, prompts)

    key = lambda js: [(j["prompt"].prompt_id, j["seed"], j["trajectory_id"]) for j in js]  # noqa: E731
    full, tail = build(0, 12), build(8, 4)
    assert key(tail) == [x for x in key(full) if x[2] >= 8]
    stub = SimpleNamespace()
    for seed, jobs in E1Runner._group_jobs_by_seed(stub, tail).items():
        mine = E1Runner._make_length_buckets(stub, jobs)
        theirs = E1Runner._make_length_buckets(stub, E1Runner._group_jobs_by_seed(stub, full)[seed])
        assert [[key([j])[0] for j in b] for b in mine] == [[key([j])[0] for j in b] for b in theirs]
