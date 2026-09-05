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
