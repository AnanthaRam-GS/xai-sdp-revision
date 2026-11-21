"""Tests for the reproduction entry points (quick mode configuration, plan)."""

from __future__ import annotations

from pathlib import Path

import yaml

from sdp import repro


def test_quick_configuration_never_writes_to_the_real_results(tmp_path):
    d = repro.write_quick_configs(tmp_path / "configs")
    paths = yaml.safe_load((d / "default.yaml").read_text())["paths"]
    for key in ("results_dir", "reports_dir", "artifacts_dir", "processed_dir"):
        assert not any(paths[key] == p or paths[key].startswith(p + "/") for p in repro.PROTECTED), (key, paths[key])
    assert repro.quick_paths_are_separate(d)
    real = yaml.safe_load((repro.REPO_ROOT / "configs" / "default.yaml").read_text())
    real_dir = tmp_path / "real"
    (real_dir).mkdir()
    (real_dir / "default.yaml").write_text(yaml.safe_dump(real))
    assert not repro.quick_paths_are_separate(real_dir)


def test_quick_configuration_is_reduced(tmp_path):
    d = repro.write_quick_configs(tmp_path / "configs")
    default = yaml.safe_load((d / "default.yaml").read_text())
    assert set(default["projects"]) == {"ant", "synapse"}
    assert default["tuning"]["n_trials"] == {"time_aware": 5, "within_release": 5}
    main = yaml.safe_load((d / "experiments" / "main_time_aware.yaml").read_text())
    assert main["seeds"] == [0, 1] and main["n_trials"] == 5
    ex = yaml.safe_load((d / "explain.yaml").read_text())
    assert ex["expensive_instances"] == 20 and ex["seeds"] == [0, 1]
    # every real configuration file has a quick counterpart
    real = {p.relative_to(repro.REPO_ROOT / "configs").as_posix() for p in (repro.REPO_ROOT / "configs").rglob("*.yaml")}
    quick = {p.relative_to(d).as_posix() for p in d.rglob("*.yaml")}
    assert real <= quick


def test_plan_covers_steps_one_to_eight(tmp_path):
    steps = repro.plan("quick", tmp_path, offline=True)
    assert [s.number for s in steps] == list(range(1, 9))
    flat = [" ".join(c) for s in steps for c in s.commands]
    assert any("scripts.prepare_data --offline" in c for c in flat)
    assert any("make_paper_assets --collect --out results_quick/paper" in c for c in flat)


def test_expensive_instance_sample_is_seeded_and_shared():
    from sdp.explain.runner import ExplainSettings, expensive_sample

    s = ExplainSettings("e", "class_weight", {}, {}, [0], [], [0], 100, {}, expensive_instances=20)
    a, b = expensive_sample("p:1>2", 300, s), expensive_sample("p:1>2", 300, s)
    assert len(a) == 20 and (a == b).all() and (a == sorted(a)).all()
    assert expensive_sample("p:1>2", 15, s) is None
    assert expensive_sample("p:1>2", 300, ExplainSettings("e", "c", {}, {}, [0], [], [0], 100, {})) is None
