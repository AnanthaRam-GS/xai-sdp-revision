"""Tests for the study inclusion rule and release pairing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sdp.config import Config, Paths, StudySettings, ZenodoSettings
from sdp.data.study import apply_inclusion_rule


def make_cfg(projects: dict[str, list[str]]) -> Config:
    p = Path("/nonexistent")
    return Config(
        seed=42,
        paths=Paths(p, p, p, p),
        zenodo=ZenodoSettings(),
        projects=projects,
        study=StudySettings(0.05, 0.75, 3),
    )


def stats(rows: list[tuple[str, str, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["project", "version", "final_rows", "defective_count"])


def test_ratio_rule_then_min_releases():
    cfg = make_cfg({"a": ["init", "1.2", "1.10", "2.0"], "b": ["1", "2", "3"]})
    s = stats([
        ("a", "init", 100, 20), ("a", "1.2", 100, 4),   # 4% -> excluded
        ("a", "1.10", 100, 30), ("a", "2.0", 100, 50),
        ("b", "1", 100, 20), ("b", "2", 100, 80), ("b", "3", 100, 30),  # 80% -> excluded, then 2 left
    ])  # fmt: skip
    study = apply_inclusion_rule(s, cfg)
    assert study.releases == {"a": ["init", "1.10", "2.0"]}
    reasons = study.table.set_index(["project", "version"])["reason"]
    assert "< 5%" in reasons[("a", "1.2")]
    assert "> 75%" in reasons[("b", "2")]
    assert "only 2 release(s)" in reasons[("b", "1")]


def test_boundaries_are_inclusive():
    cfg = make_cfg({"a": ["1", "2", "3"]})
    study = apply_inclusion_rule(stats([("a", "1", 100, 5), ("a", "2", 100, 75), ("a", "3", 100, 40)]), cfg)
    assert study.releases == {"a": ["1", "2", "3"]}


def test_pairs_follow_configured_order_not_string_order():
    cfg = make_cfg({"x": ["init", "1.2", "1.10"]})
    study = apply_inclusion_rule(stats([("x", v, 100, 30) for v in ["1.10", "init", "1.2"]]), cfg)
    cons = [(p.train, p.test) for p in study.pairs("consecutive")]
    cum = [(p.train, p.test) for p in study.pairs("cumulative")]
    assert cons == [(("init",), "1.2"), (("1.2",), "1.10")]
    assert cum == [(("init",), "1.2"), (("init", "1.2"), "1.10")]
