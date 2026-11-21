"""Smoke tests of the paper asset generators on the committed inputs (reports/ and paper/data/)."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORTS, DATA = ROOT / "reports", ROOT / "paper" / "data"
needs_inputs = pytest.mark.skipif(not (REPORTS / "step4_global_importance.csv").exists() or not DATA.exists(),
                                  reason="committed reports and paper inputs are required")  # fmt: skip


@needs_inputs
def test_every_figure_is_written_as_pdf_and_png(tmp_path):
    from sdp.paper import figures

    figures.make_all(REPORTS, DATA, tmp_path)
    for name in figures.FIGURES:
        assert (tmp_path / f"{name}.pdf").stat().st_size > 0 and (tmp_path / f"{name}.png").stat().st_size > 0
    captions = (tmp_path / "CAPTIONS.md").read_text()
    assert all(f"`{name}.pdf`" in captions for name in figures.FIGURES)


def test_metric_labels_are_upper_case_feature_names():
    from sdp.eval.preprocess import FEATURES
    from sdp.paper.style import METRIC_LABEL

    assert METRIC_LABEL["loc"] == "LOC" and METRIC_LABEL["avg_cc"] == "AVG_CC" and set(METRIC_LABEL) == set(FEATURES)


@needs_inputs
def test_tables_are_well_formed_booktabs(tmp_path):
    import re

    from sdp.paper import tables

    written = tables.make_all(REPORTS, DATA, tmp_path)
    assert {p.stem for p in written} == set(tables.TABLES)
    for path in written:
        tex = path.read_text()
        assert tex.startswith("% Suggested caption:") and "% Source:" in tex
        assert all(rule in tex for rule in (r"\toprule", r"\midrule", r"\bottomrule"))
        assert tex.count("{") == tex.count("}") - tex.count(r"\}") + tex.count(r"\{")
        spec = re.search(r"\\begin\{tabular\}\{(.*)\}\s*$", tex, flags=re.MULTILINE).group(1)
        ncols = len(re.findall(r"p\{[^}]*\}", spec)) + len(re.findall(r"[lrc]", re.sub(r"p\{[^}]*\}", "", spec)))
        assert "|" not in spec  # no vertical rules
        body = tex.split(r"\toprule")[1].split(r"\bottomrule")[0]
        for line in body.splitlines():
            if line.endswith(r"\\"):
                assert len(re.findall(r"(?<!\\)&", line)) == ncols - 1, (path.name, line[:80])
        assert re.search(r"(?<![\\$])[<>](?!\$)", body.replace("$<$", "").replace("$>$", "")) is None, path.name


@needs_inputs
def test_check_numbers_script_passes(tmp_path):
    import subprocess
    import sys

    res = subprocess.run([sys.executable, "-m", "scripts.check_numbers"], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "match their sources" in res.stdout


def test_check_numbers_detects_a_changed_macro(tmp_path):
    from sdp.paper import numbers

    src = ROOT / "paper" / "numbers.tex"
    if not src.exists():
        pytest.skip("numbers.tex not generated")
    bad = tmp_path / "numbers.tex"
    bad.write_text(src.read_text().replace(r"\newcommand{\NumPairs}{24}", r"\newcommand{\NumPairs}{25}"))
    problems = numbers.check_numbers(ROOT, bad)
    assert any(p.startswith("NumPairs") for p in problems)
