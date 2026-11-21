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
