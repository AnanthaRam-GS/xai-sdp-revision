"""Offline tests for GitHub-mirror support: blob hashing, layout handling, comparison."""

from __future__ import annotations

import subprocess

import pytest

from sdp.data.clean import ID_COL, METRICS, clean_release
from sdp.data.download import git_blob_sha, read_raw_csv, split_by_version
from sdp.data.legacy import compare_release

METRIC_HDR = ",".join(METRICS)
VALS_A = ",".join(["1"] * len(METRICS))
VALS_B = ",".join(["2"] * len(METRICS))

# Same two classes in both layouts; rows in a different order; CRLF vs LF.
ZENODO = (
    f"name,version,name,{METRIC_HDR},bug\r\n"
    f"ant,1.7,pkg.B,{VALS_B},0\r\n"
    f"ant,1.7,pkg.A,{VALS_A},2\r\n"
).encode()
MIRROR = (f"name,{METRIC_HDR},bug\npkg.A,{VALS_A},2\npkg.B,{VALS_B},0\n").encode()


def test_git_blob_sha_matches_git(tmp_path):
    p = tmp_path / "f.csv"
    p.write_bytes(MIRROR)
    try:
        expected = subprocess.run(["git", "hash-object", str(p)], capture_output=True, text=True, check=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("git not available")
    assert git_blob_sha(MIRROR) == expected


def test_mirror_layout_is_stored_unchanged():
    assert split_by_version(MIRROR, "1.3") == {"1.3": (MIRROR, "1.3")}


def test_layouts_compare_identical_regardless_of_order_and_line_endings():
    res = compare_release(MIRROR, ZENODO)
    assert res["identical"] and res["differing_rows"] == 0
    assert not res["byte_identical"]


def test_comparison_detects_a_changed_value():
    changed = MIRROR.replace(b"pkg.A,1,", b"pkg.A,9,", 1)
    res = compare_release(changed, ZENODO)
    assert not res["identical"] and res["differing_rows"] == 1


def test_cleaning_handles_mirror_layout():
    clean, _ = clean_release(read_raw_csv(MIRROR))
    assert sorted(clean[ID_COL]) == ["pkg.A", "pkg.B"]
    assert clean["defective"].tolist() == [1, 0]
