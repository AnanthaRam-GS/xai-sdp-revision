"""Offline tests for the downloader helpers (no network access)."""

from __future__ import annotations

import hashlib

import pandas as pd

from sdp.config import version_sort_key
from sdp.data.download import compare_with_expected, csv_pattern, md5_of, split_by_version

HEADER = b"name,version,name,wmc,bug\n"


def test_md5_of_bytes_and_file(tmp_path):
    data = b"abc\n"
    p = tmp_path / "f.csv"
    p.write_bytes(data)
    assert md5_of(data) == md5_of(p) == hashlib.md5(data).hexdigest()


def test_single_release_file_is_kept_byte_for_byte():
    content = HEADER + b"p,2.0RC1,A,1,0\n"
    out = split_by_version(content, "2.0")
    assert list(out) == ["2.0"]
    assert out["2.0"] == (content, "2.0RC1")


def test_multi_release_file_is_split_by_version():
    content = HEADER + b"p,1.1,A,1,0\np,1.0,A,1,0\np,1.1,B,2,1\n"
    out = split_by_version(content, "all")
    assert list(out) == ["1.0", "1.1"]
    assert out["1.1"][0].count(b"\n") == 3  # header + 2 rows


def test_csv_pattern_and_version_order():
    m = csv_pattern("xerces").match("xerces-init.csv")
    assert m and m.group("version") == "init"
    assert csv_pattern("lucene").match("lucene V2.4.csv") is None
    assert sorted(["1.4", "init", "1.2", "1.10"], key=version_sort_key) == ["init", "1.2", "1.4", "1.10"]


def test_compare_with_expected():
    manifest = pd.DataFrame({"project": ["a", "a", "b"], "version": ["1.0", "1.2", "2.0"]})
    missing, unexpected = compare_with_expected(manifest, {"a": ["1.0", "1.1"], "b": ["2.0"]})
    assert missing == {"a": ["1.1"]}
    assert unexpected == {"a": ["1.2"]}
