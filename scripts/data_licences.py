"""Write data/LICENSE-DATA.md from the licences stated by the data sources.

Usage::

    python -m scripts.data_licences

Reads the licence of every Zenodo record listed in ``data/raw/MANIFEST.csv`` from the
Zenodo REST API and the licence of the GitHub mirror from the GitHub API (``none stated``
when the repository declares none). Nothing is assumed.
"""

from __future__ import annotations

import datetime as dt
import sys

import pandas as pd
import requests

from sdp.config import REPO_ROOT, load_config

HEADERS = {"User-Agent": "sdp-replication-package (licence check)", "Accept": "application/json"}


def zenodo_licence(api: str, record: int) -> dict:
    r = requests.get(f"{api}/records/{record}", headers=HEADERS, timeout=60)
    r.raise_for_status()
    meta = r.json().get("metadata", {})
    lic = meta.get("license") or {}
    lic_id = lic.get("id") if isinstance(lic, dict) else lic
    return {"title": meta.get("title", ""), "licence": lic_id or "none stated", "access": meta.get("access_right", "")}


def github_licence(api: str, repo: str) -> str:
    r = requests.get(f"{api}/repos/{repo}", headers=HEADERS, timeout=60)
    r.raise_for_status()
    lic = r.json().get("license")
    return (lic or {}).get("spdx_id") or (lic or {}).get("name") or "none stated"


def main() -> int:
    cfg = load_config()
    m = pd.read_csv(REPO_ROOT / "data" / "raw" / "MANIFEST.csv")
    rows = []
    for rec, g in m[m["source"] == "zenodo"].groupby("zenodo_record_id"):
        info = zenodo_licence(cfg.zenodo.api_url, int(rec))
        rows.append({"source": f"Zenodo record {int(rec)} (doi:{g['doi'].iloc[0]})", "project": ", ".join(sorted(g["project"].unique())),
                     "releases": ", ".join(g["version"].astype(str)), "licence (as stated)": info["licence"],
                     "access": info["access"], "title": info["title"]})  # fmt: skip
    mirror = m[m["source"] == "github-mirror"]
    if len(mirror):
        repo = cfg.github_mirror.repository
        rows.append({"source": f"GitHub mirror {repo} @ {mirror['commit_sha'].iloc[0][:12]}",
                     "project": ", ".join(sorted(mirror["project"].unique())), "releases": ", ".join(mirror["version"].astype(str)),
                     "licence (as stated)": github_licence("https://api.github.com", repo), "access": "public", "title": ""})  # fmt: skip
    t = pd.DataFrame(rows)
    lines = ["# Data licences", "",
             f"Read from the Zenodo REST API and the GitHub API on {dt.date.today().isoformat()} "
             "(`python -m scripts.data_licences`). The licence column is copied as stated by each source; "
             "nothing is assumed. The data files in `data/raw/` are redistributed unchanged (checksums in "
             "`data/raw/MANIFEST.csv`); the code licence (`LICENSE`, MIT) does not apply to them.", "",
             "| source | project | releases | licence (as stated) | access | record title |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['source']} | {r['project']} | {r['releases']} | {r['licence (as stated)']} | {r['access']} | {r['title']} |"
              for r in rows]
    (REPO_ROOT / "data" / "LICENSE-DATA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
