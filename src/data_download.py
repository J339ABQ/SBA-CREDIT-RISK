"""Download the SBA 7(a) and 504 FOIA loan-level CSVs from data.sba.gov (CKAN API).

    python -m src.data_download            # downloads into data/raw/real/

Dataset page: https://data.sba.gov/dataset/7-a-504-foia   (landing: https://www.sba.gov/about-sba/open-government/foia)

If automated access is blocked (proxy/firewall/403 - as happened in the sandbox this repo was built in)
the script prints the manual steps and exits with status 2. Manual fallback:
  1. Open https://data.sba.gov/dataset/7-a-504-foia in a browser.
  2. Download every CSV resource whose name contains '7a' or '504' (FY1991-FY2009, FY2010-FY2019, FY2020-present ...).
  3. Put them, unmodified, in data/raw/real/  (keep '504' in the file name for the 504 files).
  4. Run `make train` (or `python -m src.train`) - real files in data/raw/real/ take precedence over synthetic data.
NOTE: file names/columns change between SBA refreshes; src/data_prep.py harmonises the known variants
and fills missing optional columns with NaN.
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

from . import config

PACKAGE_API = "https://data.sba.gov/api/3/action/package_show?id=7-a-504-foia"
MANUAL = "\n" + __doc__.split("Manual fallback:")[1]


def list_resources(session: requests.Session) -> list[dict]:
    r = session.get(PACKAGE_API, timeout=60)
    r.raise_for_status()
    res = r.json()["result"]["resources"]
    keep = []
    for x in res:
        name = (x.get("name") or x.get("url", "")).lower()
        url = x.get("url", "")
        if url.lower().endswith(".csv") and ("7a" in name or "7(a)" in name or "504" in name or "foia" in name):
            if "data dictionary" in name or "dictionary" in name:
                continue
            keep.append(x)
    return keep


def download(dest: Path = config.DATA_REAL) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = "sba-credit-risk-portfolio/1.0"
    out = []
    for x in list_resources(s):
        url = x["url"]
        target = dest / Path(url).name
        if target.exists():
            print(f"skip (exists): {target.name}")
            out.append(target)
            continue
        print(f"downloading {url}")
        with s.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(target, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        out.append(target)
    if not out:
        raise RuntimeError("SBA package listed no matching CSV resources")
    return out


if __name__ == "__main__":
    try:
        files = download()
        print(f"done: {len(files)} file(s) in {config.DATA_REAL}")
    except Exception as e:  # noqa: BLE001 - user-facing message for any network / parsing failure
        print(f"\nAutomated download failed: {type(e).__name__}: {e}\n\nManual steps:{MANUAL}", file=sys.stderr)
        sys.exit(2)
