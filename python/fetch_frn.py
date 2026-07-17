#!/usr/bin/env python3
"""Fetch U.S. Treasury FRN market quotes (index rate + quoted spread).

Sibling of fetch_treasury.py (same design rules: stdlib only, no API key).
Source: Treasury Fiscal Data "FRN Daily Indexes" API, which publishes for every
outstanding 2-year Treasury FRN:

  - spread          : the fixed quoted spread set at auction (percent)
  - daily_index     : the current index rate = highest accepted discount rate
                      of the most recent 13-week bill auction (percent)
  - maturity_date   : calendar maturity

Writes data/curves/frn_latest.csv with one row per outstanding FRN CUSIP:

  date,cusip,maturity_date,maturity_years,spread,index_rate

maturity_years is the year-fraction from the record date (ACT/365.25), matching
the v0 "time in years from valuation" convention of the C++ engine.

Usage:
    python fetch_frn.py                # latest business day
    python fetch_frn.py --out ../data/curves
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import urllib.request

API = ("https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
       "/v1/accounting/od/frn_daily_indexes")


def fetch_latest() -> list[dict]:
    url = f"{API}?sort=-record_date&page%5Bsize%5D=300"
    req = urllib.request.Request(url, headers={"User-Agent": "tensor-bond-pricer/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.load(resp)
    rows = payload.get("data", [])
    if not rows:
        return []
    latest_date = rows[0]["record_date"]
    # One row per CUSIP for the latest record date (the feed repeats each CUSIP
    # once per accrual day of the payment period).
    seen: dict[str, dict] = {}
    for r in rows:
        if r["record_date"] != latest_date:
            continue
        seen.setdefault(r["cusip"], r)
    return sorted(seen.values(), key=lambda r: r["maturity_date"])


def year_fraction(d0: str, d1: str) -> float:
    a = dt.date.fromisoformat(d0)
    b = dt.date.fromisoformat(d1)
    return (b - a).days / 365.25


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str,
                   default=os.path.join(os.path.dirname(__file__), "..", "data", "curves"))
    args = p.parse_args(argv)

    rows = fetch_latest()
    if not rows:
        print("[fetch] no FRN rows returned", file=sys.stderr)
        return 1

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "frn_latest.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "cusip", "maturity_date", "maturity_years",
                    "spread", "index_rate"])
        for r in rows:
            w.writerow([
                r["record_date"], r["cusip"], r["maturity_date"],
                f"{year_fraction(r['record_date'], r['maturity_date']):.6f}",
                r["spread"], r["daily_index"],
            ])
    print(f"[fetch] wrote {len(rows)} FRN quotes ({rows[0]['record_date']}) -> {path}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
