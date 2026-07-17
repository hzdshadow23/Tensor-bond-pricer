#!/usr/bin/env python3
"""Fetch daily U.S. Treasury par yield curve rates from treasury.gov.

Primary data source for the tensor-bond-pricer project. No API key or signup
required. Writes tidy CSV files into ../data/curves/ that the C++/libtorch
engine consumes.

Treasury publishes the "Daily Treasury Par Yield Curve Rates" as an Atom/XML
feed, one document per calendar year:

  https://home.treasury.gov/resource-center/data-chart-center/interest-rates/
      pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value=YYYY

Each entry holds one business day with fields BC_1MONTH ... BC_30YEAR (percent).

Usage:
    python fetch_treasury.py                 # current year, write full-year CSV + latest.csv
    python fetch_treasury.py --year 2025     # a specific year
    python fetch_treasury.py --date 2026-07-15   # keep only that row in latest.csv
    python fetch_treasury.py --out ../data/curves

Design note: the fetcher is deliberately dependency-free (stdlib only) so it
runs anywhere. Agency / FRED / muni sources plug in as sibling fetch_*.py
modules that emit the same CSV schema.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET

BASE = ("https://home.treasury.gov/resource-center/data-chart-center/"
        "interest-rates/pages/xml")

# Treasury field name -> (tenor label, tenor in years). Ordered short -> long.
TENORS = [
    ("BC_1MONTH", "1M", 1 / 12),
    ("BC_2MONTH", "2M", 2 / 12),
    ("BC_3MONTH", "3M", 3 / 12),
    ("BC_4MONTH", "4M", 4 / 12),
    ("BC_6MONTH", "6M", 6 / 12),
    ("BC_1YEAR", "1Y", 1.0),
    ("BC_2YEAR", "2Y", 2.0),
    ("BC_3YEAR", "3Y", 3.0),
    ("BC_5YEAR", "5Y", 5.0),
    ("BC_7YEAR", "7Y", 7.0),
    ("BC_10YEAR", "10Y", 10.0),
    ("BC_20YEAR", "20Y", 20.0),
    ("BC_30YEAR", "30Y", 30.0),
]

# Atom / OData namespaces used by the Treasury feed.
NS = {
    "a": "http://www.w3.org/2005/Atom",
    "m": "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata",
    "d": "http://schemas.microsoft.com/ado/2007/08/dataservices",
}


def fetch_year(year: int) -> str:
    url = f"{BASE}?data=daily_treasury_yield_curve&field_tdr_date_value={year}"
    req = urllib.request.Request(url, headers={"User-Agent": "tensor-bond-pricer/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8")


def parse_feed(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    rows: list[dict] = []
    for entry in root.findall("a:entry", NS):
        props = entry.find(".//m:properties", NS)
        if props is None:
            continue
        date_el = props.find("d:NEW_DATE", NS)
        if date_el is None or not date_el.text:
            continue
        date = date_el.text.split("T")[0]
        row = {"date": date}
        for field, label, _yrs in TENORS:
            el = props.find(f"d:{field}", NS)
            val = el.text if el is not None and el.text not in (None, "") else ""
            row[label] = val
        rows.append(row)
    rows.sort(key=lambda r: r["date"])
    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    labels = [lbl for _f, lbl, _y in TENORS]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date"] + labels)
        for r in rows:
            w.writerow([r["date"]] + [r.get(lbl, "") for lbl in labels])


def write_tenor_map(path: str) -> None:
    """Emit the tenor->years mapping so the C++ side never hard-codes it."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tenor", "years"])
        for _f, lbl, yrs in TENORS:
            w.writerow([lbl, f"{yrs:.6f}"])


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, default=dt.date.today().year)
    p.add_argument("--date", type=str, default=None,
                   help="ISO date YYYY-MM-DD; latest.csv is filtered to this row")
    p.add_argument("--out", type=str,
                   default=os.path.join(os.path.dirname(__file__), "..", "data", "curves"))
    args = p.parse_args(argv)

    out = os.path.abspath(args.out)
    print(f"[fetch] Treasury par curve {args.year} -> {out}", file=sys.stderr)
    rows = parse_feed(fetch_year(args.year))
    if not rows:
        print("[fetch] no rows parsed", file=sys.stderr)
        return 1

    full = os.path.join(out, f"treasury_par_{args.year}.csv")
    write_csv(full, rows)

    if args.date:
        latest = [r for r in rows if r["date"] == args.date]
        if not latest:
            print(f"[fetch] {args.date} not found; using most recent", file=sys.stderr)
            latest = rows[-1:]
    else:
        latest = rows[-1:]
    write_csv(os.path.join(out, "latest.csv"), latest)
    write_tenor_map(os.path.join(out, "tenors.csv"))

    print(f"[fetch] wrote {len(rows)} rows; latest = {latest[-1]['date']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
