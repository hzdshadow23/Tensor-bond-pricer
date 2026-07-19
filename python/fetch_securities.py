#!/usr/bin/env python3
"""Fetch daily quotes for the on-the-run Treasury securities (stdlib only).

Covers every tenor of the coupon curve with an actual instrument:

  Bills  4/6/8/13/17/26/52-week   quote = daily closing discount rate and
                                  coupon-equivalent yield straight from the
                                  treasury.gov "Daily Treasury Bill Rates"
                                  XML feed (per-CUSIP, published every
                                  business day, no key).
  Notes  2/3/5/7/10-year          CUSIP/coupon/maturity of the latest auction
  Bonds  20/30-year               from the TreasuryDirect TA_WS API; the daily
                                  yield quote is the treasury.gov par-yield
                                  (CMT) curve at the matching tenor.

v0 approximation, documented: an on-the-run note/bond trades near par, so the
CMT par yield at its tenor is used as its street YTM quote. A true per-CUSIP
EOD price source (FedInvest, savingsbonds.gov "Prices and rates") returns an
error page to every scripted request (verified 2026-07: 5 domains, GET+POST,
cookies, referer) and is therefore not used.

Output: data/curves/securities_latest.csv, one row per on-the-run security,
sorted by maturity. Columns:

  date            quote date (bill close date / CMT curve date)
  type            Bill | Note | Bond
  term            e.g. 13-Week, 10-Year
  cusip
  coupon          percent; 0 for bills
  maturity_date   YYYY-MM-DD
  maturity_years  ACT/365.25 from the quote date
  quote_yield     percent, bond-equivalent basis (bills: coupon-equivalent
                  close; notes/bonds: CMT par yield at tenor)
  quote_disc      percent, bills only: closing discount rate (blank otherwise)
  quote_source    bill_rates_close | par_yield_cmt

All three feeds are keyless and refresh daily, so this script can run on a
daily cron alongside fetch_treasury.py / fetch_frn.py.
"""
import csv
import datetime as dt
import json
import re
import sys
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "curves" / "securities_latest.csv"

TA_WS = "https://www.treasurydirect.gov/TA_WS/securities/auctioned?format=json&type={typ}&days={days}"
XML_FEED = ("https://home.treasury.gov/resource-center/data-chart-center/"
            "interest-rates/pages/xml?data={data}&field_tdr_date_value={year}")

BILL_TENORS = ["4WK", "6WK", "8WK", "13WK", "17WK", "26WK", "52WK"]
NOTE_TERMS = ["2-Year", "3-Year", "5-Year", "7-Year", "10-Year"]
BOND_TERMS = ["20-Year", "30-Year"]
CMT_FIELD = {"2-Year": "BC_2YEAR", "3-Year": "BC_3YEAR", "5-Year": "BC_5YEAR",
             "7-Year": "BC_7YEAR", "10-Year": "BC_10YEAR",
             "20-Year": "BC_20YEAR", "30-Year": "BC_30YEAR"}


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "tensor-bond-pricer/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def xml_props(feed: bytes) -> list[dict]:
    """Parse the OData-style treasury.gov feed into a list of field dicts."""
    out = []
    for block in re.findall(rb"<m:properties>(.*?)</m:properties>", feed, re.S):
        row = {}
        for m in re.finditer(rb"<d:([A-Za-z0-9_]+)[^>]*>([^<]*)</d:\1>", block):
            row[m.group(1).decode()] = m.group(2).decode().strip()
        out.append(row)
    return out


def yf(maturity: str, asof: dt.date) -> float:
    """ACT/365.25 year-fraction from asof to an ISO date(-time) string."""
    d = dt.date.fromisoformat(maturity[:10])
    return round((d - asof).days / 365.25, 6)


def fetch_bills() -> list[dict]:
    year = dt.date.today().year
    rows = xml_props(get(XML_FEED.format(data="daily_treasury_bill_rates", year=year)))
    if not rows:  # early January: current year feed can be empty
        rows = xml_props(get(XML_FEED.format(data="daily_treasury_bill_rates", year=year - 1)))
    latest = max(rows, key=lambda r: r.get("INDEX_DATE", ""))
    date = latest["INDEX_DATE"][:10]
    asof = dt.date.fromisoformat(date)
    out = []
    for t in BILL_TENORS:
        disc = latest.get(f"ROUND_B1_CLOSE_{t}_2")
        bey = latest.get(f"ROUND_B1_YIELD_{t}_2")
        cusip = latest.get(f"CUSIP_{t}")
        mat = latest.get(f"MATURITY_DATE_{t}")
        if not (disc and bey and cusip and mat):
            continue
        out.append(dict(date=date, type="Bill", term=t.replace("WK", "-Week"),
                        cusip=cusip, coupon=0.0, maturity_date=mat[:10],
                        maturity_years=yf(mat, asof), quote_yield=float(bey),
                        quote_disc=float(disc), quote_source="bill_rates_close"))
    return out


def fetch_otr_coupons() -> dict[str, dict]:
    """Most recent auction per original term from TA_WS (rows newest-first)."""
    otr = {}
    for typ, terms in (("Note", NOTE_TERMS), ("Bond", BOND_TERMS)):
        for r in json.loads(get(TA_WS.format(typ=typ, days=400))):
            term = r.get("originalSecurityTerm")
            if term in terms and term not in otr:
                otr[term] = dict(type=typ, cusip=r["cusip"],
                                 coupon=float(r["interestRate"] or 0.0),
                                 maturity_date=r["maturityDate"][:10])
    return otr


def fetch_cmt() -> tuple[str, dict[str, float]]:
    """Latest par-yield (CMT) curve row: (date, {BC_field: percent})."""
    year = dt.date.today().year
    rows = xml_props(get(XML_FEED.format(data="daily_treasury_yield_curve", year=year)))
    if not rows:
        rows = xml_props(get(XML_FEED.format(data="daily_treasury_yield_curve", year=year - 1)))
    latest = max(rows, key=lambda r: r.get("NEW_DATE", ""))
    yields = {k: float(v) for k, v in latest.items()
              if k.startswith("BC_") and v and not k.endswith("DISPLAY")}
    return latest["NEW_DATE"][:10], yields


def main() -> int:
    bills = fetch_bills()
    otr = fetch_otr_coupons()
    cmt_date, cmt = fetch_cmt()
    asof = dt.date.fromisoformat(cmt_date)

    rows = list(bills)
    for term in NOTE_TERMS + BOND_TERMS:
        sec = otr.get(term)
        y = cmt.get(CMT_FIELD[term])
        if not sec or y is None:
            print(f"warning: no quote for {term}", file=sys.stderr)
            continue
        rows.append(dict(date=cmt_date, type=sec["type"], term=term,
                         cusip=sec["cusip"], coupon=sec["coupon"],
                         maturity_date=sec["maturity_date"],
                         maturity_years=yf(sec["maturity_date"], asof),
                         quote_yield=y, quote_disc="",
                         quote_source="par_yield_cmt"))

    rows.sort(key=lambda r: r["maturity_years"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["date", "type", "term", "cusip", "coupon", "maturity_date",
            "maturity_years", "quote_yield", "quote_disc", "quote_source"]
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} securities, bills as of {bills[0]['date'] if bills else 'n/a'}, "
          f"coupon curve as of {cmt_date})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
