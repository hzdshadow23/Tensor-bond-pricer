#!/usr/bin/env python3
"""On-the-run Treasury pricing test: every tenor, priced by CUSIP, checked
against its quote — plus an optional self-contained HTML report (the UI).

Runs the price_cusip binary over data/curves/securities_latest.csv with
--json, then verifies:
  1. every expected on-the-run term is present (7 bills, 5 notes, 2 bonds)
  2. every security's model clean price MATCHES the clean price implied by
     its quoted yield (|diff| < tolerance; the curve reprices its own
     instruments, so a mismatch means stale data or a broken bootstrap)
  3. cashflow schedules are sane: non-empty, final payment at maturity,
     final amount = face (+ coupon), accrued within [0, coupon/freq)

Exit 0 if all checks pass, 1 otherwise (usable in CI).

Usage:
  python3 python/test_otr_pricing.py                 # terminal verdicts only
  python3 python/test_otr_pricing.py --html build/otr_report.html
  open build/otr_report.html                          # the UI

Stdlib only, like the fetchers.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_TERMS = [
    ("Bill", "4-Week"), ("Bill", "6-Week"), ("Bill", "8-Week"),
    ("Bill", "13-Week"), ("Bill", "17-Week"), ("Bill", "26-Week"),
    ("Bill", "52-Week"),
    ("Note", "2-Year"), ("Note", "3-Year"), ("Note", "5-Year"),
    ("Note", "7-Year"), ("Note", "10-Year"),
    ("Bond", "20-Year"), ("Bond", "30-Year"),
]

failures = []


def check(ok, name, detail=""):
    tag = "ok  " if ok else "FAIL"
    print(f"{tag}: {name}" + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(name)


def run_pricer(binary, csv, json_out):
    cmd = [str(binary), str(csv), "--all", "--json", str(json_out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # exit 0 = all match, 2 = priced but mismatches (still produces json)
    check(proc.returncode in (0, 2), "price_cusip ran",
          f"exit {proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()[:200]}")
    if proc.returncode not in (0, 2):
        sys.exit(1)
    return json.loads(Path(json_out).read_text())


def verify(data):
    secs = data["securities"]
    tol = data["tolerance"]
    have = {(s["type"], s["term"]) for s in secs}
    for t in EXPECTED_TERMS:
        check(t in have, f"on-the-run present: {t[0]} {t[1]}")
    check(len(secs) == len(EXPECTED_TERMS),
          f"exactly {len(EXPECTED_TERMS)} securities", f"got {len(secs)}")

    for s in secs:
        tag = f'{s["cusip"]} ({s["term"]})'
        check(s["match"] and abs(s["diff_clean"]) < tol,
              f"{tag} model clean == quote clean",
              f'diff {s["diff_clean"]:+.6f} vs tol {tol}')
        check(50.0 < s["npv_dirty"] < 150.0, f"{tag} NPV sane",
              f'npv {s["npv_dirty"]}')
        sched = s["schedule"]
        check(len(sched) > 0, f"{tag} schedule non-empty")
        if not sched:
            continue
        last = sched[-1]
        # Final payment lands on (or is holiday-rolled just after) maturity.
        check(last["date"] >= s["maturity_date"] and
              last["date"][:7] == s["maturity_date"][:7],
              f"{tag} final payment at maturity",
              f'{last["date"]} vs {s["maturity_date"]}')
        cpn = 100.0 * s["coupon"] / 2.0
        check(abs(last["amount"] - (100.0 + cpn)) < 1e-9,
              f"{tag} final amount = face + coupon", f'{last["amount"]}')
        check(0.0 <= s["accrued"] < max(cpn, 1e-9) or s["coupon"] == 0.0,
              f"{tag} accrued in [0, coupon/2)", f'{s["accrued"]}')


# --------------------------------------------------------------------------
# HTML report (self-contained; light+dark; no external resources).
# --------------------------------------------------------------------------
TEMPLATE = r"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OTR Treasury Pricing</title>
<style>
:root {
  --bg: #f7f8f5; --surface: #ffffff; --ink: #1b2420; --muted: #5c6a63;
  --border: #e2e7e1; --accent: #1471b8; --accent-soft: #e3eef7;
  --good: #1f7a4d; --good-bg: #e4f2ea; --bad: #b3261e; --bad-bg: #f9e7e5;
  --grid: #edf0ec;
}
@media (prefers-color-scheme: dark) { :root {
  --bg: #101513; --surface: #181f1c; --ink: #e4eae6; --muted: #96a49d;
  --border: #2a332e; --accent: #459cd9; --accent-soft: #1c2d3a;
  --good: #4cc38a; --good-bg: #16281f; --bad: #ef8078; --bad-bg: #331b19;
  --grid: #1f2723;
} }
:root[data-theme="light"] {
  --bg: #f7f8f5; --surface: #ffffff; --ink: #1b2420; --muted: #5c6a63;
  --border: #e2e7e1; --accent: #1471b8; --accent-soft: #e3eef7;
  --good: #1f7a4d; --good-bg: #e4f2ea; --bad: #b3261e; --bad-bg: #f9e7e5;
  --grid: #edf0ec;
}
:root[data-theme="dark"] {
  --bg: #101513; --surface: #181f1c; --ink: #e4eae6; --muted: #96a49d;
  --border: #2a332e; --accent: #459cd9; --accent-soft: #1c2d3a;
  --good: #4cc38a; --good-bg: #16281f; --bad: #ef8078; --bad-bg: #331b19;
  --grid: #1f2723;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
}
.mono { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
        font-variant-numeric: tabular-nums; }
.wrap { max-width: 1100px; margin: 0 auto; padding: 32px 20px 64px; }
header .eyebrow { font-size: 11px; letter-spacing: .14em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 6px; }
h1 { font-size: 26px; margin: 0 0 4px; text-wrap: balance; }
.sub { color: var(--muted); margin: 0 0 24px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px; margin-bottom: 24px; }
.tile { background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 14px 16px; }
.tile .k { font-size: 11px; letter-spacing: .12em; text-transform: uppercase;
  color: var(--muted); }
.tile .v { font-size: 22px; font-weight: 600; margin-top: 2px; }
.tile .v.good { color: var(--good); } .tile .v.bad { color: var(--bad); }
.card { background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 20px; margin-bottom: 24px; }
.card h2 { font-size: 15px; margin: 0 0 2px; }
.card .note { font-size: 12.5px; color: var(--muted); margin: 0 0 14px; }
.chart-box { position: relative; }
.chart-box svg { display: block; width: 100%; height: auto; }
.tip { position: absolute; pointer-events: none; background: var(--ink);
  color: var(--bg); font-size: 12px; padding: 5px 9px; border-radius: 5px;
  transform: translate(-50%, -130%); white-space: nowrap; opacity: 0;
  transition: opacity .12s; }
@media (prefers-reduced-motion: reduce) { .tip { transition: none; } }
.tbl-box { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
th { font-size: 11px; letter-spacing: .1em; text-transform: uppercase;
  color: var(--muted); text-align: right; padding: 8px 10px;
  border-bottom: 1px solid var(--border); white-space: nowrap; }
th.l, td.l { text-align: left; }
td { padding: 8px 10px; text-align: right; border-bottom: 1px solid var(--grid);
  white-space: nowrap; }
tr.sec { cursor: pointer; }
tr.sec:hover td { background: var(--accent-soft); }
tr.sec:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.pill { display: inline-block; font-size: 11px; font-weight: 600;
  letter-spacing: .06em; padding: 2px 9px; border-radius: 999px; }
.pill.match { color: var(--good); background: var(--good-bg); }
.pill.mismatch { color: var(--bad); background: var(--bad-bg); }
tr.detail > td { background: var(--bg); padding: 14px 18px; text-align: left; }
.d-stats { display: flex; flex-wrap: wrap; gap: 18px; font-size: 13px;
  margin-bottom: 10px; color: var(--muted); }
.d-stats b { color: var(--ink); font-weight: 600; }
.sched { max-height: 240px; overflow: auto; border: 1px solid var(--border);
  border-radius: 6px; display: inline-block; min-width: 300px; }
.sched table { font-size: 12.5px; } .sched td, .sched th { padding: 4px 14px; }
footer { font-size: 12.5px; color: var(--muted); }
footer code { font-family: ui-monospace, Menlo, monospace; }
</style>
<div class="wrap">
<header>
  <div class="eyebrow">tensor-bond-pricer &middot; YC_TSY instrument curve</div>
  <h1>On-the-run Treasury pricing</h1>
  <p class="sub">Every on-the-run bill, note and bond priced by CUSIP off the
  bootstrapped discount curve, close of <b class="mono" id="h-date"></b> —
  NPV by discounted cashflow, ACT/ACT accrued, model clean vs the clean price
  implied by the day&rsquo;s quote.</p>
</header>
<div class="tiles" id="tiles"></div>
<div class="card">
  <h2>YC_TSY zero curve</h2>
  <p class="note">Continuously-compounded zeros at the 14 instrument
  maturities (&radic;t axis so the bill end stays readable). Hover a node for
  the exact rate; the table below is the data view.</p>
  <div class="chart-box" id="chart"></div>
</div>
<div class="card">
  <h2>Pricing vs quotes</h2>
  <p class="note">Click a row for the future payment schedule, accrued detail
  and risk. Prices per 100 face. Verdict: |model clean &minus; market clean|
  &lt; <span id="h-tol"></span>.</p>
  <div class="tbl-box"><table id="tbl"></table></div>
</div>
<footer>
  Bills quote as per-CUSIP closes (Daily Treasury Bill Rates); note/bond
  yields are the CMT par yield at the matching tenor (documented v0
  approximation — on-the-runs trade near par). Regenerate:
  <code>python3 python/test_otr_pricing.py --html build/otr_report.html</code>
</footer>
</div>
<script>
const DATA = __DATA__;
const fmt = (x, d=4) => x.toLocaleString("en-US",
  {minimumFractionDigits: d, maximumFractionDigits: d});

// ---- header + tiles ----
document.getElementById("h-date").textContent = DATA.date;
document.getElementById("h-tol").textContent = DATA.tolerance.toFixed(2);
const secs = DATA.securities;
const nMatch = secs.filter(s => s.match).length;
const maxDiff = Math.max(...secs.map(s => Math.abs(s.diff_clean)));
const tiles = [
  ["Securities priced", secs.length, ""],
  ["Quote matches", nMatch + " / " + secs.length,
    nMatch === secs.length ? "good" : "bad"],
  ["Max |clean diff|", fmt(maxDiff), maxDiff < DATA.tolerance ? "good" : "bad"],
  ["Curve nodes", DATA.curve.times.length, ""],
];
document.getElementById("tiles").innerHTML = tiles.map(([k, v, c]) =>
  `<div class="tile"><div class="k">${k}</div>` +
  `<div class="v mono ${c}">${v}</div></div>`).join("");

// ---- zero curve chart (single series; sqrt-t x axis) ----
(function chart() {
  const W = 900, H = 300, m = {t: 16, r: 16, b: 34, l: 52};
  const ts = DATA.curve.times, zs = DATA.curve.zeros.map(z => z * 100);
  const xmax = Math.sqrt(Math.max(...ts)) * 1.03;
  const zlo = Math.floor(Math.min(...zs) * 4) / 4 - 0.25;
  const zhi = Math.ceil(Math.max(...zs) * 4) / 4 + 0.25;
  const X = t => m.l + Math.sqrt(t) / xmax * (W - m.l - m.r);
  const Y = z => m.t + (zhi - z) / (zhi - zlo) * (H - m.t - m.b);
  let g = "";
  for (let z = zlo + 0.25; z < zhi; z += 0.25) {
    g += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(z)}" y2="${Y(z)}"` +
         ` stroke="var(--grid)"/>` +
         `<text x="${m.l - 8}" y="${Y(z) + 4}" text-anchor="end"` +
         ` class="mono" font-size="11" fill="var(--muted)">${z.toFixed(2)}</text>`;
  }
  for (const t of [0.25, 1, 2, 5, 10, 20, 30]) {
    if (Math.sqrt(t) > xmax) continue;
    g += `<text x="${X(t)}" y="${H - 10}" text-anchor="middle" class="mono"` +
         ` font-size="11" fill="var(--muted)">${t}y</text>`;
  }
  const path = ts.map((t, i) =>
    (i ? "L" : "M") + X(t).toFixed(1) + " " + Y(zs[i]).toFixed(1)).join(" ");
  let dots = "";
  ts.forEach((t, i) => {
    dots += `<circle cx="${X(t)}" cy="${Y(zs[i])}" r="4" fill="var(--accent)"` +
            ` stroke="var(--surface)" stroke-width="2"/>` +
            `<circle cx="${X(t)}" cy="${Y(zs[i])}" r="12" fill="transparent"` +
            ` data-i="${i}"/>`;
  });
  const box = document.getElementById("chart");
  box.innerHTML =
    `<svg viewBox="0 0 ${W} ${H}" role="img"` +
    ` aria-label="YC_TSY zero curve, ${DATA.date}">` +
    `${g}<path d="${path}" fill="none" stroke="var(--accent)"` +
    ` stroke-width="2"/>${dots}</svg><div class="tip" id="tip"></div>`;
  const tip = document.getElementById("tip");
  box.querySelectorAll("circle[data-i]").forEach(c => {
    c.addEventListener("mouseenter", () => {
      const i = +c.dataset.i, s = secs[i];
      tip.innerHTML = `${s.term} &middot; ${s.cusip}<br>` +
        `z(${ts[i].toFixed(2)}y) = ${zs[i].toFixed(3)}%`;
      tip.style.left = (c.cx.baseVal.value / W * 100) + "%";
      tip.style.top = (c.cy.baseVal.value / H * 100) + "%";
      tip.style.opacity = 1;
    });
    c.addEventListener("mouseleave", () => { tip.style.opacity = 0; });
  });
})();

// ---- table with expandable detail ----
(function table() {
  const cols = ["CUSIP", "Term", "Coupon", "Maturity", "Quote yld", "Zero",
                "NPV dirty", "Accrued", "Model clean", "Mkt clean", "Diff",
                "Verdict"];
  let html = "<thead><tr>" + cols.map((c, i) =>
    `<th${i < 2 ? ' class="l"' : ""}>${c}</th>`).join("") + "</tr></thead><tbody>";
  secs.forEach((s, i) => {
    html += `<tr class="sec" data-i="${i}" tabindex="0" role="button"` +
      ` aria-expanded="false"><td class="l mono">${s.cusip}</td>` +
      `<td class="l">${s.term} ${s.type}</td>` +
      `<td class="mono">${(s.coupon * 100).toFixed(3)}%</td>` +
      `<td class="mono">${s.maturity_date}</td>` +
      `<td class="mono">${(s.quote_yield * 100).toFixed(3)}%</td>` +
      `<td class="mono">${(DATA.curve.zeros[i] * 100).toFixed(3)}%</td>` +
      `<td class="mono">${fmt(s.npv_dirty)}</td>` +
      `<td class="mono">${fmt(s.accrued)}</td>` +
      `<td class="mono">${fmt(s.model_clean)}</td>` +
      `<td class="mono">${fmt(s.market_clean)}</td>` +
      `<td class="mono">${(s.diff_clean >= 0 ? "+" : "") + fmt(s.diff_clean)}</td>` +
      `<td><span class="pill ${s.match ? "match" : "mismatch"}">` +
      `${s.match ? "MATCH" : "MISMATCH"}</span></td></tr>`;
    const rows = s.schedule.map(cf =>
      `<tr><td class="mono l">${cf.date}</td>` +
      `<td class="mono">${fmt(cf.amount, 4)}</td></tr>`).join("");
    const acc = s.coupon > 0
      ? `accrued <b class="mono">${fmt(s.accrued, 6)}</b> ` +
        `(${s.accrued_days}/${s.period_days} days ACT/ACT)`
      : "no accrued (bill)";
    html += `<tr class="detail" hidden><td colspan="${cols.length}">` +
      `<div class="d-stats"><span>next payment ` +
      `<b class="mono">${s.next_coupon}</b></span><span>${acc}</span>` +
      `<span>DV01 <b class="mono">${fmt(s.dv01, 6)}</b></span>` +
      `<span>mod duration <b class="mono">${fmt(s.mod_duration, 4)}</b></span>` +
      `<span>${s.schedule.length} future payment` +
      `${s.schedule.length > 1 ? "s" : ""}</span></div>` +
      `<div class="sched"><table><thead><tr><th class="l">Pay date</th>` +
      `<th>Amount</th></tr></thead><tbody>${rows}</tbody></table></div>` +
      `</td></tr>`;
  });
  const tbl = document.getElementById("tbl");
  tbl.innerHTML = html + "</tbody>";
  tbl.querySelectorAll("tr.sec").forEach(tr => {
    const toggle = () => {
      const d = tr.nextElementSibling;
      d.hidden = !d.hidden;
      tr.setAttribute("aria-expanded", String(!d.hidden));
    };
    tr.addEventListener("click", toggle);
    tr.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); }
    });
  });
})();
</script>
"""


def write_html(data, path):
    html = TEMPLATE.replace("__DATA__", json.dumps(data))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(html)
    print(f"\nwrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", default=ROOT / "build" / "price_cusip")
    ap.add_argument("--csv", default=ROOT / "data" / "curves" / "securities_latest.csv")
    ap.add_argument("--json-out", default=ROOT / "build" / "otr_pricing.json")
    ap.add_argument("--html", default=None, help="also write the UI report here")
    args = ap.parse_args()

    data = run_pricer(args.binary, args.csv, args.json_out)
    verify(data)
    if args.html:
        write_html(data, args.html)

    n = len(failures)
    print(f"\n{'TESTS FAILED' if n else 'ALL OTR PRICING CHECKS PASSED'} ({n} failure(s))")
    sys.exit(1 if n else 0)


if __name__ == "__main__":
    main()
