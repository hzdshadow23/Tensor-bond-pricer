# CLAUDE.md — tensor-bond-pricer

Guidance for Claude Code (and any AI agent) working in this repo. Read this
before making changes. Keep it up to date when the architecture moves.

## What this project is

A bond pricing engine built on a **tensor curve framework**. Discount curves and
bond cashflows are represented as **libtorch** (the C++ distribution of PyTorch)
tensors, so:

- bonds are priced as batched tensor operations, and
- **risk (DV01, key-rate/bucketed DV01, duration) comes from `torch::autograd`**
  — analytic gradients, no bump-and-reprice loop.

`libtorch` is not a wrapper over Python PyTorch. It is the same C++ core
(ATen + autograd); Python PyTorch binds *to* it. So this project links the C++
library directly and needs no Python at runtime.

## Scope (deliberately small first)

- **In scope now:** U.S. **Treasury** and **Agency** bonds, **option-free /
  bullet only** (non-callable, non-putable), fixed coupon; **Treasury FRNs**
  (2y floaters indexed to the 13-week bill) priced dual-curve
  (discount + forecast) with a discount margin; the **on-the-run 20Y and 30Y
  bonds** priced as products off an **instrument curve** bootstrapped from
  actual on-the-run bill/note/bond quotes (`bootstrap_from_quotes`).
- **Later (do not build yet unless asked):** callable agencies, **muni**,
  **corporate**, non-Treasury floaters, option-adjusted spread / lattice
  models, full calendar & day-count conventions, intraday data.

When asked to extend scope, prefer adding a new sector spread curve or a new
`fetch_*.py` over changing the core tensor math.

## Layout

```
include/tbp/          Public headers (src/ mirrors this tree)
  core/curve.hpp        DiscountCurve: nodes as tensors, interp, discount,
                        forward_rate(); bootstrap_from_par (published par grid),
                        bootstrap_from_quotes (actual on-the-run instruments);
                        make_forecast_curve() for projection curves; named
                        (e.g. "YC_TSY") + save()/load() as a libtorch archive
  core/curve_store.hpp  CurveStore: named curve registry (YC_TSY today,
                        YC_MUNI / YC_CORP later; FC_* for forecast curves)
  core/schedule.hpp     coupon time generation (v0 works in year-fractions)
  instruments/bond.hpp  FixedRateBond + price() + analyze() (autograd risk)
  instruments/frn.hpp   Treasury FRN: dual-curve pricing + discount margin + risk
  io/csv.hpp            loaders for the data/curves cache (std-only, no torch)
src/
  core/curve.cpp        curve + both bootstraps + forecast-curve builder
  instruments/bond.cpp  pricing + autograd DV01 / key-rate risk
  instruments/frn.cpp   FRN pricing/risk (separate module, off the fixed path)
  io/csv.cpp            CSV parsing for par curve / FRN / securities quotes
apps/
  price_bonds.cpp       demo: par curve + UST/Agency/TFRN, then instrument
                        curve from on-the-run quotes -> price 20Y + 30Y
tests/
  test_pricing.cpp      dependency-free sanity checks (ctest)
python/
  fetch_treasury.py     daily par-curve fetcher (stdlib only) -> latest.csv
  fetch_frn.py          TFRN quotes (index + spread per CUSIP, fiscaldata API)
  fetch_securities.py   on-the-run bill/note/bond daily quotes (TA_WS +
                        bill-rates feed + CMT) -> securities_latest.csv
data/curves/          raw quote cache (CSV): latest.csv, tenors.csv,
                      frn_latest.csv, securities_latest.csv,
                      treasury_par_YYYY.csv; plus persisted curve OBJECTS
                      (YC_TSY.pt, libtorch archive, gitignored — the curve
                      itself is never stored as csv)
docs/
  tensor_curve_framework.md   design + math + roadmap
CMakeLists.txt        find_package(Torch); builds lib, demo, tests
```

## Build & run

```bash
# 1. Get libtorch once (CPU build is fine): https://pytorch.org/get-started
#    unzip somewhere, note the absolute path.
# 2. Configure + build
cmake -DCMAKE_PREFIX_PATH=/abs/path/to/libtorch -B build -S .
cmake --build build -j
# 3. Refresh today's curve + FRN + security quotes, then price
python3 python/fetch_treasury.py
python3 python/fetch_frn.py
python3 python/fetch_securities.py
./build/price_bonds data/curves/latest.csv data/curves/tenors.csv \
                    data/curves/frn_latest.csv data/curves/securities_latest.csv
# 4. Tests
cd build && ctest --output-on-failure
```

The demo falls back to a built-in synthetic par curve if no CSV is passed, so
`./build/price_bonds` always runs.

## Data

Primary source: **treasury.gov Daily Treasury Par Yield Curve Rates** (XML feed,
no API key, no signup). `python/fetch_treasury.py` parses it to a tidy CSV with
tenor columns `1M..30Y` (percent). `tenors.csv` maps each label to its
year-fraction so C++ never hard-codes the grid. Only daily granularity is
needed; there is no intraday path.

TFRN quotes come from the **Treasury Fiscal Data "FRN Daily Indexes" API**
(JSON, no key): `python/fetch_frn.py` writes one row per outstanding FRN CUSIP
(maturity, quoted spread, current 13-week bill index) to `frn_latest.csv`.

Security-level quotes come from three keyless daily feeds combined by
`python/fetch_securities.py` into `securities_latest.csv`:

- **Bills (4/6/8/13/17/26/52-week):** treasury.gov **Daily Treasury Bill
  Rates** XML — a true per-CUSIP daily quote (closing discount rate +
  coupon-equivalent yield + CUSIP + maturity, every business day).
- **Notes (2/3/5/7/10Y) and Bonds (20/30Y):** CUSIP/coupon/maturity of the
  latest auction from the **TreasuryDirect TA_WS API**
  (`/TA_WS/securities/auctioned?format=json&type=Note|Bond&days=N`); the daily
  yield quote is the par-yield (CMT) curve at the matching tenor (documented
  v0 approximation — on-the-runs trade near par). **FedInvest**, the only
  public per-CUSIP EOD *price* source, blocks scripted access (verified
  2026-07); do not spend time on it again.

Agency-specific and FRED (needs a free key) fetchers are future siblings that
must emit the **same CSV schema**.

## Conventions & gotchas

- All tensors are **float64** (`torch::kFloat64`). Money math wants the
  precision; keep it consistent or autograd/interp will silently upcast.
- Zero rates are **continuously compounded**, decimals (0.042 = 4.2%).
  `DF(t) = exp(-z(t) * t)`, linear interpolation in zero-rate space.
- `DiscountCurve::node_zeros()` is the **differentiable leaf** — never call
  `.item()`, `.detach()`, or `.to()` inside the pricing path or you break the
  autograd graph that key-rate risk depends on.
- DV01 sign convention: **positive DV01 = loss for a +1bp rate rise.**
- Curves are **named objects**: `YC_<SECTOR>` for discount curves (`YC_TSY`;
  `YC_TSY_PAR` for the par-grid build; future `YC_MUNI`, `YC_CORP`),
  `FC_<SECTOR>` for forecast curves. Register/look up via `CurveStore`;
  persist with `curve.save()/DiscountCurve::load()` (`.pt`), never as csv —
  csv is only for raw fetched quotes.
- v0 measures time in **year-fractions from valuation**, not calendar dates.
  Adding real day-count/business-day logic is a scoped task in `docs/`.

## Definition of done for a change

1. `ctest` passes.
2. `price_bonds` runs on `data/curves/latest.csv` without error.
3. New math has at least one test in `tests/test_pricing.cpp`.
4. If scope/architecture changed, update this file and `docs/`.
