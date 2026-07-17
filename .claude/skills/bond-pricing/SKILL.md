---
name: bond-pricing
description: >
  Price option-free Treasury/Agency bonds and compute risk in this repo's tensor
  curve framework (libtorch). Use when asked to fetch the daily curve, price a
  bond, compute DV01/key-rate/duration, add an instrument or sector, or extend
  the curve math. Covers build/run, data refresh, and the autograd risk path.
---

# Bond pricing (tensor curve framework)

This skill operates the tensor-bond-pricer engine. Read `CLAUDE.md` and
`docs/tensor_curve_framework.md` for the full picture; this is the operational
checklist.

## Refresh the daily curve

```bash
python3 python/fetch_treasury.py            # today (treasury.gov, no API key)
python3 python/fetch_treasury.py --year 2025
python3 python/fetch_treasury.py --date 2026-07-15
```
Writes `data/curves/latest.csv`, `tenors.csv`, and `treasury_par_YYYY.csv`.

## Build & price

```bash
cmake -DCMAKE_PREFIX_PATH=/abs/path/to/libtorch -B build -S .
cmake --build build -j
./build/price_bonds data/curves/latest.csv data/curves/tenors.csv
cd build && ctest --output-on-failure
```

## Price a bond programmatically

```cpp
#include "tbp/bond.hpp"
#include "tbp/curve.hpp"
using namespace tbp;

auto curve = DiscountCurve::bootstrap_from_par(tenors, par_yields, /*freq=*/2);
FixedRateBond b{/*face=*/100.0, /*coupon=*/0.04, /*freq=*/2, /*maturity_years=*/10.0};
RiskReport r = analyze(b, curve, /*spread=*/0.0);   // PV, DV01, duration, key-rate DV01
```

## Rules that protect the autograd risk path

- Keep everything **float64**.
- Zero rates are **continuously compounded decimals**; `DF = exp(-z*t)`.
- Never `.item()` / `.detach()` / `.to()` on `curve.node_zeros()` inside pricing
  — it severs the graph and key-rate DV01 goes to zero/garbage.
- **Positive DV01 = loss for +1bp.**

## Common tasks

- **New sector (e.g. agency spread curve):** add an additive spread on the
  Treasury zero curve; generalize the scalar `spread` toward a `[n_sectors,
  n_nodes]` tensor. Do not fork the core pricing loop.
- **New data source:** add `python/fetch_<source>.py` emitting the **same CSV
  schema** (date + tenor columns, plus a tenor->years map).
- **New instrument math:** add a test in `tests/test_pricing.cpp` first
  (par reprices ~100; key-rate DV01 sums to parallel DV01).

## Out of scope (needs explicit go-ahead)

Callable/putable bonds, muni, corporate, floaters, OAS/lattice models, intraday
data. These live behind new modules, not inside the option-free path.
