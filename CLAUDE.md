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
  bullet only** (non-callable, non-putable), fixed coupon.
- **Later (do not build yet unless asked):** callable agencies, **muni**,
  **corporate**, floaters, option-adjusted spread / lattice models, full
  calendar & day-count conventions, intraday data.

When asked to extend scope, prefer adding a new sector spread curve or a new
`fetch_*.py` over changing the core tensor math.

## Layout

```
include/tbp/     Public headers
  curve.hpp        DiscountCurve: nodes as tensors, interp, discount, bootstrap
  bond.hpp         FixedRateBond + price() + analyze() (autograd risk)
  schedule.hpp     coupon time generation (v0 works in year-fractions)
src/
  curve.cpp        curve + par->zero bootstrap
  bond.cpp         pricing + autograd DV01 / key-rate risk
  main.cpp         demo: load par curve -> bootstrap -> price UST + Agency
tests/
  test_pricing.cpp dependency-free sanity checks (ctest)
python/
  fetch_treasury.py  daily par-curve fetcher (stdlib only), writes data/curves/
data/curves/       CSV cache: latest.csv, tenors.csv, treasury_par_YYYY.csv
docs/
  tensor_curve_framework.md   design + math + roadmap
CMakeLists.txt     find_package(Torch); builds lib, demo, tests
```

## Build & run

```bash
# 1. Get libtorch once (CPU build is fine): https://pytorch.org/get-started
#    unzip somewhere, note the absolute path.
# 2. Configure + build
cmake -DCMAKE_PREFIX_PATH=/abs/path/to/libtorch -B build -S .
cmake --build build -j
# 3. Refresh today's curve, then price
python3 python/fetch_treasury.py
./build/price_bonds data/curves/latest.csv data/curves/tenors.csv
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
- v0 measures time in **year-fractions from valuation**, not calendar dates.
  Adding real day-count/business-day logic is a scoped task in `docs/`.

## Definition of done for a change

1. `ctest` passes.
2. `price_bonds` runs on `data/curves/latest.csv` without error.
3. New math has at least one test in `tests/test_pricing.cpp`.
4. If scope/architecture changed, update this file and `docs/`.
