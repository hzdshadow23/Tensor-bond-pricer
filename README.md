# tensor-bond-pricer

Price option-free **Treasury** and **Agency** bonds with a **tensor curve
framework**: discount curves and cashflows are [libtorch] tensors, so pricing is
vectorized and **risk (DV01, key-rate DV01, duration) comes straight from
automatic differentiation** — no bump-and-reprice.

libtorch is the **C++ distribution of PyTorch** — the same ATen/autograd engine
that Python PyTorch binds to, used here directly from C++.

## Status / scope

- ✅ U.S. Treasury & Agency, **option-free (bullet, non-callable)**, fixed coupon
- ✅ Daily par-curve fetch from treasury.gov (no API key)
- ✅ Par→zero bootstrap, discounting, autograd DV01 + key-rate DV01 + duration
- 🚧 Later: callable agencies, muni, corporate, day-count/calendar, spline curves, OAS

## Quickstart

```bash
# 1. libtorch (CPU build is enough) — download & unzip from pytorch.org
# 2. Build
cmake -DCMAKE_PREFIX_PATH=/abs/path/to/libtorch -B build -S .
cmake --build build -j

# 3. Fetch today's Treasury par curve (stdlib only, no key)
python3 python/fetch_treasury.py

# 4. Price the demo bonds (falls back to a built-in curve if no CSV given)
./build/price_bonds data/curves/latest.csv data/curves/tenors.csv

# 5. Tests
cd build && ctest --output-on-failure
```

## Layout

| Path | What |
|------|------|
| `include/tbp/`, `src/` | C++/libtorch core: `curve`, `bond`, `schedule` |
| `python/fetch_treasury.py` | daily par-curve fetcher → `data/curves/*.csv` |
| `tests/test_pricing.cpp` | ctest sanity checks (par reprices, DV01 sums) |
| `docs/tensor_curve_framework.md` | design, math, roadmap (incl. 中文小结) |
| `CLAUDE.md` / `AGENTS.md` | instructions for AI agents working in the repo |
| `.claude/skills/bond-pricing/` | Claude Code skill for common tasks |

## Data

**Source:** treasury.gov *Daily Treasury Par Yield Curve Rates* (XML feed, no
signup, daily granularity). The fetcher writes tenor columns `1M…30Y` plus a
`tenors.csv` mapping labels to year-fractions. FRED (needs a free key) and
agency/muni/corporate sources plug in later as sibling fetchers with the same
schema.

## Design in one line

Curve nodes are a differentiable tensor → every price is a function of them →
`torch::autograd::grad(price, nodes)` = key-rate risk, exact.

---

**中文小结**：本项目用 libtorch（PyTorch 的 C++ 版）以「张量曲线」框架为无期权的
美国国债与机构债定价。曲线节点为可微张量，价格对其自动微分即得 DV01 / 关键期限
DV01 / 久期，无需扰动重定价。数据取自 treasury.gov 每日到期收益率曲线（免 key、
日频即可）。当前仅做**无期权（不可赎回）固定票息**；市政债、公司债、可赎回债及
日历计息、样条曲线、OAS 等列入后续路线图。详见 `docs/tensor_curve_framework.md`
与 `CLAUDE.md`。

[libtorch]: https://pytorch.org/cppdocs/installing.html
