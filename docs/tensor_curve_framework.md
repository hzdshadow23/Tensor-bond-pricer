# Tensor Curve Framework — design notes

*张量曲线框架设计文档 — 见文末中文小结。*

## 1. Why "tensor"

Everything in the pricing path is a libtorch tensor so that two things come for
free:

1. **Vectorized pricing.** A portfolio of bonds is a set of (times, cashflows)
   tensors; pricing is a couple of elementwise ops and a reduction. Batching
   over bonds, curves, or scenarios is just adding a leading dimension.
2. **Analytic risk via autograd.** The curve's node zero rates are a
   differentiable leaf tensor. Any price is a function of that tensor, so
   `torch::autograd::grad(PV, node_zeros)` returns **key-rate (bucketed) DV01**
   directly — exact to machine precision, no bump-and-reprice.

This mirrors how you'd build a differentiable pricer in Python PyTorch, but in
C++ (libtorch) for deployment. Same ATen/autograd engine underneath.

## 2. Objects

**DiscountCurve** — `include/tbp/curve.hpp`

- State: `node_times` `[N]` (years, no grad) and `node_zeros` `[N]`
  (continuously-compounded zero rates, `requires_grad = true`).
- `zero_rate(t)`: linear interpolation in zero-rate space, flat extrapolation
  past the ends. Differentiable w.r.t. every node.
- `discount(t) = exp(-zero_rate(t) * t)`.
- `bootstrap_from_par(tenors, par_yields, freq)`: standard sequential par
  bootstrap. Points below one coupon period are treated as simple money-market
  zeros; longer points are par bonds solved node-by-node, sweeping the grid a
  few times so the interpolation used for intermediate coupons is self-consistent.
- `bootstrap_from_quotes(maturities, coupons, yields, freq)`: same sweep
  machinery, but the inputs are **actual on-the-run securities** (the daily
  quotes in `securities_latest.csv`) rather than the published par grid. Each
  quote is first converted to a target dirty price — bills via the simple
  money-market convention `DF = 1/(1+yT)`, coupon securities via the street
  formula `P = Σ cf_k (1+y/f)^(-f·t_k)` on the front-stub schedule counted
  back from maturity — and node zeros are then solved so the curve reprices
  every instrument. When every instrument is an on-grid par bond this
  degenerates exactly to `bootstrap_from_par` (unit test). The curve nodes sit
  at the instruments' true maturities (e.g. 19.83y for the current 20Y), so
  key-rate risk buckets line up with the hedge instruments themselves.

**FixedRateBond** — `include/tbp/bond.hpp`

- Bullet, fixed coupon, option-free. Emits aligned `times()` and `cashflows()`
  tensors (final cashflow includes redemption).
- `price(bond, curve, spread)` = `sum(cf * exp(-(z(t)+spread) * t))`.
- `analyze(bond, curve, spread)` returns PV, parallel DV01, modified duration,
  and per-node key-rate DV01 via autograd.

**Forecast (projection) curve** — `make_forecast_curve(base, basis)` in
`include/tbp/curve.hpp`

- A second `DiscountCurve` with its own grad leaf: same node times, zeros =
  base zeros + additive basis. `forward_rate(t0, t1)` returns the
  simple-compounded forward `(DF(t0)/DF(t1) - 1)/(t1 - t0)` used to project
  floating coupons. Discount risk and projection risk separate cleanly in
  autograd because the two curves are distinct leaves.

**FloatingRateNote (Treasury FRN)** — `include/tbp/frn.hpp`

- Quarterly coupons = index + quoted spread; the index is the highest accepted
  discount rate of the most recent 13-week bill auction (resets weekly; v0
  fixes the current period at `current_index`). Dual-curve pricing: coupons
  projected off the forecast curve, discounted off the discount curve plus a
  **discount margin** (the FRN's quoted valuation spread).
- The telescoping identity `sum(DF(t_{i-1}) - DF(t_i)) = 1 - DF(T)` means a
  spread-free FRN with forecast == discount reprices exactly to face at a
  reset — this is a unit test.
- `analyze(frn, discount, forecast, dm)` returns dirty/clean PV, accrued,
  rate DV01 (tiny, by design of a floater), DM DV01 (sized like a fixed bond
  of the same maturity), and key-rate DV01 per node of *each* curve.
- Market quotes: `python/fetch_frn.py` pulls index + quoted spread per
  outstanding CUSIP from the Treasury Fiscal Data "FRN Daily Indexes" API into
  `data/curves/frn_latest.csv`.

## 3. Sector dimension (Treasury → Agency → …)

A sector is modelled as an **additive spread** on the Treasury zero curve. Today
that spread is a scalar placeholder (`+25bp` for agency in the demo). The natural
generalization is a spread *curve* per sector, giving a `[n_sectors, n_nodes]`
tensor — the "tensor curve" surface. Muni and corporate slot in as further rows,
each with its own spread curve (and, eventually, its own credit/tax adjustments).

## 4. Data pipeline

```
treasury.gov par XML  ──►  fetch_treasury.py  ──►  latest.csv + tenors.csv
fiscaldata FRN API    ──►  fetch_frn.py       ──►  frn_latest.csv
TA_WS auctions        ─┐
bill-rates XML        ─┼─► fetch_securities.py ──► securities_latest.csv
par-yield (CMT) XML   ─┘
                            │
                            ▼
        apps/price_bonds.cpp ──► bootstrap_from_par (published grid)
                             ──► bootstrap_from_quotes (on-the-run instruments)
                             ──► price/analyze (UST, Agency, TFRN, 20Y, 30Y)
```

Per-security daily quotes: bills have a true per-CUSIP daily close (Daily
Treasury Bill Rates feed). Notes/bonds have no scriptable public per-CUSIP
price feed (FedInvest blocks scripts — verified 2026-07), so the on-the-run
CUSIP/coupon/maturity from TA_WS is paired with the CMT par yield at its tenor
as the daily yield quote (near-par v0 approximation).

Daily granularity only. Additional sources (FRED for agency spreads, EMMA/MSRB
for muni, TRACE for corporate) attach as sibling `fetch_*.py` writing the same
schema.

## 5. Known simplifications in v0 (roadmap)

- **Time = year-fractions**, not calendar dates. Add `Date`, day-count
  (ACT/ACT, 30/360), business-day rolls, and accrued interest for clean/dirty
  price split.
- **Interpolation** is linear-in-zero. Consider log-linear on discount factors
  or monotone-convex; a **tensor-product spline** across (tenor × sector) is the
  eventual home for the "tensor curve" name.
- **Bootstrap from real issues** exists (`bootstrap_from_quotes`) but takes
  the CMT par yield as each note/bond's YTM (near-par approximation) and
  ignores accrued-interest day-count detail; next step is true quoted
  prices/yields per CUSIP if a scriptable source appears, and a spline fit to
  more than one issue per tenor.
- **Sector spread** is flat; move to a fitted spread curve, then OAS once
  options (callable agencies) are introduced.
- **No options.** Callable/putable need a short-rate lattice or Monte Carlo;
  keep that behind a separate module so the option-free path stays simple.
- **FRN approximations.** Current coupon fully fixed at today's index (real
  TFRNs reset weekly within the period, lagged by the auction schedule);
  accrual in year-fractions rather than ACT/360 daily; bill discount rate
  converted to cc with a flat 0.25y period. Refine alongside the calendar
  work.

## 6. Validation ideas

- Reprice par bonds → ~100 (in `tests/`).
- Key-rate DV01s sum to parallel DV01 (in `tests/`).
- Cross-check PV/duration against QuantLib or a spreadsheet for a few known bonds
  before trusting the engine on new instruments.

---

## 中文小结

本框架用 libtorch（PyTorch 的 C++ 版，底层是同一套 ATen/autograd 引擎，并非对
Python 的封装）把**贴现曲线**和**债券现金流**都表示为张量，从而获得两大好处：

1. **向量化定价**：一组债券即一组 (时间, 现金流) 张量，定价就是逐元素运算加求和，
   天然支持对债券/曲线/情景做批处理。
2. **自动微分求风险**：曲线节点零息利率是可微叶子张量，价格对其求梯度即得**关键
   期限 DV01（分桶敏感度）**，精确且无需扰动重定价。

当前范围：美国**国债与机构债**、**无期权（子弹型、不可赎回）**、固定票息。
市政债、公司债、可赎回债为后续工作。数据源：treasury.gov 每日国债到期收益率
曲线（XML，无需 API key），仅需日频。板块（国债→机构债→…）用**加性利差**建模，
未来扩展为 `[板块, 节点]` 的利差曲线张量面。v0 用「距估值日的年数」而非日历日期，
日历/计息惯例、样条插值、利差曲线、期权（格点/蒙特卡洛）均列入路线图。
