// Treasury Floating Rate Note (TFRN) pricing off dual tensor curves.
//
// Structure (per the Treasury FRN term sheet): quarterly interest payments,
// coupon = index rate + quoted spread, where the index is the highest accepted
// discount rate of the most recent 13-week bill auction and the quoted spread
// is fixed at auction. Valuation is quoted as a discount margin (DM): the
// spread over the discount curve that reprices the note.
//
// Dual-curve setup:
//   - forecast curve projects the future index fixings as simple forwards,
//   - discount curve (+ DM) present-values the cashflows.
// Both are tensor DiscountCurves, so DV01 / key-rate risk on EACH curve falls
// out of torch::autograd.
//
// v0 simplifications (consistent with the rest of the engine): time in year
// fractions, the current coupon is fully fixed at `current_index` (the real
// note resets weekly within the period), and ACT/360 daily accrual is
// approximated by year-fraction accrual.
#pragma once
#include <torch/torch.h>
#include <vector>
#include "tbp/core/curve.hpp"
#include "tbp/instruments/bond.hpp"  // Sector, BondTerms (an FRN IS a bond)

namespace tbp {

// Floating-leg reset terms. The Treasury FRN index resets with every 13-week
// bill auction: auctioned weekly (generally Monday), effective from the
// following issue day (generally Thursday), with a lockout before each
// payment date. v0 pricing fixes the whole current period at `current_index`;
// these fields DECLARE the true reset schedule so the calendar work can
// refine the projection in place without changing the instrument's shape.
struct FloatingTerms {
    int resets_per_year = 52;              // weekly index resets
    Weekday reset_weekday = Weekday::Thu;  // index effective day (bill issue)
    int lockout_days = 2;                  // business days before payment
};

struct FloatingRateNote {
    double face = 100.0;
    double quoted_spread = 0.0;   // decimal, fixed at auction (e.g. 0.00103)
    int freq = 4;                 // Treasury FRNs pay quarterly
    double maturity_years = 0.0;  // years from valuation
    double current_index = 0.0;   // decimal simple rate fixed for the current period

    // Trailing members with defaults keep aggregate init working. Treasury
    // FRNs accrue ACT/360 (v0 approximates with year-fraction accrual).
    Sector sector = Sector::Treasury;
    BondTerms terms{DayCount::ACT_360};
    FloatingTerms floating{};

    // Payment times in years from valuation, ascending, last == maturity.
    // Uses a front stub: the first payment lands < 1/freq away when the
    // maturity is off the coupon grid (the usual case mid-period).
    torch::Tensor times() const;
};

struct FrnRiskReport {
    double dirty_pv = 0.0;
    double accrued = 0.0;    // interest accrued in the current period
    double clean_pv = 0.0;   // dirty - accrued
    double dv01 = 0.0;       // +1bp parallel shift of BOTH curves (>0 = loss)
    double spread_dv01 = 0.0;  // +1bp discount margin (credit-style risk)
    double mod_duration = 0.0; // rate duration, from dv01
    // Key-rate DV01 per node of each curve (aligned with node_times()).
    std::vector<double> key_rate_dv01_discount;
    std::vector<double> key_rate_dv01_forecast;
};

// Dirty PV. `discount_margin` is a decimal continuously-compounded spread on
// the discount curve. `discount` and `forecast` must be distinct objects (use
// make_forecast_curve) so their risks are separable.
torch::Tensor price(const FloatingRateNote& frn,
                    const DiscountCurve& discount,
                    const DiscountCurve& forecast,
                    const torch::Tensor& discount_margin =
                        torch::zeros({}, torch::kFloat64));

// Full autograd risk report.
FrnRiskReport analyze(const FloatingRateNote& frn,
                      DiscountCurve& discount,
                      DiscountCurve& forecast,
                      double discount_margin = 0.0);

}  // namespace tbp
