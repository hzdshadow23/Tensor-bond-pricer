// Option-free (bullet) fixed-coupon bond and its pricing/risk analytics.
//
// Taxonomy: every instrument in this library IS a bond — including floaters.
// The split is (1) by SECTOR (Treasury / Agency / Muni / Corporate, the
// `Sector` tag below and the instruments/treasury|muni|corporate/ directories)
// and (2) within a sector, by which pricing engine the cashflows need:
//   fixed cashflows    -> FixedRateBond   (bills, notes, bonds; TIPS later)
//   floating cashflows -> FloatingRateNote (frn.hpp — dual-curve, reset terms)
// See instruments/treasury/treasury.hpp for the concrete Treasury product
// types (TBill / TNote / TBond / TFRN / TIPS placeholder).
//
// Scope for v0: Treasury and Agency, NON-callable / NON-putable only. Callable
// agencies, munis, and corporates are explicit future work.
#pragma once
#include <torch/torch.h>
#include "tbp/core/conventions.hpp"
#include "tbp/core/curve.hpp"

namespace tbp {

// Sector of the issuer. Selects the discount curve family (YC_TSY, later
// YC_MUNI / YC_CORP) and, eventually, sector-specific conventions.
enum class Sector { Treasury, Agency, Muni, Corporate };

// Market-convention terms carried by every bond: day count, business-day
// roll, end-of-month rule, holiday calendar, all defined in
// core/conventions.hpp. v0 pricing still runs in year-fractions; these terms
// are what build_coupon_dates() / year_fractions() apply when turning real
// calendar dates into those year-fractions.
struct BondTerms {
    DayCount day_count = DayCount::ACT_ACT;  // Treasury notes/bonds: ACT/ACT
    BusinessDayRoll roll = BusinessDayRoll::MODIFIED_FOLLOWING;
    bool eom = false;                        // end-of-month coupon rule
    Calendar calendar = Calendar::US;
};

struct FixedRateBond {
    double face = 100.0;         // redemption / notional
    double coupon_rate = 0.0;    // annual coupon rate (decimal, e.g. 0.04)
    int freq = 2;                // coupons (pay frequency) per year
    double maturity_years = 0.0; // time to maturity in years from valuation

    // Trailing members with defaults so existing aggregate init
    // ({face, coupon, freq, maturity}) keeps compiling unchanged.
    Sector sector = Sector::Treasury;
    BondTerms terms{};           // day count / roll / EOM / calendar

    // Payment times (years) and cashflow amounts as aligned tensors.
    torch::Tensor times() const;
    torch::Tensor cashflows() const;  // coupons + face at maturity
};

struct RiskReport {
    double clean_pv = 0.0;   // present value (per `face`)
    double dv01 = 0.0;       // price change for +1bp parallel zero shift (>0 = loss)
    double mod_duration = 0.0;
    // Key-rate DV01s: sensitivity to +1bp at each curve node, aligned with
    // curve.node_times(). Sums (approximately) to the parallel dv01.
    std::vector<double> key_rate_dv01;
};

// Present value of a bond off a discount curve, plus an optional additive
// sector/OAS spread (decimal, continuously compounded). Returns a scalar tensor
// so callers can keep differentiating.
torch::Tensor price(const FixedRateBond& bond,
                    const DiscountCurve& curve,
                    const torch::Tensor& spread = torch::zeros({}, torch::kFloat64));

// Full risk report using autograd for dv01 and key-rate buckets.
RiskReport analyze(const FixedRateBond& bond,
                   DiscountCurve& curve,
                   double spread = 0.0);

}  // namespace tbp
