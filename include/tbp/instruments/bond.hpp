// Option-free (bullet) fixed-coupon bond and its pricing/risk analytics.
//
// Scope for v0: Treasury and Agency, NON-callable / NON-putable only. Callable
// agencies, munis, and corporates are explicit future work.
#pragma once
#include <torch/torch.h>
#include "tbp/core/curve.hpp"

namespace tbp {

struct FixedRateBond {
    double face = 100.0;         // redemption / notional
    double coupon_rate = 0.0;    // annual coupon rate (decimal, e.g. 0.04)
    int freq = 2;                // coupons per year
    double maturity_years = 0.0; // time to maturity in years from valuation

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
