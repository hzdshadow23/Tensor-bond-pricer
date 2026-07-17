#include "tbp/frn.hpp"
#include <torch/autograd.h>
#include <cmath>
#include <stdexcept>

namespace tbp {

namespace {
constexpr auto kF64 = torch::kFloat64;
}  // namespace

torch::Tensor FloatingRateNote::times() const {
    const double step = 1.0 / freq;
    // ceil, not round: a payment period already in progress still pays its
    // full coupon at the period end, so the first time is a front stub.
    const int n = static_cast<int>(std::ceil(maturity_years * freq - 1e-9));
    std::vector<double> t;
    t.reserve(n);
    for (int i = 1; i <= n; ++i) {
        double ti = maturity_years - (n - i) * step;
        if (ti > 1e-9) t.push_back(ti);
    }
    return torch::tensor(t, kF64);
}

torch::Tensor price(const FloatingRateNote& frn,
                    const DiscountCurve& discount,
                    const DiscountCurve& forecast,
                    const torch::Tensor& discount_margin) {
    auto t = frn.times();
    const int64_t n = t.size(0);
    if (n == 0) throw std::invalid_argument("FRN has no remaining payments");
    const double step = 1.0 / frn.freq;

    // Period start times; the first period began in the past (or at 0) and its
    // rate is already fixed, so forwards are only projected for rows 1..n-1.
    auto starts = t - step;
    auto rates = torch::full({n}, frn.current_index, kF64);
    if (n > 1) {
        auto fwd = forecast.forward_rate(starts.slice(0, 1, n), t.slice(0, 1, n));
        rates = torch::cat({rates.slice(0, 0, 1), fwd});
    }

    // Coupon for period i accrues over the full period 1/freq at rate+spread.
    auto coupons = frn.face * (rates + frn.quoted_spread) * step;

    // Discount with the margin as an additive cc spread on the zero rate.
    auto z = discount.zero_rate(t) + discount_margin.to(kF64);
    auto df = torch::exp(-z * t);

    return (coupons * df).sum() + frn.face * df[n - 1];
}

FrnRiskReport analyze(const FloatingRateNote& frn,
                      DiscountCurve& discount,
                      DiscountCurve& forecast,
                      double discount_margin) {
    if (&discount == &forecast ||
        discount.node_zeros().is_same(forecast.node_zeros())) {
        throw std::invalid_argument(
            "discount and forecast must be distinct curves (make_forecast_curve)");
    }
    FrnRiskReport rep;

    auto dm = torch::full({}, discount_margin,
                          torch::TensorOptions().dtype(kF64).requires_grad(true));
    auto pv = price(frn, discount, forecast, dm);
    rep.dirty_pv = pv.item<double>();

    // Accrued interest: the current period started at t1 - 1/freq (in the
    // past); interest to date at the fixed rate + spread.
    auto t = frn.times();
    double t1 = t[0].item<double>();
    double accrual_elapsed = std::max(0.0, 1.0 / frn.freq - t1);
    rep.accrued = frn.face * (frn.current_index + frn.quoted_spread) * accrual_elapsed;
    rep.clean_pv = rep.dirty_pv - rep.accrued;

    // One backward pass gives every sensitivity: both curves' nodes + the DM.
    auto grads = torch::autograd::grad(
        {pv}, {discount.node_zeros(), forecast.node_zeros(), dm},
        /*grad_outputs=*/{}, /*retain_graph=*/false, /*create_graph=*/false);

    auto krd_d = (-grads[0] * 1e-4).contiguous();
    auto krd_f = (-grads[1] * 1e-4).contiguous();
    rep.spread_dv01 = -grads[2].item<double>() * 1e-4;

    rep.key_rate_dv01_discount.resize(krd_d.size(0));
    rep.key_rate_dv01_forecast.resize(krd_f.size(0));
    double total = 0.0;
    for (int64_t i = 0; i < krd_d.size(0); ++i) {
        rep.key_rate_dv01_discount[i] = krd_d[i].item<double>();
        total += rep.key_rate_dv01_discount[i];
    }
    for (int64_t i = 0; i < krd_f.size(0); ++i) {
        rep.key_rate_dv01_forecast[i] = krd_f[i].item<double>();
        total += rep.key_rate_dv01_forecast[i];
    }
    // Linear in the zeros, so the sum over every node of both curves IS the
    // parallel DV01 -- no separate shift variable needed.
    rep.dv01 = total;
    rep.mod_duration = (rep.dirty_pv != 0.0) ? rep.dv01 / rep.dirty_pv * 1e4 : 0.0;

    return rep;
}

}  // namespace tbp
