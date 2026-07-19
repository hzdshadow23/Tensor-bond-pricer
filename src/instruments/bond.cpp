#include "tbp/instruments/bond.hpp"
#include "tbp/core/schedule.hpp"
#include <torch/autograd.h>

namespace tbp {

namespace {
constexpr auto kF64 = torch::kFloat64;
}  // namespace

torch::Tensor FixedRateBond::times() const {
    auto t = coupon_times(maturity_years, freq);
    return torch::tensor(t, kF64);
}

torch::Tensor FixedRateBond::cashflows() const {
    auto t = coupon_times(maturity_years, freq);
    const double cpn = face * coupon_rate / freq;
    std::vector<double> cf(t.size(), cpn);
    if (!cf.empty()) cf.back() += face;  // redemption at maturity
    return torch::tensor(cf, kF64);
}

torch::Tensor price(const FixedRateBond& bond,
                    const DiscountCurve& curve,
                    const torch::Tensor& spread) {
    auto t = bond.times();
    auto cf = bond.cashflows();
    // Discount with an additive spread on the zero rate (sector / OAS).
    auto z = curve.zero_rate(t) + spread.to(kF64);
    auto df = torch::exp(-z * t);
    return (cf * df).sum();
}

RiskReport analyze(const FixedRateBond& bond, DiscountCurve& curve, double spread) {
    RiskReport rep;

    // --- PV and parallel DV01 via autograd on a scalar shift `s` ---
    auto s = torch::zeros({}, torch::TensorOptions().dtype(kF64).requires_grad(true));
    auto pv = price(bond, curve, s + spread);
    rep.clean_pv = pv.item<double>();

    auto g_shift = torch::autograd::grad({pv}, {s}, /*grad_outputs=*/{},
                                         /*retain_graph=*/true,
                                         /*create_graph=*/false)[0];
    double dpv_dshift = g_shift.item<double>();  // dPV per unit (100%) shift
    rep.dv01 = -dpv_dshift * 1e-4;               // per +1bp; >0 means loss
    rep.mod_duration = (rep.clean_pv != 0.0) ? -dpv_dshift / rep.clean_pv : 0.0;

    // --- Key-rate DV01s via autograd on the node zeros ---
    auto pv2 = price(bond, curve, torch::full({}, spread, kF64));
    auto g_nodes = torch::autograd::grad({pv2}, {curve.node_zeros()},
                                         /*grad_outputs=*/{},
                                         /*retain_graph=*/false,
                                         /*create_graph=*/false)[0];
    auto krd = (-g_nodes * 1e-4).contiguous();
    rep.key_rate_dv01.resize(krd.size(0));
    for (int64_t i = 0; i < krd.size(0); ++i) rep.key_rate_dv01[i] = krd[i].item<double>();

    return rep;
}

}  // namespace tbp
