#include "tbp/curve.hpp"
#include <stdexcept>

namespace tbp {

namespace {
constexpr auto kF64 = torch::kFloat64;
}  // namespace

DiscountCurve::DiscountCurve(torch::Tensor node_times, torch::Tensor node_zeros)
    : node_times_(node_times.to(kF64).contiguous()),
      node_zeros_(node_zeros.to(kF64).contiguous()) {
    if (node_times_.dim() != 1 || node_zeros_.dim() != 1 ||
        node_times_.size(0) != node_zeros_.size(0)) {
        throw std::invalid_argument("node_times and node_zeros must be 1-D, same length");
    }
    node_zeros_.set_requires_grad(true);
}

torch::Tensor DiscountCurve::zero_rate(const torch::Tensor& t) const {
    const int64_t n = node_times_.size(0);
    auto tt = t.to(kF64);
    // Bracket each query time: idx in [1, n-1] so idx-1, idx are valid nodes.
    auto idx = torch::searchsorted(node_times_, tt, /*out_int32=*/false,
                                   /*right=*/true)
                   .clamp(1, n - 1);
    auto t0 = node_times_.index_select(0, idx - 1);
    auto t1 = node_times_.index_select(0, idx);
    auto z0 = node_zeros_.index_select(0, idx - 1);
    auto z1 = node_zeros_.index_select(0, idx);
    auto w = (tt - t0) / (t1 - t0);          // may be <0 or >1 -> flat extrapolation
    w = w.clamp(0.0, 1.0);
    return z0 + w * (z1 - z0);
}

torch::Tensor DiscountCurve::discount(const torch::Tensor& t) const {
    auto tt = t.to(kF64);
    return torch::exp(-zero_rate(tt) * tt);
}

DiscountCurve DiscountCurve::bootstrap_from_par(const torch::Tensor& tenors_in,
                                                const torch::Tensor& par_in,
                                                int freq) {
    auto tenors = tenors_in.to(kF64).contiguous();
    auto par = par_in.to(kF64).contiguous();
    const int64_t n = tenors.size(0);
    const double step = 1.0 / freq;

    // Zeros we solve for, node by node. Start from par as a warm guess.
    auto zeros = par.clone();

    // The par bond for node i has coupon dates on the freq grid up to tenor_i.
    // Intermediate coupon discount factors are taken from the curve built so
    // far (linear-in-zero interpolation). Because that interpolation depends on
    // later nodes near the boundary, we sweep the whole grid a few times to
    // converge. This is a standard sequential par bootstrap.
    for (int iter = 0; iter < 8; ++iter) {
        DiscountCurve cur(tenors, zeros);
        auto new_zeros = zeros.clone();
        for (int64_t i = 0; i < n; ++i) {
            double T = tenors[i].item<double>();
            double y = par[i].item<double>();
            if (T <= step + 1e-9) {
                // Money-market / zero-coupon point: DF = 1/(1 + y*T), simple.
                double df = 1.0 / (1.0 + y * T);
                new_zeros[i] = -std::log(df) / T;
                continue;
            }
            // Semiannual coupon par bond, face = 1, price = 1.
            double c = y * step;  // coupon per period
            int m = static_cast<int>(T * freq + 0.5);
            double pv_known = 0.0;
            for (int k = 1; k < m; ++k) {
                double tk = T - (m - k) * step;  // ascending coupon time < T
                double dfk = cur.discount(torch::tensor({tk}, kF64)).item<double>();
                pv_known += c * dfk;
            }
            // 1 = pv_known + (1 + c) * DF(T)  ->  DF(T)
            double df_T = (1.0 - pv_known) / (1.0 + c);
            df_T = std::max(df_T, 1e-8);
            new_zeros[i] = -std::log(df_T) / T;
        }
        zeros = new_zeros;
    }
    return DiscountCurve(tenors, zeros);
}

}  // namespace tbp
