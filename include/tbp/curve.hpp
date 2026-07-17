// Tensor discount curve.
//
// The curve is a small set of node points (time, zero_rate) held as libtorch
// tensors. Continuously-compounded zero rates are interpolated linearly in
// time; discount factors are DF(t) = exp(-z(t) * t).
//
// Because node_zeros is a differentiable leaf tensor, any price computed from
// this curve is differentiable w.r.t. every node -> key-rate (bucketed) risk
// falls straight out of torch::autograd, no bump-and-reprice loop needed.
//
// A "sector" dimension (Treasury, Agency, ...) is layered on top as an additive
// spread curve, so a full multi-sector surface is a [n_sectors, n_nodes] tensor.
#pragma once
#include <torch/torch.h>
#include <vector>

namespace tbp {

class DiscountCurve {
public:
    // node_times: [N] strictly ascending, in years.
    // node_zeros: [N] continuously-compounded zero rates (decimal, e.g. 0.042).
    DiscountCurve(torch::Tensor node_times, torch::Tensor node_zeros);

    // Continuously-compounded zero rate at arbitrary times `t` ([M] tensor).
    // Linear interpolation in zero-rate space; flat extrapolation past the ends.
    torch::Tensor zero_rate(const torch::Tensor& t) const;

    // Discount factor(s) at times `t`.
    torch::Tensor discount(const torch::Tensor& t) const;

    // Simple-compounded forward rate over [t0, t1] (aligned [M] tensors):
    //   f = (DF(t0)/DF(t1) - 1) / (t1 - t0)
    // This is the rate a forecast (projection) curve implies for a floating
    // coupon that fixes at t0 and pays at t1. Differentiable w.r.t. node_zeros.
    torch::Tensor forward_rate(const torch::Tensor& t0,
                               const torch::Tensor& t1) const;

    // Differentiable leaf tensor of node zero rates (requires_grad = true).
    torch::Tensor& node_zeros() { return node_zeros_; }
    const torch::Tensor& node_zeros() const { return node_zeros_; }
    const torch::Tensor& node_times() const { return node_times_; }
    int64_t size() const { return node_times_.size(0); }

    // Bootstrap a zero curve from par yields (the shape Treasury publishes).
    //   tenors     : [N] years (e.g. {0.25, 0.5, 1, 2, 5, 10, 30})
    //   par_yields : [N] decimal par yields at those tenors
    //   freq       : coupon frequency for the >=1y par bonds (2 = semiannual)
    // Points with tenor < 1/freq are treated as zero-coupon money-market rates.
    static DiscountCurve bootstrap_from_par(const torch::Tensor& tenors,
                                            const torch::Tensor& par_yields,
                                            int freq = 2);

private:
    torch::Tensor node_times_;  // [N], float64, no grad
    torch::Tensor node_zeros_;  // [N], float64, requires_grad
};

// Build a forecast (projection) curve from a discount curve: same node times,
// zeros = base zeros + an additive basis (decimal, continuously compounded).
// The result owns a NEW differentiable leaf, so forecast risk and discount
// risk can be separated by autograd even when basis == 0. For Treasury FRNs
// the index is itself a Treasury rate, so basis is ~0; a nonzero basis anchors
// the short end to the observed 13-week bill index.
DiscountCurve make_forecast_curve(const DiscountCurve& base, double basis = 0.0);

}  // namespace tbp
