// Minimal dependency-free tests (no gtest). Returns non-zero on first failure;
// wired into ctest via CMakeLists. Run: ctest --output-on-failure
#include <torch/torch.h>
#include <cmath>
#include <cstdio>

#include "tbp/bond.hpp"
#include "tbp/curve.hpp"
#include "tbp/frn.hpp"

static int failures = 0;
#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) { std::printf("FAIL: %s\n", msg); ++failures; }        \
        else { std::printf("ok  : %s\n", msg); }                           \
    } while (0)

static bool close(double a, double b, double tol) { return std::fabs(a - b) < tol; }

int main() {
    using namespace tbp;
    const auto f64 = torch::kFloat64;

    // 1) Flat curve: zero-coupon reprices to face * exp(-r t).
    {
        auto t = torch::tensor({0.25, 1.0, 5.0, 30.0}, f64);
        auto z = torch::full({4}, 0.04, f64);
        DiscountCurve c(t, z);
        double df5 = c.discount(torch::tensor({5.0}, f64)).item<double>();
        CHECK(close(df5, std::exp(-0.04 * 5.0), 1e-10), "flat curve DF = exp(-rt)");
    }

    // 2) Bootstrap from flat par yields, then a par-coupon bond ~ 100.
    {
        auto tenors = torch::tensor({0.5, 1.0, 2.0, 5.0, 10.0}, f64);
        auto par = torch::full({5}, 0.04, f64);
        auto curve = DiscountCurve::bootstrap_from_par(tenors, par, 2);
        FixedRateBond b{100.0, 0.04, 2, 10.0};  // coupon == par yield
        double pv = price(b, curve).item<double>();
        CHECK(close(pv, 100.0, 0.20), "par-coupon bond reprices ~100");
    }

    // 3) DV01 > 0 and modified duration in a sane range for a 10y 4% bond.
    {
        auto tenors = torch::tensor({0.5, 1.0, 2.0, 5.0, 10.0}, f64);
        auto par = torch::full({5}, 0.04, f64);
        auto curve = DiscountCurve::bootstrap_from_par(tenors, par, 2);
        FixedRateBond b{100.0, 0.04, 2, 10.0};
        auto r = analyze(b, curve, 0.0);
        CHECK(r.dv01 > 0.0, "10y bond DV01 positive");
        CHECK(r.mod_duration > 6.0 && r.mod_duration < 9.0, "10y mod duration in [6,9]");

        // Key-rate DV01s approximately sum to the parallel DV01.
        double s = 0.0;
        for (double k : r.key_rate_dv01) s += k;
        CHECK(close(s, r.dv01, 0.02 * r.dv01 + 1e-6), "sum(key-rate DV01) ~ parallel DV01");
    }

    // 4) FRN telescoping identity: when the coupon rates are the discount
    //    curve's own simple forwards (forecast == discount, zero spread, zero
    //    DM) the floater reprices EXACTLY to face at a reset date.
    {
        auto t = torch::tensor({0.25, 1.0, 2.0, 5.0}, f64);
        auto z = torch::tensor({0.038, 0.040, 0.042, 0.044}, f64);
        DiscountCurve disc(t, z);
        auto fcst = make_forecast_curve(disc, 0.0);
        // Current-period fixing = the curve's own first simple quarterly rate.
        double idx = disc.forward_rate(torch::tensor({0.0}, f64),
                                       torch::tensor({0.25}, f64)).item<double>();
        FloatingRateNote frn{100.0, 0.0, 4, 2.0, idx};
        double pv = price(frn, disc, fcst).item<double>();
        CHECK(close(pv, 100.0, 1e-8), "FRN at reset with fwd fixings reprices to 100");
    }

    // 5) FRN risk shape: tiny rate DV01 (only the fixed current coupon and the
    //    spread annuity carry curve risk) but a fixed-bond-sized DM DV01; the
    //    autograd parallel DV01 matches a 1bp bump-and-reprice of both curves.
    {
        auto t = torch::tensor({0.25, 1.0, 2.0, 5.0}, f64);
        auto z = torch::full({4}, 0.04, f64);
        DiscountCurve disc(t, z);
        auto fcst = make_forecast_curve(disc, 0.0);
        FloatingRateNote frn{100.0, 0.00103, 4, 1.79, 0.0380};
        auto r = analyze(frn, disc, fcst, /*discount_margin=*/0.0);

        CHECK(r.spread_dv01 > 0.01 && r.spread_dv01 < 0.03,
              "FRN DM DV01 sized like a ~2y fixed bond");
        CHECK(std::fabs(r.dv01) < 0.25 * r.spread_dv01,
              "FRN rate DV01 much smaller than DM DV01");

        DiscountCurve disc_up(t, (disc.node_zeros() + 1e-4).detach());
        DiscountCurve fcst_up(t, (fcst.node_zeros() + 1e-4).detach());
        double pv_up = price(frn, disc_up, fcst_up).item<double>();
        double bump_dv01 = r.dirty_pv - pv_up;
        CHECK(close(r.dv01, bump_dv01, 1e-6 + 0.01 * std::fabs(bump_dv01)),
              "FRN autograd DV01 == bump-and-reprice (both curves +1bp)");
    }

    std::printf("\n%s (%d failure(s))\n", failures ? "TESTS FAILED" : "ALL TESTS PASSED", failures);
    return failures ? 1 : 0;
}
