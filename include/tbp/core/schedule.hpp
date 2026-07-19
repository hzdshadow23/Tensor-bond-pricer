// Cashflow schedule + day-count helpers.
//
// v0 works in "time in years from the valuation date" (double) rather than
// full calendar dates. This keeps the tensor pricing path clean and
// differentiable. Calendar/business-day/day-count refinements (ACT/ACT,
// 30/360, holiday calendars) are a documented next step -- see
// docs/tensor_curve_framework.md.
#pragma once
#include <vector>

namespace tbp {

// Generate semiannual (or freq/yr) coupon times for a bullet, option-free bond
// maturing in `maturity_years`, measured in years from valuation. Times are
// counted back from maturity so the final coupon lands exactly on maturity.
inline std::vector<double> coupon_times(double maturity_years, int freq = 2) {
    std::vector<double> t;
    const double step = 1.0 / freq;
    const int n = static_cast<int>(maturity_years * freq + 0.5);
    for (int i = n; i >= 1; --i) {
        double ti = maturity_years - (i - 1) * step;
        if (ti > 1e-9) t.push_back(ti);
    }
    return t;  // ascending, last element == maturity_years
}

}  // namespace tbp
