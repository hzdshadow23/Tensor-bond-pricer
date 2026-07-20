// Minimal dependency-free tests (no gtest). Returns non-zero on first failure;
// wired into ctest via CMakeLists. Run: ctest --output-on-failure
#include <torch/torch.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "tbp/core/conventions.hpp"
#include "tbp/core/curve.hpp"
#include "tbp/core/curve_store.hpp"
#include "tbp/instruments/bond.hpp"
#include "tbp/instruments/frn.hpp"
#include "tbp/instruments/treasury/treasury.hpp"

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

    // 6) bootstrap_from_quotes degenerates to bootstrap_from_par when every
    //    instrument is an on-grid par bond (coupon == street yield -> price
    //    100 exactly), so the two bootstraps must produce the same zeros.
    {
        auto tenors = torch::tensor({0.5, 1.0, 2.0, 5.0, 10.0}, f64);
        auto par = torch::tensor({0.040, 0.041, 0.042, 0.044, 0.046}, f64);
        auto from_par = DiscountCurve::bootstrap_from_par(tenors, par, 2);
        auto from_qts = DiscountCurve::bootstrap_from_quotes(tenors, par, par, 2);
        double max_diff =
            (from_par.node_zeros() - from_qts.node_zeros()).abs().max().item<double>();
        CHECK(max_diff < 1e-8, "quote bootstrap == par bootstrap for par instruments");
    }

    // 7) Round-trip on a realistic on-the-run set (bills + seasoned coupon
    //    securities off the coupon grid): the bootstrapped curve must reprice
    //    every instrument to the dirty price implied by its quoted yield.
    {
        auto mats = torch::tensor({0.0876, 0.2464, 0.4956, 0.9747,
                                   1.9548, 4.9528, 9.8289, 19.8275, 29.8289}, f64);
        auto cpns = torch::tensor({0.0, 0.0, 0.0, 0.0,
                                   0.04125, 0.04125, 0.04375, 0.05, 0.05}, f64);
        auto ylds = torch::tensor({0.0371, 0.0380, 0.0392, 0.0400,
                                   0.0418, 0.0428, 0.0455, 0.0507, 0.0506}, f64);
        auto curve = DiscountCurve::bootstrap_from_quotes(mats, cpns, ylds, 2);

        double worst = 0.0;
        for (int64_t i = 0; i < mats.size(0); ++i) {
            double T = mats[i].item<double>();
            double c = cpns[i].item<double>();
            double y = ylds[i].item<double>();
            double market, model;
            if (c == 0.0) {  // bill: simple money-market discounting
                market = 1.0 / (1.0 + y * T);
                model = curve.discount(torch::tensor({T}, f64)).item<double>();
            } else {         // street dirty price vs curve PV, same schedule
                FixedRateBond b{1.0, c, 2, T};
                auto t = b.times();
                market = 0.0;
                for (int64_t k = 0; k < t.size(0); ++k) {
                    double tk = t[k].item<double>();
                    double cf = c / 2 + (k + 1 == t.size(0) ? 1.0 : 0.0);
                    market += cf * std::pow(1.0 + y / 2, -2.0 * tk);
                }
                model = price(b, curve).item<double>();
            }
            worst = std::max(worst, std::fabs(model - market));
        }
        CHECK(worst < 1e-6, "instrument curve reprices every quote (bills+notes+bonds)");

        // 20Y and 30Y products off the instrument curve: sane risk profile.
        FixedRateBond b20{100.0, 0.05, 2, 19.8275};
        FixedRateBond b30{100.0, 0.05, 2, 29.8289};
        auto r20 = analyze(b20, curve, 0.0);
        auto r30 = analyze(b30, curve, 0.0);
        CHECK(r20.dv01 > 0.0 && r30.dv01 > r20.dv01,
              "30y DV01 > 20y DV01 > 0 off instrument curve");
        CHECK(r20.mod_duration > 10.0 && r20.mod_duration < 15.0,
              "20y mod duration in [10,15]");
        CHECK(r30.mod_duration > 13.0 && r30.mod_duration < 19.0,
              "30y mod duration in [13,19]");
    }

    // 8) Named curve objects: a curve persists as a native libtorch archive
    //    (not csv) and round-trips exactly; CurveStore hands back a reference
    //    to the registered object by name.
    {
        auto t = torch::tensor({0.5, 2.0, 10.0}, f64);
        auto z = torch::tensor({0.040, 0.042, 0.046}, f64);
        DiscountCurve c(t, z, "YC_TSY_TEST");
        const char* path = "yc_tsy_test.pt";
        c.save(path);
        auto loaded = DiscountCurve::load(path);
        std::remove(path);

        CHECK(loaded.name() == "YC_TSY_TEST", "curve name survives save/load");
        double dz = (loaded.node_zeros() - c.node_zeros()).abs().max().item<double>();
        double dt = (loaded.node_times() - c.node_times()).abs().max().item<double>();
        CHECK(dz == 0.0 && dt == 0.0, "curve nodes survive save/load exactly");
        CHECK(loaded.node_zeros().requires_grad(), "loaded curve is a fresh grad leaf");

        CurveStore store;
        store.add(loaded);
        FixedRateBond b{100.0, 0.04, 2, 10.0};
        double pv_store = price(b, store.get("YC_TSY_TEST")).item<double>();
        double pv_direct = price(b, c).item<double>();
        CHECK(close(pv_store, pv_direct, 1e-12), "pricing off store lookup == direct");

        bool threw = false;
        try { store.get("YC_MUNI"); } catch (const std::out_of_range&) { threw = true; }
        CHECK(threw, "unknown curve name throws with known names listed");
    }

    // 9) Conventions layer: dates, holidays, business-day rolls, day counts,
    //    EOM rule, and date-based coupon schedules.
    {
        // Date arithmetic round-trips through the serial representation.
        Date d{2026, 7, 17};
        CHECK(Date::from_serial(d.serial()) == d, "date <-> serial round-trip");
        CHECK(Date::from_iso(d.iso()) == d, "date <-> iso round-trip");
        CHECK(d.weekday() == Weekday::Fri, "2026-07-17 is a Friday");

        // July 4 2026 is a Saturday -> observed Friday 2026-07-03 is a holiday.
        CHECK((Date{2026, 7, 4}.weekday() == Weekday::Sat), "2026-07-04 is a Saturday");
        CHECK((!is_business_day(Date{2026, 7, 3})), "observed July 4th (Fri) not a business day");
        CHECK((is_us_holiday(Date{2026, 11, 26})), "Thanksgiving 2026 = Nov 26");
        CHECK((is_business_day(Date{2026, 7, 17})), "a plain Friday is a business day");

        // Rolls: Sat 2026-07-04 -> FOLLOWING = Mon 7/6; month-end Sat
        // 2026-10-31 -> MODIFIED_FOLLOWING rolls BACK to Fri 10/30.
        CHECK((adjust(Date{2026, 7, 4}, BusinessDayRoll::FOLLOWING) == Date{2026, 7, 6}),
              "FOLLOWING rolls Sat to Mon");
        CHECK((adjust(Date{2026, 10, 31}, BusinessDayRoll::MODIFIED_FOLLOWING) ==
                  Date{2026, 10, 30}),
              "MODIFIED_FOLLOWING stays in month at month-end");

        // End-of-month rule vs plain clamping.
        CHECK((Date{2026, 1, 31}.add_months(1) == Date{2026, 2, 28}),
              "add_months clamps to month length");
        CHECK((Date{2026, 4, 30}.add_months(1, /*eom=*/true) == Date{2026, 5, 31}),
              "EOM rule: month-end stays month-end");
        CHECK((Date{2026, 4, 30}.add_months(1, /*eom=*/false) == Date{2026, 5, 30}),
              "no EOM: day-of-month kept");

        // Day counts.
        CHECK((close(year_fraction(Date{2026, 1, 30}, Date{2026, 7, 30},
                                   DayCount::THIRTY_360), 0.5, 1e-12)),
              "30/360: Jan 30 -> Jul 30 = 0.5y");
        CHECK((close(year_fraction(Date{2026, 1, 1}, Date{2027, 1, 1},
                                   DayCount::ACT_360), 365.0 / 360.0, 1e-12)),
              "ACT/360: one non-leap year = 365/360");

        // Date-based schedule for the on-the-run 30Y (5% 2056-05-15) as seen
        // from 2026-07-17: 60 semiannual payments, first on 2026-11-16
        // (Nov 15 is a Sunday -> MODIFIED_FOLLOWING), last at maturity.
        tbp::treasury::TBond b30{100.0, 0.05, 2, 29.83};
        auto pay = build_coupon_dates(Date{2026, 7, 17}, Date{2056, 5, 15}, b30.freq,
                                      b30.terms.eom, b30.terms.roll, b30.terms.calendar);
        CHECK(pay.size() == 60, "30Y bond: 60 remaining semiannual payments");
        CHECK((pay.front() == Date{2026, 11, 16}), "first coupon rolled off Sunday Nov 15");
        CHECK((pay.back() == Date{2056, 5, 15}), "final payment at maturity");
        auto yf = year_fractions(Date{2026, 7, 17}, pay, b30.terms.day_count);
        bool ascending = true;
        for (size_t i = 1; i < yf.size(); ++i) ascending &= yf[i] > yf[i - 1];
        CHECK(ascending && yf.back() > 29.7 && yf.back() < 29.95,
              "year-fractions ascending, maturity ~29.83y (feeds the tensor engine)");

        // Taxonomy: trailing terms defaulted, aggregate init unchanged; the
        // TFRN carries ACT/360 accrual + weekly reset terms.
        tbp::treasury::TFRN tfrn{100.0, 0.00103, 4, 1.79, 0.0380};
        CHECK(tfrn.sector == Sector::Treasury && tfrn.terms.day_count == DayCount::ACT_360 &&
                  tfrn.floating.resets_per_year == 52,
              "TFRN defaults: Treasury sector, ACT/360, weekly resets");
        auto bill = tbp::treasury::make_bill(100.0, 0.25);
        CHECK(bill.coupon_rate == 0.0, "T-bill is a zero-coupon FixedRateBond");
    }

    std::printf("\n%s (%d failure(s))\n", failures ? "TESTS FAILED" : "ALL TESTS PASSED", failures);
    return failures ? 1 : 0;
}
