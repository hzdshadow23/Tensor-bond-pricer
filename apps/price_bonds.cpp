// Demo driver: load the daily Treasury market data, build tensor curves, and
// price option-free bonds + the on-the-run TFRN, 20Y and 30Y with autograd risk.
//
//   ./price_bonds data/curves/latest.csv data/curves/tenors.csv \
//                 data/curves/frn_latest.csv data/curves/securities_latest.csv
//
// If no paths are given it falls back to a small built-in par curve so the
// binary always runs.
#include <torch/torch.h>
#include <cstdio>
#include <string>
#include <vector>

#include "tbp/core/curve.hpp"
#include "tbp/core/curve_store.hpp"
#include "tbp/instruments/bond.hpp"
#include "tbp/instruments/frn.hpp"
#include "tbp/io/csv.hpp"

namespace {

void print_report(const std::string& name, const tbp::RiskReport& r,
                  const tbp::DiscountCurve& curve) {
    std::printf("\n%s\n  PV            = %.4f\n  DV01 (per bp) = %.6f\n"
                "  Mod duration  = %.4f\n  Key-rate DV01:\n",
                name.c_str(), r.clean_pv, r.dv01, r.mod_duration);
    auto tvec = curve.node_times();
    for (size_t i = 0; i < r.key_rate_dv01.size(); ++i)
        std::printf("    t=%6.2fy : %+.6f\n", tvec[i].item<double>(), r.key_rate_dv01[i]);
}

}  // namespace

int main(int argc, char** argv) {
    torch::manual_seed(0);
    std::vector<double> years, pars;
    std::string date = "built-in";

    bool loaded = false;
    if (argc >= 3)
        loaded = tbp::io::load_latest_curve(argv[1], argv[2], years, pars, date);
    if (!loaded) {
        // Fallback synthetic par curve (decimal).
        years = {0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30};
        pars  = {0.0425, 0.0410, 0.0395, 0.0380, 0.0378,
                 0.0385, 0.0400, 0.0415, 0.0445, 0.0455};
    }

    auto tenors = torch::tensor(years, torch::kFloat64);
    auto par    = torch::tensor(pars, torch::kFloat64);
    auto curve  = tbp::DiscountCurve::bootstrap_from_par(tenors, par, /*freq=*/2);
    curve.set_name("YC_TSY_PAR");
    tbp::CurveStore::instance().add(curve);

    std::printf("Valuation curve: %s = %s  (%lld nodes)\n", curve.name().c_str(),
                date.c_str(), static_cast<long long>(curve.size()));

    // Treasury 4% 10y, semiannual, option-free.
    tbp::FixedRateBond ust{100.0, 0.04, 2, 10.0};
    print_report("UST 4.00% 10y", tbp::analyze(ust, curve, /*spread=*/0.0), curve);

    // Agency 4% 10y priced at Treasury + 25bp (flat spread placeholder).
    tbp::FixedRateBond agy{100.0, 0.04, 2, 10.0};
    print_report("Agency 4.00% 10y (+25bp)", tbp::analyze(agy, curve, /*spread=*/0.0025), curve);

    // --- Treasury FRN: dual-curve (discount + forecast) pricing -------------
    tbp::io::FrnQuote q{"synthetic", "n/a", 2.0, 0.0010, 0.0380};
    if (argc >= 4 && tbp::io::load_latest_frn(argv[3], q)) {
        std::printf("\nTFRN quote: %s mat %s (%.3fy)  spread=%.1fbp  index=%.4f%%\n",
                    q.cusip.c_str(), q.maturity_date.c_str(), q.maturity_years,
                    q.spread * 1e4, q.index_rate * 1e2);
    } else {
        std::printf("\nTFRN quote: synthetic fallback\n");
    }

    // Forecast curve = Treasury zero curve re-anchored so its first-quarter
    // simple forward reproduces the observed 13-week bill index (cc basis).
    auto q3m = torch::tensor({0.25}, torch::kFloat64);
    double z3m = curve.zero_rate(q3m).item<double>();
    double z_index = -std::log(1.0 - q.index_rate * 0.25) / 0.25;  // bill discount rate -> cc
    auto fcst = tbp::make_forecast_curve(curve, z_index - z3m);
    fcst.set_name("FC_TSY");
    tbp::CurveStore::instance().add(fcst);

    tbp::FloatingRateNote frn{100.0, q.spread, 4, q.maturity_years, q.index_rate};
    auto fr = tbp::analyze(frn, curve, fcst, /*discount_margin=*/0.0);
    std::printf("TFRN %s (DM = 0bp)\n"
                "  Dirty PV      = %.4f\n  Accrued       = %.4f\n"
                "  Clean PV      = %.4f\n  DV01 (rates)  = %.6f\n"
                "  DV01 (DM)     = %.6f\n  Mod duration  = %.4f\n",
                q.cusip.c_str(), fr.dirty_pv, fr.accrued, fr.clean_pv,
                fr.dv01, fr.spread_dv01, fr.mod_duration);
    auto tvec = curve.node_times();
    std::printf("  Key-rate DV01 (discount | forecast):\n");
    for (size_t i = 0; i < fr.key_rate_dv01_discount.size(); ++i)
        std::printf("    t=%6.2fy : %+.6f | %+.6f\n", tvec[i].item<double>(),
                    fr.key_rate_dv01_discount[i], fr.key_rate_dv01_forecast[i]);

    // --- Instrument curve: bootstrap from actual on-the-run quotes ----------
    // securities_latest.csv holds every on-the-run bill/note/bond with its
    // daily quote (bill closes + CMT yields; see python/fetch_securities.py).
    // The curve is built so each security reprices to its market dirty price;
    // the 20Y and 30Y bonds are then priced as products off that curve.
    std::vector<tbp::io::SecurityQuote> secs;
    std::string sec_date;
    if (argc >= 5 && tbp::io::load_securities(argv[4], secs, sec_date)) {
        std::vector<double> m, c, y;
        for (const auto& s : secs) {
            m.push_back(s.maturity_years);
            c.push_back(s.coupon);
            y.push_back(s.quote_yield);
        }
        auto built = tbp::DiscountCurve::bootstrap_from_quotes(
            torch::tensor(m, torch::kFloat64), torch::tensor(c, torch::kFloat64),
            torch::tensor(y, torch::kFloat64), /*freq=*/2);
        built.set_name("YC_TSY");

        // The curve is a named object, not csv state: persist it as a native
        // libtorch archive and register it; downstream pricing looks it up by
        // name (YC_MUNI / YC_CORP will slot in beside it later).
        built.save("data/curves/YC_TSY.pt");
        tbp::CurveStore::instance().add(tbp::DiscountCurve::load("data/curves/YC_TSY.pt"));
        auto& icurve = tbp::CurveStore::instance().get("YC_TSY");

        std::printf("\n%s: %s, bootstrapped from %zu on-the-run securities, "
                    "saved to data/curves/YC_TSY.pt\n  %-8s %-9s  %9s %8s %9s\n",
                    icurve.name().c_str(), sec_date.c_str(), secs.size(),
                    "term", "cusip", "maturity", "quote%", "zero%");
        for (size_t i = 0; i < secs.size(); ++i)
            std::printf("  %-8s %-9s  %8.3fy %8.3f %9.4f\n",
                        secs[i].term.c_str(), secs[i].cusip.c_str(),
                        secs[i].maturity_years, secs[i].quote_yield * 100.0,
                        icurve.node_zeros()[i].item<double>() * 100.0);

        for (const auto& s : secs) {
            if (s.type != "Bond") continue;  // the 20Y and 30Y products
            tbp::FixedRateBond b{100.0, s.coupon, 2, s.maturity_years};
            double model_dirty = tbp::price(b, icurve).item<double>();
            char name[160];
            std::snprintf(name, sizeof name,
                          "UST %s %.3f%% %s (%s)  [model dirty %.4f]",
                          s.term.c_str(), s.coupon * 100.0,
                          s.maturity_date.c_str(), s.cusip.c_str(), model_dirty);
            print_report(name, tbp::analyze(b, icurve, /*spread=*/0.0), icurve);
        }
    } else {
        std::printf("\nYC_TSY: no securities CSV given, skipped\n");
    }

    std::printf("\nRegistered curves:");
    for (const auto& n : tbp::CurveStore::instance().names())
        std::printf(" %s", n.c_str());
    std::printf("\n");

    return 0;
}
