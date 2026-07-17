// Demo driver: load the daily Treasury par curve, bootstrap a tensor discount
// curve, and price a couple of option-free bonds with autograd risk.
//
//   ./price_bonds ../data/curves/latest.csv ../data/curves/tenors.csv
//
// If no paths are given it falls back to a small built-in par curve so the
// binary always runs.
#include <torch/torch.h>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "tbp/bond.hpp"
#include "tbp/curve.hpp"

namespace {

std::map<std::string, double> load_tenor_years(const std::string& path) {
    std::map<std::string, double> m;
    std::ifstream f(path);
    std::string line;
    std::getline(f, line);  // header
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        std::string tenor, yrs;
        std::getline(ss, tenor, ',');
        std::getline(ss, yrs, ',');
        if (!tenor.empty() && !yrs.empty()) m[tenor] = std::stod(yrs);
    }
    return m;
}

// Read the most recent row of latest.csv into aligned (years, par-yield) vecs.
bool load_latest_curve(const std::string& csv, const std::string& tenor_csv,
                       std::vector<double>& years, std::vector<double>& pars,
                       std::string& date) {
    auto tmap = load_tenor_years(tenor_csv);
    std::ifstream f(csv);
    if (!f) return false;
    std::string header, row, last;
    std::getline(f, header);
    while (std::getline(f, row))
        if (!row.empty()) last = row;
    if (last.empty()) return false;

    std::vector<std::string> cols;
    { std::stringstream hs(header); std::string c;
      while (std::getline(hs, c, ',')) cols.push_back(c); }
    std::vector<std::string> vals;
    { std::stringstream rs(last); std::string c;
      while (std::getline(rs, c, ',')) vals.push_back(c); }

    for (size_t i = 0; i < cols.size() && i < vals.size(); ++i) {
        if (cols[i] == "date") { date = vals[i]; continue; }
        if (vals[i].empty()) continue;                 // tenor not published today
        auto it = tmap.find(cols[i]);
        if (it == tmap.end()) continue;
        years.push_back(it->second);
        pars.push_back(std::stod(vals[i]) / 100.0);    // percent -> decimal
    }
    return !years.empty();
}

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
        loaded = load_latest_curve(argv[1], argv[2], years, pars, date);
    if (!loaded) {
        // Fallback synthetic par curve (decimal).
        years = {0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30};
        pars  = {0.0425, 0.0410, 0.0395, 0.0380, 0.0378,
                 0.0385, 0.0400, 0.0415, 0.0445, 0.0455};
    }

    auto tenors = torch::tensor(years, torch::kFloat64);
    auto par    = torch::tensor(pars, torch::kFloat64);
    auto curve  = tbp::DiscountCurve::bootstrap_from_par(tenors, par, /*freq=*/2);

    std::printf("Valuation curve: %s  (%lld nodes)\n", date.c_str(),
                static_cast<long long>(curve.size()));

    // Treasury 4% 10y, semiannual, option-free.
    tbp::FixedRateBond ust{100.0, 0.04, 2, 10.0};
    print_report("UST 4.00% 10y", tbp::analyze(ust, curve, /*spread=*/0.0), curve);

    // Agency 4% 10y priced at Treasury + 25bp (flat spread placeholder).
    tbp::FixedRateBond agy{100.0, 0.04, 2, 10.0};
    print_report("Agency 4.00% 10y (+25bp)", tbp::analyze(agy, curve, /*spread=*/0.0025), curve);

    return 0;
}
