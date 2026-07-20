// Price a Treasury security BY CUSIP off the on-the-run instrument curve.
//
//   ./price_cusip data/curves/securities_latest.csv 912810UV8   # one CUSIP, full detail
//   ./price_cusip data/curves/securities_latest.csv --all       # every on-the-run tenor
//   ./price_cusip data/curves/securities_latest.csv --all --json report.json
//
// For each security it reports:
//   - NPV = discounted cashflows off the YC_TSY curve (bootstrapped fresh
//     from every on-the-run quote in the csv, so the discount curve and the
//     priced bond come from the same market close)
//   - the future coupon schedule on REAL dates (conventions layer: counted
//     back from maturity, holiday-rolled)
//   - accrued interest, ACT/ACT actual-day, unadjusted coupon dates
//     (Treasury street convention)
//   - model clean = NPV - accrued, compared against the market clean price
//     implied by the day's quoted yield (street formula). The curve reprices
//     its own instruments, so |diff| should be sub-cent; anything above the
//     tolerance is flagged MISMATCH and the exit code is non-zero.
//
// `tbp` = tensor-bond-pricer (see include/tbp/core/curve.hpp).
#include <torch/torch.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "tbp/core/conventions.hpp"
#include "tbp/core/curve.hpp"
#include "tbp/core/curve_store.hpp"
#include "tbp/instruments/treasury/treasury.hpp"
#include "tbp/io/csv.hpp"

namespace {

constexpr double kMatchTolerance = 0.01;  // 1 cent per 100 face

struct Cashflow {
    tbp::Date date;
    double amount = 0.0;
};

struct CusipReport {
    tbp::io::SecurityQuote sec;
    double npv_dirty = 0.0;      // DCF off the curve
    double accrued = 0.0;        // ACT/ACT actual-day
    int accrued_days = 0, period_days = 0;
    tbp::Date prev_coupon{}, next_coupon{};
    double model_clean = 0.0;    // npv_dirty - accrued
    double market_dirty = 0.0;   // street price from the quoted yield
    double market_clean = 0.0;   // market_dirty - accrued
    double diff_clean = 0.0;     // model_clean - market_clean
    bool match = false;
    double dv01 = 0.0, mod_duration = 0.0;
    std::vector<Cashflow> schedule;  // rolled payment dates + amounts
};

// Street dirty price from the quoted yield on the bond's own year-fraction
// schedule — identical to the bootstrap's target, so this IS the market side.
double street_dirty(const tbp::FixedRateBond& b, double y) {
    if (b.coupon_rate == 0.0)  // bill: simple money-market discounting
        return b.face / (1.0 + y * b.maturity_years);
    auto t = b.times();
    double p = 0.0;
    for (int64_t k = 0; k < t.size(0); ++k) {
        double cf = b.face * b.coupon_rate / b.freq +
                    (k + 1 == t.size(0) ? b.face : 0.0);
        p += cf * std::pow(1.0 + y / b.freq, -b.freq * t[k].item<double>());
    }
    return p;
}

CusipReport price_cusip(const tbp::io::SecurityQuote& s, tbp::DiscountCurve& curve,
                        const tbp::Date& valuation) {
    CusipReport r;
    r.sec = s;
    tbp::treasury::TBond b{100.0, s.coupon, 2, s.maturity_years};

    r.npv_dirty = tbp::price(b, curve).item<double>();
    auto risk = tbp::analyze(b, curve, /*spread=*/0.0);
    r.dv01 = risk.dv01;
    r.mod_duration = risk.mod_duration;

    const tbp::Date mat = tbp::Date::from_iso(s.maturity_date);
    if (s.coupon > 0.0) {
        // Accrual uses UNADJUSTED coupon dates (Treasury convention).
        auto unadj = tbp::build_coupon_dates(valuation, mat, b.freq, b.terms.eom,
                                             tbp::BusinessDayRoll::NONE,
                                             b.terms.calendar);
        const int n = static_cast<int>(unadj.size());
        r.prev_coupon = mat.add_months(-6 * n, b.terms.eom);
        r.next_coupon = unadj.front();
        r.accrued_days = static_cast<int>(valuation.serial() - r.prev_coupon.serial());
        r.period_days = static_cast<int>(unadj.front().serial() - r.prev_coupon.serial());
        r.accrued = 100.0 * s.coupon / b.freq *
                    static_cast<double>(r.accrued_days) / r.period_days;

        // Payment schedule on rolled dates per the bond's terms.
        auto rolled = tbp::build_coupon_dates(valuation, mat, b.freq, b.terms.eom,
                                              b.terms.roll, b.terms.calendar);
        for (size_t k = 0; k < rolled.size(); ++k)
            r.schedule.push_back({rolled[k], 100.0 * s.coupon / b.freq +
                                             (k + 1 == rolled.size() ? 100.0 : 0.0)});
    } else {
        r.next_coupon = mat;  // single redemption cashflow
        r.schedule.push_back({mat, 100.0});
    }

    r.model_clean = r.npv_dirty - r.accrued;
    r.market_dirty = street_dirty(b, s.quote_yield);
    r.market_clean = r.market_dirty - r.accrued;
    r.diff_clean = r.model_clean - r.market_clean;
    r.match = std::fabs(r.diff_clean) < kMatchTolerance;
    return r;
}

void print_detail(const CusipReport& r, const std::string& date) {
    const auto& s = r.sec;
    std::printf("\nCUSIP %s — UST %s %s %.3f%% %s\n", s.cusip.c_str(),
                s.type.c_str(), s.term.c_str(), s.coupon * 100.0,
                s.maturity_date.c_str());
    std::printf("  Quote (%s): yield %.3f%%  [%s]\n", date.c_str(),
                s.quote_yield * 100.0,
                s.type == "Bill" ? "per-CUSIP close" : "CMT par at tenor, v0");
    std::printf("  NPV dirty (DCF off YC_TSY)   = %10.6f\n", r.npv_dirty);
    if (s.coupon > 0.0)
        std::printf("  Accrued (ACT/ACT, %d/%dd since %s) = %.6f\n",
                    r.accrued_days, r.period_days, r.prev_coupon.iso().c_str(),
                    r.accrued);
    else
        std::printf("  Accrued                      =   0 (bill)\n");
    std::printf("  Model clean                  = %10.6f\n", r.model_clean);
    std::printf("  Market dirty | clean (quote) = %10.6f | %.6f\n",
                r.market_dirty, r.market_clean);
    std::printf("  Diff clean (model - market)  = %+.6f  => %s (tol %.2f)\n",
                r.diff_clean, r.match ? "MATCH" : "MISMATCH", kMatchTolerance);
    std::printf("  DV01 = %.6f   Mod duration = %.4f\n", r.dv01, r.mod_duration);
    std::printf("  Future payments (%zu):\n", r.schedule.size());
    for (const auto& cf : r.schedule)
        std::printf("    %s  %10.4f\n", cf.date.iso().c_str(), cf.amount);
}

void print_row(const CusipReport& r) {
    const auto& s = r.sec;
    std::printf("  %-9s %-4s %-7s %6.3f%%  %s  %10.4f %8.4f %10.4f %10.4f  %+8.4f  %s\n",
                s.cusip.c_str(), s.type.c_str(), s.term.c_str(),
                s.coupon * 100.0, s.maturity_date.c_str(), r.npv_dirty,
                r.accrued, r.model_clean, r.market_clean, r.diff_clean,
                r.match ? "MATCH" : "MISMATCH");
}

void write_json(const std::string& path, const std::string& date,
                const tbp::DiscountCurve& curve,
                const std::vector<CusipReport>& reports) {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) { std::printf("cannot write %s\n", path.c_str()); return; }
    std::fprintf(f, "{\n  \"date\": \"%s\",\n  \"curve\": {\"name\": \"%s\", \"times\": [",
                 date.c_str(), curve.name().c_str());
    auto t = curve.node_times();
    auto z = curve.node_zeros();
    for (int64_t i = 0; i < t.size(0); ++i)
        std::fprintf(f, "%s%.6f", i ? ", " : "", t[i].item<double>());
    std::fprintf(f, "], \"zeros\": [");
    for (int64_t i = 0; i < z.size(0); ++i)
        std::fprintf(f, "%s%.8f", i ? ", " : "", z[i].item<double>());
    std::fprintf(f, "]},\n  \"tolerance\": %.4f,\n  \"securities\": [\n", kMatchTolerance);
    for (size_t i = 0; i < reports.size(); ++i) {
        const auto& r = reports[i];
        const auto& s = r.sec;
        std::fprintf(f,
            "    {\"cusip\": \"%s\", \"type\": \"%s\", \"term\": \"%s\", "
            "\"coupon\": %.5f, \"maturity_date\": \"%s\", \"maturity_years\": %.6f, "
            "\"quote_yield\": %.6f, \"npv_dirty\": %.6f, \"accrued\": %.6f, "
            "\"accrued_days\": %d, \"period_days\": %d, \"model_clean\": %.6f, "
            "\"market_dirty\": %.6f, \"market_clean\": %.6f, \"diff_clean\": %.6f, "
            "\"match\": %s, \"dv01\": %.6f, \"mod_duration\": %.4f, "
            "\"next_coupon\": \"%s\", \"schedule\": [",
            s.cusip.c_str(), s.type.c_str(), s.term.c_str(), s.coupon,
            s.maturity_date.c_str(), s.maturity_years, s.quote_yield,
            r.npv_dirty, r.accrued, r.accrued_days, r.period_days,
            r.model_clean, r.market_dirty, r.market_clean, r.diff_clean,
            r.match ? "true" : "false", r.dv01, r.mod_duration,
            r.next_coupon.iso().c_str());
        for (size_t k = 0; k < r.schedule.size(); ++k)
            std::fprintf(f, "%s{\"date\": \"%s\", \"amount\": %.4f}",
                         k ? ", " : "", r.schedule[k].date.iso().c_str(),
                         r.schedule[k].amount);
        std::fprintf(f, "]}%s\n", i + 1 == reports.size() ? "" : ",");
    }
    std::fprintf(f, "  ]\n}\n");
    std::fclose(f);
    std::printf("\nwrote %s\n", path.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: %s <securities_latest.csv> <CUSIP...|--all> [--json out.json]\n",
                    argv[0]);
        return 1;
    }
    bool all = false;
    std::string json_path;
    std::vector<std::string> want;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--all") == 0) all = true;
        else if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) json_path = argv[++i];
        else want.emplace_back(argv[i]);
    }

    std::vector<tbp::io::SecurityQuote> secs;
    std::string date;
    if (!tbp::io::load_securities(argv[1], secs, date) || secs.empty()) {
        std::printf("failed to load securities from %s\n", argv[1]);
        return 1;
    }
    const tbp::Date valuation = tbp::Date::from_iso(date);

    // Discount curve: bootstrap YC_TSY from EVERY on-the-run quote of the day.
    std::vector<double> m, c, y;
    for (const auto& s : secs) {
        m.push_back(s.maturity_years);
        c.push_back(s.coupon);
        y.push_back(s.quote_yield);
    }
    auto f64 = torch::kFloat64;
    auto curve = tbp::DiscountCurve::bootstrap_from_quotes(
        torch::tensor(m, f64), torch::tensor(c, f64), torch::tensor(y, f64), 2);
    curve.set_name(tbp::curves::YC_TSY);
    tbp::CurveStore::instance().add(curve);
    auto& yc = tbp::CurveStore::instance().get(tbp::curves::YC_TSY);

    std::vector<CusipReport> reports;
    std::vector<std::string> not_found = want;
    for (const auto& s : secs) {
        bool selected = all;
        for (auto it = not_found.begin(); it != not_found.end(); ++it)
            if (*it == s.cusip) { selected = true; not_found.erase(it); break; }
        if (selected) reports.push_back(price_cusip(s, yc, valuation));
    }
    for (const auto& miss : not_found)
        std::printf("CUSIP %s not in %s (on-the-run set only)\n", miss.c_str(), argv[1]);
    if (reports.empty()) return 1;

    std::printf("YC_TSY bootstrapped from %zu on-the-run securities, close %s\n",
                secs.size(), date.c_str());
    if (reports.size() == 1) {
        print_detail(reports.front(), date);
    } else {
        std::printf("\n  %-9s %-4s %-7s %8s  %-10s  %10s %8s %10s %10s  %8s  verdict\n",
                    "cusip", "type", "term", "coupon", "maturity", "npv_dirty",
                    "accrued", "mdl_clean", "mkt_clean", "diff");
        for (const auto& r : reports) print_row(r);
    }

    int mismatches = 0;
    for (const auto& r : reports) mismatches += r.match ? 0 : 1;
    if (mismatches)
        std::printf("\n%d MISMATCH(es) beyond %.2f — stale quotes or broken bootstrap.\n",
                    mismatches, kMatchTolerance);
    else
        std::printf("\nAll %zu securities MATCH their quotes (tol %.2f).\n",
                    reports.size(), kMatchTolerance);

    if (!json_path.empty()) write_json(json_path, date, yc, reports);
    return mismatches ? 2 : 0;
}
