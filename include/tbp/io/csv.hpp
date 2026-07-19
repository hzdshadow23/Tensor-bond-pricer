// CSV loaders for the data/curves/ cache written by the python/fetch_*.py
// fetchers. Deliberately dependency-free (std only, no libtorch): io is about
// getting market quotes into plain vectors; turning them into tensors is the
// caller's job.
#pragma once
#include <map>
#include <string>
#include <vector>

namespace tbp::io {

// tenors.csv: label ("1M".."30Y") -> year fraction.
std::map<std::string, double> load_tenor_years(const std::string& path);

// latest.csv (daily par-yield curve): most recent row, aligned (years, decimal
// par yield) vectors. Returns false if the file is missing/empty.
bool load_latest_curve(const std::string& csv, const std::string& tenor_csv,
                       std::vector<double>& years, std::vector<double>& pars,
                       std::string& date);

// frn_latest.csv: on-the-run Treasury FRN quote (last row = longest maturity).
struct FrnQuote {
    std::string cusip, maturity_date;
    double maturity_years = 0.0;
    double spread = 0.0;      // decimal
    double index_rate = 0.0;  // decimal
};
bool load_latest_frn(const std::string& csv, FrnQuote& q);

// securities_latest.csv: one row per on-the-run bill/note/bond with its daily
// quote (see python/fetch_securities.py for sources and conventions).
struct SecurityQuote {
    std::string type;           // "Bill" | "Note" | "Bond"
    std::string term;           // e.g. "13-Week", "20-Year"
    std::string cusip, maturity_date;
    double coupon = 0.0;        // decimal annual rate; 0 for bills
    double maturity_years = 0.0;
    double quote_yield = 0.0;   // decimal, bond-equivalent basis
};
bool load_securities(const std::string& csv, std::vector<SecurityQuote>& out,
                     std::string& date);

}  // namespace tbp::io
