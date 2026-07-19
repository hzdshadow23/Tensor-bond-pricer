#include "tbp/io/csv.hpp"
#include <fstream>
#include <sstream>

namespace tbp::io {

namespace {

std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string c;
    while (std::getline(ss, c, ',')) out.push_back(c);
    if (!line.empty() && line.back() == ',') out.push_back("");
    return out;
}

// Header + last non-empty data row of a CSV.
bool last_row(const std::string& path, std::vector<std::string>& cols,
              std::vector<std::string>& vals) {
    std::ifstream f(path);
    if (!f) return false;
    std::string header, row, last;
    std::getline(f, header);
    while (std::getline(f, row))
        if (!row.empty()) last = row;
    if (last.empty()) return false;
    cols = split(header);
    vals = split(last);
    return true;
}

}  // namespace

std::map<std::string, double> load_tenor_years(const std::string& path) {
    std::map<std::string, double> m;
    std::ifstream f(path);
    std::string line;
    std::getline(f, line);  // header
    while (std::getline(f, line)) {
        auto v = split(line);
        if (v.size() >= 2 && !v[0].empty() && !v[1].empty()) m[v[0]] = std::stod(v[1]);
    }
    return m;
}

bool load_latest_curve(const std::string& csv, const std::string& tenor_csv,
                       std::vector<double>& years, std::vector<double>& pars,
                       std::string& date) {
    auto tmap = load_tenor_years(tenor_csv);
    std::vector<std::string> cols, vals;
    if (!last_row(csv, cols, vals)) return false;
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

bool load_latest_frn(const std::string& csv, FrnQuote& q) {
    std::vector<std::string> cols, v;
    if (!last_row(csv, cols, v) || v.size() < 6) return false;
    q.cusip = v[1];
    q.maturity_date = v[2];
    q.maturity_years = std::stod(v[3]);
    q.spread = std::stod(v[4]) / 100.0;      // percent -> decimal
    q.index_rate = std::stod(v[5]) / 100.0;
    return true;
}

bool load_securities(const std::string& csv, std::vector<SecurityQuote>& out,
                     std::string& date) {
    std::ifstream f(csv);
    if (!f) return false;
    std::string line;
    std::getline(f, line);  // header:
    // date,type,term,cusip,coupon,maturity_date,maturity_years,quote_yield,...
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto v = split(line);
        if (v.size() < 8) continue;
        SecurityQuote s;
        date = v[0];
        s.type = v[1];
        s.term = v[2];
        s.cusip = v[3];
        s.coupon = std::stod(v[4]) / 100.0;         // percent -> decimal
        s.maturity_date = v[5];
        s.maturity_years = std::stod(v[6]);
        s.quote_yield = std::stod(v[7]) / 100.0;
        out.push_back(s);
    }
    return !out.empty();
}

}  // namespace tbp::io
