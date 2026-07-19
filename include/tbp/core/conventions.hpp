// Market conventions: calendar dates, day counts, business-day rolls, holiday
// calendar, end-of-month rule, and date-based schedule generation.
//
// This is the bridge from real bond terms to the tensor engine: the pricing
// path stays in year-fractions (differentiable, batched), and this layer is
// where those year-fractions come FROM. build_coupon_dates() applies pay
// frequency + EOM + holiday roll to produce actual payment dates;
// year_fractions() converts them to times under a day count.
//
// v0 approximations (documented; refine as the calendar work deepens):
//  - US holiday calendar covers the federal holidays with Sat->Fri / Sun->Mon
//    observance; SIFMA's Good Friday is not yet included.
//  - ACT/ACT is approximated as ACT/365.25 (proper ISDA period-splitting is a
//    scoped follow-up).
// Header-only on purpose: no torch dependency, usable from io and python glue.
#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace tbp {

// ---------------------------------------------------------------------------
// Date: proleptic Gregorian, stored as a serial day count (1970-01-01 == 0).
// ---------------------------------------------------------------------------
enum class Weekday { Sun = 0, Mon, Tue, Wed, Thu, Fri, Sat };

struct Date {
    int y = 1970, m = 1, d = 1;

    // Howard Hinnant's civil-days algorithm.
    int64_t serial() const {
        int yy = y - (m <= 2);
        const int era = (yy >= 0 ? yy : yy - 399) / 400;
        const unsigned yoe = static_cast<unsigned>(yy - era * 400);
        const unsigned doy = (153u * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
        const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        return era * 146097LL + static_cast<int64_t>(doe) - 719468;
    }
    static Date from_serial(int64_t z) {
        z += 719468;
        const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
        const unsigned doe = static_cast<unsigned>(z - era * 146097);
        const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        const int yy = static_cast<int>(yoe) + static_cast<int>(era) * 400;
        const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        const unsigned mp = (5 * doy + 2) / 153;
        const unsigned dd = doy - (153 * mp + 2) / 5 + 1;
        const unsigned mm = mp + (mp < 10 ? 3 : -9);
        return Date{yy + (mm <= 2), static_cast<int>(mm), static_cast<int>(dd)};
    }

    static bool is_leap(int y) { return y % 4 == 0 && (y % 100 != 0 || y % 400 == 0); }
    static int days_in_month(int y, int m) {
        static constexpr std::array<int, 12> n{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        return m == 2 && is_leap(y) ? 29 : n[m - 1];
    }

    Weekday weekday() const {
        return static_cast<Weekday>(((serial() % 7) + 11) % 7);  // 1970-01-01 = Thu
    }
    bool is_eom() const { return d == days_in_month(y, m); }

    Date add_days(int n) const { return from_serial(serial() + n); }

    // Month arithmetic with the end-of-month rule: if `eom` and the start date
    // is the last day of its month, the result is the last day of the target
    // month; otherwise the day-of-month is kept (clamped to month length).
    Date add_months(int n, bool eom = false) const {
        int idx = y * 12 + (m - 1) + n;
        int yy = idx / 12, mm = idx % 12 + 1;
        if (idx < 0 && idx % 12 != 0) { yy -= 1; mm = idx % 12 + 13; }
        int dd = (eom && is_eom()) ? days_in_month(yy, mm)
                                   : std::min(d, days_in_month(yy, mm));
        return Date{yy, mm, dd};
    }

    bool operator==(const Date& o) const { return y == o.y && m == o.m && d == o.d; }
    bool operator<(const Date& o) const { return serial() < o.serial(); }
    bool operator<=(const Date& o) const { return serial() <= o.serial(); }

    std::string iso() const {
        char b[16];
        std::snprintf(b, sizeof b, "%04d-%02d-%02d", y, m, d);
        return b;
    }
    static Date from_iso(const std::string& s) {  // "YYYY-MM-DD"
        if (s.size() < 10) throw std::invalid_argument("Date::from_iso: " + s);
        return Date{std::stoi(s.substr(0, 4)), std::stoi(s.substr(5, 2)),
                    std::stoi(s.substr(8, 2))};
    }
};

// ---------------------------------------------------------------------------
// Holiday calendar. US = federal holidays with weekend observance shifts.
// ---------------------------------------------------------------------------
enum class Calendar { NONE, US };

namespace detail {
inline Date nth_weekday(int y, int m, Weekday wd, int nth) {  // nth >= 1
    Date first{y, m, 1};
    int delta = (static_cast<int>(wd) - static_cast<int>(first.weekday()) + 7) % 7;
    return first.add_days(delta + 7 * (nth - 1));
}
inline Date last_weekday(int y, int m, Weekday wd) {
    Date last{y, m, Date::days_in_month(y, m)};
    int delta = (static_cast<int>(last.weekday()) - static_cast<int>(wd) + 7) % 7;
    return last.add_days(-delta);
}
inline Date observed(Date h) {  // Sat -> Fri, Sun -> Mon
    if (h.weekday() == Weekday::Sat) return h.add_days(-1);
    if (h.weekday() == Weekday::Sun) return h.add_days(1);
    return h;
}
}  // namespace detail

inline bool is_us_holiday(const Date& dt) {
    using namespace detail;
    const int y = dt.y;
    const Date fixed[] = {observed({y, 1, 1}),  observed({y, 6, 19}),
                          observed({y, 7, 4}),  observed({y, 11, 11}),
                          observed({y, 12, 25})};
    for (const auto& h : fixed)
        if (dt == h) return true;
    return dt == nth_weekday(y, 1, Weekday::Mon, 3)    // MLK
        || dt == nth_weekday(y, 2, Weekday::Mon, 3)    // Presidents
        || dt == last_weekday(y, 5, Weekday::Mon)      // Memorial
        || dt == nth_weekday(y, 9, Weekday::Mon, 1)    // Labor
        || dt == nth_weekday(y, 10, Weekday::Mon, 2)   // Columbus
        || dt == nth_weekday(y, 11, Weekday::Thu, 4);  // Thanksgiving
}

inline bool is_business_day(const Date& dt, Calendar cal = Calendar::US) {
    auto wd = dt.weekday();
    if (wd == Weekday::Sat || wd == Weekday::Sun) return false;
    return cal == Calendar::NONE || !is_us_holiday(dt);
}

// ---------------------------------------------------------------------------
// Business-day roll.
// ---------------------------------------------------------------------------
enum class BusinessDayRoll { NONE, FOLLOWING, MODIFIED_FOLLOWING, PRECEDING };

inline Date adjust(Date dt, BusinessDayRoll roll, Calendar cal = Calendar::US) {
    if (roll == BusinessDayRoll::NONE) return dt;
    Date out = dt;
    int dir = (roll == BusinessDayRoll::PRECEDING) ? -1 : 1;
    while (!is_business_day(out, cal)) out = out.add_days(dir);
    if (roll == BusinessDayRoll::MODIFIED_FOLLOWING && out.m != dt.m) {
        out = dt;
        while (!is_business_day(out, cal)) out = out.add_days(-1);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Day count.
// ---------------------------------------------------------------------------
enum class DayCount { ACT_ACT, ACT_360, ACT_365F, THIRTY_360 };

inline double year_fraction(const Date& d1, const Date& d2, DayCount dc) {
    const double days = static_cast<double>(d2.serial() - d1.serial());
    switch (dc) {
        case DayCount::ACT_360:  return days / 360.0;
        case DayCount::ACT_365F: return days / 365.0;
        case DayCount::THIRTY_360: {  // US (Bond Basis)
            int dd1 = std::min(d1.d, 30);
            int dd2 = (d2.d == 31 && dd1 == 30) ? 30 : d2.d;
            return (360.0 * (d2.y - d1.y) + 30.0 * (d2.m - d1.m) + (dd2 - dd1)) / 360.0;
        }
        case DayCount::ACT_ACT:  // v0 approximation (proper ISDA is roadmap)
        default:                 return days / 365.25;
    }
}

// ---------------------------------------------------------------------------
// Date-based coupon schedules.
// ---------------------------------------------------------------------------
// Payment dates counted BACK from maturity every 12/freq months (so the final
// payment is exactly at maturity), EOM rule applied to the unadjusted dates,
// then each date rolled per the business-day convention. Only dates strictly
// after `valuation` are returned, ascending.
inline std::vector<Date> build_coupon_dates(const Date& valuation, const Date& maturity,
                                            int freq, bool eom = false,
                                            BusinessDayRoll roll = BusinessDayRoll::MODIFIED_FOLLOWING,
                                            Calendar cal = Calendar::US) {
    if (freq <= 0 || 12 % freq != 0)
        throw std::invalid_argument("build_coupon_dates: freq must divide 12");
    const int step = 12 / freq;
    std::vector<Date> out;
    for (int k = 0;; ++k) {
        Date unadj = maturity.add_months(-step * k, eom);
        if (unadj <= valuation) break;
        out.push_back(adjust(unadj, roll, cal));
    }
    std::vector<Date> asc(out.rbegin(), out.rend());
    return asc;
}

// Convert payment dates to the engine's year-fraction times.
inline std::vector<double> year_fractions(const Date& valuation,
                                          const std::vector<Date>& dates,
                                          DayCount dc = DayCount::ACT_ACT) {
    std::vector<double> t;
    t.reserve(dates.size());
    for (const auto& dt : dates) t.push_back(year_fraction(valuation, dt, dc));
    return t;
}

}  // namespace tbp
