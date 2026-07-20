// Named curve registry: curves are first-class objects looked up by name,
// not csv state re-parsed at each use.
//
// Curve TYPES and their naming convention (prefix encodes the type):
//   YC_<SECTOR>[_<VARIANT>]  discount (yield) curve — PVs cashflows
//   FC_<SECTOR|INDEX>        forecast (projection) curve — projects floating
//                            fixings; distinct grad leaf so discount risk and
//                            projection risk separate cleanly in autograd
//   future: real-yield curves (TIPS), fitted sector SPREAD curves over
//   Treasury — same DiscountCurve machinery, reserved names below.
//
// The registry hands out references, so the stored curve's node_zeros grad
// leaf is shared with every user — pricing off store.get("YC_TSY") produces
// key-rate risk on the same tensor the curve was registered with.
#pragma once
#include <map>
#include <string>
#include <vector>

#include "tbp/core/curve.hpp"

namespace tbp {

// Well-known curve names. Use these constants instead of string literals so
// a typo is a compile error, not an empty curve lookup.
namespace curves {
// Registered today by the demo pipeline:
inline constexpr const char* YC_TSY      = "YC_TSY";      // Treasury discount, bootstrapped from on-the-run quotes
inline constexpr const char* YC_TSY_PAR  = "YC_TSY_PAR";  // Treasury discount, from the published par grid
inline constexpr const char* FC_TSY      = "FC_TSY";      // forecast curve projecting the TFRN 13-week bill index
// Reserved placeholders for the sector roadmap (not yet built):
inline constexpr const char* YC_TSY_REAL = "YC_TSY_REAL"; // Treasury REAL-yield curve (TIPS pricing)
inline constexpr const char* YC_MUNI     = "YC_MUNI";     // muni discount curve (tax-adjusted spread over YC_TSY)
inline constexpr const char* YC_CORP     = "YC_CORP";     // corporate discount curve (credit spread over YC_TSY)
inline constexpr const char* FC_SOFR     = "FC_SOFR";     // SOFR forecast curve (corporate floaters)
}  // namespace curves

class CurveStore {
public:
    // Process-wide registry (a plain instance can be made for tests).
    static CurveStore& instance();

    // Register under curve.name(); throws if the name is empty. Replaces any
    // existing curve with the same name (a fresh day's build supersedes).
    void add(DiscountCurve curve);

    // Lookup by name; throws std::out_of_range with the known names listed.
    DiscountCurve& get(const std::string& name);

    bool has(const std::string& name) const { return curves_.count(name) > 0; }
    std::vector<std::string> names() const;

private:
    std::map<std::string, DiscountCurve> curves_;
};

}  // namespace tbp
