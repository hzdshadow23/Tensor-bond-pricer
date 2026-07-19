// Named curve registry: curves are first-class objects looked up by name,
// not csv state re-parsed at each use.
//
// Naming convention:
//   YC_<SECTOR>   discount curve  (YC_TSY today; YC_MUNI / YC_CORP later)
//   FC_<SECTOR>   forecast (projection) curve (FC_TSY for the TFRN index)
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
