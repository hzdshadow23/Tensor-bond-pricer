#include "tbp/core/curve_store.hpp"
#include <stdexcept>

namespace tbp {

CurveStore& CurveStore::instance() {
    static CurveStore store;
    return store;
}

void CurveStore::add(DiscountCurve curve) {
    if (curve.name().empty())
        throw std::invalid_argument("CurveStore::add: curve must be named (e.g. YC_TSY)");
    curves_.insert_or_assign(curve.name(), std::move(curve));
}

DiscountCurve& CurveStore::get(const std::string& name) {
    auto it = curves_.find(name);
    if (it == curves_.end()) {
        std::string known;
        for (const auto& [k, v] : curves_) known += (known.empty() ? "" : ", ") + k;
        throw std::out_of_range("CurveStore: no curve named '" + name +
                                "' (known: " + (known.empty() ? "<none>" : known) + ")");
    }
    return it->second;
}

std::vector<std::string> CurveStore::names() const {
    std::vector<std::string> out;
    out.reserve(curves_.size());
    for (const auto& [k, v] : curves_) out.push_back(k);
    return out;
}

}  // namespace tbp
