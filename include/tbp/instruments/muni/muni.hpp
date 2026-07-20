// Muni sector placeholder — future work, deliberately empty of pricing code.
//
// When municipals come into scope (see CLAUDE.md "Scope"), this directory
// gets the muni product types the way instruments/treasury/ has the Treasury
// ones: GO and revenue bonds as FixedRateBond aliases with muni conventions
// (30/360, EOM common), priced off the YC_MUNI discount curve (a tax-adjusted
// spread curve over YC_TSY registered in CurveStore) — plus callables behind
// the option module once it exists.
#pragma once
#include "tbp/instruments/bond.hpp"

namespace tbp {
namespace muni {

// using GoBond      = FixedRateBond;  // unlocked when YC_MUNI exists
// using RevenueBond = FixedRateBond;

}  // namespace muni
}  // namespace tbp
