// Corporate sector placeholder — future work, deliberately empty of pricing
// code.
//
// When corporates come into scope (see CLAUDE.md "Scope"), this directory
// gets the corporate product types: fixed-coupon bullets as FixedRateBond
// with 30/360 conventions off YC_CORP (credit spread curve over YC_TSY, per
// rating/sector bucket eventually), corporate floaters as FloatingRateNote
// off an FC_SOFR forecast curve, and callables behind the option module.
#pragma once
#include "tbp/instruments/bond.hpp"

namespace tbp {
namespace corporate {

// using CorpBond = FixedRateBond;       // unlocked when YC_CORP exists
// using CorpFRN  = FloatingRateNote;    // unlocked when FC_SOFR exists

}  // namespace corporate
}  // namespace tbp
