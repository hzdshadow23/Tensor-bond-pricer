// Treasury sector taxonomy: the concrete product types issued by the U.S.
// Treasury, mapped onto the library's two pricing engines.
//
//   TBill  4/6/8/13/17/26/52-week   zero-coupon discount security -> FixedRateBond
//   TNote  2/3/5/7/10-year          semiannual fixed coupon        -> FixedRateBond
//   TBond  20/30-year               semiannual fixed coupon        -> FixedRateBond
//   TFRN   2-year floater           quarterly, 13-week-bill index  -> FloatingRateNote
//   TIPS   5/10/30-year             CPI-indexed principal          -> placeholder (below)
//
// An FRN is not a separate asset class — it is a bond whose coupons float;
// it prices through FloatingRateNote (dual-curve) instead of FixedRateBond.
// Muni and corporate sectors get sibling directories (instruments/muni/,
// instruments/corporate/) with their own product types when they come into
// scope; the Treasury types here are the template.
#pragma once
#include "tbp/instruments/bond.hpp"
#include "tbp/instruments/frn.hpp"

namespace tbp {
namespace treasury {

// Bills are zero-coupon: coupon_rate = 0, quoted on a discount basis
// (fetch_securities.py stores both the discount rate and the
// coupon-equivalent yield; the bootstrap consumes the yield).
using TBill = FixedRateBond;
inline TBill make_bill(double face, double maturity_years) {
    return TBill{face, 0.0, /*freq=*/1, maturity_years};
}

// Notes (2/3/5/7/10Y) and bonds (20/30Y) are the same instrument shape —
// semiannual fixed coupon, bullet — differing only in original term.
using TNote = FixedRateBond;
using TBond = FixedRateBond;

// 2-year floater indexed to the 13-week bill auction rate (see frn.hpp for
// the reset terms: weekly resets, quarterly pay, lockout).
using TFRN = FloatingRateNote;

// TIPS placeholder: principal indexed to CPI (index_ratio), real coupon rate.
// NOT priceable yet BY DESIGN — pricing needs the real-yield curve
// (YC_TSY_REAL) plus an inflation carry assumption, so no price() overload
// exists; that prevents a TIPS from being silently mispriced off the nominal
// curve. Scoped future work alongside YC_TSY_REAL.
struct TIPS {
    double face = 100.0;
    double coupon_rate = 0.0;     // REAL coupon rate (decimal)
    int freq = 2;
    double maturity_years = 0.0;
    double index_ratio = 1.0;     // CPI index ratio applied to principal
    BondTerms terms{};
};

}  // namespace treasury
}  // namespace tbp
