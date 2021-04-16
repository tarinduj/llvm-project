//===- Fraction.h - MLIR Fraction Class -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a simple class to represent fractions. It supports multiplication,
// comparison, floor, and ceiling operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_FRACTION_H
#define MLIR_ANALYSIS_PRESBURGER_FRACTION_H

#include "mlir/Analysis/Presburger/SafeInteger.h"
#include "mlir/Support/MathExtras.h"

namespace mlir {
namespace analysis {
namespace presburger {

/// A class to represent fractions. The sign of the fraction is represented
/// in the sign of the numerator; the denominator is always positive.
///
/// Note that overflows may occur if the numerator or denominator are not
/// representable by 64-bit integers.
template <typename Int>
struct Fraction {
  /// Default constructor initializes the represented rational number to zero.
  Fraction() : num(0), den(1) {}

  /// Construct a Fraction from a numerator and denominator.
  Fraction(Int oNum, Int oDen) : num(oNum), den(oDen) {
    if (den < 0) {
      num = -num;
      den = -den;
    }
  }

  /// The numerator and denominator, respectively. The denominator is always
  /// positive.
  Int num, den;
};

template <typename Int>
inline int sign(Int x) {
  if (x > 0)
    return +1;
  if (x < 0)
    return -1;
  return 0;
}

/// Three-way comparison between two fractions.
/// Returns +1, 0, and -1 if the first fraction is greater than, equal to, or
/// less than the second fraction, respectively.
template <typename Int>
inline int compare(Fraction<Int> x, Fraction<Int> y) {
  Int p = x.num * y.den;
  assert(sign(p) == sign(x.num) * sign(y.den));
  Int q = y.num * x.den;
  assert(sign(q) == sign(y.num) * sign(x.den));
  Int diff = p - q;
  return sign(diff);
}
template <typename Int>
inline int compare(Fraction<Int> x, Int y) {
  return compare(x, Fraction<Int>(y, 1));
}
template <typename Int>
inline int compare(Int x, Fraction<Int> y) {
  return compare(Fraction<Int>(x, 1), y);
}

template <typename Int>
inline Int floor(Fraction<Int> f) { return floorDiv(f.num, f.den); }
template <typename Int>
inline Int ceil(Fraction<Int> f) { return ceilDiv(f.num, f.den); }
template <typename Int>
inline Fraction<Int> operator-(Fraction<Int> x) { return Fraction<Int>(-x.num, x.den); }
template <typename Int>
inline bool operator<(Fraction<Int> x, Fraction<Int> y) { return compare(x, y) < 0; }
template <typename Int>
inline bool operator<=(Fraction<Int> x, Fraction<Int> y) { return compare(x, y) <= 0; }
template <typename Int>
inline bool operator==(Fraction<Int> x, Fraction<Int> y) { return compare(x, y) == 0; }
template <typename Int>
inline bool operator!=(Fraction<Int> x, Fraction<Int> y) { return compare(x, y) != 0; }
template <typename Int>
inline bool operator>(Fraction<Int> x, Fraction<Int> y) { return compare(x, y) > 0; }
template <typename Int>
inline bool operator>=(Fraction<Int> x, Fraction<Int> y) { return compare(x, y) >= 0; }

template <typename Int>
inline bool operator<(Fraction<Int> x, Int y) { return compare(x, y) < 0; }
template <typename Int>
inline bool operator<=(Fraction<Int> x, Int y) { return compare(x, y) <= 0; }
template <typename Int>
inline bool operator==(Fraction<Int> x, Int y) { return compare(x, y) == 0; }
template <typename Int>
inline bool operator!=(Fraction<Int> x, Int y) { return compare(x, y) != 0; }
template <typename Int>
inline bool operator>(Fraction<Int> x, Int y) { return compare(x, y) > 0; }
template <typename Int>
inline bool operator>=(Fraction<Int> x, Int y) { return compare(x, y) >= 0; }

template <typename Int>
inline bool operator<(Int x, Fraction<Int> y) { return compare(x, y) < 0; }
template <typename Int>
inline bool operator<=(Int x, Fraction<Int> y) { return compare(x, y) <= 0; }
template <typename Int>
inline bool operator==(Int x, Fraction<Int> y) { return compare(x, y) == 0; }
template <typename Int>
inline bool operator!=(Int x, Fraction<Int> y) { return compare(x, y) != 0; }
template <typename Int>
inline bool operator>(Int x, Fraction<Int> y) { return compare(x, y) > 0; }
template <typename Int>
inline bool operator>=(Int x, Fraction<Int> y) { return compare(x, y) >= 0; }

template <typename Int>
inline Fraction<Int> operator*(Fraction<Int> x, Fraction<Int> y) {
  return Fraction<Int>(x.num * y.num, x.den * y.den);
}

} // namespace presburger
} // namespace analysis
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_FRACTION_H
