//===- SafeInteger.h - MLIR SafeInteger Class -------------------*- C++ -*-===//
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

#ifndef MLIR_ANALYSIS_PRESBURGER_SAFE_INTEGER_H
#define MLIR_ANALYSIS_PRESBURGER_SAFE_INTEGER_H

#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <iostream>
#include <limits>

namespace mlir {
namespace analysis {
namespace presburger {

using DefaultInt = __int128_t;

/// A class to overflow-aware 64-bit integers.
template <typename Int>
struct SafeInteger {
  /// Construct a SafeInteger from a numerator and denominator.
  SafeInteger(int64_t oVal) {
    if constexpr (std::is_same<Int, __int128_t>::value) {
      val = oVal;
    } else {
      Int min = std::numeric_limits<Int>::min();
      Int max = std::numeric_limits<Int>::max();
      throwOverflowIf(oVal < min);
      throwOverflowIf(max < oVal);
      val = oVal;
    }
  }

  /// Default constructor initializes the number to zero.
  SafeInteger() : SafeInteger(0) {}

  inline explicit operator bool();

  /// The stored value.
  Int val;

  static bool overflow;

  static void throwOverflowIf(bool cond) {
    if (cond)
      overflow = true;
  }
};

template <typename Int>
bool SafeInteger<Int>::overflow = false;

template <typename Int>
inline bool operator<(const SafeInteger<Int> &x, const SafeInteger<Int> &y) {
  return x.val < y.val;
}
template <typename Int>
inline bool operator<=(const SafeInteger<Int> &x, const SafeInteger<Int> &y) {
  return x.val <= y.val;
}
template <typename Int>
inline bool operator==(const SafeInteger<Int> &x, const SafeInteger<Int> &y) {
  return x.val == y.val;
}
template <typename Int>
inline bool operator!=(const SafeInteger<Int> &x, const SafeInteger<Int> &y) {
  return x.val != y.val;
}
template <typename Int>
inline bool operator>(const SafeInteger<Int> &x, const SafeInteger<Int> &y) {
  return x.val > y.val;
}
template <typename Int>
inline bool operator>=(const SafeInteger<Int> &x, const SafeInteger<Int> &y) {
  return x.val >= y.val;
}

template <typename Int>
inline bool operator<(const SafeInteger<Int> &x, int y) {
  return x.val < y;
}
template <typename Int>
inline bool operator<=(const SafeInteger<Int> &x, int y) {
  return x.val <= y;
}
template <typename Int>
inline bool operator==(const SafeInteger<Int> &x, int y) {
  return x.val == y;
}
template <typename Int>
inline bool operator!=(const SafeInteger<Int> &x, int y) {
  return x.val != y;
}
template <typename Int>
inline bool operator>(const SafeInteger<Int> &x, int y) {
  return x.val > y;
}
template <typename Int>
inline bool operator>=(const SafeInteger<Int> &x, int y) {
  return x.val >= y;
}

template <typename Int>
inline SafeInteger<Int> operator+(const SafeInteger<Int> &x, const SafeInteger<Int> &y) {
  Int result;
  bool overflow = __builtin_add_overflow(x.val, y.val, &result);
  SafeInteger<Int>::throwOverflowIf(overflow);
  return SafeInteger<Int>(result);
}

template <typename Int>
inline SafeInteger<Int> operator-(const SafeInteger<Int> &x, const SafeInteger<Int> &y) {
  Int result;
  bool overflow = __builtin_sub_overflow(x.val, y.val, &result);
  SafeInteger<Int>::throwOverflowIf(overflow);
  return SafeInteger<Int>(result);
}

template <typename Int>
inline SafeInteger<Int> operator-(const SafeInteger<Int> &x) {
  return SafeInteger<Int>(0) - x;
}

template <typename Int>
inline SafeInteger<Int> operator*(const SafeInteger<Int> &x, const SafeInteger<Int> &y) {
  Int result;
  bool overflow = __builtin_mul_overflow(x.val, y.val, &result);
  SafeInteger<Int>::throwOverflowIf(overflow);
  return SafeInteger<Int>(result);
}

template <typename Int>
inline SafeInteger<Int> operator/(const SafeInteger<Int> &x, const SafeInteger<Int> &y) {
  // overflow only possible if y == -1
  if (y.val == -1)
    return -x;
  // Divide by zeros should only occur due to overflows. We return some garbage in such a case.
  if (y.val == 0) {
    assert(SafeInteger<Int>::overflow);
    return 0;
  }
  return x.val / y.val;
}

template <typename Int>
inline SafeInteger<Int> operator%(const SafeInteger<Int> &x, const SafeInteger<Int> &y) {
  // The denominator should only become zero if an overflow occurred, in which case we return some garbage.
  if (y.val == 0) {
    assert(SafeInteger<Int>::overflow);
    return x.val;
  }
  return x.val % y.val;
}

template <typename Int>
inline void operator+=(SafeInteger<Int> &x, const SafeInteger<Int> &y) { x = x + y; }

template <typename Int>
inline void operator-=(SafeInteger<Int> &x, const SafeInteger<Int> &y) { x = x - y; }

template <typename Int>
inline void operator*=(SafeInteger<Int> &x, const SafeInteger<Int> &y) { x = x * y; }

template <typename Int>
inline void operator/=(SafeInteger<Int> &x, const SafeInteger<Int> &y) { x = x / y; }

template <typename Int>
inline void operator%=(SafeInteger<Int> &x, const SafeInteger<Int> &y) { x = x % y; }

template <typename Int>
inline void operator+=(SafeInteger<Int> &x, int y) { x = x + SafeInteger<Int>(y); }

template <typename Int>
inline void operator-=(SafeInteger<Int> &x, int y) { x = x - SafeInteger<Int>(y); }

template <typename Int>
inline void operator*=(SafeInteger<Int> &x, int y) { x = x * SafeInteger<Int>(y); }

template <typename Int>
inline void operator/=(SafeInteger<Int> &x, int y) { x = x / SafeInteger<Int>(y); }

template <typename Int>
inline void operator%=(SafeInteger<Int> &x, int y) { x = x % SafeInteger<Int>(y); }

template <typename Int>
inline SafeInteger<Int> operator+(const SafeInteger<Int> &x, int y) { return x + SafeInteger<Int>(y); }

template <typename Int>
inline SafeInteger<Int> operator-(const SafeInteger<Int> &x, int y) { return x - SafeInteger<Int>(y); }

template <typename Int>
inline SafeInteger<Int> operator*(const SafeInteger<Int> &x, int y) { return x * SafeInteger<Int>(y); }

template <typename Int>
inline SafeInteger<Int> operator/(const SafeInteger<Int> &x, int y) { return x / SafeInteger<Int>(y); }

template <typename Int>
inline SafeInteger<Int> operator%(const SafeInteger<Int> &x, int y) { return x % SafeInteger<Int>(y); }
/// Returns MLIR's mod operation on constants. MLIR's mod operation yields the
/// remainder of the Euclidean division of 'lhs' by 'rhs', and is therefore not
/// C's % operator.  The RHS is always expected to be positive, and the result
/// is always non-negative.
template <typename Int>
inline SafeInteger<Int> mod(SafeInteger<Int> lhs, SafeInteger<Int> rhs) {
  assert(rhs >= 1);
  return lhs % rhs < 0 ? lhs % rhs + rhs : lhs % rhs;
}

template <typename Int>
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const SafeInteger<Int> &x) {
  if (x == 0) {
    os << "0";
    return os;
  }
  std::string out;
  auto copy = x;
  if (copy < 0) {
    os << '-';
    copy = -copy;
  }
  while (copy > 0) {
    out.push_back('0' + int(copy.val % 10));
    copy /= SafeInteger<Int>(10);
  }
  std::reverse(out.begin(), out.end());

  os << out;
  return os;
}

template <typename Int>
inline std::ostream &operator<<(std::ostream &os, const SafeInteger<Int> &x) {
  os << "[SafeInteger<Int>::operator<<(std::ostream) NYI";
  // os << x.val;
  return os;
}

template <typename Int>
inline SafeInteger<Int> ceilDiv(SafeInteger<Int> lhs, SafeInteger<Int> rhs) {
  assert(rhs >= 1);
  return lhs % rhs > 0 ? lhs / rhs + 1 : lhs / rhs;
}

template <typename Int>
inline SafeInteger<Int> floorDiv(SafeInteger<Int> lhs, SafeInteger<Int> rhs) {
  assert(rhs >= 1);
  return lhs % rhs < 0 ? lhs / rhs - 1 : lhs / rhs;
}

template <typename Int>
inline SafeInteger<Int>::operator bool() { return *this != 0; }

} // namespace presburger
} // namespace analysis
} // namespace mlir

namespace std {

template <typename Int>
using SafeInteger = mlir::analysis::presburger::SafeInteger<Int>;

template <typename Int>
inline SafeInteger<Int> abs(SafeInteger<Int> x) { return x < 0 ? -x : x; }

/// Returns the least common multiple of 'a' and 'b'.
template <typename Int>
inline SafeInteger<Int> lcm(SafeInteger<Int> a, SafeInteger<Int> b) {
  SafeInteger<Int> x = abs(a);
  SafeInteger<Int> y = abs(b);
  SafeInteger<Int> lcm = (x * y) / llvm::greatestCommonDivisor(x, y);
  return lcm;
}
} // namespace std

#endif // MLIR_ANALYSIS_PRESBURGER_SAFE_INTEGER_H
