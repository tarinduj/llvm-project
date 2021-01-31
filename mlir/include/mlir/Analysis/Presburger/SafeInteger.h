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

namespace mlir {
namespace analysis {
namespace presburger {
using llvm::APInt;

/// A class to overflow-aware 64-bit integers.
///
/// Overflows are asserted to not occur.
struct SafeInteger {
  /// Construct a SafeInteger from a numerator and denominator.
  explicit SafeInteger(const __int128_t &x, int64_t y, uint8_t state) : val128(x), val64(y), state(128) {}
  static SafeInteger make128(const __int128_t &oVal) const { return SafeInteger{oVal, 0, 128}; }
  SafeInteger(int64_t oVal) : val64(oVal), state(64) {}

  /// Default constructor initializes the number to zero.
  SafeInteger() : SafeInteger(0) {}

  inline explicit operator bool();

  /// The stored value. This is always 64-bit.
  __int128_t val128;
  int64_t val64;
  uint8_t state;
};

inline void overflowErrorIf(bool overflow) {
  if (overflow) {
    llvm::errs() << "Overflow!\n";
    abort();
  }
}

template <typename T>
inline int sign(const T &x) {
  if (x == 0)
    return 0;
  return x > 0 ? +1 : -1;
}

inline int compare(const SafeInteger &x, const SafeInteger &y) {
  if (x.state == 128) {
    if (y.state == 128) {
      return sign(x.val128 - y.val128);
    } else {
      return sign(x.val128 - y.val64);
    }
  } else {
    if (y.state == 128) {
      return sign(x.val64 - y.val128);
    } else {
      return sign(x.val64 - y.val64);
    }
  }
}

inline bool operator<(const SafeInteger &x, const SafeInteger &y) {
  return compare(x, y) < 0;
}
inline bool operator<=(const SafeInteger &x, const SafeInteger &y) {
  return compare(x, y) <= 0;
}
inline bool operator==(const SafeInteger &x, const SafeInteger &y) {
  return compare(x, y) == 0;
}
inline bool operator!=(const SafeInteger &x, const SafeInteger &y) {
  return compare(x, y) != 0;
}
inline bool operator>(const SafeInteger &x, const SafeInteger &y) {
  return compare(x, y) > 0;
}
inline bool operator>=(const SafeInteger &x, const SafeInteger &y) {
  return compare(x, y) >= 0;
}

inline SafeInteger operator+(const SafeInteger &x, const SafeInteger &y) {
  if (x.state == 64) {
    if (y.state == 64) {
      int64_t result;
      bool overflow = __builtin_add_overflow(x.val64, y.val64, &result);
      if (overflow)
        return SafeInteger::make128(__int128_t(x.val64) + __int128_t(y.val64));
      return SafeInteger(result);
    } else {
      __int128_t result;
      bool overflow = __builtin_add_overflow(x.val64, y.val128, &result);
      overflowErrorIf(overflow);
      return SafeInteger(result);
    }
  } else {
    if (y.state == 64) {
      __int128_t result;
      bool overflow = __builtin_add_overflow(x.val128, y.val64, &result);
      overflowErrorIf(overflow);
      return SafeInteger(result);
    } else {
      __int128_t result;
      bool overflow = __builtin_add_overflow(x.val128, y.val128, &result);
      overflowErrorIf(overflow);
      return SafeInteger(result);
    }
  }
}


inline SafeInteger operator-(const SafeInteger &x, const SafeInteger &y) {
  if (x.state == 64) {
    if (y.state == 64) {
      int64_t result;
      bool overflow = __builtin_sub_overflow(x.val64, y.val64, &result);
      if (overflow)
        return SafeInteger::make128(__int128_t(x.val64) - __int128_t(y.val64));
      return SafeInteger(result);
    } else {
      __int128_t result;
      bool overflow = __builtin_sub_overflow(x.val64, y.val128, &result);
      overflowErrorIf(overflow);
      return SafeInteger(result);
    }
  } else {
    if (y.state == 64) {
      __int128_t result;
      bool overflow = __builtin_sub_overflow(x.val128, y.val64, &result);
      overflowErrorIf(overflow);
      return SafeInteger(result);
    } else {
      __int128_t result;
      bool overflow = __builtin_sub_overflow(x.val128, y.val128, &result);
      overflowErrorIf(overflow);
      return SafeInteger(result);
    }
  }
}

inline SafeInteger operator-(const SafeInteger &x) {
  return SafeInteger(0) - x;
}

inline SafeInteger operator*(const SafeInteger &x, const SafeInteger &y) {
  if (x.state == 64) {
    if (y.state == 64) {
      int64_t result;
      bool overflow = __builtin_mul_overflow(x.val64, y.val64, &result);
      if (overflow)
        return SafeInteger::make128(__int128_t(x.val64) * __int128_t(y.val64));
      return SafeInteger(result);
    } else {
      __int128_t result;
      bool overflow = __builtin_mul_overflow(x.val64, y.val128, &result);
      overflowErrorIf(overflow);
      return SafeInteger(result);
    }
  } else {
    if (y.state == 64) {
      __int128_t result;
      bool overflow = __builtin_mul_overflow(x.val128, y.val64, &result);
      overflowErrorIf(overflow);
      return SafeInteger(result);
    } else {
      __int128_t result;
      bool overflow = __builtin_mul_overflow(x.val128, y.val128, &result);
      overflowErrorIf(overflow);
      return SafeInteger(result);
    }
  }
}

inline SafeInteger operator/(const SafeInteger &x, const SafeInteger &y) {
  // overflow only possible if y == -1
  if (x.state == 64) {
    if (y.state == 64) {
      if (y.val64 == -1)
        return -x;
      return x.val64 % y.val64;
    } else {
      if (y.val128 == -1)
        return -x;
      return x.val64 % y.val128;
    }
  } else {
    if (y.state == 64) {
      if (y.val64 == -1)
        return -x;
      return x.val128 % y.val64;
    } else {
      if (y.val128 == -1)
        return -x;
      return x.val128 % y.val128;
    }
  }
}

inline SafeInteger operator%(const SafeInteger &x, const SafeInteger &y) {
  if (x.state == 64) {
    if (y.state == 64) {
      return x.val64 % y.val64;
    } else {
      return x.val64 % y.val128;
    }
  } else {
    if (y.state == 64) {
      return x.val128 % y.val64;
    } else {
      return x.val128 % y.val128;
    }
  }
}

inline void operator+=(SafeInteger &x, const SafeInteger &y) {
  x = x + y;
}
inline void operator-=(SafeInteger &x, const SafeInteger &y) {
  x = x - y;
}
inline void operator*=(SafeInteger &x, const SafeInteger &y) {
  x = x * y;
}
inline void operator/=(SafeInteger &x, const SafeInteger &y) {
  x = x / y;
}
inline void operator%=(SafeInteger &x, const SafeInteger &y) {
  x = x % y;
}

inline SafeInteger abs(const SafeInteger &x) { return x < 0 ? -x : x; }

/// Returns the least common multiple of 'a' and 'b'.
inline SafeInteger lcm(SafeInteger a, SafeInteger b) {
  SafeInteger x = abs(a);
  SafeInteger y = abs(b);
  SafeInteger lcm = (x * y) / llvm::greatestCommonDivisor(x, y);
  return lcm;
}

/// Returns MLIR's mod operation on constants. MLIR's mod operation yields the
/// remainder of the Euclidean division of 'lhs' by 'rhs', and is therefore not
/// C's % operator.  The RHS is always expected to be positive, and the result
/// is always non-negative.
inline SafeInteger mod(SafeInteger lhs, SafeInteger rhs) {
  assert(rhs >= 1);
  return lhs % rhs < 0 ? lhs % rhs + rhs : lhs % rhs;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const SafeInteger &x) {
  std::string out;
  auto copy = x;
  if (copy < 0) {
    os << '-';
    copy = -copy;
  }
  while (copy > 0) {
    out.push_back('0' + int(copy.state == 64 ? copy.val64 % 10 : copy.val128 % 10));
    copy /= 10;
  }
  std::reverse(out.begin(), out.end());

  os << out;
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const SafeInteger &x) {
  os << "[SafeInteger::operator<<(std::ostream) NYI";
  // os << x.val;
  return os;
}

inline SafeInteger ceilDiv(SafeInteger lhs, SafeInteger rhs) {
  assert(rhs >= 1);
  return lhs % rhs > 0 ? lhs / rhs + 1 : lhs / rhs;
}

inline SafeInteger floorDiv(SafeInteger lhs, SafeInteger rhs) {
  assert(rhs >= 1);
  return lhs % rhs < 0 ? lhs / rhs - 1 : lhs / rhs;
}

inline SafeInteger::operator bool() { return *this != 0; }

} // namespace presburger
} // namespace analysis
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_SAFE_INTEGER_H
