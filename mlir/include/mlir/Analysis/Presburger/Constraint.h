//===- Constraint.h - MLIR Constraint Class -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Constraint class. Supports inequality, equality, and division constraints.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_CONSTRAINT_H
#define MLIR_ANALYSIS_PRESBURGER_CONSTRAINT_H

#include "mlir/Analysis/Presburger/SafeInteger.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {



using namespace mlir::analysis::presburger;

template <typename Int>
class Constraint {
public:
  Constraint() = delete;

  template <typename OInt>
  friend class Constraint;

  template <typename OInt>
  Constraint(const Constraint<OInt> &o) : coeffs(convert<Int>(o.coeffs)) {}

  unsigned getNumDims() const {
    // The last element of the coefficient vector is the constant term and does
    // not correspond to any dimension.
    return coeffs.size() - 1;
  }

  /// Insert `count` empty dimensions before the `pos`th dimension, or at the
  /// end if `pos` is equal to `getNumDims()`.
  void insertDimensions(unsigned pos, unsigned count) {
    assert(pos <= getNumDims());
    coeffs.insert(coeffs.begin() + pos, count, 0);
  }

  /// Erase `count` dimensions starting at the `pos`-th one.
  void eraseDimensions(unsigned pos, unsigned count) {
    assert(pos + count - 1 < getNumDims() &&
           "Dimension to be erased does not exist!");
    coeffs.erase(coeffs.begin() + pos, coeffs.begin() + pos + count);
  }

  ArrayRef<Int> getCoeffs() const { return coeffs; }

  void shiftToOrigin() { coeffs.back() = 0; }

  void substitute(ArrayRef<Int> values) {
    assert(values.size() <= getNumDims() && "Too many values to substitute!");
    for (size_t i = 0; i < values.size(); i++)
      coeffs.back() += values[i] * coeffs[i];

    coeffs = SmallVector<Int, 8>(coeffs.begin() + values.size(),
                                         coeffs.end());
  }

  void shift(Int x) { coeffs.back() += x; }

  void appendDimension() { insertDimensions(getNumDims(), 1); }

  void removeLastDimension() { eraseDimensions(getNumDims() - 1, 1); }

  void print(raw_ostream &os) const {
    bool first = true;
    if (coeffs.back() != 0) {
      os << coeffs.back();
      first = false;
    }

    for (unsigned i = 0; i < coeffs.size() - 1; ++i) {
      if (coeffs[i] == 0)
        continue;

      if (first) {
        if (coeffs[i] == -1)
          os << '-';
        else if (coeffs[i] != 1)
          os << coeffs[i];
        first = false;
      } else if (coeffs[i] > 0) {
        os << " + ";
        if (coeffs[i] != 1)
          os << coeffs[i];
      } else {
        os << " - " << -coeffs[i];
        if (-coeffs[i] != 1)
          os << -coeffs[i];
      }

      os << "x" << i;
    }
  }

  void dump() const { print(llvm::errs()); }

  void dumpCoeffs() const {
    for (auto coeff : coeffs) {
      llvm::errs() << coeff << ' ';
    }
    llvm::errs() << '\n';
  }

protected:
  Constraint(ArrayRef<Int> oCoeffs)
      : coeffs(oCoeffs.begin(), oCoeffs.end()) {}
  SmallVector<Int, 8> coeffs;
};

template <typename Int>
class InequalityConstraint : public Constraint<Int> {
public:
  InequalityConstraint(ArrayRef<Int> oCoeffs) : Constraint<Int>(oCoeffs) {}
  template <typename OInt>
  InequalityConstraint(const InequalityConstraint<OInt> &o) : Constraint<Int>(o) {}

  void print(raw_ostream &os) const {
    Constraint<Int>::print(os);
    os << " >= 0";
  }
  void dump() const { print(llvm::errs()); }
};

template <typename Int>
class EqualityConstraint : public Constraint<Int> {
public:
  EqualityConstraint(ArrayRef<Int> oCoeffs) : Constraint<Int>(oCoeffs) {}
  template <typename OInt>
  EqualityConstraint(const EqualityConstraint<OInt> &o) : Constraint<Int>(o) {}
  void print(raw_ostream &os) const {
    Constraint<Int>::print(os);
    os << " = 0";
  }
  void dump() const { print(llvm::errs()); }
};

template <typename Int>
class DivisionConstraint : public Constraint<Int> {
using Constraint<Int>::coeffs;
public:
  DivisionConstraint(ArrayRef<Int> oCoeffs, Int oDenom,
                     unsigned oVariable)
      : Constraint<Int>(oCoeffs), denom(oDenom), variable(oVariable) {}
  void print(raw_ostream &os) const {
    os << "x" << variable << " = floor((";
    Constraint<Int>::print(os);
    os << ")/" << denom << ')';
  }

  template <typename OInt>
  friend class DivisionConstraint;

  template <typename OInt>
  DivisionConstraint(const DivisionConstraint<OInt> &o) : Constraint<Int>(o), denom(o.denom), variable(o.variable) {}

  Int getDenominator() const { return denom; }

  InequalityConstraint<Int> getInequalityLowerBound() const {
    SmallVector<Int, 8> ineqCoeffs = coeffs;
    ineqCoeffs[variable] -= denom;
    return InequalityConstraint<Int>(ineqCoeffs);
  }

  InequalityConstraint<Int> getInequalityUpperBound() const {
    SmallVector<Int, 8> ineqCoeffs;
    ineqCoeffs.reserve(coeffs.size());
    for (Int coeff : coeffs)
      ineqCoeffs.push_back(-coeff);
    ineqCoeffs[variable] += denom;
    ineqCoeffs.back() += denom - 1;
    return InequalityConstraint<Int>(ineqCoeffs);
  }

  void insertDimensions(unsigned pos, unsigned count) {
    if (pos <= variable)
      variable += count;
    Constraint<Int>::insertDimensions(pos, count);
  }

  void eraseDimensions(unsigned pos, unsigned count) {
    assert(!(pos <= variable && variable < pos + count) &&
           "cannot erase division variable!");
    Constraint<Int>::eraseDimensions(pos, count);
  }

  void substitute(ArrayRef<Int> values) {
    assert(variable >= values.size() && "Not yet implemented");
  }

  void dump() const { print(llvm::errs()); }

private:
  Int denom;
  unsigned variable;
};
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_CONSTRAINT_H
