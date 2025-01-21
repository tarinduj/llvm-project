//===- Simplex.h - MLIR ParamLexSimplex Class -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functionality to perform analysis on FlatAffineConstraints. In particular,
// support for performing emptiness checks.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PARAMLEXSIMPLEX_H
#define MLIR_ANALYSIS_PRESBURGER_PARAMLEXSIMPLEX_H

#include "mlir/Analysis/Presburger/PresburgerBasicSet.h"
#include "mlir/Analysis/Presburger/Simplex.h"
// #include "mlir/Analysis/AffineStructures.h"
// #include "mlir/Analysis/Presburger/Fraction.h"
// #include "mlir/Analysis/Presburger/Matrix.h"
// #include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
// #include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
// #include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace analysis {
namespace presburger {

template <typename Int>
struct pwaFunction {
  SmallVector<PresburgerBasicSet<Int>, 8> domain;
  SmallVector<SmallVector<SmallVector<Int, 8>, 8>, 8> value;

  void dump() {
    for (unsigned i = 0; i < value.size(); ++i) {
      domain[i].dump();
      llvm::errs() << "\n";
      for (unsigned j = 0; j < value[i].size(); ++j) {
        llvm::errs() << "a" << j << " = ";
        for (unsigned k = 0; k < value[i][j].size() - 1; ++k) {
          if (value[i][j][k] == 0)
            continue;
          llvm::errs() << value[i][j][k] << "x" << k << " + ";
        }
        llvm::errs() << value[i][j].back() << '\n';
      }
      llvm::errs() << '\n';
    }
  }
};

template <typename Int>
class ParamLexSimplex : public Simplex<Int> {
public:
  using Simplex<Int>::nRow;
  using Simplex<Int>::nCol;
  using Simplex<Int>::nRedundant;
  using Simplex<Int>::liveColBegin;
  using Simplex<Int>::tableau;
  using Simplex<Int>::empty;
  using Simplex<Int>::undoLog;
  using Simplex<Int>::savedBases;
  using Simplex<Int>::rowUnknown;
  using Simplex<Int>::colUnknown;
  using Simplex<Int>::con;
  using Simplex<Int>::var;
  using Unknown = typename Simplex<Int>::Unknown;
  using Direction = typename Simplex<Int>::Direction;
  using Orientation = typename Simplex<Int>::Orientation;
  using Simplex<Int>::getSnapshotBasis;
  using Simplex<Int>::unknownFromRow;
  using Simplex<Int>::unknownFromColumn;
  using Simplex<Int>::unknownFromIndex;
  using Simplex<Int>::rollback;
  using Simplex<Int>::addVariable;
  using Simplex<Int>::addDivisionVariable;
  using Simplex<Int>::addZeroConstraint;
  using Simplex<Int>::pivot;
  using Simplex<Int>::markEmpty;
  using Simplex<Int>::addRow;

  ParamLexSimplex() = delete;
  ParamLexSimplex(unsigned nDim, unsigned nParam);
  explicit ParamLexSimplex(const FlatAffineConstraints &constraints);

  void addInequality(ArrayRef<Int> coeffs);
  void addEquality(ArrayRef<Int> coeffs);
  void addDivisionVariable(ArrayRef<Int> coeffs, Int denom);

  pwaFunction<Int> findParamLexmin();
  void findParamLexminRecursively(Simplex<Int> &domainSimplex,
                                  PresburgerBasicSet<Int> &domainSet,
                                  pwaFunction<Int> &result,
                                  int depth = 0);

private:
  SmallVector<Int, 8> getRowParamSample(unsigned row);
  LogicalResult moveRowUnknownToColumn(unsigned row);
  void restoreConsistency();
  unsigned getSnapshot();
  // SmallVector<Int, 8> varCoeffsFromRowCoeffs(ArrayRef<Int>
  // rowCoeffs) const;
  Optional<unsigned> findPivot(unsigned row) const;

  unsigned nParam, nDiv;
  SmallVector<SmallVector<Int, 8>, 8> originalCoeffs;
};
} // namespace presburger
} // namespace analysis
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PARAMLEXSIMPLEX_H
