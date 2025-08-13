//===- Matrix<Int>.cpp - MLIR Matrix<Int> Class -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/SMEMatrix.h"

#ifndef MLIR_ANALYSIS_PRESBURGER_SMEMATRIX_IMPL_H
#define MLIR_ANALYSIS_PRESBURGER_SMEMATRIX_IMPL_H

using namespace mlir;
using namespace analysis::presburger;

// Utility functions are defined in Matrix-impl.h

template <typename Int>
SMEMatrix<Int>::SMEMatrix(unsigned rows, unsigned columns)
    : nRows(rows), nColumns(columns), 
    nReservedColumns(16), nReservedRows(nextMultipleOfFour(nRows)) {

  if (columns > MatrixSize || rows > MatrixSize) {
    std::cerr << "Size exceeds matrix size limit.\n";
    std::abort();
  }
}

template <typename Int>
unsigned SMEMatrix<Int>::getNumRows() const { return nRows; }

template <typename Int>
unsigned SMEMatrix<Int>::getNumColumns() const { return nColumns; }

template <typename Int>
unsigned SMEMatrix<Int>::getNReservedColumns() const {return nReservedColumns; }

template <typename Int>
void SMEMatrix<Int>::resize(unsigned newNRows, unsigned newNColumns) {
  assert(newNRows <= MatrixSize);
  assert(newNColumns <= MatrixSize);
  nRows = newNRows;
  nColumns = newNColumns;
}

template <typename Int>
void SMEMatrix<Int>::reserveRows(unsigned newNRows) {
  assert(newNRows >= nRows);
  assert(newNRows <= MatrixSize);
}

template <typename Int>
void SMEMatrix<Int>::swapRows(unsigned row, unsigned otherRow) {
  assert((row < getNumRows() && otherRow < getNumRows()) &&
         "Given row out of bounds");
  if (row == otherRow)
    return;
  __asm__ __volatile__(
    "smstart sm                                               \n"
    "mov w12, %w[row]                                         \n" // Move row to w12
    "mov w13, %w[otherRow]                                    \n" // Move otherRow to w13
    "ptrue	p0.s                                              \n"
    "mov z0.s, p0/m, za0h.s[w12, 0]                           \n" // Load the value at (row, col) from ZA0 to z0
    "mov z1.s, p0/m, za0h.s[w13, 0]                           \n" // Load the value at (otherRow, col) from ZA0 to z1
    "mov za0h.s[w12, 0], p0/m, z1.s                           \n" // Load the value at (otherRow, col) from ZA0 to z1
    "mov za0h.s[w13, 0], p0/m, z0.s                           \n" // Load the value at (row, col) from ZA0 to z0
    "smstop sm                                                \n"
    :
    : [row] "r"(row),
      [otherRow] "r"(otherRow)
    : "x12", "x13",
      "z0", "z1",
      "za",
      "p0",
      "memory"
    );
}

template <typename Int>
void SMEMatrix<Int>::swapColumns(unsigned column, unsigned otherColumn) {
  assert((column < getNumColumns() && otherColumn < getNumColumns()) &&
         "Given column out of bounds");
  if (column == otherColumn)
    return;
  __asm__ __volatile__(
    "smstart sm                                               \n"
    "mov w12, %w[column]                                      \n" // Move column to w12
    "mov w13, %w[otherColumn]                                 \n" // Move otherColumn to w13
    "ptrue	p0.s                                              \n"
    "mov z0.s, p0/m, za0v.s[w12, 0]                           \n" // Load the value at (row, col) from ZA0 to z0
    "mov z1.s, p0/m, za0v.s[w13, 0]                           \n" // Load the value at (row, otherColumn) from ZA0 to z1
    "mov za0v.s[w12, 0], p0/m, z1.s                           \n" // Load the value at (row, otherColumn) from ZA0 to z1
    "mov za0v.s[w13, 0], p0/m, z0.s                           \n" // Load the value at (row, col) from ZA0 to z0
    "smstop sm                                                \n"
    :
    : [column] "r"(column),
      [otherColumn] "r"(otherColumn)
    : "x12", "x13",
      "z0", "z1",
      "za",
      "p0",
      "memory"
    );
}


// get Row vector from ZA0 register into Z0
// put it into ArrayRef<Int> variable
template <typename Int>
ArrayRef<Int> SMEMatrix<Int>::getRow(unsigned row) const {
  Int *rowData = new Int[16]; // Size 16 since MatrixSize is 16
          
  __asm__ __volatile__(
    "smstart sm                                               \n"
    "mov w12, %w[row]                                         \n" // Move row to w12
    "ptrue	p0.s                                              \n"
    "mov z0.s, p0/m, za0h.s[w12, 0]                           \n" // Load the value at (row, col) from ZA1 to za0
    "mov x0, %[rowData]                                       \n" // Move rowData to x0
    "mov x1, #0                                               \n"
    "st1w {z0.s}, p0, [x0, x1, lsl #2]                        \n"
    "smstop sm                                                \n"
    :
    : [row] "r"(row),
      [rowData] "r"(rowData)
    : "x0", "x1", "x12",
      "z0",
      "za",
      "p0",
      "memory"
  );

  ArrayRef<Int> rowVector = ArrayRef<Int>(rowData, 16);

  // print rowVector
  // std::cout << "rowVector: ";
  // for (int i = 0; i < 16; i++) {
  //   std::cout << rowVector[i] << " ";
  // }
  // std::cout << std::endl;

  return rowVector;
}


template <typename Int>
void SMEMatrix<Int>::print(raw_ostream &os) const {
  for (unsigned row = 0; row < nRows; ++row) {
    for (unsigned column = 0; column < nColumns; ++column)
      os << (*this)(row, column) << '\t';
    os << '\n';
  }
}

template <typename Int>
void SMEMatrix<Int>::dump() const { print(llvm::errs()); }

#endif // MLIR_ANALYSIS_PRESBURGER_SMEMATRIX_IMPL_H
