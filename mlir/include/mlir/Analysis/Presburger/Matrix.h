//===- Matrix.h - MLIR Matrix Class -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a simple 2D matrix class that supports reading, writing, resizing,
// swapping rows, and swapping columns.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_MATRIX_H
#define MLIR_ANALYSIS_PRESBURGER_MATRIX_H

#include "mlir/Analysis/Presburger/AlignedAllocator.h"
#include "mlir/Analysis/Presburger/SafeInteger.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <vector>

namespace mlir {
namespace analysis {
namespace presburger {

#define MATRIX_COLUMN_COUNT 32

typedef int16_t Vector __attribute__((ext_vector_type(MATRIX_COLUMN_COUNT)));

/// This is a simple class to represent a resizable matrix.
///
/// The data is stored in the form of a vector of vectors.

template <typename Int>
class Matrix {
public:
  Matrix() = delete;

  /// Construct a matrix with the specified number of rows and columns.
  /// Initially, the values are default initialized.
  Matrix(unsigned rows, unsigned columns);

  /// Return the identity matrix of the specified dimension.
  static Matrix identity(unsigned dimension);

  /// Access the element at the specified row and column.

  __attribute__((always_inline))
  SafeInteger<Int> &at(unsigned row, unsigned column) {
    assert(row < getNumRows() && "Row outside of range");
    assert(column < getNumColumns() && "Column outside of range");
    return data[row * MATRIX_COLUMN_COUNT + column];
  }

  __attribute__((always_inline))
  SafeInteger<Int> at(unsigned row, unsigned column) const {
    assert(row < getNumRows() && "Row outside of range");
    assert(column < getNumColumns() && "Column outside of range");
    return data[row * MATRIX_COLUMN_COUNT + column];
  }

  __attribute__((always_inline))
  SafeInteger<Int> &operator()(unsigned row, unsigned column) {
    return at(row, column);
  }

  __attribute__((always_inline))
  SafeInteger<Int> operator()(unsigned row, unsigned column) const {
    return at(row, column);
  }

  /// Swap the given columns.
  void swapColumns(unsigned column, unsigned otherColumn);

  /// Swap the given rows.
  void swapRows(unsigned row, unsigned otherRow);

  /// Negate the column.
  ///
  /// \returns True if overflow occurs, False otherwise.
  void negateColumn(unsigned column);

  unsigned getNumRows() const;

  unsigned getNumColumns() const;

  __attribute__((always_inline))
  Vector &getRowVector(unsigned row) {
    static_assert(std::is_same<Int, int16_t>::value, "getRowVector is only valid for int16_t matrices!");
    return *(Vector *)&data[row * MATRIX_COLUMN_COUNT];
  }


  /// Get an ArrayRef corresponding to the specified row.
  ArrayRef<SafeInteger<Int>> getRow(unsigned row) const;

  /// Add `scale` multiples of the source row to the target row.
  void addToRow(unsigned sourceRow, unsigned targetRow, SafeInteger<Int> scale);

  void scaleColumn(unsigned column, SafeInteger<Int> scale);

  void addToColumn(unsigned sourceColumn, unsigned targetColumn,
                   SafeInteger<Int> scale);

  /// Resize the matrix to the specified dimensions. If a dimension is smaller,
  /// the values are truncated; if it is bigger, the new values are default
  /// initialized.
  void resize(unsigned newNRows, unsigned newNColumns);

  // Reserve space for newNRows in total. This number must be greater than the current number of rows.
  void reserveRows(unsigned newNRows);

  /// Print the matrix.
  void print(raw_ostream &os) const;
  void dump() const;

private:
  unsigned nRows, nColumns;

  /// Stores the data. data.size() is equal to nRows * nColumns.
  std::vector<SafeInteger<Int>, AlignedAllocator<SafeInteger<Int>, 64>> data;
};

} // namespace presburger
} // namespace analysis
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_MATRIX_H
