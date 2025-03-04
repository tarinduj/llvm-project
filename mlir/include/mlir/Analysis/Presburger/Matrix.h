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

#include <cfenv>

namespace mlir {
namespace analysis {
namespace presburger {

void checkFloatingPointExceptions() {
  if (std::fetestexcept(FE_DIVBYZERO)) {
      std::cerr << "Floating point exception: Division by zero!\n";
  }
  if (std::fetestexcept(FE_INVALID)) {
      std::cerr << "Floating point exception: Invalid operation!\n";
  }
  if (std::fetestexcept(FE_OVERFLOW)) {
      std::cerr << "Floating point exception: Overflow!\n";
  }
  if (std::fetestexcept(FE_UNDERFLOW)) {
      std::cerr << "Floating point exception: Underflow!\n";
  }
  if (std::fetestexcept(FE_INEXACT)) {
      std::cerr << "Floating point exception: Inexact result!\n";
  }
}

static inline void assert_aligned(const void *pointer, size_t byte_count)
{ assert((uintptr_t)pointer % byte_count == 0); }

template <typename T, typename Int>
inline constexpr bool isInt = std::is_same_v<T, SafeInteger<Int>> || std::is_same_v<Int, T>;

/// This is a simple class to represent a resizable matrix.
///
/// The data is stored as one big vector.
template <typename Int>
class Matrix {
public:
#ifdef ENABLE_VECTORIZATION
  static constexpr bool isVectorized = isInt<Int, int16_t> || isInt<Int, int32_t>;
#else
  static constexpr bool isVectorized = false;
#endif

#ifdef ENABLE_SME
  // TODO: set this using IsInt equiavlent
  static constexpr bool isMatrixized = true;
#else
  static constexpr bool isMatrixized = false;
#endif

  // using Vector = typename std::conditional<isInt<Int, int16_t>,
  //   Vector16x32,
  //   void>::type;
  // using Vector = Vector16x32;
  static constexpr unsigned MatrixVectorColumns = isInt<Int, int16_t> ? 32 : 16;
  static constexpr unsigned MatrixSize = 16;
  typedef int16_t Vector __attribute__((ext_vector_type(MatrixVectorColumns)));
  static constexpr bool isChecked = std::is_same_v<Int, SafeInteger<int16_t>> ||
                                    std::is_same_v<Int, SafeInteger<int32_t>> ||
                                    std::is_same_v<Int, SafeInteger<int64_t>> ;

  Matrix() = delete;

  /// Construct a matrix with the specified number of rows and columns.
  /// Initially, the values are default initialized.
  Matrix(unsigned rows, unsigned columns);

  /// Return the identity matrix of the specified dimension.
  static Matrix identity(unsigned dimension);

  /// Access the element at the specified row and column.

  __attribute__((always_inline))
  Int &at(unsigned row, unsigned column) {
    assert(row < getNumRows() && "Row outside of range");
    assert(column < getNumColumns() && "Column outside of range");
    return data[row * nReservedColumns + column];
  }

  __attribute__((always_inline))
  Int at(unsigned row, unsigned column) const {
    assert(row < getNumRows() && "Row outside of range");
    assert(column < getNumColumns() && "Column outside of range");
    return data[row * nReservedColumns + column];
  }

  __attribute__((always_inline))
  Int &operator()(unsigned row, unsigned column) {
    return at(row, column);
  }

  __attribute__((always_inline))
  Int operator()(unsigned row, unsigned column) const {
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

  unsigned getNReservedColumns() const;

  __attribute__((always_inline))
  Vector &getRowVector(unsigned row) {
    static_assert(isVectorized, "getRowVector is only valid for int16_t matrices!");
    assert_aligned(&data[row * nReservedColumns], 64);
    return *(Vector *)&data[row * nReservedColumns];
  }


  /// Get an ArrayRef corresponding to the specified row.
  ArrayRef<Int> getRow(unsigned row) const;

  /// Add `scale` multiples of the source row to the target row.
  void addToRow(unsigned sourceRow, unsigned targetRow, Int scale);

  void scaleColumn(unsigned column, Int scale);

  void addToColumn(unsigned sourceColumn, unsigned targetColumn,
                   Int scale);

  /// Resize the matrix to the specified dimensions. If a dimension is smaller,
  /// the values are truncated; if it is bigger, the new values are default
  /// initialized.
  void resize(unsigned newNRows, unsigned newNColumns);

  // Reserve space for newNRows in total. This number must be greater than the current number of rows.
  void reserveRows(unsigned newNRows);

  /// Print the matrix.
  void print(raw_ostream &os) const;
  void dump() const;

  template <typename OtherInt>
  Matrix<OtherInt> castTo() const {
    Matrix<OtherInt> result(getNumRows(), getNumColumns());
    for (unsigned i = 0; i < getNumRows(); ++i) {
      for (unsigned j = 0; j < getNumColumns(); ++j) {
        std::feclearexcept(FE_ALL_EXCEPT); // Clear all exceptions
        result(i, j) = static_cast<OtherInt>(at(i, j));
        if (std::fetestexcept(FE_ALL_EXCEPT)) {
          std::cerr << "Floating point exception in castTo!\n";
          std::cerr << "from: " << at(i, j) << " to: " << result(i, j) << '\n';
          checkFloatingPointExceptions();
          abort();
        }
      }
    }
    return result;
  }

  Int* getDataPointer() {
    return data.data();
  }

private:
  unsigned nRows, nColumns, nReservedColumns, nReservedRows;

  // using VectorType = typename std::conditional<isVectorized,
  //     std::vector<Int, AlignedAllocator<Int, 64>>,
  //     llvm::SmallVector<Int, 16>
  // >::type;
  using VectorType = llvm::SmallVector<Int, 16>;
  /// Stores the data. data.size() is equal to nRows * nColumns.
  VectorType data;
};

} // namespace presburger
} // namespace analysis
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_MATRIX_H
