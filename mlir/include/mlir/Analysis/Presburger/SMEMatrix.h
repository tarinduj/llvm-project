//===- SMEMatrix.h - MLIR SMEMatrix Class -----------------------------*- C++ -*-===//
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

#ifndef MLIR_ANALYSIS_PRESBURGER_SMEMATRIX_H
#define MLIR_ANALYSIS_PRESBURGER_SMEMATRIX_H

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

/// This is a simple class to represent a matrix resident in ZA.
///
/// The data is stored on the ZA register.
template <typename Int>
class SMEMatrix {
public:
  class ElementProxy {
  public:
    ElementProxy(SMEMatrix &matrix, unsigned row, unsigned col)
        : matrix(matrix), row(row), col(col) {}

    // This is called for assignments: matrix(r, c) = value
    ElementProxy &operator=(Int value) {
      // std::cout << "Assigning value" << std::endl;
      __asm__ __volatile__(
        "smstart sm                                               \n"
        "mov w12, %w[row]                                         \n" // Move row to w12
        "mov w13, %w[col]                                         \n" // Move col to w13
        "mov w14, %w[value]                                       \n" // Move val to w14
        "ptrue	p0.s                                              \n"
        "index z1.s, #0, #1                                       \n" // z1 = [0,1,2,...]
        "dup z2.s, w13                                            \n" // broadcast `col`
        "cmpeq p1.s, p0/z, z1.s, z2.s                             \n" // p1 true only at lane == col
        "dup z0.s, w14                                            \n" // broadcast val
        "mov za0h.s[w12, 0], p1/m, z0.s                           \n" // write z0.s into ZA0 row-slice at (row, col) for lanes in p1
        "smstop sm                                                \n"
        : 
        : [row] "r"(row),
          [col] "r"(col),
          [value] "r"(value)
        : "x12", "x13", "x14",
          "z0", "z1", "z2",
          "za",
          "p0", "p1",
          "memory"
      );
      // std::cout << "Assigned value" << std::endl;
      return *this;
    }

    // This is called for retrievals: x = matrix(r, c)
    operator Int() const {
      // std::cout << "Retrieving element" << std::endl;
      Int val;
      __asm__ __volatile__(
        "smstart sm                                               \n"
        "mov w12, %w[row]                                         \n" // Move row to w12
        "mov w13, %w[col]                                         \n" // Move col to w13
        "ptrue	p0.s                                              \n"
        "mov z0.s, p0/m, za0h.s[w12, 0]                           \n" // Load the value at (row, col) from ZA0 to za0
        "index z1.s, #0, #1                                       \n" // z1 = [0,1,2,...]
        "dup z2.s, w13                                            \n" // broadcast `col`
        "cmpeq p1.s, p0/z, z1.s, z2.s                             \n" // p1 true only at lane == col
        "lastb %w[val], p1, z0.s                                  \n" // store the value at (row, col) from ZA0 to val
        "smstop sm                                                \n"
        : [val] "=r"(val)
        : [row] "r"(row),
          [col] "r"(col)
        : "x12", "x13",
          "z0", "z1", "z2",
          "za",
          "p0", "p1",
          "memory"
        );
      return val;
    }

    ElementProxy &operator+=(Int rhs) {
      return *this = static_cast<Int>(*this) + rhs;
    }
    ElementProxy &operator-=(Int rhs) {
      return *this = static_cast<Int>(*this) - rhs;
    }
    ElementProxy &operator*=(Int rhs) {
      return *this = static_cast<Int>(*this) * rhs;
    }
    ElementProxy &operator/=(Int rhs) {
      return *this = static_cast<Int>(*this) / rhs;
    }
    ElementProxy &operator=(const ElementProxy &rhs) {
      return *this = static_cast<Int>(rhs);
    }

    // ADL-friendly swap for proxy temporaries
    friend inline void matrixSwap(ElementProxy a, ElementProxy b) noexcept {
      Int tmp = static_cast<Int>(a);
      a = static_cast<Int>(b);
      b = tmp;
    }

    friend Int lcm(ElementProxy a, ElementProxy b) {
      return std::lcm(static_cast<Int>(a), static_cast<Int>(b));
    }

    friend Int abs(ElementProxy a) {
      return std::abs(static_cast<Int>(a));
    }

    friend Int mod(ElementProxy a, Int b) {
      assert(b >= 1);
      return static_cast<Int>(a) % b < 0 ? static_cast<Int>(a) % b + b : static_cast<Int>(a) % b;
    }

  private:
    SMEMatrix &matrix;
    unsigned row;
    unsigned col;
  };


  static constexpr bool isMatrixized = true;

  static constexpr unsigned MatrixSize = 16; // platform specific

  SMEMatrix() = delete;

  /// Construct a matrix with the specified number of rows and columns.
  /// Initially, the values are default initialized.
  SMEMatrix(unsigned rows, unsigned columns);

  /// Access the element at the specified row and column.

  __attribute__((always_inline))
  ElementProxy at(unsigned row, unsigned column) {
    // std::cout << "At ElementProxy" << std::endl;
    assert(row < getNumRows() && "Row outside of range");
    assert(column < getNumColumns() && "Column outside of range");
    return ElementProxy(*this, row, column);
  }

  __attribute__((always_inline))
  Int at(unsigned row, unsigned column) const {
    // std::cout << "At Int const" << std::endl;
    assert(row < getNumRows() && "Row outside of range");
    assert(column < getNumColumns() && "Column outside of range");
    
    Int val;
    __asm__ __volatile__(
      "smstart sm                                               \n"
      "mov w12, %w[row]                                         \n" // Move row to w12
      "mov w13, %w[col]                                         \n" // Move col to w13
      "ptrue	p0.s                                              \n"
      "mov z0.s, p0/m, za0h.s[w12, 0]                           \n" // Load the value at (row, col) from ZA1 to za0
      "index z1.s, #0, #1                                       \n" // z1 = [0,1,2,...]
      "dup z2.s, w13                                            \n" // broadcast `col`
      "cmpeq p1.s, p0/z, z1.s, z2.s                             \n" // p1 true only at lane == col
      "lastb %w[val], p1, z0.s                                  \n" // store the value at (row, col) from ZA1 to val
      "smstop sm                                                \n"
      : [val] "=r"(val)
      : [row] "r"(row), 
        [col] "r"(column)
      : "x12", "x13",
        "z0", "z1", "z2",
        "za",
        "p0", "p1",
        "memory"
      );

    return val;
    
  }

  __attribute__((always_inline))
  ElementProxy operator()(unsigned row, unsigned column) {
    // std::cout << "At ElementProxy operator()" << std::endl;
    return at(row, column);
  }

  __attribute__((always_inline))
  Int operator()(unsigned row, unsigned column) const {
    // std::cout << "Int at operator() const" << std::endl;
    return at(row, column);         // TODO: check if this is correct             
  }

  /// Swap the given columns.
  void swapColumns(unsigned column, unsigned otherColumn);

  /// Swap the given rows.
  void swapRows(unsigned row, unsigned otherRow);

  unsigned getNumRows() const;

  unsigned getNumColumns() const;

  unsigned getNReservedColumns() const;

  /// Resize the matrix to the specified dimensions. If a dimension is smaller,
  /// the values are truncated; if it is bigger, the new values are default
  /// initialized.
  void resize(unsigned newNRows, unsigned newNColumns);

  // Reserve space for newNRows in total. This number must be greater than the current number of rows.
  void reserveRows(unsigned newNRows);

  /// Get an ArrayRef corresponding to the specified row.
  ArrayRef<Int> getRow(unsigned row) const;

  /// Print the matrix.
  void print(raw_ostream &os) const;
  void dump() const;

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

// template <typename Int>
// inline Int lcm(
//     typename SMEMatrix<Int>::ElementProxy a,
//     typename SMEMatrix<Int>::ElementProxy b) {
//   return std::lcm(static_cast<Int>(a), static_cast<Int>(b));
// }

// template <typename Int>
// inline void swap(typename SMEMatrix<Int>::ElementProxy a,
//                  typename SMEMatrix<Int>::ElementProxy b) noexcept {
//   Int tmp = static_cast<Int>(a);
//   a = static_cast<Int>(b);
//   b = tmp;
// }

} // namespace presburger
} // namespace analysis
} // namespace mlir

namespace std {

// template <typename Int>
// inline Int abs(typename mlir::analysis::presburger::SMEMatrix<Int>::ElementProxy a) {
//   Int x = static_cast<Int>(a);
//   return x < 0 ? -x : x; 
// }
} // namespace std

#endif // MLIR_ANALYSIS_PRESBURGER_MATRIX_H
