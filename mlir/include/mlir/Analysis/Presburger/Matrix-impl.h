//===- Matrix<Int>.cpp - MLIR Matrix<Int> Class -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Matrix.h"

#ifndef MLIR_ANALYSIS_PRESBURGER_MATRIX_IMPL_H
#define MLIR_ANALYSIS_PRESBURGER_MATRIX_IMPL_H

using namespace mlir;
using namespace analysis::presburger;

inline unsigned nextPowOfTwo(unsigned n) {
  unsigned ret = 1;
  while (n > ret)
    ret *= 2;
  return ret;
}

inline unsigned nextMultipleOfFour(unsigned n) {
  unsigned ret = 4;
  while (n > ret)
    ret += 4;
  return ret;
}

template <typename Int>
Matrix<Int>::Matrix(unsigned rows, unsigned columns)
    : nRows(rows), nColumns(columns), 
    nReservedColumns(nextPowOfTwo(nColumns)), nReservedRows(nextMultipleOfFour(nRows)), 
    data(nReservedRows * nReservedColumns) {

  if (isMatrixized) {
    if (columns > MatrixSize || rows > MatrixSize) {
      std::cerr << "Size exceeds matrix size limit.\n";
      std::abort();
    }
  }
}

template <typename Int>
Matrix<Int> Matrix<Int>::identity(unsigned dimension) {
  Matrix<Int> matrix(dimension, dimension);
  for (unsigned i = 0; i < dimension; ++i)
    matrix(i, i) = 1;
  return matrix;
}

template <typename Int>
unsigned Matrix<Int>::getNumRows() const { return nRows; }

template <typename Int>
unsigned Matrix<Int>::getNumColumns() const { return nColumns; }

template <typename Int>
unsigned Matrix<Int>::getNReservedColumns() const {return nReservedColumns; }

template <typename Int>
void Matrix<Int>::resize(unsigned newNRows, unsigned newNColumns) {
  if (isMatrixized) {
    if (newNColumns > MatrixSize || newNRows > MatrixSize) { 
      std::cerr << "Size exceeds matrix size limit.\n";
      std::abort();
    }
  }
  if (newNColumns > nReservedColumns) {
    unsigned newNReservedColumns = nextPowOfTwo(newNColumns);
    unsigned newNReservedRows = nextMultipleOfFour(newNRows);
    data.resize(newNReservedRows * newNReservedColumns);
    for (int row = newNRows - 1; row >= 0; --row)
      for (int col = newNReservedColumns - 1; col >= 0; --col)
        data[row * newNReservedColumns + col] = unsigned(row) < nRows && unsigned(col) < nColumns ? at(row, col) : 0;
    nRows = newNRows;
    nColumns = newNColumns;
    nReservedColumns = newNReservedColumns;
    nReservedRows = newNReservedRows;
  } else {
    unsigned newNReservedRows = nextMultipleOfFour(newNRows);
    data.resize(newNReservedRows * nReservedColumns);
    if (newNColumns < nColumns) {
      for (unsigned row = 0; row < newNRows; ++row) {
        for (unsigned col = newNColumns; col < nColumns; ++col) {
          at(row, col) = 0;
        }
      }
    }
    nRows = newNRows;
    nColumns = newNColumns;
    nReservedRows = newNReservedRows;
  }
}

template <typename Int>
void Matrix<Int>::reserveRows(unsigned newNRows) {
  assert(newNRows >= nRows);
  data.reserve(newNRows * nReservedColumns);
}

template <typename Int>
void Matrix<Int>::swapRows(unsigned row, unsigned otherRow) {
  assert((row < getNumRows() && otherRow < getNumRows()) &&
         "Given row out of bounds");
  if (row == otherRow)
    return;
  for (unsigned col = 0; col < nColumns; col++)
    std::swap(at(row, col), at(otherRow, col));
}

template <typename Int>
void Matrix<Int>::swapColumns(unsigned column, unsigned otherColumn) {
  assert((column < getNumColumns() && otherColumn < getNumColumns()) &&
         "Given column out of bounds");
  if (column == otherColumn)
    return;
  for (unsigned row = 0; row < nRows; row++)
    std::swap(at(row, column), at(row, otherColumn));
}

template <typename Int>
void Matrix<Int>::negateColumn(unsigned column) {
  assert(column < getNumColumns() && "Given column out of bounds");
  for (unsigned row = 0, e = getNumRows(); row < e; ++row) {
    // TODO not overflow safe
    at(row, column) = -at(row, column);
  }
}

template <typename Int>
ArrayRef<Int> Matrix<Int>::getRow(unsigned row) const {
  return {&data[row * nReservedColumns], nColumns};
}

template <typename Int>
void Matrix<Int>::addToRow(unsigned sourceRow, unsigned targetRow,
                      Int scale) {
  if (scale == 0)
    return;
  for (unsigned col = 0; col < nColumns; ++col)
    at(targetRow, col) += scale * at(sourceRow, col);
  return;
}

template <typename Int>
void Matrix<Int>::scaleColumn(unsigned column, Int scale) {
  if (scale == 0)
    return;
  for (unsigned row = 0, e = getNumRows(); row < e; ++row)
    at(row, column) *= scale;
  return;
}

template <typename Int>
void Matrix<Int>::addToColumn(unsigned sourceColumn, unsigned targetColumn,
                         Int scale) {
  if (scale == 0)
    return;
  for (unsigned row = 0, e = getNumRows(); row < e; ++row)
    at(row, targetColumn) += scale * at(row, sourceColumn);
  return;
}

template <typename Int>
void Matrix<Int>::print(raw_ostream &os) const {
  for (unsigned row = 0; row < nRows; ++row) {
    for (unsigned column = 0; column < nColumns; ++column)
      os << at(row, column) << '\t';
    os << '\n';
  }
}

template <typename Int>
void Matrix<Int>::dump() const { print(llvm::errs()); }

#endif // MLIR_ANALYSIS_PRESBURGER_MATRIX_IMPL_H
