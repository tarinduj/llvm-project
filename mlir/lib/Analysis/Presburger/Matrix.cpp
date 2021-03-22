//===- Matrix<Int>.cpp - MLIR Matrix<Int> Class -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Matrix.h"

using namespace mlir;
using namespace analysis::presburger;

template <typename Int>
Matrix<Int>::Matrix(unsigned rows, unsigned columns)
    : nRows(rows), nColumns(columns), data(nRows * MATRIX_COLUMN_COUNT) {
  if (nColumns > MATRIX_COLUMN_COUNT) {
    llvm::errs() << "Cannot construct matrix with " << nColumns << " columns; limit is " << MATRIX_COLUMN_COUNT << ".\n";
    abort();
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
void Matrix<Int>::resize(unsigned newNRows, unsigned newNColumns) {
  nRows = newNRows;
  data.resize(nRows * MATRIX_COLUMN_COUNT);
  if (newNColumns < nColumns) {
    for (unsigned row = 0; row < nRows; ++row) {
      for (unsigned col = newNColumns; col < nColumns; ++col) {
        at(row, col) = 0;
      }
    }
  }
  nColumns = newNColumns;
}

template <typename Int>
void Matrix<Int>::reserveRows(unsigned newNRows) {
  assert(newNRows >= nRows);
  data.reserve(newNRows * MATRIX_COLUMN_COUNT);
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
ArrayRef<SafeInteger<Int>> Matrix<Int>::getRow(unsigned row) const {
  return {&data[row * MATRIX_COLUMN_COUNT], nColumns};
}

template <typename Int>
void Matrix<Int>::addToRow(unsigned sourceRow, unsigned targetRow,
                      SafeInteger<Int> scale) {
  if (scale == 0)
    return;
  for (unsigned col = 0; col < nColumns; ++col)
    at(targetRow, col) += scale * at(sourceRow, col);
  return;
}

template <typename Int>
void Matrix<Int>::scaleColumn(unsigned column, SafeInteger<Int> scale) {
  if (scale == 0)
    return;
  for (unsigned row = 0, e = getNumRows(); row < e; ++row)
    at(row, column) *= scale;
  return;
}

template <typename Int>
void Matrix<Int>::addToColumn(unsigned sourceColumn, unsigned targetColumn,
                         SafeInteger<Int> scale) {
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
