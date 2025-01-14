//===- Simplex.h - MLIR Simplex Class ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functionality to perform analysis on FlatAffineConstraints. In particular,
// support for performing emptiness checks and constraint simplilifaction to
// detect redundant constraints.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_SIMPLEX_H
#define MLIR_ANALYSIS_PRESBURGER_SIMPLEX_H

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Presburger/Fraction.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>

namespace mlir {
namespace analysis {
namespace presburger {

template <typename Int>
class GBRSimplex;
template <typename Int>
class PresburgerBasicSet;

/// This class implements a version of the Simplex and Generalized Basis
/// Reduction algorithms, which can perform analysis of integer sets with affine
/// inequalities and equalities. A Simplex can be constructed
/// by specifying the dimensionality of the set. It supports adding affine
/// inequalities and equalities, and can perform emptiness checks, i.e., it can
/// find a solution to the set of constraints if one exists, or say that the
/// set is empty if no solution exists. Furthermore it can find a subset of
/// these constraint that are redundant, i.e. a subset of constraints that
/// doesn't constrain the affine set further after addinf the non-redundant
/// constraints. Currently, this only works for bounded sets. Simplex can also
/// be constructed from a FlatAffineConstraints object.
///
/// The implementation of this Simplex class, other than the functionality
/// for sampling, is based on the paper
/// "Simplify: A Theorem Prover for Program Checking"
/// by D. Detlefs, G. Nelson, J. B. Saxe.
///
/// We define variables, constraints, and unknowns. Consider the example of a
/// two-dimensional set defined by 1 + 2x + 3y >= 0 and 2x - 3y >= 0. Here,
/// x, y, are variables while 1 + 2x + 3y >= 0, 2x - 3y >= 0 are
/// constraints. Unknowns are either variables or constraints, i.e., x, y,
/// 1 + 2x + 3y >= 0, 2x - 3y >= 0 are all unknowns.
///
/// The implementation involves a matrix called a tableau, which can be thought
/// of as a 2D matrix of rational numbers having number of rows equal to the
/// number of constraints and number of columns equal to one plus the number of
/// variables. In our implementation, instead of storing rational numbers, we
/// store a common denominator for each row, so it is in fact a matrix of
/// integers with number of rows equal to number of constraints and number of
/// columns equal to _two_ plus the number of variables. For example, instead of
/// storing a row of three rationals [1/2, 2/3, 3], we would store [6, 3, 4, 18]
/// since 3/6 = 1/2, 4/6 = 2/3, and 18/6 = 3.
///
/// Every row and column except the first and second columns is associated with
/// an unknown and every unknown is associated with a row or column. The second
/// column represents the constant, explained in more detail below. An unknown
/// associated with a row or column is said to be in row or column position
/// respectively.
///
/// The vectors var and con store information about the variables and
/// constraints respectively, namely, whether they are in row or column
/// position, which row or column they are associated with, and whether they
/// correspond to a variable or a constraint.
///
/// An unknown is addressed by its index. If the index i is non-negative, then
/// the variable var[i] is being addressed. If the index i is negative, then
/// the constraint con[~i] is being addressed. Effectively this maps
/// 0 -> var[0], 1 -> var[1], -1 -> con[0], -2 -> con[1], etc. rowUnknown[r] and
/// colUnknown[c] are the indexes of the unknowns associated with row r and
/// column c, respectively.
///
/// The unknowns in column position are together called the basis. Initially the
/// basis is the set of variables -- in our example above, the initial basis is
/// x, y.
///
/// The unknowns in row position are represented in terms of the basis unknowns.
/// If the basis unknowns are u_1, u_2, ... u_m, and a row in the tableau is
/// d, c, a_1, a_2, ... a_m, this representats the unknown for that row as
/// (c + a_1*u_1 + a_2*u_2 + ... + a_m*u_m)/d. In our running example, if the
/// basis is the initial basis of x, y, then the constraint 1 + 2x + 3y >= 0
/// would be represented by the row [1, 1, 2, 3].
///
/// The association of unknowns to rows and columns can be changed by a process
/// called pivoting, where a row unknown and a column unknown exchange places
/// and the remaining row variables' representation is changed accordingly
/// by eliminating the old column unknown in favour of the new column unknown.
/// If we had pivoted the column for x with the row for 2x - 3y >= 0,
/// the new row for x would be [2, 1, 3] since x = (1*(2x - 3y) + 3*y)/2.
/// See the documentation for the pivot member function for details.
///
/// The association of unknowns to rows and columns is called the _tableau
/// configuration_. The _sample value_ of an unknown in a particular tableau
/// configuration is its value if all the column unknowns were set to zero.
/// Concretely, for unknowns in column position the sample value is zero and
/// for unknowns in row position the sample value is the constant term divided
/// by the common denominator.
///
/// The tableau configuration is called _consistent_ if the sample value of all
/// restricted unknowns is non-negative. Initially there are no constraints, and
/// the tableau is consistent. When a new constraint is added, its sample value
/// in the current tableau configuration may be negative. In that case, we try
/// to find a series of pivots to bring us to a consistent tableau
/// configuration, i.e. we try to make the new constraint's sample value
/// non-negative without making that of any other constraints negative. (See
/// findPivot and findPivotRow for details.) If this is not possible, then the
/// set of constraints is mutually contradictory and the tableau is marked
/// _empty_, which means the set of constraints has no solution.
///
/// This Simplex class also supports taking snapshots of the current state
/// and rolling back to prior snapshots. This works by maintaing an undo log
/// of operations. Snapshots are just pointers to a particular location in the
/// log, and rolling back to a snapshot is done by reverting each log entry's
/// operation from the end until we reach the snapshot's location.
///
/// Finding an integer sample is done with the Generalized Basis Reduction
/// algorithm. See the documentation for findIntegerSample and reduceBasis.

template <typename Int>
class Simplex {
public:
  using BaseInt = UnderlyingInt<Int>;
  static constexpr bool isVectorized = Matrix<Int>::isVectorized;
  static constexpr bool isMatrixized = Matrix<Int>::isMatrixized;
  static constexpr bool isChecked = Matrix<Int>::isChecked;
  using Vector = typename Matrix<Int>::Vector;
  using MatrixRow = typename Matrix<Int>::MatrixRow;

  enum class Direction { Up, Down };

  enum class IneqType { Redundant, Separate, Cut, AdjEq, AdjIneq };

  Simplex() = delete;
  explicit Simplex(unsigned nVar);
  explicit Simplex(const FlatAffineConstraints &constraints);
  explicit Simplex(const PresburgerBasicSet<Int> &bs);

  /// Returns true if the tableau is empty (has conflicting constraints),
  /// false otherwise.
  bool isEmpty() const;

  /// Given a constraint con >= 0, add another constraint to force con = 0.
  /// Before the function returns the added constraint is removed again, but the
  /// effects on the other unknowns remain.
  void cutToHyperplane(int con_index);

  /// Check for redundant constraints and mark them as redundant.
  void detectRedundant();

  /// Check whether the constraint has been marked redundant.
  bool isMarkedRedundant(int conIndex) const;

  void extendConstraints(unsigned n_new);

  ~Simplex();

  /// Add an inequality to the tableau. If coeffs is c_0, c_1, ... c_n, where n
  /// is the current number of variables, then the corresponding inequality is
  /// c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} >= 0.
  void addInequality(ArrayRef<Int> coeffs);

  /// Returns the number of variables in the tableau.
  unsigned numVariables() const;

  /// Returns the number of constraints in the tableau.
  unsigned numConstraints() const;

  /// Add an equality to the tableau. If coeffs is c_0, c_1, ... c_n, where n
  /// is the current number of variables, then the corresponding equality is
  /// c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} == 0.
  void addEquality(ArrayRef<Int> coeffs);

  void addVariable();
  void addDivisionVariable(ArrayRef<Int> coeffs, Int denom);

  /// Mark the tableau as being empty.
  void markEmpty();

  /// Get a snapshot of the current state. This is used for rolling back.
  unsigned getSnapshot() const;
  unsigned getSnapshotBasis();

  /// Rollback to a snapshot. This invalidates all later snapshots.
  void rollback(unsigned snapshot);

  /// Checks the type of the inequality. If coeffs is c_0, c_1, ... c_n, where n
  /// is the current number of variables, then the corresponding equality is
  /// c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} >= 0.
  ///
  /// The possible results are:
  /// Redundant   The inequality is already satisfied
  /// Cut         The inequality is satisfied by some points but not others
  /// Separate    The inequality is satisfied by no points
  ///
  /// Special cases of separate when the tableau is in integer mode:
  /// AdjEq      The value of the expression is always -1
  /// AdjIneq    The inequality is c(-u - 1) >= 0 where u is an existing
  ///             inequality
  ///
  /// \returns an IneqType, the type of the specified inequality.
  IneqType ineqType(ArrayRef<Int> coeffs);

  /// Compute the maximum or minimum value of the given row, depending on
  /// direction.
  ///
  /// Returns a (num, den) pair denoting the optimum, or None if no
  /// optimum exists, i.e., if the expression is unbounded in this direction.
  Optional<Fraction<Int>> computeRowOptimum(Direction direction, unsigned row);

  /// Compute the maximum or minimum value of the given expression, depending on
  /// direction.
  ///
  /// Returns a (num, den) pair denoting the optimum, or a null value if no
  /// optimum exists, i.e., if the expression is unbounded in this direction.
  Optional<Fraction<Int>> computeOptimum(Direction direction,
                                    ArrayRef<Int> coeffs);

  /// Returns a (min, max) pair denoting the minimum and maximum integer values
  /// of the given expression.
  std::pair<Int, Int>
  computeIntegerBounds(ArrayRef<Int> coeffs);

  /// Returns true if the polytope is unbounded, i.e., extends to infinity in
  /// some direction. Otherwise, returns false.
  bool isUnbounded();

  /// Make a tableau to represent a pair of points in the given tableaus, one in
  /// tableau A and one in B.
  static Simplex makeProduct(const Simplex &a, const Simplex &b);

  /// Returns the current (possibly fractional) sample point. This should not
  /// be called when the simplex is empty.
  SmallVector<Fraction<Int>, 8> getSamplePoint() const;

  /// Returns the current sample point if it is integral. Otherwise, returns
  /// None.
  Optional<SmallVector<Int, 8>> getSamplePointIfIntegral() const;

  /// Returns an integer sample point if one exists, or None
  /// otherwise. This should only be called for bounded sets.
  Optional<SmallVector<Int, 8>> findIntegerSample();

  std::pair<Int, SmallVector<Int, 8>>
  findRationalSample() const;

  /// Print the tableau's internal state.
  void print(raw_ostream &os) const;

  void dump() const;

  void addFlatAffineConstraints(const FlatAffineConstraints &cs);
  void addBasicSet(const PresburgerBasicSet<Int> &bs);
  // void addFlatAffineConstraintsAsIneqs(const FlatAffineConstraints &cs);

  void detectImplicitEqualities();

  /// Check whether the constraint is an equality.
  ///
  /// \returns True if the constraint is forced to be equal to zero,
  /// False otherwise.
  bool constraintIsEquality(int con_index) const;

protected:
  friend class GBRSimplex<Int>;

  enum class Orientation { Row, Column };

  /// An Unknown is either a variable or a constraint. It is always associated
  /// with either a row or column. Whether it's a row or a column is specified
  /// by the orientation and pos identifies the specific row or column it is
  /// associated with. If the unknown is restricted, then it has a
  /// non-negativity constraint associated with it, i.e., its sample value must
  /// always be non-negative and if it cannot be made non-negative without
  /// violating other constraints, the tableau is empty.
  struct Unknown {
    Unknown(Orientation oOrientation, bool oRestricted, unsigned oPos)
        : pos(oPos), orientation(oOrientation), restricted(oRestricted),
          redundant(false), marked(false), zero(false) {}
    unsigned pos;
    Orientation orientation;
    bool restricted : 1;
    bool redundant;
    bool marked;
    bool zero;

    void print(raw_ostream &os) const {
      os << (orientation == Orientation::Row ? "r" : "c");
      os << pos;
      if (restricted)
        os << " [>=0]";
    }
  };

  struct Pivot {
    unsigned row, column;
  };

  /// Find a pivot to change the sample value of row in the specified
  /// direction. The returned pivot row will be row if and only
  /// if the unknown is unbounded in the specified direction.
  ///
  /// Returns a (row, col) pair denoting a pivot, or an empty Optional if
  /// no valid pivot exists.
  Optional<Pivot> findPivot(int row, Direction direction) const;

  /// Swap the row with the column in the tableau's data structures but not the
  /// tableau itself. This is used by pivot.
  void swapRowWithCol(unsigned row, unsigned col);

  /// Pivot the row with the column.
  inline void SMEPivotHelper(Int *matrix, int rows, int cols, int pivot_row, int pivot_col);
  void pivot(unsigned row, unsigned col);
  void pivot(Pivot pair);

  /// Pivot \p unknown down or up to row position depending on \p direction.
  ///
  /// If \p direction is empty, both directions are allowed. \p unknown is
  /// assumed to be bounded in the allowed directions.
  void toRow(Unknown &unknown, Direction direction);

  /// Check if the row can attain non-negative values.
  /// On return, the unknown remains at the same row it was initially.
  ///
  /// \returns True if the row can attain non-negative values, False otherwise.
  bool rowIsAtLeastZero(Unknown &unknown);

  /// Check if the constraint is redundant by computing its minimum value in
  /// the tableau. If this returns true, the constraint is left in row position
  /// upon return.
  ///
  /// \param conIndex must be a constraint that is not a dead column
  ///
  /// \returns True if the constraint is redundant, False otherwise.
  bool constraintIsRedundant(unsigned conIndex);

  /// Compare the maximum value of \p u with \p origin.
  ///
  /// \returns +1 if the maximum value is greater than \p origin, 0 if they are
  // equal, and -1 if it is less than \p origin.
  template <int origin>
  Int signOfMax(Unknown &u);

  /// Returns the unknown associated with index.
  const Unknown &unknownFromIndex(int index) const;
  /// Returns the unknown associated with col.
  const Unknown &unknownFromColumn(unsigned col) const;
  /// Returns the unknown associated with row.
  const Unknown &unknownFromRow(unsigned row) const;
  /// Returns the unknown associated with index.
  Unknown &unknownFromIndex(int index);
  /// Returns the unknown associated with col.
  Unknown &unknownFromColumn(unsigned col);
  /// Returns the unknown associated with row.
  Unknown &unknownFromRow(unsigned row);

  /// Check if there is obviously no lower bound on \p unknown.
  ///
  /// \returns True if \p unknown is obviously unbounded from below, False
  /// otherwise.
  bool minIsObviouslyUnbounded(Unknown &unknown) const;

  /// Check if there is obviously no upper bound on \p unknown.
  ///
  /// \returns True if \p unknown is obviously unbounded from above, False
  /// otherwise.
  bool maxIsObviouslyUnbounded(Unknown &unknown) const;

  /// Checks if \p row is not obviously constrained to be zero
  ///
  /// \returns True if \p row is obviously not constrained to be zero,
  /// False otherwise.
  bool rowIsObviouslyNotZero(unsigned row) const;

  /// Check if \p row is obviously equal to zero
  ///
  /// \returns True if \p row is obviously equal to zero, False otherwise.
  bool rowIsObviouslyZero(unsigned row) const;

  /// Check if \p row is obviously non-integral.
  ///
  /// \returns True if \p unknown is obviously non-integral, False otherwise.
  bool rowIsObviouslyNonIntegral(unsigned row) const;

  int indexFromUnknown(const Unknown &u) const;

  /// Called by ineqType. Checks for special cases of separate inequalities for
  /// integral tableaus. Must only be called for separate inequalities.
  ///
  /// \returns the separation type, IneqType::Separate, IneqType::AdjEq, or
  /// IneqType::AdjIneq.
  IneqType separationType(unsigned row);

  /// Checks that \p unknown is neither a redundant row or a dead column
  ///
  /// \returns True if \p unknown is neither a redundant row nor a dead column,
  /// False otherwise.
  bool unknownIsRelevant(Unknown &unknown) const;

  /// Add a new zero constraint to the tableau and the associated data
  /// structures.
  void addZeroConstraint();

  /// Add a new row to the tableau and the associated data structures.
  unsigned addRow(ArrayRef<Int> coeffs);

  /// Normalize the given row by removing common factors between the numerator
  /// and the denominator.
  void normalizeRow(unsigned row, Vector &rowVec);
  void normalizeRow(unsigned row);
  void normalizeRowScalar(unsigned row);

  /// Mark the column as zero.
  ///
  /// \returns True if the column is interchanged with a later column, False
  /// otherwise. This is used when iterating through the columns; if the return
  /// is true, the same column index must be processed again.
  bool killCol(unsigned col);
  void closeRow(unsigned row, bool temp_row);

  /// Mark the row as being redundant.
  ///
  /// \returns True if the row is interchanged with a later row, False
  /// otherwise. This is used when iterating through the rows; if the return is
  /// true, the same row index must be processed again.
  bool markRedundant(unsigned row);

  /// Drop row \p row from the tableau.
  void dropRow(unsigned row);

  /// Swap the two rows in the tableau and associated data structures.
  void swapRows(unsigned i, unsigned j);

  /// Swap the two cols in the tableau and associated data structures.
  void swapColumns(unsigned i, unsigned j);

  /// Restore the unknown to a non-negative sample value.
  ///
  /// Returns true if the unknown was successfully restored to a non-negative
  /// sample value, false otherwise.
  LogicalResult restoreRow(Unknown &u);

  enum class UndoLogEntry {
    RemoveLastConstraint,
    RestoreBasis,
    RemoveLastVariable,
    UnmarkEmpty,
    UnmarkRedundant,
    UnmarkZero
  };

  /// Undo the operation represented by the log entry.
  void undo(UndoLogEntry entry, Optional<int> index);

  /// Find a row that can be used to pivot the column in the specified
  /// direction. If skipRow is not null, then this row is excluded
  /// from consideration. The returned pivot will maintain all constraints
  /// except the column itself and skipRow, if it is set. (if these unknowns
  /// are restricted).
  ///
  /// Returns the row to pivot to, or an empty Optional if the column
  /// is unbounded in the specified direction.
  Optional<unsigned> findPivotRow(Optional<unsigned> skipRow,
                                  Direction direction, unsigned col) const;

  Int sign(Int num, Int den = 1,
                   Int origin = 0) const;

  /// Reduce the given basis, starting at the specified level, using general
  /// basis reduction.
  void reduceBasis(Matrix<Int> &basis, unsigned level);

  /// The number of rows in the tableau.
  unsigned nRow;

  /// The number of columns in the tableau, including the common denominator
  /// and the constant column.
  unsigned nCol;

  /// The number of constraints marked redundant.
  unsigned nRedundant;

  /// The index of the first live column.
  unsigned liveColBegin;

  /// The matrix representing the tableau.
  Matrix<Int> tableau;

  /// This is true if the tableau has been detected to be empty, false
  /// otherwise.
  bool empty;

  /// Holds a log of operations, used for rolling back to a previous state.
  SmallVector<std::pair<UndoLogEntry, Optional<int>>, 8> undoLog;
  SmallVector<SmallVector<int, 8>, 8> savedBases;

  /// These hold the indexes of the unknown at a given row or column position.
  /// We keep these as signed integers since that makes it convenient to check
  /// if an index corresponds to a variable or a constraint by checking the
  /// sign.
  ///
  /// colUnknown is padded with two null indexes at the front since the first
  /// two columns don't correspond to any unknowns.
  SmallVector<int, 8> rowUnknown, colUnknown;

  /// These hold information about each unknown.
  SmallVector<Unknown, 8> con, var;

  unsigned numPivots;

  void printAndResetNumPivots() {
#ifdef PRINT_PIVOTS
    std::cout << numPivots << '\n';
    numPivots = 0;
#endif
  }
};

} // namespace presburger
} // namespace analysis
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_SIMPLEX_H
