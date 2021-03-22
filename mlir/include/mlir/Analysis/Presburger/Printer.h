#ifndef PRESBURGER_PRINTER_H
#define PRESBURGER_PRINTER_H

#include "mlir/Analysis/Presburger/Expr.h"
#include "mlir/Analysis/Presburger/Set.h"

namespace mlir {
namespace analysis {
namespace presburger {

/// This file provides printing functions for the different Presburger
/// datastructures.
/// TODO: move this to the head documentation

/// Prints the set into the stream
template <typename Int>
void printPresburgerSet(raw_ostream &os, const PresburgerSet<Int> &set);

/// Prints the basic set into the stream
template <typename Int>
void printPresburgerBasicSet(raw_ostream &os, const PresburgerBasicSet<Int> &set);

/// Prints the Presburger expression into the stream
void printPresburgerExpr(raw_ostream &os, const PresburgerExpr &expr);

} // namespace presburger
} // namespace analysis
} // namespace mlir

#endif // PRESBURGER_PRINTER_H
