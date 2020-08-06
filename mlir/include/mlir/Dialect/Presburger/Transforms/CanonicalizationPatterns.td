#ifndef MLIR_DIALECT_PRESBURGER_TRANSFORMS_SIMPLIFICATION_PATTERNS
#define MLIR_DIALECT_PRESBURGER_TRANSFORMS_SIMPLIFICATION_PATTERNS

include "mlir/Dialect/Presburger/PresburgerOps.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"

//===----------------------------------------------------------------------===//
// Presburger dialect transformations
//===----------------------------------------------------------------------===//

// These patterns detect trivial identities of operations on Presburger sets.

def ValuesAreIdentical : Constraint<CPred<"$0 == $1">>;

def DoubleComplementOptPattern : Pat<
  (Presburger_ComplementOp(Presburger_ComplementOp $arg)),
  (replaceWithValue $arg)>;

def SameArgUnionOptPattern : Pat<
  (Presburger_UnionOp $arg1, $arg2),
  (replaceWithValue $arg1),
  [(ValuesAreIdentical $arg1, $arg2)]>;

def SameArgIntersectOptPattern : Pat<
  (Presburger_IntersectOp $arg1, $arg2),
  (replaceWithValue $arg1),
  [(ValuesAreIdentical $arg1, $arg2)]>;

def createEmptySetOfEqualDim : NativeCodeCall<
  "createEmptySetOfEqualDim($_builder, $0.getType().cast<PresburgerSetType>())">;

def SameArgSubtractOptPattern : Pat<
  (Presburger_SubtractOp $arg1, $arg2),
  (createEmptySetOfEqualDim $arg1),
  [(ValuesAreIdentical $arg1, $arg2)]>;

def createConstantTrue : NativeCodeCall<
  "createConstantTrue($_builder)">;

def SameArgEqualOptPattern : Pat<
  (Presburger_EqualOp $arg1, $arg2),
  (createConstantTrue),
  [(ValuesAreIdentical $arg1, $arg2)]>;
#endif // MLIR_DIALECT_PRESBURGER_TRANSFORMS_SIMPLIFICATION_PATTERNS

