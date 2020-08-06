#include "mlir/Dialect/Presburger/Attributes.h"
#include "mlir/Dialect/Presburger/PresburgerOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::presburger;

static SetOp createEmptySetOfEqualDim(PatternRewriter &rewriter,
                                      PresburgerSetType type) {
  Location loc = rewriter.getInsertionPoint()->getLoc();
  PresburgerSet set(type.getDimCount(), type.getSymbolCount(), true);

  PresburgerSetAttr attr = PresburgerSetAttr::get(type, set);
  return rewriter.create<SetOp>(loc, type, attr);
}

static ConstantOp createConstantTrue(PatternRewriter &rewriter) {
  Location loc = rewriter.getInsertionPoint()->getLoc();
  return rewriter.create<ConstantIntOp>(loc, 1, 1);
}

namespace {
#include "mlir/Dialect/Presburger/Transforms/CanonicalizationPatterns.cpp.inc"
}

void ComplementOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<DoubleComplementOptPattern>(context);
}

void UnionOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
  results.insert<SameArgUnionOptPattern>(context);
}

void IntersectOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<SameArgIntersectOptPattern>(context);
}

void SubtractOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results.insert<SameArgSubtractOptPattern>(context);
}

void EqualOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
  results.insert<SameArgEqualOptPattern>(context);
}
