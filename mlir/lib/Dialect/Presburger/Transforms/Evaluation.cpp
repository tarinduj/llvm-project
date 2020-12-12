#include "PassDetail.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Presburger/Coalesce.h"
#include "mlir/Analysis/Presburger/Set.h"
#include "mlir/Dialect/Presburger/Attributes.h"
#include "mlir/Dialect/Presburger/Passes.h"
#include "mlir/Dialect/Presburger/PresburgerOps.h"
#include "mlir/Dialect/Presburger/PresburgerOptions.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

#include <chrono>

using namespace mlir;
using namespace mlir::presburger;
using namespace llvm;

static SetOp unionSets(PatternRewriter &rewriter, Operation *op,
                       PresburgerSetAttr attr1, PresburgerSetAttr attr2) {
  registerPresburgerCLOptions();
  PresburgerSet ps(attr1.getValue());

  if (printPresburgerRuntimes()) {
    auto start = std::chrono::system_clock::now();
    ps.unionSet(attr2.getValue());
    auto end = std::chrono::system_clock::now();
    llvm::errs() << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                          start)
                        .count();
  } else {
    ps.unionSet(attr2.getValue());
  }

  if (dumpResults()) {
    ps.dumpISL();
  }

  PresburgerSetType type = PresburgerSetType::get(
      rewriter.getContext(), ps.getNumDims(), ps.getNumSyms());

  PresburgerSetAttr newAttr = PresburgerSetAttr::get(type, ps);
  return rewriter.create<SetOp>(op->getLoc(), type, newAttr);
}

static SetOp intersectSets(PatternRewriter &rewriter, Operation *op,
                           PresburgerSetAttr attr1, PresburgerSetAttr attr2) {
  PresburgerSet ps(attr1.getValue());

  if (printPresburgerRuntimes()) {
    auto start = std::chrono::system_clock::now();
    ps.intersectSet(attr2.getValue());
    auto end = std::chrono::system_clock::now();
    llvm::errs() << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                          start)
                        .count();
  } else {
    ps.intersectSet(attr2.getValue());
  }

  if (dumpResults()) {
    ps.dumpISL();
  }
  PresburgerSetType type = PresburgerSetType::get(
      rewriter.getContext(), ps.getNumDims(), ps.getNumSyms());

  PresburgerSetAttr newAttr = PresburgerSetAttr::get(type, ps);
  return rewriter.create<SetOp>(op->getLoc(), type, newAttr);
}

static SetOp subtractSets(PatternRewriter &rewriter, Operation *op,
                          PresburgerSetAttr attr1, PresburgerSetAttr attr2) {
  PresburgerSet ps(attr1.getValue());

  if (printPresburgerRuntimes()) {
    auto start = std::chrono::system_clock::now();
    ps.subtract(attr2.getValue());
    auto end = std::chrono::system_clock::now();
    llvm::errs() << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                          start)
                        .count();
  } else {
    ps.subtract(attr2.getValue());
  }

  if (dumpResults()) {
    ps.dumpISL();
  }
  PresburgerSetType type = PresburgerSetType::get(
      rewriter.getContext(), ps.getNumDims(), ps.getNumSyms());

  PresburgerSetAttr newAttr = PresburgerSetAttr::get(type, ps);
  return rewriter.create<SetOp>(op->getLoc(), type, newAttr);
}

static SetOp coalesceSet(PatternRewriter &rewriter, Operation *op,
                         PresburgerSetAttr attr) {
  // TODO: change Namespace of coalesce
  PresburgerSet in = attr.getValue();
  PresburgerSet ps;
  if (printPresburgerRuntimes()) {
    auto start = std::chrono::system_clock::now();
    ps = coalesce(in);
    auto end = std::chrono::system_clock::now();
    llvm::errs() << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                          start)
                        .count();
  } else {
    ps = coalesce(in);
  }

  if (dumpResults()) {
    ps.dumpISL();
  }
  PresburgerSetType type = PresburgerSetType::get(
      rewriter.getContext(), ps.getNumDims(), ps.getNumSyms());

  PresburgerSetAttr newAttr = PresburgerSetAttr::get(type, ps);
  return rewriter.create<SetOp>(op->getLoc(), type, newAttr);
}

static SetOp eliminateExistentialsSet(PatternRewriter &rewriter, Operation *op,
                                      PresburgerSetAttr attr) {
  // TODO: change Namespace of coalesce
  PresburgerSet in = attr.getValue();

  PresburgerSet ps;
  if (printPresburgerRuntimes()) {
    auto start = std::chrono::system_clock::now();
    ps = PresburgerSet::eliminateExistentials(in);
    auto end = std::chrono::system_clock::now();
    llvm::errs() << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                          start)
                        .count();
  } else {
    ps = PresburgerSet::eliminateExistentials(in);
  }

  if (dumpResults()) {
    ps.dumpISL();
  }
  PresburgerSetType type = PresburgerSetType::get(
      rewriter.getContext(), ps.getNumDims(), ps.getNumSyms());

  PresburgerSetAttr newAttr = PresburgerSetAttr::get(type, ps);
  ps.dumpISL();
  return rewriter.create<SetOp>(op->getLoc(), type, newAttr);
}

static SetOp complementSet(PatternRewriter &rewriter, Operation *op,
                           PresburgerSetAttr attr) {
  PresburgerSet ps;
  if (printPresburgerRuntimes()) {
    auto start = std::chrono::system_clock::now();
    ps = PresburgerSet::complement(attr.getValue());
    auto end = std::chrono::system_clock::now();
    llvm::errs() << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                          start)
                        .count();
  } else {
    ps = PresburgerSet::complement(attr.getValue());
  }

  if (dumpResults()) {
    ps.dumpISL();
  }
  PresburgerSetType type = PresburgerSetType::get(
      rewriter.getContext(), ps.getNumDims(), ps.getNumSyms());

  PresburgerSetAttr newAttr = PresburgerSetAttr::get(type, ps);
  return rewriter.create<SetOp>(op->getLoc(), type, newAttr);
}

static ConstantOp areEqualSets(PatternRewriter &rewriter, Operation *op,
                               PresburgerSetAttr attr1,
                               PresburgerSetAttr attr2) {
  bool eq = PresburgerSet::equal(attr1.getValue(), attr2.getValue());
  if (printPresburgerRuntimes()) {
    auto start = std::chrono::system_clock::now();
    eq = PresburgerSet::equal(attr1.getValue(), attr2.getValue());
    auto end = std::chrono::system_clock::now();
    llvm::errs() << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                          start)
                        .count();
  } else {
    eq = PresburgerSet::equal(attr1.getValue(), attr2.getValue());
  }

  if (dumpResults()) {
    llvm::errs() << eq;
  }
  IntegerType type = rewriter.getI1Type();
  IntegerAttr attr = IntegerAttr::get(type, eq);

  return rewriter.create<ConstantOp>(op->getLoc(), type, attr);
}

static ConstantOp emptySet(PatternRewriter &rewriter, Operation *op,
                           PresburgerSetAttr attr) {
  PresburgerSet ps = attr.getValue();
  bool empty;
  if (printPresburgerRuntimes()) {
    auto start = std::chrono::system_clock::now();
    empty = ps.isIntegerEmpty();
    auto end = std::chrono::system_clock::now();
    llvm::errs() << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                          start)
                        .count();
  } else {
    empty = ps.isIntegerEmpty();
  }

  if (dumpResults()) {
    llvm::errs() << empty;
  }

  IntegerType type = rewriter.getI1Type();
  IntegerAttr iAttr = IntegerAttr::get(type, empty);

  return rewriter.create<ConstantOp>(op->getLoc(), type, iAttr);
}
namespace {

#include "mlir/Dialect/Presburger/Transforms/EvaluationPatterns.cpp.inc"

} // end anonymous namespace

void mlir::populatePresburgerEvaluatePatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  // clang-format off
  patterns.insert<
    FoldIntersectPattern,
    FoldUnionPattern,
    FoldSubtractPattern,
    FoldCoalescePattern,
    FoldEmptyPattern,
    FoldEliminateExPattern,
    FoldComplementPattern,
    FoldEqualPattern
    >(ctx);
  // clang-format on
}

struct PresburgerEvaluatePass
    : public PresburgerEvaluateBase<PresburgerEvaluatePass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    populatePresburgerEvaluatePatterns(patterns, &getContext());
    applyPatternsAndFoldGreedily(getFunction(), patterns);
  }
};

std::unique_ptr<OperationPass<FuncOp>> mlir::createPresburgerEvaluatePass() {
  return std::make_unique<PresburgerEvaluatePass>();
}
