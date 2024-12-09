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
  DialectSet ps(attr1.getValue());

  if (printPresburgerRuntimes()) {
    DialectSet set1(attr1.getValue());
    DialectSet set2(attr2.getValue());
    unsigned int dummy;
    auto start = std::chrono::high_resolution_clock::now();
    set1.unionSet(std::move(set2));
    auto end = std::chrono::high_resolution_clock::now();
    llvm::errs() << static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) << '\n';
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
  DialectSet ps(attr1.getValue());

  if (printPresburgerRuntimes()) {
    unsigned int dummy;
    auto start = std::chrono::high_resolution_clock::now();
    ps.intersectSet(attr2.getValue());
    auto end = std::chrono::high_resolution_clock::now();
    llvm::errs() << static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) << '\n';
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
  DialectSet ps(attr1.getValue());

  if (printPresburgerRuntimes()) {
    unsigned int dummy;
    auto start = std::chrono::high_resolution_clock::now();
    ps.subtract(attr2.getValue());
    auto end = std::chrono::high_resolution_clock::now();
    llvm::errs() << static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) << '\n';
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
  DialectSet in = attr.getValue();
  DialectSet ps;
  if (printPresburgerRuntimes()) {
    unsigned int dummy;
    auto start = std::chrono::high_resolution_clock::now();
    ps = coalesce(in);
    auto end = std::chrono::high_resolution_clock::now();
    llvm::errs() << static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) << '\n';
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
  DialectSet in = attr.getValue();

  DialectSet ps;
  if (printPresburgerRuntimes()) {
    DialectSet in2 = attr.getValue();
    unsigned int dummy;
    auto start = std::chrono::high_resolution_clock::now();
    ps = DialectSet::eliminateExistentials(std::move(in2));
    auto end = std::chrono::high_resolution_clock::now();
    llvm::errs() << static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) << '\n';
  } else {
    ps = DialectSet::eliminateExistentials(in);
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
  DialectSet ps;
  if (printPresburgerRuntimes()) {
    unsigned int dummy;
    auto start = std::chrono::high_resolution_clock::now();
    ps = DialectSet::complement(attr.getValue());
    auto end = std::chrono::high_resolution_clock::now();
    llvm::errs() << static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) << '\n';
  } else {
    ps = DialectSet::complement(attr.getValue());
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
  bool eq = DialectSet::equal(attr1.getValue(), attr2.getValue());
  if (printPresburgerRuntimes()) {
    unsigned int dummy;
    auto start = std::chrono::high_resolution_clock::now();
    eq = DialectSet::equal(attr1.getValue(), attr2.getValue());
    auto end = std::chrono::high_resolution_clock::now();
    llvm::errs() << static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) << '\n';
  } else {
    eq = DialectSet::equal(attr1.getValue(), attr2.getValue());
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
  DialectSet ps = attr.getValue();
  bool empty;
  if (printPresburgerRuntimes()) {
    unsigned int dummy;
    auto start = std::chrono::high_resolution_clock::now();
    empty = ps.isIntegerEmpty();
    auto end = std::chrono::high_resolution_clock::now();
    llvm::errs() << static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) << '\n';
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
