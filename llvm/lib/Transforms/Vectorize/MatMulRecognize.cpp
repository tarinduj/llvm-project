//===- MatMulRecognize.cpp - MatMul pattern recognition + SME lowering ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Recognizes matrix multiply loop nests in LLVM IR and lowers them to tiled
// AArch64 SME FMOPA (outer product accumulate) sequences. Fully opt-in.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/MatMulRecognize.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopNestAnalysis.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "matmul-recognize"

STATISTIC(NumMatMulsRecognized, "Number of matmul patterns recognized");
STATISTIC(NumMatMulsLowered, "Number of matmul patterns lowered to SME FMOPA");

static cl::opt<bool> RecognizeOnly(
    "matmul-recognize-only", cl::init(false), cl::Hidden,
    cl::desc("Only recognize matmul patterns (no SME lowering). "
             "Useful for analysis and debugging."));

static cl::opt<bool> MatMulVerbose(
    "matmul-recognize-verbose", cl::init(false), cl::Hidden,
    cl::desc("Print recognized matmul patterns to stderr."));

static cl::opt<unsigned> MinTripCount(
    "matmul-min-trip-count", cl::init(4), cl::Hidden,
    cl::desc("Minimum known trip count to consider for SME lowering."));

namespace {

//===----------------------------------------------------------------------===//
// Pattern Recognition
//===----------------------------------------------------------------------===//

/// Given a loop, return its canonical induction variable (the PHI that
/// starts at 0 and increments by 1), or nullptr.
static PHINode *getCanonicalInductionVariable(Loop *L) {
  BasicBlock *Header = L->getHeader();
  for (PHINode &PN : Header->phis()) {
    if (!PN.getType()->isIntegerTy())
      continue;

    // Check: one incoming value is 0 (from preheader), the other is PHI+1
    // (from latch).
    Value *StartVal = PN.getIncomingValueForBlock(L->getLoopPreheader());
    Value *LatchVal = PN.getIncomingValueForBlock(L->getLoopLatch());

    auto *StartConst = dyn_cast<ConstantInt>(StartVal);
    if (!StartConst || !StartConst->isZero())
      continue;

    // Latch value should be PHI + 1.
    ConstantInt *C;
    if (match(LatchVal, m_Add(m_Specific(&PN), m_ConstantInt(C))) &&
        C->isOne())
      return &PN;
  }
  return nullptr;
}

/// Holds the result of decomposing a GEP-based array access into
/// base pointer + linear combination of induction variables.
struct ArrayAccess {
  Value *Base = nullptr;
  // Which loop induction variables appear in the address, and with what
  // multipliers. We track at most 2 dimensions for matmul.
  // Pairs of (PHINode* inductionVar, Value* stride).
  SmallVector<std::pair<PHINode *, const SCEV *>, 2> Indices;
};

/// Try to decompose a memory address (from a load/store) into a base pointer
/// plus linear functions of loop induction variables. Returns true on success.
static bool decomposeAddress(Value *Ptr, ScalarEvolution &SE,
                             const SmallVectorImpl<PHINode *> &IndVars,
                             ArrayAccess &Result) {
  const SCEV *S = SE.getSCEV(Ptr);

  // Walk the SCEV looking for AddRecExprs that correspond to our induction
  // variables.
  Result.Indices.clear();

  // Helper: recursively find AddRecExprs in the SCEV.
  std::function<bool(const SCEV *)> findIndices = [&](const SCEV *Expr) -> bool {
    if (auto *AR = dyn_cast<SCEVAddRecExpr>(Expr)) {
      // Check if this AddRec's loop corresponds to one of our induction vars.
      const Loop *L = AR->getLoop();
      for (PHINode *IV : IndVars) {
        if (IV->getParent() == L->getHeader()) {
          Result.Indices.push_back({IV, AR->getStepRecurrence(SE)});
          // Recurse into the start to find outer loop contributions.
          return findIndices(AR->getStart());
        }
      }
      // AddRec for an unrelated loop — bail.
      return false;
    }
    if (auto *Add = dyn_cast<SCEVAddExpr>(Expr)) {
      for (const SCEV *Op : Add->operands()) {
        if (!findIndices(Op))
          return false;
      }
      return true;
    }
    // Cast expressions: recurse into the operand.
    if (auto *Cast = dyn_cast<SCEVCastExpr>(Expr))
      return findIndices(Cast->getOperand());
    // Mul expressions: recurse into operands (handles stride * IV).
    if (auto *Mul = dyn_cast<SCEVMulExpr>(Expr)) {
      for (const SCEV *Op : Mul->operands()) {
        if (!findIndices(Op))
          return false;
      }
      return true;
    }
    // Constants and unknowns are fine (they form part of the base).
    if (isa<SCEVConstant>(Expr) || isa<SCEVUnknown>(Expr))
      return true;
    // Anything else is too complex.
    return false;
  };

  if (!findIndices(S))
    return false;

  // Extract base pointer: strip all GEPs to get the underlying object.
  Result.Base = getUnderlyingObject(Ptr);
  return true;
}

/// Identify which induction variable plays which role (i, j, k) in a matmul
/// by examining which variables appear in the C, A, B array accesses.
///
/// Canonical matmul: C[i][j] += A[i][k] * B[k][j]
/// - C depends on i and j (not k)
/// - A depends on i and k (not j)
/// - B depends on k and j (not i)
///
/// The "reduction" variable k is the one that does NOT appear in C's index.
static bool classifyRoles(const ArrayAccess &AccC, const ArrayAccess &AccA,
                          const ArrayAccess &AccB,
                          const SmallVectorImpl<PHINode *> &AllIVs,
                          MatMulInfo &Info) {
  // Build sets of which IVs appear in each access.
  auto ivSet = [](const ArrayAccess &Acc) {
    SmallDenseSet<PHINode *, 4> S;
    for (auto &[IV, _] : Acc.Indices)
      S.insert(IV);
    return S;
  };

  auto CIVs = ivSet(AccC);
  auto AIVs = ivSet(AccA);
  auto BIVs = ivSet(AccB);

  // We expect exactly 2 IVs in C, 2 in A, 2 in B, from a set of 3 total.
  if (CIVs.size() != 2 || AIVs.size() != 2 || BIVs.size() != 2)
    return false;
  if (AllIVs.size() != 3)
    return false;

  // Find k: the IV that is NOT in C.
  PHINode *K = nullptr;
  for (PHINode *IV : AllIVs) {
    if (!CIVs.contains(IV)) {
      K = IV;
      break;
    }
  }
  if (!K)
    return false;

  // k must appear in both A and B.
  if (!AIVs.contains(K) || !BIVs.contains(K))
    return false;

  // Find i: the IV in both C and A (but not k).
  PHINode *I = nullptr;
  for (PHINode *IV : AllIVs) {
    if (IV != K && CIVs.contains(IV) && AIVs.contains(IV)) {
      I = IV;
      break;
    }
  }
  if (!I)
    return false;

  // Find j: the remaining IV (in C and B, but not i or k).
  PHINode *J = nullptr;
  for (PHINode *IV : AllIVs) {
    if (IV != K && IV != I) {
      J = IV;
      break;
    }
  }
  if (!J || !CIVs.contains(J) || !BIVs.contains(J))
    return false;

  Info.IndI = I;
  Info.IndJ = J;
  Info.IndK = K;

  // Determine which loop each IV belongs to.
  auto loopOf = [](PHINode *IV) -> Loop * {
    // The IV's parent block is the loop header.
    // We rely on LoopInfo having been computed; here we just record the PHI.
    return nullptr; // Filled in by caller.
  };
  (void)loopOf;

  // Detect transposition: In canonical form, A's indices are (i, k).
  // If A has (k, i) instead, A is transposed. Check by looking at which
  // IV has the unit stride in A's access.
  auto hasUnitStride = [](const ArrayAccess &Acc, PHINode *IV,
                          ScalarEvolution &SE) -> bool {
    for (auto &[V, Stride] : Acc.Indices) {
      if (V == IV) {
        if (auto *SC = dyn_cast<SCEVConstant>(Stride))
          return SC->getAPInt().isOne();
      }
    }
    return false;
  };

  // Unused for now — transposition detection is complex.
  // For the canonical case, A[i][k] has unit stride on k (column-major iter),
  // B[k][j] has unit stride on j.
  (void)hasUnitStride;

  return true;
}

/// Try to recognize a matmul pattern in a 3-deep loop nest.
/// Returns a filled MatMulInfo on success, std::nullopt on failure.
static std::optional<MatMulInfo>
analyzeLoopNest(Loop *OuterLoop, LoopInfo &LI, ScalarEvolution &SE) {
  // Collect the loop nest: we need exactly 3 levels.
  SmallVector<Loop *, 4> Nest;
  Value *DetectedBeta = nullptr;
  Nest.push_back(OuterLoop);

  // Walk down the single-child chain.
  Loop *Current = OuterLoop;
  while (!Current->getSubLoops().empty()) {
    if (Current->getSubLoops().size() != 1) {
      // Multiple children — could be imperfect nest (e.g., GEMM with separate
      // beta-scaling loop). For now, only handle the case where one child
      // contains the matmul and the other is a simple scaling loop.
      // TODO: Handle imperfect nests with sibling scaling loops.
      if (Current->getSubLoops().size() == 2 && Nest.size() == 1) {
        // Check if one of the two children is a 2-deep nest (the matmul)
        // and the other is a single loop (the scaling loop).
        Loop *Child0 = Current->getSubLoops()[0];
        Loop *Child1 = Current->getSubLoops()[1];

        Loop *MatMulChild = nullptr;
        Loop *ScaleChild = nullptr;
        if (!Child0->getSubLoops().empty() && Child1->getSubLoops().empty()) {
          MatMulChild = Child0;
          ScaleChild = Child1;
        } else if (Child0->getSubLoops().empty() &&
                   !Child1->getSubLoops().empty()) {
          MatMulChild = Child1;
          ScaleChild = Child0;
        }

        if (MatMulChild && MatMulChild->getSubLoops().size() == 1) {
          Nest.push_back(MatMulChild);
          Nest.push_back(MatMulChild->getSubLoops()[0]);

          // Try to extract beta from the scaling code.
          // It may be a loop (C[i][j] *= beta) or unrolled fmul sequence.
          // Scan all blocks in the i-loop that are NOT part of the matmul
          // child for a pattern: store(fmul(load(ptr), beta_const), ptr).
          {
            SmallDenseSet<BasicBlock *, 16> MatMulBlocks;
            for (BasicBlock *BB : MatMulChild->blocks())
              MatMulBlocks.insert(BB);
            for (BasicBlock *BB : Current->blocks()) {
              if (MatMulBlocks.contains(BB))
                continue;
              for (Instruction &I : *BB) {
                auto *SI = dyn_cast<StoreInst>(&I);
                if (!SI) continue;
                Value *MulLHS, *MulRHS;
                if (!match(SI->getValueOperand(),
                           m_FMul(m_Value(MulLHS), m_Value(MulRHS))))
                  continue;
                if (auto *LI = dyn_cast<LoadInst>(MulLHS)) {
                  if (LI->getPointerOperand() == SI->getPointerOperand()) {
                    DetectedBeta = MulRHS;
                    break;
                  }
                }
                if (auto *LI = dyn_cast<LoadInst>(MulRHS)) {
                  if (LI->getPointerOperand() == SI->getPointerOperand()) {
                    DetectedBeta = MulLHS;
                    break;
                  }
                }
              }
              if (DetectedBeta) break;
            }
          }
          break;
        }
      }
      LLVM_DEBUG(dbgs() << "MMR: Loop nest has " << Current->getSubLoops().size()
                        << " children, expected 1\n");
      return std::nullopt;
    }
    Current = Current->getSubLoops()[0];
    Nest.push_back(Current);
  }

  if (Nest.size() != 3) {
    LLVM_DEBUG(dbgs() << "MMR: Loop nest depth is " << Nest.size()
                      << ", expected 3\n");
    if (MatMulVerbose)
      errs() << "MMR: Loop nest depth is " << Nest.size()
             << ", expected 3\n";
    return std::nullopt;
  }

  // Detect beta scaling: scan the outermost loop's blocks that are NOT
  // part of any sub-loop. Look for store(fmul(load(ptr), constant), ptr).
  // This handles both loop-based and unrolled beta scaling.
  if (!DetectedBeta) {
    SmallDenseSet<BasicBlock *, 32> SubLoopBlocks;
    for (Loop *Sub : *Nest[0]) {
      for (BasicBlock *BB : Sub->blocks())
        SubLoopBlocks.insert(BB);
    }
    for (BasicBlock *BB : Nest[0]->blocks()) {
      if (SubLoopBlocks.contains(BB))
        continue;
      for (Instruction &I : *BB) {
        auto *SI = dyn_cast<StoreInst>(&I);
        if (!SI)
          continue;
        Value *MulL, *MulR;
        if (!match(SI->getValueOperand(), m_FMul(m_Value(MulL), m_Value(MulR))))
          continue;
        if (auto *LI = dyn_cast<LoadInst>(MulL)) {
          if (LI->getPointerOperand() == SI->getPointerOperand()) {
            DetectedBeta = MulR;
            break;
          }
        }
        if (!DetectedBeta) {
          if (auto *LI = dyn_cast<LoadInst>(MulR)) {
            if (LI->getPointerOperand() == SI->getPointerOperand()) {
              DetectedBeta = MulL;
              break;
            }
          }
        }
      }
      if (DetectedBeta)
        break;
    }
    if (DetectedBeta && MatMulVerbose)
      errs() << "MMR: Detected beta = " << *DetectedBeta << "\n";
  }

  // Get canonical induction variables for all 3 loops.
  SmallVector<PHINode *, 3> IndVars;
  for (Loop *L : Nest) {
    PHINode *IV = getCanonicalInductionVariable(L);
    if (!IV) {
      LLVM_DEBUG(dbgs() << "MMR: No canonical IV for loop at depth "
                        << L->getLoopDepth() << "\n");
      if (MatMulVerbose)
        errs() << "MMR: No canonical IV for loop at depth "
               << L->getLoopDepth() << "\n";
      return std::nullopt;
    }
    IndVars.push_back(IV);
  }

  // Find the innermost loop and look for the accumulation pattern.
  // Two forms are supported:
  //
  // Pattern 1 — PHI-based scalar reduction:
  //   %acc = phi [init, preheader], [%add, latch]
  //   %mul = fmul load_A, load_B
  //   %add = fadd %acc, %mul
  //   (store after the inner loop)
  //
  // Pattern 2 — Load-modify-store (array accumulation):
  //   %c = load C[i][j]
  //   %mul = fmul load_A, load_B
  //   %add = fadd %c, %mul
  //   store %add, C[i][j]
  //
  Loop *InnerLoop = Nest[2];

  LoadInst *LoadLHS = nullptr;   // Load from A
  LoadInst *LoadRHS = nullptr;   // Load from B
  Value *AlphaVal = nullptr;     // Optional alpha scaling
  BinaryOperator *AddOp = nullptr;
  StoreInst *CStore = nullptr;
  LoadInst *CLoad = nullptr;     // For pattern 2

  // Helper lambdas for extracting loads through optional alpha scaling.
  auto extractLoad = [](Value *V) -> LoadInst * {
    if (auto *LI = dyn_cast<LoadInst>(V))
      return LI;
    return nullptr;
  };

  auto extractAlphaLoad = [](Value *V, Value *&Alpha) -> LoadInst * {
    Value *A, *B;
    if (match(V, m_FMul(m_Value(A), m_Value(B)))) {
      if (auto *LI = dyn_cast<LoadInst>(B)) {
        Alpha = A;
        return LI;
      }
      if (auto *LI = dyn_cast<LoadInst>(A)) {
        Alpha = B;
        return LI;
      }
    }
    return nullptr;
  };

  // Try to match the multiply-accumulate from an fadd instruction.
  // Returns true if %Addend = fmul(load, load) or fmul(alpha*load, load).
  auto matchMulAccum = [&](Value *Addend) -> bool {
    Value *MulLHS, *MulRHS;
    if (!match(Addend, m_FMul(m_Value(MulLHS), m_Value(MulRHS))))
      return false;

    LoadInst *L1 = extractLoad(MulLHS);
    LoadInst *L2 = extractLoad(MulRHS);
    Value *Alpha1 = nullptr;

    if (L1 && L2) {
      LoadLHS = L1;
      LoadRHS = L2;
    } else if (L1 && !L2) {
      LoadLHS = L1;
      LoadRHS = extractAlphaLoad(MulRHS, Alpha1);
      if (!LoadRHS)
        return false;
      AlphaVal = Alpha1;
    } else if (!L1 && L2) {
      LoadRHS = L2;
      LoadLHS = extractAlphaLoad(MulLHS, Alpha1);
      if (!LoadLHS)
        return false;
      AlphaVal = Alpha1;
    } else {
      LoadLHS = extractAlphaLoad(MulLHS, Alpha1);
      if (LoadLHS) {
        LoadRHS = extractLoad(MulRHS);
        if (!LoadRHS) {
          Value *Alpha2 = nullptr;
          LoadRHS = extractAlphaLoad(MulRHS, Alpha2);
        }
        AlphaVal = Alpha1;
      }
      if (!LoadLHS || !LoadRHS)
        return false;
    }
    return true;
  };

  bool Found = false;

  // --- Pattern 1: PHI-based reduction ---
  BasicBlock *InnerHeader = InnerLoop->getHeader();
  for (PHINode &PN : InnerHeader->phis()) {
    if (!PN.getType()->isFloatingPointTy())
      continue;

    Value *LatchVal =
        PN.getIncomingValueForBlock(InnerLoop->getLoopLatch());

    Value *AddLHS, *AddRHS;
    if (!match(LatchVal, m_FAdd(m_Value(AddLHS), m_Value(AddRHS))))
      continue;

    Value *Addend = nullptr;
    if (AddLHS == &PN)
      Addend = AddRHS;
    else if (AddRHS == &PN)
      Addend = AddLHS;
    else
      continue;

    if (!matchMulAccum(Addend))
      continue;

    AddOp = cast<BinaryOperator>(LatchVal);
    Found = true;
    break;
  }

  // --- Pattern 2: Load-modify-store (no reduction PHI) ---
  // Matches: store(fadd(load(ptr), fmul(...)), ptr)
  if (!Found) {
    for (BasicBlock *BB : InnerLoop->blocks()) {
      for (Instruction &I : *BB) {
        auto *SI = dyn_cast<StoreInst>(&I);
        if (!SI)
          continue;

        Value *StoreVal = SI->getValueOperand();
        Value *AddLHSVal, *AddRHSVal;
        if (!match(StoreVal, m_FAdd(m_Value(AddLHSVal), m_Value(AddRHSVal))))
          continue;

        LoadInst *LoadFromC = nullptr;
        Value *Addend = nullptr;
        if (auto *LI = dyn_cast<LoadInst>(AddLHSVal)) {
          if (LI->getPointerOperand() == SI->getPointerOperand()) {
            LoadFromC = LI;
            Addend = AddRHSVal;
          }
        }
        if (!LoadFromC) {
          if (auto *LI = dyn_cast<LoadInst>(AddRHSVal)) {
            if (LI->getPointerOperand() == SI->getPointerOperand()) {
              LoadFromC = LI;
              Addend = AddLHSVal;
            }
          }
        }
        if (!LoadFromC || !Addend)
          continue;

        if (!matchMulAccum(Addend))
          continue;

        CLoad = LoadFromC;
        CStore = SI;
        AddOp = cast<BinaryOperator>(StoreVal);
        Found = true;
        break;
      }
      if (Found)
        break;
    }
  }

  // --- Pattern 3: fmuladd/fma intrinsic ---
  // Matches: store(fmuladd(A, B, load(C_ptr)), C_ptr)
  //      or: store(fmuladd(alpha*A, B, load(C_ptr)), C_ptr)
  // The fmuladd computes A*B + C in one operation.
  if (!Found) {
    for (BasicBlock *BB : InnerLoop->blocks()) {
      for (Instruction &I : *BB) {
        auto *SI = dyn_cast<StoreInst>(&I);
        if (!SI)
          continue;

        auto *CI = dyn_cast<CallInst>(SI->getValueOperand());
        if (!CI)
          continue;

        Function *Callee = CI->getCalledFunction();
        if (!Callee)
          continue;

        auto IID = Callee->getIntrinsicID();
        if (IID != Intrinsic::fmuladd && IID != Intrinsic::fma)
          continue;

        // fmuladd(a, b, c) = a*b + c
        // The 'c' argument should be a load from the same address as the store.
        Value *MulArg0 = CI->getArgOperand(0);
        Value *MulArg1 = CI->getArgOperand(1);
        Value *AddArg = CI->getArgOperand(2);

        auto *LoadFromC = dyn_cast<LoadInst>(AddArg);
        if (!LoadFromC ||
            LoadFromC->getPointerOperand() != SI->getPointerOperand())
          continue;

        // The multiply operands (arg0, arg1) should be loads (possibly
        // with alpha scaling).
        LoadInst *L0 = extractLoad(MulArg0);
        LoadInst *L1 = extractLoad(MulArg1);
        Value *Alpha1 = nullptr;

        if (L0 && L1) {
          LoadLHS = L0;
          LoadRHS = L1;
        } else if (L0 && !L1) {
          LoadLHS = L0;
          LoadRHS = extractAlphaLoad(MulArg1, Alpha1);
          if (!LoadRHS) continue;
          AlphaVal = Alpha1;
        } else if (!L0 && L1) {
          LoadRHS = L1;
          LoadLHS = extractAlphaLoad(MulArg0, Alpha1);
          if (!LoadLHS) continue;
          AlphaVal = Alpha1;
        } else {
          LoadLHS = extractAlphaLoad(MulArg0, Alpha1);
          if (LoadLHS) {
            LoadRHS = extractLoad(MulArg1);
            AlphaVal = Alpha1;
          }
          if (!LoadLHS || !LoadRHS) continue;
        }

        CLoad = LoadFromC;
        CStore = SI;
        // AddOp is not a BinaryOperator for fmuladd; store the call instead.
        // We only need CStore for address decomposition.
        Found = true;
        break;
      }
      if (Found)
        break;
    }
  }

  if (!Found) {
    LLVM_DEBUG(dbgs() << "MMR: No matmul accumulation pattern in innermost loop\n");
    if (MatMulVerbose)
      errs() << "MMR: No matmul accumulation pattern in innermost loop\n";
    return std::nullopt;
  }

  // Decompose the load addresses to identify array access patterns.
  ArrayAccess AccLHS, AccRHS;
  if (!decomposeAddress(LoadLHS->getPointerOperand(), SE, IndVars, AccLHS) ||
      !decomposeAddress(LoadRHS->getPointerOperand(), SE, IndVars, AccRHS)) {
    LLVM_DEBUG(dbgs() << "MMR: Failed to decompose load addresses\n");
    if (MatMulVerbose)
      errs() << "MMR: Failed to decompose load addresses\n";
    return std::nullopt;
  }

  // Find the store to C.
  //
  // For pattern 2 (load-modify-store), CStore is already set above.
  // For pattern 1 (PHI reduction), we need to find the store:
  //   Case 1a: Store inside the inner loop (using AddOp)
  //   Case 1b: Store after the inner loop (scalar accumulation)
  ArrayAccess AccC;

  if (!CStore) {
    // Case 1a: Store inside inner loop.
    for (BasicBlock *BB : InnerLoop->blocks()) {
      for (Instruction &I : *BB) {
        if (auto *SI = dyn_cast<StoreInst>(&I)) {
          if (SI->getValueOperand() == AddOp) {
            CStore = SI;
            break;
          }
        }
      }
      if (CStore)
        break;
    }
  }

  // Case 1b: Scalar accumulation — store is after the inner loop.
  if (!CStore) {
    BasicBlock *InnerExit = InnerLoop->getExitBlock();
    if (InnerExit) {
      for (Instruction &I : *InnerExit) {
        if (auto *SI = dyn_cast<StoreInst>(&I)) {
          if (auto *PN = dyn_cast<PHINode>(SI->getValueOperand())) {
            if (PN->getNumIncomingValues() == 1 &&
                PN->getIncomingValue(0) == AddOp) {
              CStore = SI;
              break;
            }
          }
          if (SI->getValueOperand() == AddOp) {
            CStore = SI;
            break;
          }
        }
      }
    }
  }

  if (!CStore) {
    LLVM_DEBUG(dbgs() << "MMR: No store to C array found\n");
    if (MatMulVerbose)
      errs() << "MMR: No store to C array found\n";
    return std::nullopt;
  }

  if (!decomposeAddress(CStore->getPointerOperand(), SE, IndVars, AccC)) {
    LLVM_DEBUG(dbgs() << "MMR: Failed to decompose C store address\n");
    if (MatMulVerbose)
      errs() << "MMR: Failed to decompose C store address\n";
    return std::nullopt;
  }

  // Debug: print decomposed addresses.
  if (MatMulVerbose) {
    auto printAcc = [](raw_ostream &OS, const char *Name,
                       const ArrayAccess &Acc) {
      OS << "  " << Name << ": base=" << *Acc.Base << " indices=[";
      for (auto &[IV, Stride] : Acc.Indices)
        OS << "(" << IV->getName() << ", " << *Stride << ") ";
      OS << "]\n";
    };
    errs() << "MMR: Address decomposition:\n";
    printAcc(errs(), "C", AccC);
    printAcc(errs(), "LHS", AccLHS);
    printAcc(errs(), "RHS", AccRHS);
  }

  // Now classify roles: which access is A, B, C based on IV usage.
  MatMulInfo Info;
  Info.BaseC = AccC.Base;
  Info.ElemTy = LoadLHS->getType();
  Info.Alpha = AlphaVal;
  Info.Beta = DetectedBeta;

  // Try both orderings of the two loads: (LHS=A, RHS=B) or (LHS=B, RHS=A).
  bool Classified = false;
  if (classifyRoles(AccC, AccLHS, AccRHS, IndVars, Info)) {
    Info.BaseA = AccLHS.Base;
    Info.BaseB = AccRHS.Base;
    Classified = true;
  } else if (classifyRoles(AccC, AccRHS, AccLHS, IndVars, Info)) {
    Info.BaseA = AccRHS.Base;
    Info.BaseB = AccLHS.Base;
    Classified = true;
  }

  if (!Classified) {
    LLVM_DEBUG(dbgs() << "MMR: Failed to classify I/J/K roles\n");
    if (MatMulVerbose)
      errs() << "MMR: Failed to classify I/J/K roles\n";
    return std::nullopt;
  }

  // Map induction variables back to their loops.
  for (Loop *L : Nest) {
    PHINode *IV = getCanonicalInductionVariable(L);
    if (IV == Info.IndI)
      Info.LoopI = L;
    else if (IV == Info.IndJ)
      Info.LoopJ = L;
    else if (IV == Info.IndK)
      Info.LoopK = L;
  }

  if (!Info.LoopI || !Info.LoopJ || !Info.LoopK) {
    LLVM_DEBUG(dbgs() << "MMR: Failed to map IVs to loops\n");
    return std::nullopt;
  }

  // Extract trip counts.
  if (auto TC = SE.getSmallConstantTripCount(Info.LoopI))
    Info.TripCountI = TC;
  if (auto TC = SE.getSmallConstantTripCount(Info.LoopJ))
    Info.TripCountJ = TC;
  if (auto TC = SE.getSmallConstantTripCount(Info.LoopK))
    Info.TripCountK = TC;

  // Extract leading dimensions from SCEV strides.
  // For C[i][j], the stride of i gives us LdC (number of columns in C).
  for (auto &[IV, Stride] : AccC.Indices) {
    if (IV == Info.IndI) {
      if (auto *SC = dyn_cast<SCEVConstant>(Stride))
        Info.LdC = ConstantInt::get(IV->getType(), SC->getAPInt());
      else if (auto *SU = dyn_cast<SCEVUnknown>(Stride))
        Info.LdC = SU->getValue();
    }
  }

  return Info;
}

//===----------------------------------------------------------------------===//
// SME FMOPA Lowering
//===----------------------------------------------------------------------===//

/// Lower a recognized matmul to tiled SME FMOPA operations.
///
/// Strategy:
/// 1. Extract trip count bounds and base pointers from the original loops.
/// 2. Create a new standalone function with the tiled FMOPA body.
/// 3. Replace the original loop nest with a call to the new function.
///
/// The original loops become dead code after the replacement.
static bool lowerMatMulToSME(MatMulInfo &MMI, Function &F, LoopInfo &LI,
                              DominatorTree &DT, ScalarEvolution &SE,
                              AssumptionCache *AC) {
  if (!MMI.ElemTy->isFloatTy()) {
    LLVM_DEBUG(dbgs() << "MMR: SME lowering only supports f32 currently\n");
    return false;
  }

  // 1. Extract trip count bounds from loop exit conditions.
  Value *BoundI = getLoopBoundValue(MMI.LoopI);
  Value *BoundJ = getLoopBoundValue(MMI.LoopJ);
  Value *BoundK = getLoopBoundValue(MMI.LoopK);
  if (!BoundI || !BoundJ || !BoundK) {
    LLVM_DEBUG(dbgs() << "MMR: Failed to extract loop bounds\n");
    if (MatMulVerbose)
      errs() << "MMR: Failed to extract loop bounds\n";
    return false;
  }

  // For canonical GEMM with C[NI][NJ], A[NI][NK], B[NK][NJ]:
  //   LdC = NJ, LdA = NK, LdB = NJ (number of columns in each matrix).
  // These are the trip counts of the appropriate loops.
  Value *LdC = BoundJ;
  Value *LdA = BoundK;
  Value *LdB = BoundJ;

  // 2. Find the outermost loop of the matmul nest.
  Loop *OuterLoop = MMI.LoopI;
  if (MMI.LoopJ->getLoopDepth() < OuterLoop->getLoopDepth())
    OuterLoop = MMI.LoopJ;
  if (MMI.LoopK->getLoopDepth() < OuterLoop->getLoopDepth())
    OuterLoop = MMI.LoopK;

  // 3. Create the FMOPA function.
  Module &M = *F.getParent();
  std::string FnName = (F.getName() + ".matmul_fmopa").str();
  Function *FmopaFn = createAndBuildFMOPAFunction(M, FnName);

  // Propagate target-cpu if present.
  if (F.hasFnAttribute("target-cpu"))
    FmopaFn->addFnAttr(F.getFnAttribute("target-cpu"));
  if (F.hasFnAttribute("target-features")) {
    StringRef Features =
        F.getFnAttribute("target-features").getValueAsString();
    if (!Features.contains("+sme")) {
      std::string NewFeatures = (Features + ",+sme").str();
      FmopaFn->addFnAttr("target-features", NewFeatures);
    } else {
      FmopaFn->addFnAttr("target-features", Features);
    }
  }

  LLVM_DEBUG(dbgs() << "MMR: Created FMOPA function " << FmopaFn->getName()
                    << "\n");

  // 4. Insert a call to the FMOPA function before the loop and bypass the
  //    original loop.
  BasicBlock *Preheader = OuterLoop->getLoopPreheader();
  if (!Preheader) {
    LLVM_DEBUG(dbgs() << "MMR: No loop preheader\n");
    return false;
  }

  IRBuilder<> B(Preheader->getTerminator());
  Type *I64 = B.getInt64Ty();

  // Widen bounds to i64 if needed.
  auto toI64 = [&](Value *V) -> Value * {
    if (V->getType() == I64)
      return V;
    return B.CreateSExt(V, I64, V->getName() + ".i64");
  };

  // Alpha defaults to 1.0, Beta defaults to 0.0 (pure C = A*B).
  // If the recognizer found alpha, use it. Beta comes from the separate
  // scaling loop — for the full GEMM (C = beta*C + alpha*A*B), we pass
  // beta from the original code. If no beta was detected, use 0.0
  // (meaning C is initialized to zero before accumulation).
  Type *F32Ty = B.getFloatTy();
  Value *AlphaVal = MMI.Alpha ? MMI.Alpha : ConstantFP::get(F32Ty, 1.0);
  // For PolyBench GEMM, beta scaling happens in a sibling loop that we're
  // bypassing. We need to fold it into the FMOPA function. If MMI.Beta is
  // set, use it; otherwise use 1.0 (C += A*B pattern, C is already initialized).
  Value *BetaVal = MMI.Beta ? MMI.Beta : ConstantFP::get(F32Ty, 1.0);

  // Ensure alpha/beta are f32.
  if (AlphaVal->getType() != F32Ty)
    AlphaVal = B.CreateFPCast(AlphaVal, F32Ty, "alpha.f32");
  if (BetaVal->getType() != F32Ty)
    BetaVal = B.CreateFPCast(BetaVal, F32Ty, "beta.f32");

  Value *Args[] = {
      MMI.BaseC, MMI.BaseA, MMI.BaseB,
      toI64(BoundI), toI64(BoundJ), toI64(BoundK),
      toI64(LdC), toI64(LdA), toI64(LdB),
      AlphaVal, BetaVal,
  };
  B.CreateCall(FmopaFn, Args);

  // Redirect the preheader to skip the original loop.
  BasicBlock *ExitBlock = OuterLoop->getExitBlock();
  if (!ExitBlock) {
    LLVM_DEBUG(dbgs() << "MMR: No single exit block\n");
    return false;
  }

  // Update any PHI nodes in the exit block that receive values from
  // inside the loop — replace them with undef since the loop is dead.
  BasicBlock *Header = OuterLoop->getHeader();
  for (PHINode &PN : ExitBlock->phis()) {
    for (unsigned i = 0; i < PN.getNumIncomingValues(); i++) {
      BasicBlock *InBB = PN.getIncomingBlock(i);
      if (OuterLoop->contains(InBB)) {
        PN.setIncomingBlock(i, Preheader);
        PN.setIncomingValue(i, UndefValue::get(PN.getType()));
      }
    }
  }

  Preheader->getTerminator()->eraseFromParent();
  BranchInst::Create(ExitBlock, Preheader);

  // Make the old loop blocks unreachable and let later cleanup passes
  // (simplifycfg) remove them. We can't safely delete them here because
  // other blocks may still reference them through PHI nodes or branches.
  // Detaching the preheader is sufficient — the loop is now unreachable.

  if (MatMulVerbose)
    errs() << "MMR: Lowered matmul to FMOPA: " << FmopaFn->getName() << "\n";

  ++NumMatMulsLowered;
  return true;
}

//===----------------------------------------------------------------------===//
// Pass Entry Point
//===----------------------------------------------------------------------===//

} // anonymous namespace

PreservedAnalyses MatMulRecognizeAndLowerPass::run(Function &F,
                                                     FunctionAnalysisManager &AM) {
  // Skip if function is already streaming or has opt-none.
  if (F.hasFnAttribute("aarch64_pstate_sm_enabled") ||
      F.hasFnAttribute("aarch64_pstate_sm_body") ||
      F.hasFnAttribute(Attribute::OptimizeNone) || F.empty())
    return PreservedAnalyses::all();

  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto *AC = AM.getCachedResult<AssumptionAnalysis>(F);

  bool Changed = false;

  // Collect ALL loops as matmul candidates — not just top-level.
  // After inlining, matmul loops may be nested inside other loops.
  SmallVector<Loop *, 16> AllLoops;
  std::function<void(Loop *)> collectLoops = [&](Loop *L) {
    AllLoops.push_back(L);
    for (Loop *Sub : *L)
      collectLoops(Sub);
  };
  for (Loop *L : LI)
    collectLoops(L);

  if (MatMulVerbose)
    errs() << "MMR: Found " << AllLoops.size() << " loops in " << F.getName()
           << "\n";

  for (Loop *L : AllLoops) {
    auto MMI = analyzeLoopNest(L, LI, SE);
    if (!MMI)
      continue;

    ++NumMatMulsRecognized;

    if (MatMulVerbose) {
      errs() << "MMR: Recognized matmul in " << F.getName() << ":\n"
             << "  C[" << *MMI->IndI << "][" << *MMI->IndJ << "] += "
             << (MMI->Alpha ? "alpha * " : "")
             << "A[" << (MMI->ATransposed ? "k" : "i") << "]["
             << (MMI->ATransposed ? "i" : "k") << "] * "
             << "B[" << (MMI->BTransposed ? "j" : "k") << "]["
             << (MMI->BTransposed ? "k" : "j") << "]\n"
             << "  Element type: " << *MMI->ElemTy << "\n"
             << "  Trip counts: I=" << MMI->TripCountI
             << " J=" << MMI->TripCountJ << " K=" << MMI->TripCountK << "\n";
    }

    LLVM_DEBUG(dbgs() << "MMR: Recognized matmul pattern in " << F.getName()
                      << " with element type " << *MMI->ElemTy << "\n");

    if (RecognizeOnly)
      continue;

    // Profitability check: skip if known trip counts are too small.
    if (MMI->TripCountI && MMI->TripCountI < MinTripCount) {
      LLVM_DEBUG(dbgs() << "MMR: Skipping — I trip count " << MMI->TripCountI
                        << " below threshold " << MinTripCount << "\n");
      continue;
    }
    if (MMI->TripCountJ && MMI->TripCountJ < MinTripCount) {
      LLVM_DEBUG(dbgs() << "MMR: Skipping — J trip count " << MMI->TripCountJ
                        << " below threshold " << MinTripCount << "\n");
      continue;
    }

    if (lowerMatMulToSME(*MMI, F, LI, DT, SE, AC))
      Changed = true;
  }

  if (!Changed)
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}
