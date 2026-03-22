//===- LoopStreamingSwitcher.cpp - Per-loop SSVE/NEON switching ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass uses a trained decision tree to decide per-loop whether Streaming
// SVE (SSVE) or NEON is faster. Loops predicted to benefit from SSVE are
// outlined into separate functions marked with aarch64_pstate_sm_body.
//
// The pass runs BEFORE the Loop Vectorizer so that the LV sees the streaming
// attribute on outlined functions and uses scalable SVE vectorization.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LoopStreamingSwitcher.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopVectorFeatures.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"

using namespace llvm;

#define DEBUG_TYPE "loop-streaming-switcher"

STATISTIC(NumOutlined, "Number of loops outlined for streaming SVE");

static cl::opt<bool> AutoStreamingMode(
    "loop-vectorize-auto-streaming", cl::init(false), cl::Hidden,
    cl::desc("Automatically select SSVE vs NEON per-loop using a trained "
             "decision tree. Outlines SSVE-beneficial loops into streaming "
             "functions before the Loop Vectorizer runs."));

static cl::opt<bool> SwitcherVerbose(
    "loop-streaming-switcher-verbose", cl::init(false), cl::Hidden,
    cl::desc("Print per-loop SSVE/NEON decisions to stderr."));

//===----------------------------------------------------------------------===//
// Decision Tree Classifier
//===----------------------------------------------------------------------===//

namespace {

/// Auto-generated from decision tree (depth=11, 31 leaves, 17 features)
/// Features used: ci_num_fp_arith_ops, ci_num_int_arith_ops, ci_num_total_ops,
/// ci_ops_per_memory, dep_num_dependences, dtype_max_element_size_bytes,
/// dtype_num_32bit_accesses, dtype_num_8bit_accesses, lb_final_iv_value,
/// lb_initial_iv_value, loop_depth, num_stores, stride_max_stride_bytes,
/// stride_min_stride_bytes, stride_num_non_unit_stride,
/// stride_num_unique_base_ptrs, trip_count_value
/// Classes: 0 = NEON, 1 = SSVE
static bool shouldUseStreamingSVE(const LoopVectorFeatures &F) {
  if (F.stride_min_stride_bytes <= 384) {
    if (F.ci_ops_per_memory <= 1.225000) {
      if (F.stride_num_unique_base_ptrs <= 5) {
        if (F.ci_num_int_arith_ops <= 133) {
          if (F.dtype_num_32bit_accesses <= 190) {
            if (F.dtype_num_8bit_accesses <= 6) {
              if (F.dtype_num_32bit_accesses <= 2) {
                if (F.loop_depth <= 2) {
                  if (F.ci_num_int_arith_ops <= 1) {
                    if (F.lb_final_iv_value <= 47) {
                      if (F.trip_count_value <= 31) {
                        return false; // NEON (100%)
                      } else {
                        return false; // NEON (99%)
                      }
                    } else {
                      return false; // NEON (100%)
                    }
                  } else {
                    if (F.ci_num_total_ops <= 21) {
                      return false; // NEON (93%)
                    } else {
                      if (F.lb_initial_iv_value <= 0) {
                        return false; // NEON (100%)
                      } else {
                        return false; // NEON (100%)
                      }
                    }
                  }
                } else {
                  return false; // NEON (99%)
                }
              } else {
                if (F.num_stores <= 4) {
                  return false; // NEON (80%)
                } else {
                  if (F.dep_num_dependences <= 58) {
                    return false; // NEON (100%)
                  } else {
                    return false; // NEON (91%)
                  }
                }
              }
            } else {
              if (F.stride_num_non_unit_stride <= 39) {
                return false; // NEON (85%)
              } else {
                return false; // NEON (100%)
              }
            }
          } else {
            return false; // NEON (80%)
          }
        } else {
          if (F.ci_ops_per_memory <= 0.509273) {
            return false; // NEON (100%)
          } else {
            return true; // SSVE (90%)
          }
        }
      } else {
        if (F.dtype_max_element_size_bytes <= 3) {
          return false; // NEON (86%)
        } else {
          return true; // SSVE (69%)
        }
      }
    } else {
      if (F.ci_num_int_arith_ops <= 2) {
        return false; // NEON (90%)
      } else {
        if (F.ci_ops_per_memory <= 1.516667) {
          if (F.stride_max_stride_bytes <= 3) {
            if (F.lb_initial_iv_value <= 1) {
              return true; // SSVE (60%)
            } else {
              return false; // NEON (87%)
            }
          } else {
            if (F.stride_max_stride_bytes <= 384) {
              if (F.lb_final_iv_value <= 96) {
                if (F.stride_max_stride_bytes <= 96) {
                  return false; // NEON (73%)
                } else {
                  return true; // SSVE (88%)
                }
              } else {
                return true; // SSVE (94%)
              }
            } else {
              return false; // NEON (77%)
            }
          }
        } else {
          if (F.ci_ops_per_memory <= 2.125000) {
            return false; // NEON (100%)
          } else {
            return false; // NEON (94%)
          }
        }
      }
    }
  } else {
    if (F.ci_num_fp_arith_ops <= 0) {
      return false; // NEON (99%)
    } else {
      if (F.dtype_max_element_size_bytes <= 6) {
        if (F.stride_max_stride_bytes <= 768) {
          return true; // SSVE (96%)
        } else {
          return true; // SSVE (100%)
        }
      } else {
        if (F.trip_count_value <= 81) {
          return false; // NEON (75%)
        } else {
          return true; // SSVE (96%)
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Loop Outlining
//===----------------------------------------------------------------------===//

static Function *outlineLoopForStreaming(Loop *L, DominatorTree &DT,
                                         AssumptionCache *AC,
                                         Function &ParentFunc) {
  if (!L->isLoopSimplifyForm()) {
    LLVM_DEBUG(dbgs() << "LSS: Skipping loop not in LoopSimplify form: "
                      << L->getLocStr() << "\n");
    return nullptr;
  }

  CodeExtractorAnalysisCache CEAC(ParentFunc);
  CodeExtractor Extractor(L->getBlocks(), &DT, /*AggregateArgs=*/false,
                          /*BFI=*/nullptr, /*BPI=*/nullptr, AC);

  Function *NewFunc = Extractor.extractCodeRegion(CEAC);
  if (!NewFunc) {
    LLVM_DEBUG(dbgs() << "LSS: CodeExtractor failed for loop: "
                      << L->getLocStr() << "\n");
    return nullptr;
  }

  NewFunc->addFnAttr("aarch64_pstate_sm_body");

  if (ParentFunc.hasFnAttribute("target-features")) {
    StringRef Features =
        ParentFunc.getFnAttribute("target-features").getValueAsString();
    if (Features.contains("+sme")) {
      NewFunc->addFnAttr("target-features", Features);
    } else {
      std::string NewFeatures = (Features + ",+sme").str();
      NewFunc->addFnAttr("target-features", NewFeatures);
    }
  } else {
    NewFunc->addFnAttr("target-features", "+sme");
  }

  if (ParentFunc.hasFnAttribute("target-cpu"))
    NewFunc->addFnAttr(ParentFunc.getFnAttribute("target-cpu"));

  ++NumOutlined;
  return NewFunc;
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

PreservedAnalyses LoopStreamingSwitcherPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  if (!AutoStreamingMode)
    return PreservedAnalyses::all();

  if (F.hasFnAttribute("aarch64_pstate_sm_enabled") ||
      F.hasFnAttribute("aarch64_pstate_sm_body"))
    return PreservedAnalyses::all();

  if (F.hasOptNone() || F.empty())
    return PreservedAnalyses::all();

  auto &LI = AM.getResult<LoopAnalysis>(F);
  if (LI.empty())
    return PreservedAnalyses::all();

  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto *AC = AM.getCachedResult<AssumptionAnalysis>(F);
  auto &LAIs = AM.getResult<LoopAccessAnalysis>(F);
  const DataLayout &DL = F.getDataLayout();

  SmallVector<Loop *, 8> InnermostLoops;
  for (Loop *L : LI) {
    SmallVector<Loop *, 8> Stack;
    Stack.push_back(L);
    while (!Stack.empty()) {
      Loop *Current = Stack.pop_back_val();
      if (Current->getSubLoops().empty())
        InnermostLoops.push_back(Current);
      for (Loop *Sub : Current->getSubLoops())
        Stack.push_back(Sub);
    }
  }

  for (Loop *L : InnermostLoops) {
    if (!L->isLoopSimplifyForm())
      continue;

    {
      bool HasNonIntrinsicCall = false;
      for (BasicBlock *BB : L->blocks()) {
        for (Instruction &I : *BB) {
          if (auto *CI = dyn_cast<CallInst>(&I)) {
            Function *Callee = CI->getCalledFunction();
            if (!Callee || !Callee->isIntrinsic()) {
              HasNonIntrinsicCall = true;
              break;
            }
          }
        }
        if (HasNonIntrinsicCall)
          break;
      }
      if (HasNonIntrinsicCall) {
        LLVM_DEBUG(dbgs() << "LSS: Skipping loop with non-intrinsic calls: "
                          << L->getLocStr() << "\n");
        if (SwitcherVerbose)
          errs() << "LSS: NEON — " << F.getName() << " " << L->getLocStr() << "\n";
        continue;
      }
    }

    const LoopAccessInfo &LAI = LAIs.getInfo(*L);

    if (!LAI.canVectorizeMemory()) {
      LLVM_DEBUG(dbgs() << "LSS: Skipping loop — LAI can't vectorize memory: "
                        << L->getLocStr() << "\n");
      if (SwitcherVerbose)
        errs() << "LSS: NEON — " << F.getName() << " " << L->getLocStr() << "\n";
      continue;
    }

    LoopVectorFeatures Features = extractLoopVectorFeatures(L, SE, LAI, DL);

    if (Features.reduction_count > 0) {
      LLVM_DEBUG(dbgs() << "LSS: Skipping loop with reductions: "
                        << L->getLocStr() << "\n");
      if (SwitcherVerbose)
        errs() << "LSS: NEON — " << F.getName() << " " << L->getLocStr() << "\n";
      continue;
    }

    if (!shouldUseStreamingSVE(Features)) {
      LLVM_DEBUG(dbgs() << "LSS: Loop stays NEON: " << L->getLocStr() << "\n");
      if (SwitcherVerbose)
        errs() << "LSS: NEON — " << F.getName() << " " << L->getLocStr() << "\n";
      continue;
    }

    LLVM_DEBUG(dbgs() << "LSS: Decision tree predicts SSVE for loop: "
                      << L->getLocStr() << "\n");
    if (SwitcherVerbose)
      errs() << "LSS: SSVE — " << F.getName() << " " << L->getLocStr() << "\n";

    Function *NewFunc = outlineLoopForStreaming(L, DT, AC, F);
    if (NewFunc) {
      LLVM_DEBUG(dbgs() << "LSS: Outlined loop into streaming function: "
                        << NewFunc->getName() << "\n");
      return PreservedAnalyses::none();
    }
  }

  return PreservedAnalyses::all();
}
