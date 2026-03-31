//===- MatMulSMELower.cpp - Tiled FMOPA IR generation for SME matmul -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Creates an SME FMOPA function for a recognized matmul pattern and replaces
// the original loop nest with a call to it.
//
// The generated function has this structure (for f32):
//
//   @matmul_fmopa(ptr %C, ptr %A, ptr %B,
//                 i64 %NI, i64 %NJ, i64 %NK, i64 %LdC, i64 %LdA, i64 %LdB)
//   {
//     %svl = vscale * 4
//     for (ii = 0; ii < NI; ii += svl)
//       for (jj = 0; jj < NJ; jj += svl)
//         sme.zero(za0)
//         %pi = whilelt(ii, NI)
//         %pj = whilelt(jj, NJ)
//         for (k = 0; k < NK; k++)
//           %a = gather A[ii..ii+svl-1][k]   (stride = LdA)
//           %b = contig  B[k][jj..jj+svl-1]
//           sme.mopa(za0, pi, pj, a, b)
//         for (row = 0; row < min(svl, NI-ii); row++)
//           %zr = sme.read.horiz(za0, row)
//           %c_old = load C[ii+row][jj..jj+svl-1]
//           %c_new = fadd %c_old, %zr
//           masked.store(%c_new, &C[ii+row][jj], pj)
//   }
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/MatMulRecognize.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "matmul-sme-lower"

namespace llvm {

//===----------------------------------------------------------------------===//
// Intrinsic emission helpers
//===----------------------------------------------------------------------===//

static void emitSMEZero(IRBuilder<> &B, unsigned TileMask) {
  Module *M = B.GetInsertBlock()->getModule();
  Function *Fn =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::aarch64_sme_zero);
  B.CreateCall(Fn, {B.getInt32(TileMask)});
}

static void emitFMOPA_F32(IRBuilder<> &B, unsigned TileIdx, Value *PredN,
                           Value *PredM, Value *VecN, Value *VecM) {
  Module *M = B.GetInsertBlock()->getModule();
  auto *VecTy = ScalableVectorType::get(B.getFloatTy(), 4);
  Function *Fn = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::aarch64_sme_mopa, {VecTy});
  B.CreateCall(Fn, {B.getInt32(TileIdx), PredN, PredM, VecN, VecM});
}

static Value *emitSMEReadHoriz_F32(IRBuilder<> &B, unsigned TileIdx,
                                    Value *SliceIdx, Value *Pred) {
  Module *M = B.GetInsertBlock()->getModule();
  auto *VecTy = ScalableVectorType::get(B.getFloatTy(), 4);
  Function *Fn = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::aarch64_sme_read_horiz, {VecTy});
  Value *Passthrough = ConstantAggregateZero::get(VecTy);
  return B.CreateCall(Fn,
                      {Passthrough, Pred, B.getInt32(TileIdx), SliceIdx});
}

static Value *emitWhileLT(IRBuilder<> &B, Value *Start, Value *End) {
  Module *M = B.GetInsertBlock()->getModule();
  auto *PredTy = ScalableVectorType::get(B.getInt1Ty(), 4);
  Function *Fn = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::aarch64_sve_whilelt, {PredTy, Start->getType()});
  return B.CreateCall(Fn, {Start, End});
}

//===----------------------------------------------------------------------===//
// FMOPA function builder
//===----------------------------------------------------------------------===//

/// Build the complete FMOPA function body with pre-transposed A strips.
///
/// Structure:
///   for ii = 0..NI step SVL:
///     pre-transpose A[ii:ii+SVL][0:NK] into AT_buf (column-major)
///       AT_buf[k * SVL + lane] = A[(ii+lane) * LdA + k]
///     for jj = 0..NJ step SVL:
///       sme.zero(za0)
///       pi = whilelt(ii, NI)
///       pj = whilelt(jj, NJ)
///       for k = 0..NK:
///         a_col = contiguous_load(AT_buf + k * SVL)  // now contiguous!
///         b_row = contiguous_load(B + k * LdB + jj)
///         fmopa(za0, pi, pj, a_col, b_row)
///       store ZA tile rows back to C
///
/// The pre-transpose hoists all strided A accesses out of the jj and k loops.
/// Cost: NK * SVL scalar loads per ii tile (vs NK * SVL per (ii,jj) pair before).
///
/// Arguments: (ptr %C, ptr %A, ptr %B, i64 %NI, i64 %NJ, i64 %NK,
///             i64 %LdC, i64 %LdA, i64 %LdB)
static void buildFMOPABody(Function *Fn) {
  LLVMContext &Ctx = Fn->getContext();
  Module *M = Fn->getParent();
  IRBuilder<> B(Ctx);

  auto *ArgC = Fn->getArg(0);   ArgC->setName("C");
  auto *ArgA = Fn->getArg(1);   ArgA->setName("A");
  auto *ArgB = Fn->getArg(2);   ArgB->setName("B");
  auto *ArgNI = Fn->getArg(3);  ArgNI->setName("NI");
  auto *ArgNJ = Fn->getArg(4);  ArgNJ->setName("NJ");
  auto *ArgNK = Fn->getArg(5);  ArgNK->setName("NK");
  auto *ArgLdC = Fn->getArg(6); ArgLdC->setName("LdC");
  auto *ArgLdA = Fn->getArg(7); ArgLdA->setName("LdA");
  auto *ArgLdB = Fn->getArg(8); ArgLdB->setName("LdB");
  auto *ArgAlpha = Fn->getArg(9); ArgAlpha->setName("alpha");
  auto *ArgBeta = Fn->getArg(10); ArgBeta->setName("beta");

  Type *I64 = Type::getInt64Ty(Ctx);
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *F32 = Type::getFloatTy(Ctx);
  auto *SVF32 = ScalableVectorType::get(F32, 4);
  Value *Zero64 = ConstantInt::get(I64, 0);
  Value *One64 = ConstantInt::get(I64, 1);

  // Create all blocks.
  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn);
  BasicBlock *TileIHdr = BasicBlock::Create(Ctx, "tile.i.hdr", Fn);
  // Pre-transpose A strip blocks.
  BasicBlock *XposeKHdr = BasicBlock::Create(Ctx, "xpose.k.hdr", Fn);
  BasicBlock *XposeLaneHdr = BasicBlock::Create(Ctx, "xpose.lane.hdr", Fn);
  BasicBlock *XposeLaneBody = BasicBlock::Create(Ctx, "xpose.lane.body", Fn);
  BasicBlock *XposeLaneLatch = BasicBlock::Create(Ctx, "xpose.lane.latch", Fn);
  BasicBlock *XposeKLatch = BasicBlock::Create(Ctx, "xpose.k.latch", Fn);
  // FMOPA tile loops.
  BasicBlock *TileJHdr = BasicBlock::Create(Ctx, "tile.j.hdr", Fn);
  BasicBlock *KHdr = BasicBlock::Create(Ctx, "k.hdr", Fn);
  BasicBlock *KBody = BasicBlock::Create(Ctx, "k.body", Fn);
  BasicBlock *KLatch = BasicBlock::Create(Ctx, "k.latch", Fn);
  BasicBlock *StoreHdr = BasicBlock::Create(Ctx, "store.hdr", Fn);
  BasicBlock *StoreBody = BasicBlock::Create(Ctx, "store.body", Fn);
  BasicBlock *StoreLatch = BasicBlock::Create(Ctx, "store.latch", Fn);
  BasicBlock *TileJLatch = BasicBlock::Create(Ctx, "tile.j.latch", Fn);
  BasicBlock *TileILatch = BasicBlock::Create(Ctx, "tile.i.latch", Fn);
  BasicBlock *Exit = BasicBlock::Create(Ctx, "exit", Fn);

  // --- Entry block ---
  B.SetInsertPoint(Entry);
  Value *VScale = B.CreateVScale(I64, "vscale");
  Value *SVL = B.CreateMul(VScale, ConstantInt::get(I64, 4), "svl");
  // Allocate pre-transpose buffer: SVL * NK floats.
  // AT_buf[k * SVL + lane] = A[(ii+lane) * LdA + k]
  // Use malloc/free since the size is runtime-determined and can be large.
  Value *BufSize = B.CreateMul(SVL, ArgNK, "at.buf.size");
  Value *BufBytes = B.CreateMul(BufSize, ConstantInt::get(I64, 4), "at.bytes");
  Value *ATBufRaw = B.CreateCall(
      M->getOrInsertFunction(
          "malloc", FunctionType::get(PointerType::get(Ctx, 0), {I64}, false)),
      {BufBytes}, "at.buf.raw");
  Value *ATBuf = ATBufRaw;
  B.CreateBr(TileIHdr);

  // --- Tile I loop header ---
  B.SetInsertPoint(TileIHdr);
  PHINode *II = B.CreatePHI(I64, 2, "ii");
  II->addIncoming(Zero64, Entry);
  // Compute clamp for this tile: min(SVL, NI - ii).
  Function *UMinFn = Intrinsic::getOrInsertDeclaration(M, Intrinsic::umin, {I64});
  Value *RemI = B.CreateSub(ArgNI, II, "rem.i");
  Value *TileRows = B.CreateCall(UMinFn, {SVL, RemI}, "tile.rows");
  B.CreateBr(XposeKHdr);

  // ===================================================================
  // Pre-transpose A strip: A[ii:ii+SVL][0:NK] → AT_buf (column-major)
  //   for k = 0..NK:
  //     for lane = 0..TileRows:
  //       AT_buf[k * SVL + lane] = A[(ii+lane) * LdA + k]
  // ===================================================================

  // --- Xpose K loop header ---
  B.SetInsertPoint(XposeKHdr);
  PHINode *XK = B.CreatePHI(I64, 2, "xk");
  XK->addIncoming(Zero64, TileIHdr);
  B.CreateBr(XposeLaneHdr);

  // --- Xpose Lane loop header ---
  B.SetInsertPoint(XposeLaneHdr);
  PHINode *XLane = B.CreatePHI(I64, 2, "xlane");
  XLane->addIncoming(Zero64, XposeKHdr);
  B.CreateBr(XposeLaneBody);

  // --- Xpose Lane body ---
  B.SetInsertPoint(XposeLaneBody);
  // Source: A[(ii + xlane) * LdA + xk]
  Value *ASrcRow = B.CreateAdd(II, XLane, "a.src.row");
  Value *ASrcOff = B.CreateAdd(B.CreateMul(ASrcRow, ArgLdA), XK, "a.src.off");
  Value *ASrcPtr = B.CreateGEP(F32, ArgA, ASrcOff, "a.src.ptr");
  Value *AVal = B.CreateLoad(F32, ASrcPtr, "a.val");
  // Dest: AT_buf[xk * SVL + xlane]
  Value *ADstOff = B.CreateAdd(B.CreateMul(XK, SVL), XLane, "at.dst.off");
  Value *ADstPtr = B.CreateGEP(F32, ATBuf, ADstOff, "at.dst.ptr");
  B.CreateStore(AVal, ADstPtr);
  B.CreateBr(XposeLaneLatch);

  // --- Xpose Lane latch ---
  B.SetInsertPoint(XposeLaneLatch);
  Value *XLaneNext = B.CreateAdd(XLane, One64, "xlane.next");
  Value *XLaneCond = B.CreateICmpULT(XLaneNext, TileRows, "xlane.cond");
  B.CreateCondBr(XLaneCond, XposeLaneHdr, XposeKLatch);
  XLane->addIncoming(XLaneNext, XposeLaneLatch);

  // --- Xpose K latch ---
  B.SetInsertPoint(XposeKLatch);
  Value *XKNext = B.CreateAdd(XK, One64, "xk.next");
  Value *XKCond = B.CreateICmpSLT(XKNext, ArgNK, "xk.cond");
  B.CreateCondBr(XKCond, XposeKHdr, TileJHdr);
  XK->addIncoming(XKNext, XposeKLatch);

  // ===================================================================
  // FMOPA computation with pre-transposed A
  // ===================================================================

  // --- Tile J loop header ---
  B.SetInsertPoint(TileJHdr);
  PHINode *JJ = B.CreatePHI(I64, 2, "jj");
  JJ->addIncoming(Zero64, XposeKLatch);

  emitSMEZero(B, 255); // 0xFF = zero ALL of ZA (all d-tile overlaps)
  Value *PredI = emitWhileLT(B, II, ArgNI);
  Value *PredJ = emitWhileLT(B, JJ, ArgNJ);
  B.CreateBr(KHdr);

  // --- K loop header ---
  B.SetInsertPoint(KHdr);
  PHINode *K = B.CreatePHI(I64, 2, "k");
  K->addIncoming(Zero64, TileJHdr);
  B.CreateBr(KBody);

  // --- K loop body ---
  B.SetInsertPoint(KBody);

  // Load A column from pre-transposed buffer: AT_buf[k * SVL .. k * SVL + SVL-1]
  Value *ATOff = B.CreateMul(K, SVL, "at.off");
  Value *ATPtr = B.CreateGEP(F32, ATBuf, ATOff, "at.ptr");
  Value *AVec = B.CreateMaskedLoad(SVF32, ATPtr, Align(4), PredI,
                                   ConstantAggregateZero::get(SVF32), "a.vec");

  // Load B row: B[k][jj..jj+SVL-1] (contiguous).
  Value *BBaseOff = B.CreateAdd(B.CreateMul(K, ArgLdB, "b.row.off"), JJ, "b.off");
  Value *BPtr = B.CreateGEP(F32, ArgB, BBaseOff, "b.ptr");
  Value *BVec = B.CreateMaskedLoad(SVF32, BPtr, Align(4), PredJ,
                                   ConstantAggregateZero::get(SVF32), "b.vec");

  // Scale A vector by alpha: a_vec = alpha * a_vec.
  Value *AlphaSplat = B.CreateVectorSplat(ElementCount::getScalable(4), ArgAlpha,
                                          "alpha.splat");
  Value *AScaled = B.CreateFMul(AVec, AlphaSplat, "a.scaled");

  // FMOPA: za0 += (alpha * A_col) ⊗ B_row
  emitFMOPA_F32(B, /*TileIdx=*/0, PredI, PredJ, AScaled, BVec);
  B.CreateBr(KLatch);

  // --- K loop latch ---
  B.SetInsertPoint(KLatch);
  Value *KNext = B.CreateAdd(K, One64, "k.next");
  Value *KCond = B.CreateICmpSLT(KNext, ArgNK, "k.cond");
  B.CreateCondBr(KCond, KHdr, StoreHdr);
  K->addIncoming(KNext, KLatch);

  // --- Store loop: read ZA rows back to C ---
  B.SetInsertPoint(StoreHdr);
  Value *StoreBound = B.CreateCall(UMinFn, {SVL, RemI}, "store.bound");
  B.CreateBr(StoreBody);

  B.SetInsertPoint(StoreBody);
  PHINode *Row = B.CreatePHI(I64, 2, "row");
  Row->addIncoming(Zero64, StoreHdr);

  Value *RowI32 = B.CreateTrunc(Row, I32, "row.i32");
  Value *ZARow = emitSMEReadHoriz_F32(B, /*TileIdx=*/0, RowI32, PredJ);

  Value *AbsRow = B.CreateAdd(II, Row, "abs.row");
  Value *CBaseOff = B.CreateAdd(B.CreateMul(AbsRow, ArgLdC, "c.row.off"), JJ, "c.off");
  Value *CPtr = B.CreateGEP(F32, ArgC, CBaseOff, "c.ptr");

  Value *COld = B.CreateMaskedLoad(SVF32, CPtr, Align(4), PredJ,
                                   ConstantAggregateZero::get(SVF32), "c.old");
  // C_new = beta * C_old + ZA_row  (implements C = beta*C + alpha*A*B)
  Value *BetaSplat = B.CreateVectorSplat(ElementCount::getScalable(4), ArgBeta,
                                         "beta.splat");
  Value *CScaled = B.CreateFMul(COld, BetaSplat, "c.scaled");
  Value *CNew = B.CreateFAdd(CScaled, ZARow, "c.new");
  B.CreateMaskedStore(CNew, CPtr, Align(4), PredJ);
  B.CreateBr(StoreLatch);

  // --- Store loop latch ---
  B.SetInsertPoint(StoreLatch);
  Value *RowNext = B.CreateAdd(Row, One64, "row.next");
  Value *RowCond = B.CreateICmpULT(RowNext, StoreBound, "row.cond");
  B.CreateCondBr(RowCond, StoreBody, TileJLatch);
  Row->addIncoming(RowNext, StoreLatch);

  // --- Tile J latch ---
  B.SetInsertPoint(TileJLatch);
  Value *JJNext = B.CreateAdd(JJ, SVL, "jj.next");
  Value *JJCond = B.CreateICmpSLT(JJNext, ArgNJ, "jj.cond");
  B.CreateCondBr(JJCond, TileJHdr, TileILatch);
  JJ->addIncoming(JJNext, TileJLatch);

  // --- Tile I latch ---
  B.SetInsertPoint(TileILatch);
  Value *IINext = B.CreateAdd(II, SVL, "ii.next");
  Value *IICond = B.CreateICmpSLT(IINext, ArgNI, "ii.cond");
  B.CreateCondBr(IICond, TileIHdr, Exit);
  II->addIncoming(IINext, TileILatch);

  // --- Exit ---
  B.SetInsertPoint(Exit);
  B.CreateCall(
      M->getOrInsertFunction(
          "free", FunctionType::get(Type::getVoidTy(Ctx),
                                    {PointerType::get(Ctx, 0)}, false)),
      {ATBufRaw});
  B.CreateRetVoid();
}

//===----------------------------------------------------------------------===//
// Public entry point
//===----------------------------------------------------------------------===//

/// Extract the trip count bound (the upper limit value) from a canonical
/// loop's exit condition.  For `for (i = 0; i < N; i++)`, returns N.
static Value *getLoopBound(Loop *L) {
  BasicBlock *Latch = L->getLoopLatch();
  if (!Latch)
    return nullptr;
  auto *BI = dyn_cast<BranchInst>(Latch->getTerminator());
  if (!BI || !BI->isConditional())
    return nullptr;
  auto *Cmp = dyn_cast<ICmpInst>(BI->getCondition());
  if (!Cmp)
    return nullptr;
  // The comparison is typically: icmp slt %iv.next, %bound
  // The bound is the operand that is NOT the incremented IV.
  // Heuristic: the IV's increment is defined inside the loop; the bound
  // is loop-invariant.
  Value *Op0 = Cmp->getOperand(0);
  Value *Op1 = Cmp->getOperand(1);
  if (L->isLoopInvariant(Op1))
    return Op1;
  if (L->isLoopInvariant(Op0))
    return Op0;
  return nullptr;
}

Function *createAndBuildFMOPAFunction(Module &M, StringRef Name) {
  LLVMContext &Ctx = M.getContext();
  Type *PtrTy = PointerType::get(Ctx, 0);
  Type *I64 = Type::getInt64Ty(Ctx);

  Type *F32 = Type::getFloatTy(Ctx);
  // void @name(ptr %C, ptr %A, ptr %B,
  //            i64 %NI, i64 %NJ, i64 %NK,
  //            i64 %LdC, i64 %LdA, i64 %LdB,
  //            float %alpha, float %beta)
  FunctionType *FTy = FunctionType::get(
      Type::getVoidTy(Ctx),
      {PtrTy, PtrTy, PtrTy, I64, I64, I64, I64, I64, I64, F32, F32},
      /*isVarArg=*/false);

  Function *Fn = Function::Create(FTy, GlobalValue::InternalLinkage, Name, &M);

  // Set SME attributes.
  Fn->addFnAttr("aarch64_pstate_sm_body");
  Fn->addFnAttr("aarch64_new_za");
  Fn->addFnAttr("target-features", "+sme,+sve");

  // Build the FMOPA loop body.
  buildFMOPABody(Fn);

  // Tag with metadata.
  Fn->setMetadata("matmul.sme",
                  MDNode::get(Ctx, MDString::get(Ctx, "fmopa.f32")));

  return Fn;
}

Value *getLoopBoundValue(Loop *L) { return getLoopBound(L); }

} // namespace llvm
