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

/// Build the FMOPA function body with 2x2 multi-tile and pre-transposed A.
///
/// Uses za0-za3 to accumulate a 2×2 grid of output tiles per (ii,jj) pair.
/// Each k-iteration does 4 FMOPAs with only 4 vector loads (2 A + 2 B),
/// achieving 2x better compute-to-load ratio than single-tile.
///
/// Arguments: (ptr %C, ptr %A, ptr %B, i64 %NI, i64 %NJ, i64 %NK,
///             i64 %LdC, i64 %LdA, i64 %LdB, f32 %alpha, f32 %beta)
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
  Value *ZeroVec = ConstantAggregateZero::get(SVF32);

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn);
  BasicBlock *TileIHdr = BasicBlock::Create(Ctx, "tile.i.hdr", Fn);
  BasicBlock *XposeKHdr = BasicBlock::Create(Ctx, "xpose.k.hdr", Fn);
  BasicBlock *XposeLaneHdr = BasicBlock::Create(Ctx, "xpose.lane.hdr", Fn);
  BasicBlock *XposeLaneBody = BasicBlock::Create(Ctx, "xpose.lane.body", Fn);
  BasicBlock *XposeLaneLatch = BasicBlock::Create(Ctx, "xpose.lane.latch", Fn);
  BasicBlock *XposeKLatch = BasicBlock::Create(Ctx, "xpose.k.latch", Fn);
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

  // --- Entry ---
  B.SetInsertPoint(Entry);
  Value *VScale = B.CreateVScale(I64, "vscale");
  Value *SVL = B.CreateMul(VScale, ConstantInt::get(I64, 4), "svl");
  Value *SVL2 = B.CreateMul(SVL, ConstantInt::get(I64, 2), "svl2");
  // Pre-transpose buffer for 2 row-tiles: 2*SVL * NK floats.
  Value *BufSize = B.CreateMul(SVL2, ArgNK, "at.buf.size");
  Value *BufBytes = B.CreateMul(BufSize, ConstantInt::get(I64, 4), "at.bytes");
  Value *ATBuf = B.CreateCall(
      M->getOrInsertFunction(
          "malloc", FunctionType::get(PointerType::get(Ctx, 0), {I64}, false)),
      {BufBytes}, "at.buf");
  Function *UMinFn = Intrinsic::getOrInsertDeclaration(M, Intrinsic::umin, {I64});
  B.CreateBr(TileIHdr);

  // --- Tile I: step by 2*SVL (2 row-tiles) ---
  B.SetInsertPoint(TileIHdr);
  PHINode *II = B.CreatePHI(I64, 2, "ii");
  II->addIncoming(Zero64, Entry);
  Value *II1 = B.CreateAdd(II, SVL, "ii1"); // start of second row-tile
  Value *RemI0 = B.CreateSub(ArgNI, II, "rem.i0");
  Value *TileRows0 = B.CreateCall(UMinFn, {SVL, RemI0}, "tile.rows0");
  Value *RemI1 = B.CreateSub(ArgNI, II1, "rem.i1");
  // If ii1 >= NI, second tile has 0 rows. Use max(0, rem.i1) via umin clamped.
  Value *HasTile1 = B.CreateICmpSLT(II1, ArgNI, "has.tile1");
  Value *TileRows1Raw = B.CreateCall(UMinFn, {SVL, RemI1}, "tile.rows1.raw");
  Value *TileRows1 = B.CreateSelect(HasTile1, TileRows1Raw, Zero64, "tile.rows1");
  // Total rows to transpose
  Value *TotalXRows = B.CreateAdd(TileRows0, TileRows1, "xpose.total.rows");
  B.CreateBr(XposeKHdr);

  // === Pre-transpose 2 row-tiles of A ===
  // AT_buf[k * 2*SVL + lane] = A[(ii+lane) * LdA + k]  for lane in 0..TotalXRows
  B.SetInsertPoint(XposeKHdr);
  PHINode *XK = B.CreatePHI(I64, 2, "xk");
  XK->addIncoming(Zero64, TileIHdr);
  B.CreateBr(XposeLaneHdr);

  B.SetInsertPoint(XposeLaneHdr);
  PHINode *XLane = B.CreatePHI(I64, 2, "xlane");
  XLane->addIncoming(Zero64, XposeKHdr);
  B.CreateBr(XposeLaneBody);

  B.SetInsertPoint(XposeLaneBody);
  Value *ASrcRow = B.CreateAdd(II, XLane, "a.src.row");
  Value *ASrcOff = B.CreateAdd(B.CreateMul(ASrcRow, ArgLdA), XK, "a.src.off");
  Value *ASrcPtr = B.CreateGEP(F32, ArgA, ASrcOff, "a.src.ptr");
  Value *AVal = B.CreateLoad(F32, ASrcPtr, "a.val");
  Value *ADstOff = B.CreateAdd(B.CreateMul(XK, SVL2), XLane, "at.dst.off");
  Value *ADstPtr = B.CreateGEP(F32, ATBuf, ADstOff, "at.dst.ptr");
  B.CreateStore(AVal, ADstPtr);
  B.CreateBr(XposeLaneLatch);

  B.SetInsertPoint(XposeLaneLatch);
  Value *XLaneNext = B.CreateAdd(XLane, One64, "xlane.next");
  Value *XLaneCond = B.CreateICmpULT(XLaneNext, TotalXRows, "xlane.cond");
  B.CreateCondBr(XLaneCond, XposeLaneHdr, XposeKLatch);
  XLane->addIncoming(XLaneNext, XposeLaneLatch);

  B.SetInsertPoint(XposeKLatch);
  Value *XKNext = B.CreateAdd(XK, One64, "xk.next");
  Value *XKCond = B.CreateICmpSLT(XKNext, ArgNK, "xk.cond");
  B.CreateCondBr(XKCond, XposeKHdr, TileJHdr);
  XK->addIncoming(XKNext, XposeKLatch);

  // === FMOPA with 2x2 tile grid using za0-za3 ===
  B.SetInsertPoint(TileJHdr);
  PHINode *JJ = B.CreatePHI(I64, 2, "jj");
  JJ->addIncoming(Zero64, XposeKLatch);
  Value *JJ1 = B.CreateAdd(JJ, SVL, "jj1"); // second col-tile

  emitSMEZero(B, 255); // zero all of ZA

  Value *PredI0 = emitWhileLT(B, II, ArgNI);
  Value *PredI1 = emitWhileLT(B, II1, ArgNI);
  Value *PredJ0 = emitWhileLT(B, JJ, ArgNJ);
  Value *PredJ1 = emitWhileLT(B, JJ1, ArgNJ);

  Value *AlphaSplat = B.CreateVectorSplat(ElementCount::getScalable(4), ArgAlpha,
                                          "alpha.splat");
  B.CreateBr(KHdr);

  // --- K loop ---
  B.SetInsertPoint(KHdr);
  PHINode *K = B.CreatePHI(I64, 2, "k");
  K->addIncoming(Zero64, TileJHdr);
  B.CreateBr(KBody);

  B.SetInsertPoint(KBody);

  // Load 2 A columns from pre-transposed buffer.
  // a0 = AT[k * 2*SVL + 0 .. SVL-1]       (row-tile 0)
  // a1 = AT[k * 2*SVL + SVL .. 2*SVL-1]   (row-tile 1)
  Value *ATBase = B.CreateMul(K, SVL2, "at.base");
  Value *ATPtr0 = B.CreateGEP(F32, ATBuf, ATBase, "at.ptr0");
  Value *AVec0 = B.CreateMaskedLoad(SVF32, ATPtr0, Align(4), PredI0, ZeroVec, "a0");
  Value *AScaled0 = B.CreateFMul(AVec0, AlphaSplat, "a0s");

  Value *ATOff1 = B.CreateAdd(ATBase, SVL, "at.off1");
  Value *ATPtr1 = B.CreateGEP(F32, ATBuf, ATOff1, "at.ptr1");
  Value *AVec1 = B.CreateMaskedLoad(SVF32, ATPtr1, Align(4), PredI1, ZeroVec, "a1");
  Value *AScaled1 = B.CreateFMul(AVec1, AlphaSplat, "a1s");

  // Load 2 B rows.
  // b0 = B[k][jj .. jj+SVL-1]
  // b1 = B[k][jj+SVL .. jj+2*SVL-1]
  Value *BRowBase = B.CreateMul(K, ArgLdB, "b.row.base");
  Value *BOff0 = B.CreateAdd(BRowBase, JJ, "b.off0");
  Value *BPtr0 = B.CreateGEP(F32, ArgB, BOff0, "b.ptr0");
  Value *BVec0 = B.CreateMaskedLoad(SVF32, BPtr0, Align(4), PredJ0, ZeroVec, "b0");

  Value *BOff1 = B.CreateAdd(BRowBase, JJ1, "b.off1");
  Value *BPtr1 = B.CreateGEP(F32, ArgB, BOff1, "b.ptr1");
  Value *BVec1 = B.CreateMaskedLoad(SVF32, BPtr1, Align(4), PredJ1, ZeroVec, "b1");

  // 4 FMOPAs: 2x2 outer product grid.
  emitFMOPA_F32(B, 0, PredI0, PredJ0, AScaled0, BVec0); // za0: C[ii][jj]
  emitFMOPA_F32(B, 1, PredI0, PredJ1, AScaled0, BVec1); // za1: C[ii][jj+SVL]
  emitFMOPA_F32(B, 2, PredI1, PredJ0, AScaled1, BVec0); // za2: C[ii+SVL][jj]
  emitFMOPA_F32(B, 3, PredI1, PredJ1, AScaled1, BVec1); // za3: C[ii+SVL][jj+SVL]
  B.CreateBr(KLatch);

  // --- K latch with unroll hint ---
  B.SetInsertPoint(KLatch);
  Value *KNext = B.CreateAdd(K, One64, "k.next");
  Value *KCond = B.CreateICmpSLT(KNext, ArgNK, "k.cond");
  auto *KBr = B.CreateCondBr(KCond, KHdr, StoreHdr);
  K->addIncoming(KNext, KLatch);
  MDNode *UnrollMD = MDNode::get(Ctx, {
      MDString::get(Ctx, "llvm.loop.unroll.count"),
      ConstantAsMetadata::get(ConstantInt::get(I32, 4))});
  MDNode *LoopMD = MDNode::getDistinct(Ctx, {nullptr, UnrollMD});
  LoopMD->replaceOperandWith(0, LoopMD);
  KBr->setMetadata(LLVMContext::MD_loop, LoopMD);

  // === Store 4 tiles back to C ===
  // We store all 4 tiles in a single row loop (iterate over all rows of
  // both row-tiles). For each row, store from the appropriate tile.
  B.SetInsertPoint(StoreHdr);
  Value *StoreBound = B.CreateAdd(TileRows0, TileRows1, "store.bound");
  Value *BetaSplat = B.CreateVectorSplat(ElementCount::getScalable(4), ArgBeta,
                                         "beta.splat");
  B.CreateBr(StoreBody);

  B.SetInsertPoint(StoreBody);
  PHINode *Row = B.CreatePHI(I64, 2, "row");
  Row->addIncoming(Zero64, StoreHdr);

  Value *RowI32 = B.CreateTrunc(Row, I32, "row.i32");
  // Determine if this row belongs to tile 0 (row < TileRows0) or tile 1.
  Value *InTile0 = B.CreateICmpULT(Row, TileRows0, "in.tile0");
  // For tile 1, the ZA row index is (row - TileRows0).
  Value *Row1Offset = B.CreateSub(Row, TileRows0, "row1.off");
  Value *Row1I32 = B.CreateTrunc(Row1Offset, I32, "row1.i32");
  // Select tile index and ZA row.
  Value *TileRowI32 = B.CreateSelect(InTile0, RowI32, Row1I32, "tile.row");

  // Read from za0/za1 (tile 0 rows) or za2/za3 (tile 1 rows) for each j-tile.
  // Store to C at the corresponding (ii+row, jj) or (ii+row, jj+SVL).
  Value *AbsRow = B.CreateAdd(II, Row, "abs.row");

  // Helper: store one tile's row to C.
  auto storeOneTile = [&](unsigned ZATile, Value *Pred, Value *JJOff) {
    Value *ZARow = emitSMEReadHoriz_F32(B, ZATile, TileRowI32, Pred);
    Value *COff = B.CreateAdd(B.CreateMul(AbsRow, ArgLdC), JJOff, "c.off");
    Value *CPtr = B.CreateGEP(F32, ArgC, COff, "c.ptr");
    Value *COld = B.CreateMaskedLoad(SVF32, CPtr, Align(4), Pred, ZeroVec, "c.old");
    Value *CScaled = B.CreateFMul(COld, BetaSplat, "c.scaled");
    Value *CNew = B.CreateFAdd(CScaled, ZARow, "c.new");
    B.CreateMaskedStore(CNew, CPtr, Align(4), Pred);
  };

  // For tile row 0: store za0 (j-tile 0) and za1 (j-tile 1).
  // For tile row 1: store za2 (j-tile 0) and za3 (j-tile 1).
  // We select the tile based on InTile0.
  // za0/za2 → j-tile 0 (PredJ0, JJ)
  Value *ZA_J0_Row = B.CreateSelect(InTile0,
      emitSMEReadHoriz_F32(B, 0, TileRowI32, PredJ0),
      emitSMEReadHoriz_F32(B, 2, TileRowI32, PredJ0), "za.j0");
  Value *COff_J0 = B.CreateAdd(B.CreateMul(AbsRow, ArgLdC), JJ, "c.off.j0");
  Value *CPtr_J0 = B.CreateGEP(F32, ArgC, COff_J0, "c.ptr.j0");
  Value *COld_J0 = B.CreateMaskedLoad(SVF32, CPtr_J0, Align(4), PredJ0, ZeroVec, "c.old.j0");
  Value *CS_J0 = B.CreateFMul(COld_J0, BetaSplat, "c.s.j0");
  Value *CNew_J0 = B.CreateFAdd(CS_J0, ZA_J0_Row, "c.new.j0");
  B.CreateMaskedStore(CNew_J0, CPtr_J0, Align(4), PredJ0);

  // za1/za3 → j-tile 1 (PredJ1, JJ1)
  Value *ZA_J1_Row = B.CreateSelect(InTile0,
      emitSMEReadHoriz_F32(B, 1, TileRowI32, PredJ1),
      emitSMEReadHoriz_F32(B, 3, TileRowI32, PredJ1), "za.j1");
  Value *COff_J1 = B.CreateAdd(B.CreateMul(AbsRow, ArgLdC), JJ1, "c.off.j1");
  Value *CPtr_J1 = B.CreateGEP(F32, ArgC, COff_J1, "c.ptr.j1");
  Value *COld_J1 = B.CreateMaskedLoad(SVF32, CPtr_J1, Align(4), PredJ1, ZeroVec, "c.old.j1");
  Value *CS_J1 = B.CreateFMul(COld_J1, BetaSplat, "c.s.j1");
  Value *CNew_J1 = B.CreateFAdd(CS_J1, ZA_J1_Row, "c.new.j1");
  B.CreateMaskedStore(CNew_J1, CPtr_J1, Align(4), PredJ1);

  B.CreateBr(StoreLatch);

  // --- Store latch ---
  B.SetInsertPoint(StoreLatch);
  Value *RowNext = B.CreateAdd(Row, One64, "row.next");
  Value *RowCond = B.CreateICmpULT(RowNext, StoreBound, "row.cond");
  B.CreateCondBr(RowCond, StoreBody, TileJLatch);
  Row->addIncoming(RowNext, StoreLatch);

  // --- Tile J latch: step by 2*SVL ---
  B.SetInsertPoint(TileJLatch);
  Value *JJNext = B.CreateAdd(JJ, SVL2, "jj.next");
  Value *JJCond = B.CreateICmpSLT(JJNext, ArgNJ, "jj.cond");
  B.CreateCondBr(JJCond, TileJHdr, TileILatch);
  JJ->addIncoming(JJNext, TileJLatch);

  // --- Tile I latch: step by 2*SVL ---
  B.SetInsertPoint(TileILatch);
  Value *IINext = B.CreateAdd(II, SVL2, "ii.next");
  Value *IICond = B.CreateICmpSLT(IINext, ArgNI, "ii.cond");
  B.CreateCondBr(IICond, TileIHdr, Exit);
  II->addIncoming(IINext, TileILatch);

  // --- Exit ---
  B.SetInsertPoint(Exit);
  B.CreateCall(
      M->getOrInsertFunction(
          "free", FunctionType::get(Type::getVoidTy(Ctx),
                                    {PointerType::get(Ctx, 0)}, false)),
      {ATBuf});
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
