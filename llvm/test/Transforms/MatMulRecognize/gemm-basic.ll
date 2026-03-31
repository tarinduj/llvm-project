; RUN: opt -passes=matmul-recognize-and-lower -matmul-recognize-only -matmul-recognize-verbose -S < %s 2>&1 | FileCheck %s
;
; Test: Canonical GEMM pattern recognition.
;   for (i = 0; i < N; i++)
;     for (k = 0; k < N; k++)
;       for (j = 0; j < N; j++)
;         C[i][j] += A[i][k] * B[k][j];
;
; The pass should recognize this as a matmul with:
;   - i, j as free dimensions
;   - k as the reduction dimension
;
; CHECK: MMR: Recognized matmul

define void @gemm_basic(ptr noalias %C, ptr noalias %A, ptr noalias %B, i32 %N) {
entry:
  %cmp.i = icmp sgt i32 %N, 0
  br i1 %cmp.i, label %for.i.preheader, label %exit

for.i.preheader:
  br label %for.i

for.i:
  %i = phi i32 [ 0, %for.i.preheader ], [ %i.next, %for.i.latch ]
  br label %for.k.preheader

for.k.preheader:
  br label %for.k

for.k:
  %k = phi i32 [ 0, %for.k.preheader ], [ %k.next, %for.k.latch ]
  br label %for.j.preheader

for.j.preheader:
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.j.preheader ], [ %j.next, %for.j.latch ]

  ; C[i][j] — address = C + i*N + j
  %c.idx.i = mul nsw i32 %i, %N
  %c.idx = add nsw i32 %c.idx.i, %j
  %c.idx.ext = sext i32 %c.idx to i64
  %c.ptr = getelementptr float, ptr %C, i64 %c.idx.ext
  %c.val = load float, ptr %c.ptr, align 4

  ; A[i][k] — address = A + i*N + k
  %a.idx.i = mul nsw i32 %i, %N
  %a.idx = add nsw i32 %a.idx.i, %k
  %a.idx.ext = sext i32 %a.idx to i64
  %a.ptr = getelementptr float, ptr %A, i64 %a.idx.ext
  %a.val = load float, ptr %a.ptr, align 4

  ; B[k][j] — address = B + k*N + j
  %b.idx.k = mul nsw i32 %k, %N
  %b.idx = add nsw i32 %b.idx.k, %j
  %b.idx.ext = sext i32 %b.idx to i64
  %b.ptr = getelementptr float, ptr %B, i64 %b.idx.ext
  %b.val = load float, ptr %b.ptr, align 4

  ; C[i][j] += A[i][k] * B[k][j]
  %mul = fmul float %a.val, %b.val
  %add = fadd float %c.val, %mul
  store float %add, ptr %c.ptr, align 4

  %j.next = add nuw nsw i32 %j, 1
  br label %for.j.latch

for.j.latch:
  %j.cmp = icmp slt i32 %j.next, %N
  br i1 %j.cmp, label %for.j, label %for.k.latch

for.k.latch:
  %k.next = add nuw nsw i32 %k, 1
  %k.cmp = icmp slt i32 %k.next, %N
  br i1 %k.cmp, label %for.k, label %for.i.latch

for.i.latch:
  %i.next = add nuw nsw i32 %i, 1
  %i.cmp = icmp slt i32 %i.next, %N
  br i1 %i.cmp, label %for.i, label %exit

exit:
  ret void
}
