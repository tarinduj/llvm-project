; RUN: opt -passes=matmul-recognize-and-lower -matmul-recognize-only -matmul-recognize-verbose -S < %s 2>&1 | FileCheck %s
;
; Negative test: loops that should NOT be recognized as matmul.
;
; CHECK-NOT: MMR: Recognized matmul

; Test 1: Simple vector dot product (2 loops, not 3).
;   for (i = 0; i < N; i++)
;     sum += A[i] * B[i];
define float @dot_product(ptr noalias %A, ptr noalias %B, i32 %N) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.preheader, label %exit

for.preheader:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %for.preheader ], [ %i.next, %for.latch ]
  %sum = phi float [ 0.0, %for.preheader ], [ %add, %for.latch ]
  %i.ext = sext i32 %i to i64
  %a.ptr = getelementptr float, ptr %A, i64 %i.ext
  %b.ptr = getelementptr float, ptr %B, i64 %i.ext
  %a.val = load float, ptr %a.ptr, align 4
  %b.val = load float, ptr %b.ptr, align 4
  %mul = fmul float %a.val, %b.val
  %add = fadd float %sum, %mul
  %i.next = add nuw nsw i32 %i, 1
  br label %for.latch

for.latch:
  %cmp.i = icmp slt i32 %i.next, %N
  br i1 %cmp.i, label %for.body, label %exit

exit:
  %result = phi float [ 0.0, %entry ], [ %add, %for.latch ]
  ret float %result
}

; Test 2: Matrix addition (no reduction, 2-deep nest).
;   for (i = 0; i < N; i++)
;     for (j = 0; j < N; j++)
;       C[i][j] = A[i][j] + B[i][j];
define void @mat_add(ptr noalias %C, ptr noalias %A, ptr noalias %B, i32 %N) {
entry:
  %cmp.i = icmp sgt i32 %N, 0
  br i1 %cmp.i, label %for.i.preheader, label %exit

for.i.preheader:
  br label %for.i

for.i:
  %i = phi i32 [ 0, %for.i.preheader ], [ %i.next, %for.i.latch ]
  br label %for.j.preheader

for.j.preheader:
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.j.preheader ], [ %j.next, %for.j.latch ]
  %idx.i = mul nsw i32 %i, %N
  %idx = add nsw i32 %idx.i, %j
  %idx.ext = sext i32 %idx to i64
  %a.ptr = getelementptr float, ptr %A, i64 %idx.ext
  %b.ptr = getelementptr float, ptr %B, i64 %idx.ext
  %c.ptr = getelementptr float, ptr %C, i64 %idx.ext
  %a.val = load float, ptr %a.ptr, align 4
  %b.val = load float, ptr %b.ptr, align 4
  %add = fadd float %a.val, %b.val
  store float %add, ptr %c.ptr, align 4
  %j.next = add nuw nsw i32 %j, 1
  br label %for.j.latch

for.j.latch:
  %j.cmp = icmp slt i32 %j.next, %N
  br i1 %j.cmp, label %for.j, label %for.i.latch

for.i.latch:
  %i.next = add nuw nsw i32 %i, 1
  %i.cmp = icmp slt i32 %i.next, %N
  br i1 %i.cmp, label %for.i, label %exit

exit:
  ret void
}
