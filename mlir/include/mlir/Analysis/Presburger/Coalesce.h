#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Presburger/Set.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

/// coalesces a set according to the "integer set coalescing" by sven
/// verdoolaege.
///
/// Coalescing takes two convex BasicSets and tries to figure out, whether the
/// convex hull of those two BasicSets is the same integer set as the union of
/// those two BasicSets and if so, tries to come up with a BasicSet
/// corresponding to this convex hull.
PresburgerSet coalesce(PresburgerSet &set);

void dump(ArrayRef<SafeInteger> cons);

/// compare two constraints and give true if they are equal. Can also handle
/// cases in which for some integers a and b, c1 = a/b * c2. This is the same
/// constraint but stretched, which doesn't influence it's hyperplane.
bool sameConstraint(ArrayRef<SafeInteger> c1, ArrayRef<SafeInteger> c2);

/// rotates invalid around valid, until it becomes redundant. It does this by
/// adding the smallest multiple of valid to invalid, such that the result is
/// redundant.
Optional<SmallVector<SafeInteger, 8>>
wrapping(const PresburgerBasicSet &bs, SmallVectorImpl<SafeInteger> &valid,
         SmallVectorImpl<SafeInteger> &invalid);

/// combine two constraints c1 >= 0 and c2 >= 0 with the ratio n/d as -n c1 + d
/// c2 >= 0
SmallVector<SafeInteger, 8> combineConstraint(ArrayRef<SafeInteger> c1,
                                              ArrayRef<SafeInteger> c2,
                                              Fraction &ratio);

/// takes a BasicSet bs, a constraint ineq of that basicSet and the vector cut
/// of constraints, that were typed as cutting bs. Computes wether the part of
/// bs, that satisfies ineq with equality, is redundant for all constraints of
/// cut
bool containedFacet(ArrayRef<SafeInteger> ineq, const PresburgerBasicSet &bs,
                    ArrayRef<ArrayRef<SafeInteger>> cut);

} // namespace mlir
