#ifndef MLIR_ANALYSIS_PRESBURGER_SET_H
#define MLIR_ANALYSIS_PRESBURGER_SET_H

#include "mlir/Analysis/Presburger/PresburgerBasicSet.h"

namespace mlir {
namespace analysis {
namespace presburger {

template <typename Int>
class PresburgerSet {
public:
  PresburgerSet(unsigned nDim = 0, unsigned nSym = 0, bool markedEmpty = false)
      : nDim(nDim), nSym(nSym), markedEmpty(markedEmpty) {}
  PresburgerSet(PresburgerBasicSet<Int> cs);
  template <typename OInt>
  PresburgerSet(const PresburgerSet<OInt> &oSet);

  unsigned getNumBasicSets() const;
  unsigned getNumDims() const;
  unsigned getNumSyms() const;
  static PresburgerSet eliminateExistentials(const PresburgerBasicSet<Int> &bs);
  static PresburgerSet eliminateExistentials(const PresburgerSet &set);
  static PresburgerSet eliminateExistentials(PresburgerSet &&set);
  const SmallVector<PresburgerBasicSet<Int>, 4> &getBasicSets() const;
  void addBasicSet(const PresburgerBasicSet<Int> &cs);
  void addBasicSet(PresburgerBasicSet<Int> &&cs);
  void unionSet(const PresburgerSet &set);
  void unionSet(PresburgerSet &&set);
  void intersectSet(const PresburgerSet &set);
  static bool equal(const PresburgerSet &s, const PresburgerSet &t);
  void print(raw_ostream &os) const;
  void dump() const;
  void dumpCoeffs() const;
  void printISL(raw_ostream &os) const;
  void dumpISL() const;
  void printVariableList(raw_ostream &os) const;
  void printConstraints(raw_ostream &os) const;
  llvm::hash_code hash_value() const;
  bool isMarkedEmpty() const;
  bool isUniverse() const;

  static PresburgerSet makeEmptySet(unsigned nDim, unsigned nSym);
  static PresburgerSet complement(const PresburgerSet &set);
  void subtract(const PresburgerSet &set);
  static PresburgerSet subtract(PresburgerBasicSet<Int> c, const PresburgerSet &set);

  llvm::Optional<SmallVector<SafeInteger<Int>, 8>> findIntegerSample();
  bool isIntegerEmpty() const;
  // bool containsPoint(const std::vector<INT> &values) const;
  llvm::Optional<SmallVector<SafeInteger<Int>, 8>> maybeGetCachedSample() const;

private:
  unsigned nDim;
  unsigned nSym;
  SmallVector<PresburgerBasicSet<Int>, 4> basicSets;
  // This is NOT just cached information about the constraints in basicSets.
  // If this is set to true, then the set is empty, irrespective of the state
  // of basicSets.
  bool markedEmpty;
  Optional<SmallVector<SafeInteger<Int>, 8>> maybeSample;
  void printBasicSet(raw_ostream &os, PresburgerBasicSet<Int> cs) const;
  void printVar(raw_ostream &os, SafeInteger<Int> var, unsigned i,
                unsigned &countNonZero) const;
};

using DialectSet = PresburgerSet<DefaultInt>;

using int128_t = __int128_t;

class TransprecSet {
public:
  TransprecSet() {};
  TransprecSet(PresburgerSet<int16_t> set) : set16(std::move(set)), tag(16) {}
  TransprecSet(PresburgerSet<int64_t> set) : set64(std::move(set)), tag(64) {}
  TransprecSet(PresburgerSet<int128_t> set) : set128(std::move(set)), tag(128) {}

  bool isIntegerEmpty() {
    bool result;
    if (tag == 16) {
      SafeInteger<int16_t>::overflow = false;
      result = set16.isIntegerEmpty();
    }
    if (tag == 64) {
      SafeInteger<int64_t>::overflow = false;
      result = set64.isIntegerEmpty();
    }
    if (tag == 128) {
      SafeInteger<int128_t>::overflow = false;
      result = set128.isIntegerEmpty();
    }
    assert(tag == 16 || tag == 64 || tag == 128);
    return result;
  }

  ~TransprecSet() {
    if (tag == 16)
      set16.~PresburgerSet<int16_t>();
    else if (tag == 64)
      set64.~PresburgerSet<int64_t>();
    else if (tag == 128)
      set128.~PresburgerSet<int128_t>();
    else
      llvm_unreachable("unknown tag!");
  }

  TransprecSet(const TransprecSet &o) {
    tag = o.tag;
    if (tag == 16)
      set16 = o.set16;
    else if (tag == 64)
      set64 = o.set64;
    else if (tag == 128)
      set128 = o.set128;
    else
      llvm_unreachable("unknown tag!");
  }
private:
  union {
    PresburgerSet<int16_t> set16;
    PresburgerSet<int64_t> set64;
    PresburgerSet<int128_t> set128;
  };
  uint8_t tag;
};

} // namespace presburger
} // namespace analysis
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_SET_H
