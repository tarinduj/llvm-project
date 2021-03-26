#ifndef MLIR_ANALYSIS_PRESBURGER_SET_H
#define MLIR_ANALYSIS_PRESBURGER_SET_H

#include "mlir/Analysis/Presburger/PresburgerBasicSet.h"
#include <variant>

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

  template <typename OInt>
  friend class PresburgerSet;

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
  TransprecSet(PresburgerSet<int16_t> set) : setvar(std::move(set)) {}
  TransprecSet(PresburgerSet<int64_t> set) : setvar(std::move(set)) {}
  TransprecSet(PresburgerSet<int128_t> set) : setvar(std::move(set)) {}

  static void harmonizePrecisions(TransprecSet &a, TransprecSet &b) {
    while (a.setvar.index() < b.setvar.index())
      a.increasePrecision();
    while (a.setvar.index() > b.setvar.index())
      b.increasePrecision();
  }

  void increasePrecision() {
    if (std::holds_alternative<PresburgerSet<int16_t>>(setvar)) {
      setvar = PresburgerSet<int64_t>(std::get<PresburgerSet<int16_t>>(setvar));
    } else if (std::holds_alternative<PresburgerSet<int64_t>>(setvar)) {
      setvar = PresburgerSet<int128_t>(std::get<PresburgerSet<int64_t>>(setvar));
    } else {
      llvm_unreachable("Cannot expand beyond 128-bit!");
    }
  }

  void unionSet(TransprecSet &set) {
    harmonizePrecisions(*this, set);
    std::visit([&set](auto &&thisPS) {
      std::visit([&thisPS](auto &&oPS) {
        thisPS.unionSet(oPS);
      }, set.setvar);
    }, setvar);
  }

  void intersectSet(TransprecSet &set) {
    harmonizePrecisions(*this, set);
    std::visit([&](auto &&thisPS) {
      std::visit([&](auto &&oPS) {
        thisPS.intersectSet(oPS);
      }, set.setvar);
    }, setvar);
  }

  void subtract(TransprecSet &set) {
    harmonizePrecisions(*this, set);
    std::visit([&](auto &&thisPS) {
      std::visit([&](auto &&oPS) {
        try {
          thisPS.subtract(oPS);
        } catch (const std::overflow_error &e) {
          increasePrecision();
          set.increasePrecision();
          this->subtract(set);
        }
      }, set.setvar);
    }, setvar);
  }

  TransprecSet complement() {
    return std::visit([this](auto &&set) {
      try {
        using Set = std::decay_t<decltype(set)>;
        return TransprecSet(Set::complement(set));
      } catch (const std::overflow_error &e) {
        increasePrecision();
        return this->complement();
      }
    }, setvar);
  }

  TransprecSet eliminateExistentials() {
    return std::visit([this](auto &&set) {
      try {
        using Set = std::decay_t<decltype(set)>;
        return TransprecSet(Set::eliminateExistentials(set));
      } catch (const std::overflow_error &e) {
        increasePrecision();
        return this->eliminateExistentials();
      }
    }, setvar);
  }

  bool isIntegerEmpty() {
    return std::visit([this](auto &&set) {
      try {
        return set.isIntegerEmpty();
      } catch (const std::overflow_error &e) {
        increasePrecision();
        return this->isIntegerEmpty();
      }
    }, setvar);
  }

private:
  std::variant<PresburgerSet<int16_t>, PresburgerSet<int64_t>, PresburgerSet<int128_t>> setvar;
};

} // namespace presburger
} // namespace analysis
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_SET_H
