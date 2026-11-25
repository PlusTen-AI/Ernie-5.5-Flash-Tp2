// patches.h — runtime switches for the numbered patches listed in patches.def.
//   if (e55::patch::on(e55::patch::rdma_ar)) { … }
#pragma once
#include <string>

namespace e55::patch {

enum Id : int {
#define E55_PATCH(num, name, def, what) name = num,
#include "model/patches.def"
#undef E55_PATCH
  kNone = 0
};

bool on(Id id);           // patches.def default, overridden once by env E55_PATCHES ("+name,-name")
std::string active();     // "P001 rdma_ar, …" or "none" — logged once at engine start
void set(Id id, bool on); // tests: switch a patch inside one process (base vs patch comparisons)
void set(const std::string& name, bool on);
bool on(const std::string& name);  // by name (bindings); unknown name → fail

}  // namespace e55::patch
