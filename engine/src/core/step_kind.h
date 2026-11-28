// step_kind.h — the kind of the step being run: a prefill chunk, or a decode / verify / draft step. Host only.
//
// Who sets it: only the step's caller knows. Engine::step(…, prefill) sets it for the length of the step (StepKind
// below, also on an exception); the server passes prefill = true for prompt chunks (fill, the prompt's last chunk
// through step / run, forced-token injections) and false for decode steps; spec_round / spec_copy verify steps and
// every draft pass (spec_draft, outside any step) run with it off. Outside a step it is off (decode).
//
// Who reads it: a kernel family's host-side dispatcher, to choose a prefill-only variant (one with its own
// accumulation order under the 2026-11-28 prefill rule: the reference's rounding points, deterministic, batch- and
// chunk-size invariant within prefill, gated against the reference, not bitwise vs decode):
//     if (step_prefill() && M >= kMyMinM && patch::on(patch::my_pf_kernel)) return my_pf::run(…);
// Never choose such a kernel from M alone: a verify step of 128 rows must keep the decode bits, and a 1-row prefill
// chunk (after a prefix-checkpoint restore) must take the prefill kernels (chunk-size invariance). Kernels that are
// bitwise the decode path at every M (everything before part B) ignore it and stay chosen by M.
//
// One process per rank and one step at a time (the engine's calls are serialized), so a plain flag is enough. It is a
// function-local static of an inline function: one instance per linked module (the engine .so, or a kernel family's
// test extension — whose binding sets it with set_step_prefill for its prefill-variant tests).
#pragma once

namespace e55 {

inline bool& step_prefill_ref() {
  static bool on = false;
  return on;
}
// true: the running step is a prefill chunk (prefill-only kernels allowed); false: decode / verify / draft, or no step
inline bool step_prefill() { return step_prefill_ref(); }
inline void set_step_prefill(bool on) { step_prefill_ref() = on; }

// Scoped: the kind for one step, back to the previous kind when it ends (also when it throws, e.g. KvFull)
struct StepKind {
  bool prev;
  explicit StepKind(bool prefill) : prev(step_prefill()) { set_step_prefill(prefill); }
  ~StepKind() { set_step_prefill(prev); }
  StepKind(const StepKind&) = delete;
  StepKind& operator=(const StepKind&) = delete;
};

}  // namespace e55
