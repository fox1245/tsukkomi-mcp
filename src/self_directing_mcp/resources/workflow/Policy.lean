import Init

namespace Workflow

inductive Stage where
  | requirements | formalize | implement | verify | complete
  deriving DecidableEq, Repr

inductive Choice where
  | insufficient | revision | verification | completion
  deriving DecidableEq, Repr

structure Facts where
  approved : Bool
  obligations : Bool
  proofs : Bool
  tests : Bool
  fresh : Bool
  deriving DecidableEq, Repr

def Facts.all (f : Facts) : Bool :=
  f.approved && f.obligations && f.proofs && f.tests && f.fresh

def Facts.base (f : Facts) : Bool := f.approved && f.obligations

def Facts.verifiable (f : Facts) : Bool := f.base && f.proofs && f.fresh

def Facts.missing (f : Facts) : List String :=
  [("approved", f.approved), ("obligations", f.obligations),
   ("proofs", f.proofs), ("tests", f.tests), ("fresh", f.fresh)].filterMap
    fun (name, present) => if present then none else some name

structure Decision where
  allowed : Bool
  next : Stage
  failed : List String
  deriving Repr

def transition (s : Stage) (c : Choice) (f : Facts) : Decision := Id.run do
  -- A completed state is never retained after evidence is invalidated.
  let current := if s == .complete && !f.all then Stage.verify else s
  match c with
  | .insufficient => return ⟨false, current, ["classification"]⟩
  | .revision =>
      if f.base then
        return ⟨true, if f.proofs then .implement else .formalize, []⟩
      else return ⟨false, if current == .complete then .verify else current, f.missing⟩
  | .verification =>
      match current with
      | .requirements =>
          if f.base then return ⟨true, .formalize, []⟩
          else return ⟨false, current, f.missing⟩
      | .formalize =>
          if f.verifiable then return ⟨true, .implement, []⟩
          else return ⟨false, current, f.missing⟩
      | .implement =>
          if f.verifiable then return ⟨true, .verify, []⟩
          else return ⟨false, current, f.missing⟩
      | .verify =>
          if f.verifiable then return ⟨true, .verify, []⟩
          else return ⟨false, current, f.missing⟩
      | .complete => return ⟨false, current, ["stage"]⟩
  | .completion =>
      if current == .verify && f.all then return ⟨true, .complete, []⟩
      else return ⟨false, current, if f.all then ["stage"] else f.missing⟩

set_option maxHeartbeats 4000000 in
/-- Kernel reduction over the finite inputs, not native_decide. -/
theorem one_step_safe (s : Stage) (c : Choice) (f : Facts) :
    (transition s c f).next = .complete → f.all = true := by
  rcases f with ⟨a, o, p, t, r⟩
  cases s <;> cases c <;> cases a <;> cases o <;> cases p <;>
    cases t <;> cases r <;> decide

structure Snapshot where
  stage : Stage
  facts : Facts

structure Input where
  choice : Choice
  facts : Facts

def Safe (s : Snapshot) : Prop := s.stage = .complete → s.facts.all = true

def advance (s : Snapshot) (i : Input) : Snapshot :=
  ⟨(transition s.stage i.choice i.facts).next, i.facts⟩

theorem advance_safe (s : Snapshot) (i : Input) : Safe (advance s i) :=
  one_step_safe s.stage i.choice i.facts

def run (s : Snapshot) : List Input → Snapshot
  | [] => s
  | i :: rest => run (advance s i) rest

/-- Arbitrary classification AND evidence-update sequences preserve safety.
    Each input replaces all evidence, including evidence of a completed state. -/
theorem sequence_safe (inputs : List Input) (s : Snapshot) (h : Safe s) :
    Safe (run s inputs) := by
  induction inputs generalizing s with
  | nil => exact h
  | cons i rest ih => exact ih (advance s i) (advance_safe s i)

end Workflow
