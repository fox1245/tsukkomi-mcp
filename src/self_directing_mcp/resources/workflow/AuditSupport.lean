import Lean
import Lean.Replay

open Lean Elab Command Meta

/-- Walk kernel declarations, never a module's cached axiom extension entries. -/
private partial def kernelAxioms (env : Kernel.Environment) (pending : List Name)
    (seen : NameSet := {}) (axioms : NameSet := {}) : Except String (Array Name) := do
  match pending with
  | [] => return axioms.toArray.qsort Name.lt
  | name :: rest =>
    if seen.contains name then return ← kernelAxioms env rest seen axioms
    let some info := env.find? name | throw s!"missing kernel declaration {name}"
    let axioms := match info with
      | .axiomInfo _ => axioms.insert name
      | _ => axioms
    kernelAxioms env (info.getUsedConstantsAsSet.toList ++ rest) (seen.insert name) axioms

/-- Parse exactly one type using the caller's trusted syntax environment. -/
private def audit (name : Name) (expected : String) (syntaxEnv? : Option Environment := none)
    (kernel? : Option Kernel.Environment := none) : TermElabM Json := do
  let kernel := kernel?.getD (← getEnv).toKernelEnv
  let info ← match kernel.find? name with
    | some info => pure info
    | none => throwError "theorem is missing from checked kernel"
  unless (match info with | .thmInfo _ => true | _ => false) do
    throwError "workflow audit requires a theorem declaration"
  let stx ← match Parser.runParserCategory (syntaxEnv?.getD (← getEnv)) `term expected with
    | .ok stx => pure stx
    | .error error => throwError "invalid approved statement: {error}"
  let type ← Term.elabType stx
  Term.synthesizeSyntheticMVarsNoPostponing
  let type ← instantiateMVars type
  if type.hasMVar || type.hasFVar then
    throwError "approved statement contains unresolved variables"
  unless Kernel.isDefEqGuarded (Environment.ofKernelEnv kernel) {} info.type type do
    throwError "approved theorem type does not match actual theorem type"
  let axioms ← match kernelAxioms kernel [name] with
    | .ok axioms => pure axioms
    | .error error => throwError "{error}"
  return Json.mkObj [
    ("theorem", toJson name.toString),
    ("axioms", toJson (axioms.map Name.toString))]

/-- Used ONLY for the pinned, trusted policy sources during their native build.
    Candidate modules are never imported by a command elaboration process. -/
elab "workflow_audit " theoremName:str " against " expected:str : command => do
  let report ← liftTermElabM <| audit theoremName.getString.toName expected.getString
  logInfo m!"TSUKKOMI_AUDIT {report.compress}"

/-- A separate native process loads candidate declarations as data, without
    initializing their extensions, attributes, macros, or elaborators. Only the
    installed Lean library supplies syntax and typeclass interpretation. Native
    .olean deserialization still assumes structurally valid objects; this is
    kernel replay, not comparator's validated export plus an external kernel. -/
unsafe def main (args : List String) : IO UInt32 := do
  let [moduleName, theoremName, expected] := args |
    throw <| IO.userError "expected: module theorem approved-statement"
  let sysroot ← findSysroot
  -- Do not permit even the trusted Lean import to be shadowed in /work.
  searchPathRef.set [sysroot / "lib" / "lean"]
  enableInitializersExecution
  let trusted ← importModules #[{ module := `Lean }] {} (loadExts := true)
  searchPathRef.set [".", sysroot / "lib" / "lean"]
  let candidate ← importModules #[{ module := moduleName.toName }] {} (loadExts := false)
  let mut additions := {}
  for (name, info) in candidate.constants do
    if !trusted.contains name then
      additions := additions.insert name info
  -- Independently replay all added declarations. The imported constant map is
  -- then used for lookup; no candidate extension/notation state is initialized.
  let kernel ← trusted.toKernelEnv.replay additions
  -- Restore only state from the pinned trusted Lean import. None of the
  -- candidate's extension data, initializers, notation, or macros are loaded.
  let mut nameEnv := candidate
  for ext in (← persistentEnvExtensionsRef.get) do
    nameEnv := ext.toEnvExtension.setState (asyncMode := .sync) nameEnv
      (ext.toEnvExtension.getState (asyncMode := .sync) trusted)
  let (report, state, _, _) ← (audit theoremName.toName expected (some trusted) (some kernel)).toIO
    { fileName := "<trusted-workflow-audit>", fileMap := default,
      options := ({} : Options).setBool `autoImplicit false }
    { env := nameEnv } {} {} { errToSorry := false } {}
  if state.messages.hasErrors then
    throw <| IO.userError "approved statement elaboration failed"
  IO.println report.compress
  return 0
