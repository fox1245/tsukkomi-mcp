import Policy

open Workflow

def stageName : Stage → String
  | .requirements => "requirements"
  | .formalize => "formalize"
  | .implement => "implement"
  | .verify => "verify"
  | .complete => "complete"

def parseStage : String → Option Stage
  | "0" => some .requirements
  | "1" => some .formalize
  | "2" => some .implement
  | "3" => some .verify
  | "4" => some .complete
  | _ => none

def parseChoice : String → Option Choice
  | "0" => some .insufficient
  | "1" => some .revision
  | "2" => some .verification
  | "3" => some .completion
  | _ => none

def parseBit : String → Option Bool
  | "0" => some false
  | "1" => some true
  | _ => none

def parseRequest : List String → Option (Stage × Choice × Facts)
  | [s, c, a, o, p, t, r] => do
      return (← parseStage s, ← parseChoice c,
        ⟨← parseBit a, ← parseBit o, ← parseBit p, ← parseBit t, ← parseBit r⟩)
  | _ => none

def main (args : List String) : IO UInt32 := do
  let some (stage, choice, facts) := parseRequest args | return 2
  let result := transition stage choice facts
  -- Names and failures are fixed policy labels, never arbitrary request text.
  let failures := String.intercalate "," (result.failed.map fun reason => "\"" ++ reason ++ "\"")
  let json := "{\"allowed\":" ++ (if result.allowed then "true" else "false") ++
    ",\"next_stage\":\"" ++ stageName result.next ++ "\",\"failed\":[" ++ failures ++ "]}"
  (← IO.getStdout).putStrLn json
  return 0
