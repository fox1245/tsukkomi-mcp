"""Grok: the calls a hook blocked, joined from the host's own records.

Grok writes no structured refusal on a call. What it writes about a ``PreToolUse`` denial is (``grok.py`` explains the
shapes and why the join is made here and not in the parser):

* the ``tool_call`` (``native_tool_name``, ``call_id``);
* a ``failed`` ``tool_call_update``, the call's result, carrying ``host_event_id`` n (Grok's own sequence number);
* the host's note "`<tool>` blocked by <source>", a ``host_tool_outcome`` event carrying ``host_event_id`` n + 1, the
  tool name (``blocked_tool``) and the source (``denied_by``).

A call is blocked when its failed result and a note are numbered n and n + 1, the note names the tool of the call, and
nothing else vouches for either side. The file may show the note before the call and its result (it does in every
recording) or after them; they may be indexed by different reads. Each side waits in a small table for the other, so the
join gives the same answer for any split of the events. The one class here is the whole rule: the reference evaluator
feeds it every event, the incremental one keeps it in its shared facts (``dump``/``load``) and feeds it the new ones.

Whatever is not unambiguous is not a block, which leaves the call an attempt:

* the call must be the only one with its id that has no result yet, and the id must not have had a result before;
* a call with two results is never a block, whichever came first, and a number that two notes or two results claim is
  nobody's;
* a note names the tool of its call (``native_tool_name``), and a result and its note must lie within ``WINDOW`` events
  of each other (a number only means something next to the records it numbers: a stale note must not meet a later,
  unrelated failure, for instance if a resumed session's counter started over, which no recording shows either way);
* a result that is not ``failed`` (a call that ran) and a note without a number never join.

Nothing here reads text: a tool output, a message or the "Hook denied:" words of the failed update cannot be evidence.
"""
from __future__ import annotations

ORIGIN = "host_tool_outcome"      # ``evidence_origin`` of the note event (``grok.HOST_OUTCOME``)
WINDOW = 512                      # events between a result and its note, at most
CALLS = 256                       # calls without a result remembered
DONE = 64                         # finished calls remembered, to tell a second result from a first
PENDING = 64                      # results waiting for a note, and notes waiting for a result


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _trim(table: dict, limit: int) -> None:
    while len(table) > limit:
        del table[next(iter(table))]


class HostBlocks:
    """Feed events in order; ``rejected`` maps a blocked call's chunk id to ``[chunk id of the note, source]``."""

    def __init__(self):
        self.tick = 0
        self.calls = {}      # call id -> [call chunk id, tool name]; None when several calls use the id
        self.done = {}       # call id -> call chunk id once the call has a result; None when it cannot vouch
        self.fails = {}      # n -> [call chunk id, tool name, tick] of a failed result with no note yet
        self.notes = {}      # n -> [note chunk id, tool name, source, tick] of a note with no failed result yet
        self.rejected = {}   # call chunk id -> [note chunk id, source]

    # --- feeding ---

    def feed(self, event) -> None:
        self.tick += 1
        meta = getattr(event, "meta", None) or {}
        if event.kind == "tool_call":
            self._call(event, meta)
        elif event.kind == "tool_result":
            self._result(meta)
        elif event.kind == "meta" and meta.get("evidence_origin") == ORIGIN:
            self._note(event, meta)

    def _call(self, event, meta) -> None:
        call_id = meta.get("call_id")
        if not (isinstance(call_id, str) and call_id):
            return
        if call_id in self.calls or call_id in self.done:
            self.calls[call_id] = None                  # an id that calls share vouches for none of them
        else:
            self.calls[call_id] = [event.chunk_id, meta.get("native_tool_name")]
        _trim(self.calls, CALLS)

    def _result(self, meta) -> None:
        call_id = meta.get("call_id")
        if not (isinstance(call_id, str) and call_id):
            return
        if call_id in self.done:
            self._contest(call_id)                      # a second result: nothing about this call is certain any more
            return
        entry = self.calls.pop(call_id, False)
        if entry is False:
            return                                      # no call before this result (or it left the window)
        self.done[call_id] = entry[0] if entry else None
        _trim(self.done, DONE)
        number = meta.get("host_event_id")
        if entry and meta.get("tool_status") == "failed" and _is_int(number) and isinstance(entry[1], str) and entry[1]:
            self._fail(number, entry[0], entry[1])

    def _fail(self, number, call, name) -> None:
        note = self.notes.pop(number + 1, None)
        if note is not None and self.tick - note[3] <= WINDOW:
            self._link(call, name, note)
        elif number in self.fails:
            del self.fails[number]                      # two failed results with one number: neither is joined
        else:
            self.fails[number] = [call, name, self.tick]
            _trim(self.fails, PENDING)

    def _note(self, event, meta) -> None:
        number, tool, source = meta.get("host_event_id"), meta.get("blocked_tool"), meta.get("denied_by")
        if not (_is_int(number) and isinstance(tool, str) and tool and isinstance(source, str) and source):
            return
        note = [event.chunk_id, tool, source, self.tick]
        fail = self.fails.pop(number - 1, None)
        if fail is not None and self.tick - fail[2] <= WINDOW:
            self._link(fail[0], fail[1], note)
        elif number in self.notes:
            del self.notes[number]                      # two notes with one number: neither is joined
        else:
            self.notes[number] = note
            _trim(self.notes, PENDING)

    def _link(self, call, name, note) -> None:
        if name == note[1]:                             # the note must be about this call's tool
            self.rejected[call] = [note[0], note[2]]

    def _contest(self, call_id) -> None:
        """A call that got a second result: it is no block (an earlier decision is taken back) and never will be."""
        call = self.done.get(call_id)
        self.done[call_id] = None
        if call is None:
            return
        self.rejected.pop(call, None)
        for number in [number for number, item in self.fails.items() if item[0] == call]:
            del self.fails[number]

    # --- persistence (JSON-safe; the order of a table is the order of insertion, kept as a list of pairs) ---

    def dump(self) -> dict:
        return {"tick": self.tick, "calls": [[k, v] for k, v in self.calls.items()],
                "done": [[k, v] for k, v in self.done.items()], "fails": [[k, v] for k, v in self.fails.items()],
                "notes": [[k, v] for k, v in self.notes.items()],
                "rejected": [[k, v] for k, v in self.rejected.items()]}

    def load(self, data) -> None:
        self.tick = data["tick"]
        self.calls = {k: v for k, v in data["calls"]}
        self.done = {k: v for k, v in data["done"]}
        self.fails = {k: v for k, v in data["fails"]}
        self.notes = {k: v for k, v in data["notes"]}
        self.rejected = {k: v for k, v in data["rejected"]}
