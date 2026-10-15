"""Write-behind for the derived and advisory rows of a RESIDENT engine's index.

A hook that appended events and judged them used to pay for three synchronous SQLite commits: the events and the
cursor (authority), the incremental monitor states (a cache) and, after a full verification, the "verified in full at T"
stamp (advice). Each commit is an ``fsync`` of its own, 10 to 25 ms on the machines this was measured on, and the two
that carry no authority were most of what a warm judgement cost. A process that stays alive does not have to make them
durable before it answers, so with ``defer_derived`` they are held here and written in the transaction of the NEXT
authority commit, at ``close``, when the daemon is idle, or when the buffer is full.

What makes that safe (the argument the tests check):

* Nothing here is authority. Monitor states and stamps are validated against the commit log, the generation, the cursor
  and the block table whenever they are read (``audit.incremental.AuditPlan.prepare`` rejects a row whose generation,
  offset, hash, event count, evaluator or parser version, contract fingerprint or path does not match), and a row that is
  missing or lost is only ever "rebuild from the events" or "verify in full again", never "clean".
* The process that holds a row reads it back through the same merge rule the database would apply (``overlay_states``:
  the newer ``(generation, byte_offset)`` wins, a tie goes to the later write; ``SessionStore._source_block_record``
  applies a stamp only to the exact table, ``synced_at`` and ``root``, it was made for). So within the process a held row
  behaves exactly like a written one.
* Another process sees only what was written. For it a held row does not exist yet, which is the "rebuild or verify
  again" case above.
* Every event that makes a held row wrong removes it first: a rewrite or reindex of the session (``forget_session``), a
  dropped or demoted monitor (``forget_monitors``), a commit that replaces the block table (``forget_stamp``). When a
  held row is finally written it is validated once more against the lineage as it is then, and a row that no longer
  fits is dropped instead of written.
* A crash loses the buffer, and with it only a cache.
"""
from __future__ import annotations

from self_directing_mcp.audit.shadow import merge_flags

# A flush empties the buffer; these bound what a process that never commits again can collect in the meantime.
MAX_ENTRIES = 2048
MAX_BYTES = 16 * 1024 * 1024      # of held monitor-state text


class DerivedBuffer:
    """Held monitor-state rows and verification stamps of one ``SessionStore`` (touched only under the engine's lease)."""

    def __init__(self) -> None:
        # (provider, session_id) -> {(needs_digest, monitor_id): row}; ``row`` has the columns of audit_monitor_state.
        self.states: dict[tuple[str, str], dict[tuple[str, str], dict]] = {}
        # (provider, session_id) -> the stamp of one block table: synced_at/root name the table it was made for.
        self.stamps: dict[tuple[str, str], dict] = {}
        # (receipts database path, session_id) -> the receipts digest the index was last built on (AGY "indexed" marker).
        # It lives in another database than the rows above, so ``agy_receipts.ReceiptStore`` writes it, not the store.
        self.markers: dict[tuple[str, str], str] = {}
        # (provider, session_id) -> (epoch, state dict, request kinds touched since the last flush): the adaptive
        # shadow-comparison schedule (audit/shadow.py). It names its own premise and epoch, so no lineage check is needed
        # when it is written; the epoch guard of the table applies here as well (an older epoch never replaces a newer one).
        self.shadows: dict[tuple[str, str], tuple[int, dict, frozenset]] = {}

    def __len__(self) -> int:
        # ``list(...)`` is one atomic step: another thread may be holding a row while this one only asks "anything?".
        return (sum(len(rows) for rows in list(self.states.values())) + len(self.stamps) + len(self.markers)
                + len(self.shadows))

    # -- shadow-comparison schedules ----------------------------------------------------------------------------
    def hold_shadow(self, provider, session_id, epoch, state, touched=frozenset()) -> bool:
        """Hold a schedule unless a newer epoch is held. A writer of the held epoch keeps the flags of the request kinds it did
        not handle (another request may have saved since it read), exactly as the table does; returns whether it was held."""
        key = (provider, session_id)
        held = self.shadows.get(key)
        if held is not None:
            if held[0] > epoch:
                return False
            if held[0] == epoch:
                state = {**state, "flags": merge_flags(held[1].get("flags"), state.get("flags") or {}, touched)} \
                    if touched else state
                touched = frozenset(touched) | held[2]
            else:
                touched = frozenset(touched)
        self.shadows[key] = (epoch, state, frozenset(touched))
        return True

    def shadow_for(self, provider, session_id):
        """``(epoch, state dict)`` of the held schedule, or ``None``."""
        held = self.shadows.get((provider, session_id))
        return None if held is None else (held[0], held[1])

    def drop_shadow(self, provider, session_id) -> None:
        self.shadows.pop((provider, session_id), None)

    def drain_shadows(self) -> list[tuple[str, str, int, dict, frozenset]]:
        taken = [(provider, session, epoch, state, touched)
                 for (provider, session), (epoch, state, touched) in self.shadows.items()]
        self.shadows = {}
        return taken

    def full(self) -> bool:
        return (len(self) >= MAX_ENTRIES
                or sum(len(row["state"]) for rows in list(self.states.values()) for row in list(rows.values())) >= MAX_BYTES)

    # -- monitor states -----------------------------------------------------------------------------------------
    def hold_state(self, provider, session_id, needs_digest, monitor_id, row) -> bool:
        """Hold ``row`` unless a newer one of the same monitor is held (never over a newer state)."""
        rows = self.states.setdefault((provider, session_id), {})
        key = (needs_digest, monitor_id)
        held = rows.get(key)
        if held is not None and (held["generation"], held["byte_offset"]) > (row["generation"], row["byte_offset"]):
            return False
        rows[key] = row
        return True

    def overlay_states(self, provider, session_id, needs_digest, found: dict, decode) -> dict:
        """``found`` (``{monitor_id: stored row}``) with the held rows of this session and needs merged in.

        A held row replaces the stored one of its monitor unless the stored one is newer, the rule the database's own
        upsert applies. ``decode(row)`` turns a held row into the shape a stored row has, or ``None`` (an unreadable
        row is the same as no row)."""
        rows = self.states.get((provider, session_id))
        if not rows:
            return found
        merged = dict(found)
        for (digest, monitor_id), row in rows.items():
            if digest != needs_digest:
                continue
            stored = merged.get(monitor_id)
            if stored is not None and (row["generation"], row["byte_offset"]) < (stored["generation"],
                                                                                 stored["byte_offset"]):
                continue
            record = decode(row)
            if record is not None:
                merged[monitor_id] = record
        return merged

    def forget_session(self, provider, session_id) -> None:
        """Every held state of the session (its events were replaced, so none of them describes the stored ones)."""
        self.states.pop((provider, session_id), None)

    def forget_monitors(self, provider, session_id, monitor_ids) -> None:
        """The held states of these monitors, whatever their needs (a dropped or demoted monitor)."""
        rows = self.states.get((provider, session_id))
        if not rows:
            return
        wanted = set(monitor_ids)
        for key in [key for key in rows if key[1] in wanted]:
            del rows[key]
        if not rows:
            del self.states[(provider, session_id)]

    # -- verification stamps ------------------------------------------------------------------------------------
    def hold_stamp(self, provider, session_id, stamp: dict) -> None:
        self.stamps[(provider, session_id)] = stamp

    def stamp_for(self, provider, session_id, synced_at, root) -> dict | None:
        """The held stamp, only if it was made for exactly this table (same publication and root)."""
        stamp = self.stamps.get((provider, session_id))
        if stamp is None:
            return None
        if stamp["synced_at"] == synced_at and stamp["root"] == root:
            return stamp
        del self.stamps[(provider, session_id)]    # the table it stamped has been replaced
        return None

    def forget_stamp(self, provider, session_id) -> None:
        self.stamps.pop((provider, session_id), None)

    # -- AGY indexed markers ------------------------------------------------------------------------------------
    def hold_marker(self, database, session_id, digest) -> None:
        self.markers[(database, session_id)] = digest

    def marker_for(self, database, session_id) -> str | None:
        return self.markers.get((database, session_id))

    def drop_marker(self, database, session_id) -> None:
        self.markers.pop((database, session_id), None)

    def drain_markers(self, database) -> list[tuple[str, str]]:
        """The held markers of one receipts database as ``(session_id, digest)``, removed from the buffer."""
        taken = [(key, digest) for key, digest in self.markers.items() if key[0] == database]
        for key, _digest in taken:
            del self.markers[key]
        return [(key[1], digest) for key, digest in taken]

    # -- draining -----------------------------------------------------------------------------------------------
    def drain(self) -> tuple[list[tuple[str, str, str, str, dict]], list[tuple[str, str, dict]]]:
        """Everything held, as ``(provider, session, needs_digest, monitor_id, row)`` and ``(provider, session, stamp)``;
        the buffer is empty afterwards (a row that fails to write is a lost cache entry, not retried)."""
        states = [(provider, session, digest, monitor_id, row)
                  for (provider, session), rows in self.states.items() for (digest, monitor_id), row in rows.items()]
        stamps = [(provider, session, stamp) for (provider, session), stamp in self.stamps.items()]
        self.states, self.stamps = {}, {}
        return states, stamps
