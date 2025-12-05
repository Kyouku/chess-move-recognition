from __future__ import annotations

from typing import Dict, Optional


class SingleFrameBaseline:
    """
    Pure detection-based single-frame baseline (no python-chess).

    It keeps a reference occupancy map of the last committed board state and
    proposes a move whenever the current occupancy differs exactly on two
    squares: one source that went from occupied -> empty and one destination
    that went from empty -> occupied.

    The caller is responsible for debouncing and for calling commit() once a
    proposed move has been accepted, which updates the reference state.
    """

    def __init__(self) -> None:
        # Reference of the last committed state
        self._ref_occ: Optional[Dict[str, bool]] = None
        self._ref_pieces: Optional[Dict[str, Optional[str]]] = None

    def reset(
            self,
            initial_occ: Optional[Dict[str, bool]] = None,
            initial_pieces: Optional[Dict[str, Optional[str]]] = None,
    ) -> None:
        """Reset the internal reference to the provided maps (optional)."""
        self._ref_occ = dict(initial_occ) if initial_occ is not None else None
        self._ref_pieces = (
            dict(initial_pieces) if initial_pieces is not None else None
        )

    def update_state(
            self,
            curr_occ: Dict[str, bool],
            curr_pieces: Optional[Dict[str, Optional[str]]] = None,
    ) -> Optional[str]:
        """
        Compare current state against the reference and, if it looks like a
        single move, return the UCI-like string '<from><to>' such as 'e2e4'.
        Supported patterns:
          - Exactly one square cleared and one square filled (normal move)
          - One square cleared and one still-occupied square changed label (capture)

        Returns None if the change is ambiguous or noisy.

        This method does NOT update the internal reference. Call commit() after
        confirming the move externally.
        """
        if not curr_occ:
            return None

        # Establish reference if missing
        if self._ref_occ is None:
            self._ref_occ = dict(curr_occ)
        if self._ref_pieces is None and curr_pieces is not None:
            self._ref_pieces = dict(curr_pieces)

        if self._ref_occ is None:
            return None

        from_sq: Optional[str] = None
        to_sq: Optional[str] = None
        label_changes: int = 0
        label_sq: Optional[str] = None

        # Iterate keys from reference
        for sq, was_occ in self._ref_occ.items():
            now_occ = bool(curr_occ.get(sq, False))

            if was_occ and not now_occ:
                if from_sq is None:
                    from_sq = sq
                else:
                    # multiple sources -> ambiguous
                    return None
            elif not was_occ and now_occ:
                if to_sq is None:
                    to_sq = sq
                else:
                    # multiple destinations -> ambiguous
                    return None
            else:
                # Occupancy unchanged; check label change if we track pieces
                if was_occ and now_occ and self._ref_pieces is not None and curr_pieces is not None:
                    prev_lbl = self._ref_pieces.get(sq)
                    curr_lbl = curr_pieces.get(sq)
                    if (prev_lbl or "").lower() != (curr_lbl or "").lower():
                        label_changes += 1
                        label_sq = sq

        # Accept simple move (ignore label noise on other squares)
        # Earlier we required label_changes == 0 which proved too strict in
        # practice due to occasional reclassification noise while occupancy
        # remains the same on unrelated squares. We only need occupancy diffs
        # to identify a quiet move reliably, so accept from+to regardless of
        # label_changes.
        if from_sq and to_sq:
            return f"{from_sq}{to_sq}"

        # Accept capture heuristic: from + one label change
        if from_sq and not to_sq and label_changes == 1 and label_sq is not None:
            return f"{from_sq}{label_sq}"

        return None

    def commit(
            self,
            new_ref_occ: Dict[str, bool],
            new_ref_pieces: Optional[Dict[str, Optional[str]]] = None,
    ) -> None:
        """Update the reference state after a confirmed move."""
        self._ref_occ = dict(new_ref_occ)
        if new_ref_pieces is not None:
            self._ref_pieces = dict(new_ref_pieces)
