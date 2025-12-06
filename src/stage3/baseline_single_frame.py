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

        # Helpers
        files = "abcdefgh"

        def file_idx(s: str) -> int:
            return files.find(s[0]) if s and len(s) == 2 else -1

        def rank_num(s: str) -> int:
            try:
                return int(s[1])
            except Exception:
                return -1

        def is_diag_step(a: str, b: str) -> bool:
            return abs(file_idx(a) - file_idx(b)) == 1 and abs(rank_num(a) - rank_num(b)) == 1

        def code_piece_char(code: Optional[str]) -> str:
            return (code or "  ").lower()[1:2]  # 'p','n','b','r','q','k' or ''

        def code_color_char(code: Optional[str]) -> str:
            return (code or " ").lower()[0:1]  # 'w' or 'b' or ''

        def is_last_rank_for_color(sq: str, color: str) -> bool:
            r = rank_num(sq)
            return (color == 'w' and r == 8) or (color == 'b' and r == 1)

        # Compute diffs
        cleared: list[str] = []  # occupied -> empty
        filled: list[str] = []  # empty -> occupied
        label_change_sqs: list[str] = []  # occupied both, label changed

        for sq, was_occ in self._ref_occ.items():
            now_occ = bool(curr_occ.get(sq, False))
            if was_occ and not now_occ:
                cleared.append(sq)
            elif not was_occ and now_occ:
                filled.append(sq)
            else:
                if was_occ and now_occ and self._ref_pieces is not None and curr_pieces is not None:
                    prev_lbl = (self._ref_pieces.get(sq) or "").lower()
                    curr_lbl = (curr_pieces.get(sq) or "").lower()
                    if prev_lbl != curr_lbl:
                        label_change_sqs.append(sq)

        # 1) Simple move (quiet): exactly one cleared and one filled
        if len(cleared) == 1 and len(filled) == 1:
            from_sq = cleared[0]
            to_sq = filled[0]
            uci = f"{from_sq}{to_sq}"

            # Promotion (quiet promotion)
            if self._ref_pieces is not None:
                mover_code = self._ref_pieces.get(from_sq)
                if code_piece_char(mover_code) == 'p':
                    color = code_color_char(mover_code)
                    if is_last_rank_for_color(to_sq, color):
                        promo = 'q'
                        if curr_pieces is not None:
                            pch = code_piece_char(curr_pieces.get(to_sq))
                            if pch in ('q', 'r', 'b', 'n'):
                                promo = pch
                        uci = f"{uci}{promo}"
            return uci

        # 2) Capture where dest remained occupied: one cleared, one label-changed square
        if len(cleared) == 1 and len(filled) == 0 and len(label_change_sqs) == 1:
            from_sq = cleared[0]
            to_sq = label_change_sqs[0]
            uci = f"{from_sq}{to_sq}"

            # Promotion with capture (capture-promotion): pawn reaches last rank
            if self._ref_pieces is not None:
                mover_code = self._ref_pieces.get(from_sq)
                if code_piece_char(mover_code) == 'p':
                    color = code_color_char(mover_code)
                    if is_last_rank_for_color(to_sq, color):
                        promo = 'q'
                        if curr_pieces is not None:
                            pch = code_piece_char(curr_pieces.get(to_sq))
                            if pch in ('q', 'r', 'b', 'n'):
                                promo = pch
                        uci = f"{uci}{promo}"
            return uci

        # 3) Castling: two cleared and two filled matching castle patterns
        if len(cleared) == 2 and len(filled) == 2:
            cset = set(cleared)
            fset = set(filled)
            # White
            if cset == {"e1", "h1"} and fset == {"g1", "f1"}:
                return "e1g1"
            if cset == {"e1", "a1"} and fset == {"c1", "d1"}:
                return "e1c1"
            # Black
            if cset == {"e8", "h8"} and fset == {"g8", "f8"}:
                return "e8g8"
            if cset == {"e8", "a8"} and fset == {"c8", "d8"}:
                return "e8c8"

        # 4) En passant: two cleared (from + captured pawn), one filled (to)
        if len(cleared) == 2 and len(filled) == 1:
            to_sq = filled[0]

            # Find which cleared square is the mover (diagonal step to 'to')
            candidates = [sq for sq in cleared if is_diag_step(sq, to_sq)]
            if len(candidates) == 1:
                from_sq = candidates[0]
                captured_sq = next(s for s in cleared if s != from_sq)

                # Geometric EP check: captured square must be same file as 'to' and same rank as 'from'
                if file_idx(captured_sq) == file_idx(to_sq) and rank_num(captured_sq) == rank_num(from_sq):
                    ok = True
                    if self._ref_pieces is not None:
                        mover_code = self._ref_pieces.get(from_sq)
                        captured_code = self._ref_pieces.get(captured_sq)
                        # Must be pawn capturing pawn
                        ok = code_piece_char(mover_code) == 'p' and code_piece_char(captured_code) == 'p'
                    if ok:
                        return f"{from_sq}{to_sq}"

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
