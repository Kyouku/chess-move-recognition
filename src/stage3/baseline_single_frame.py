from __future__ import annotations

from typing import Dict, List, Optional

import chess

from src.types import PieceDetection, MoveInfo

_PIECE_TYPE_FROM_CHAR: Dict[str, int] = {
    "p": chess.PAWN,
    "n": chess.KNIGHT,
    "b": chess.BISHOP,
    "r": chess.ROOK,
    "q": chess.QUEEN,
    "k": chess.KING,
}


class SingleFrameBaseline:
    """
    Baseline recognizer that uses only the current frame.

    Pipeline idea:
      - you pass a list of PieceDetection objects for each frame
      - the recognizer compares all legal moves with the detections
      - the move whose resulting board best matches the detections
        is returned as MoveInfo

    No multi-frame filter, no history other than the current board state.
    """

    def __init__(
            self,
            start_fen: str = chess.STARTING_FEN,
            max_mismatch_squares: int = 10,
    ) -> None:
        """
        start_fen: starting position of the game
        max_mismatch_squares: maximum allowed mismatching squares between
                              simulated board and detections
        """
        self.board = chess.Board(start_fen)
        self.max_mismatch_squares = int(max_mismatch_squares)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, fen: str = chess.STARTING_FEN) -> None:
        """Reset the internal board state."""
        self.board = chess.Board(fen)

    def update(self, detections: List[PieceDetection]) -> Optional[MoveInfo]:
        """
        Update the baseline with a single-frame detection.

        Parameters:
          detections: list of all square detections for the current frame.
                      There should be at most one detection per square
                      (this is enforced here by keeping the max score).

        Returns:
          MoveInfo if a consistent move was found,
          otherwise None.
        """
        if self.board.is_game_over():
            return None

        square_to_det = self._merge_detections_per_square(detections)
        if not square_to_det:
            # nothing detected; don't attempt to infer a move
            return None

        best_move: Optional[chess.Move] = None
        best_cost: float = float("inf")

        board_before = self.board.copy(stack=False)

        for move in board_before.legal_moves:
            # simulate the position after this move
            sim_board = board_before.copy(stack=False)
            sim_board.push(move)

            cost = self._mismatch_cost(sim_board, square_to_det)
            if cost < best_cost:
                best_cost = cost
                best_move = move

        if best_move is None:
            return None

        if best_cost > self.max_mismatch_squares:
            # detections do not match any legal move well enough
            return None

        # apply the best move to the live board
        self.board.push(best_move)
        fen_after = self.board.fen()
        san = board_before.san(best_move)

        return MoveInfo(move=best_move, san=san, fen_after=fen_after)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_detections_per_square(
            detections: List[PieceDetection],
    ) -> Dict[str, PieceDetection]:
        """
        Select the detection with the highest confidence per square.
        """
        result: Dict[str, PieceDetection] = {}
        for det in detections:
            prev = result.get(det.square)
            if prev is None or det.score > prev.score:
                result[det.square] = det
        return result

    def _mismatch_cost(
            self,
            board: chess.Board,
            square_to_det: Dict[str, PieceDetection],
    ) -> int:
        """
        Simple cost function:
          - +1 if there is a piece on the board but nothing was detected
          - +1 if something was detected but the board square is empty
          - +1 if color or piece type does not match

        The lower the cost, the better the simulated position matches
        the detections.
        """
        cost = 0

        for square in chess.SQUARES:
            square_name = chess.square_name(square)
            piece = board.piece_at(square)
            det = square_to_det.get(square_name)

            if det is None and piece is None:
                continue

            if det is None and piece is not None:
                cost += 1
                continue

            if det is not None and piece is None:
                cost += 1
                continue

            assert det is not None and piece is not None

            det_color = chess.WHITE if det.color.lower() == "white" else chess.BLACK
            det_piece_type = _PIECE_TYPE_FROM_CHAR.get(det.piece_type.lower())

            if det_piece_type is None:
                # unknown type counts as a full mismatch
                cost += 1
                continue

            if piece.color != det_color or piece.piece_type != det_piece_type:
                cost += 1

        return cost
