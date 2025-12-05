from __future__ import annotations

from typing import Dict, List, Optional

import chess

from src import config
from src.app_logging import get_logger
from src.pipeline_comparison.common import PipelineResult
from src.stage3.baseline_single_frame import SingleFrameBaseline
from src.types import DetectionState

_log = get_logger(__name__)


class _BaselineRunner:
    """
    Offline Version der SingleFramePipeline.handle_detection_state.

    - Mit Start Gating auf die Anfangsstellung
    - Jeder von SingleFrameBaseline vorgeschlagene UCI kann gezählt werden
    - Keine Legalitätschecks; python-chess nur für Startpositions-Erkennung
    - Debouncing wie in der Live SingleFramePipeline optional über MOVE_MIN_CONFIRM_FRAMES
    """

    def __init__(self) -> None:
        self.baseline = SingleFrameBaseline()
        # Hinweis: Kein python-chess Boardtracking in der Single-Frame Baseline.
        # python-chess wird hier ausschließlich verwendet, um die Startbelegung
        # (Occupancy) der Anfangsstellung zu bestimmen.

        # Start Gating wie live: warte auf stabile Anfangsstellung
        self._start_occ: Dict[str, bool] = self._board_to_occupancy(
            chess.Board()
        )
        self._start_seen_frames: int = 0
        self._initialized: bool = False
        self._min_confirm_frames: int = int(
            getattr(config, "MOVE_MIN_CONFIRM_FRAMES", 2)
        )

        # Debouncing für vorgeschlagene Züge
        self._pending_uci: Optional[str] = None
        self._pending_count: int = 0
        self._confirm_frames: int = int(
            getattr(config, "MOVE_MIN_CONFIRM_FRAMES", 2)
        )

    @staticmethod
    def _board_to_occupancy(board: chess.Board) -> Dict[str, bool]:
        occ: Dict[str, bool] = {}
        for sq in chess.SQUARES:
            occ[chess.square_name(sq)] = board.piece_at(sq) is not None
        return occ

    def process_state(
            self,
            state: DetectionState,
            frame_idx: int,
    ) -> Optional[tuple[str, Optional[str], Optional[str], int]]:
        """
        Verarbeitet einen DetectionState.

        Gibt (uci, san, fen_after, frame_idx) zurück, wenn ein Zug bestätigt ist,
        sonst None. UCI bleibt erhalten, auch wenn SAN oder FEN nicht berechnet
        werden können.
        """
        # 1) Start Gating wie bisher
        if not self._initialized:
            curr_occ = state.occupancy
            is_match = all(
                self._start_occ[sq] == bool(curr_occ.get(sq, False))
                for sq in self._start_occ
            )
            if is_match:
                self._start_seen_frames += 1
                if self._start_seen_frames >= self._min_confirm_frames:
                    self._initialized = True
                    self._start_seen_frames = 0
                    # Baseline auf Startposition setzen
                    self.baseline.reset(
                        initial_occ=self._start_occ,
                        initial_pieces=state.pieces,
                    )
                    self._pending_uci = None
                    self._pending_count = 0
                    _log.info(
                        "[BASELINE OFFLINE] Initial position locked at frame %d",
                        frame_idx,
                    )
                    return None
            else:
                if self._start_seen_frames > 0:
                    self._start_seen_frames = 0
            return None

        # 2) Nach dem Gating – SingleFrameBaseline Kandidat holen
        curr_occ = state.occupancy
        curr_pieces = state.pieces
        proposed_uci = self.baseline.update_state(curr_occ, curr_pieces)

        # Kein Kandidat – Pending zurücksetzen
        if proposed_uci is None:
            if self._pending_uci is not None:
                _log.debug(
                    "[BASELINE OFFLINE] clearing pending move %s at frame %d",
                    self._pending_uci,
                    frame_idx,
                )
            self._pending_uci = None
            self._pending_count = 0
            return None

        # Debouncing wie in der Live Pipeline
        if self._pending_uci == proposed_uci:
            self._pending_count += 1
        else:
            self._pending_uci = proposed_uci
            self._pending_count = 1

        _log.debug(
            "[BASELINE OFFLINE] candidate %s (%d/%d) at frame %d",
            self._pending_uci,
            self._pending_count,
            self._confirm_frames,
            frame_idx,
        )

        if self._pending_count < self._confirm_frames:
            return None

        # Ab hier ist der Zug bestätigt
        uci = self._pending_uci
        if uci is None:
            return None

        self._pending_uci = None
        self._pending_count = 0

        # Baseline Referenz committen
        self.baseline.commit(curr_occ, curr_pieces)

        # Keine SAN-Berechnung hier – es werden nur UCI Züge ausgegeben.
        # FEN wird direkt aus der aktuellen Detektion abgeleitet (ohne Legalitätschecks).
        san: Optional[str] = None
        fen_after: Optional[str] = _state_to_fen(state)

        if san is not None:
            _log.info(
                "[BASELINE OFFLINE] frame %d move %s (%s)",
                frame_idx,
                san,
                uci,
            )
        else:
            _log.info(
                "[BASELINE OFFLINE] frame %d move (UCI only) %s | FEN: %s",
                frame_idx,
                uci,
                fen_after or "",
            )

        return uci, san, fen_after, frame_idx


def _state_to_fen(state: DetectionState) -> str:
    """
    Erzeugt eine FEN-Zeile aus dem reinen Detektionszustand (Occupancy + Labels).

    Es werden keinerlei Legalitätschecks durchgeführt. Unbekannte/fehlende Labels
    werden als leere Felder behandelt. Nebeninformationen werden mit Standard-
    Platzhaltern befüllt (w - - 0 1).
    """
    files = ["a", "b", "c", "d", "e", "f", "g", "h"]

    def code_to_fen_char(code: Optional[str]) -> Optional[str]:
        if not code or len(code) < 2:
            return None
        c0 = code[0].lower()
        c1 = code[1].lower()
        if c0 not in ("w", "b"):
            return None
        if c1 not in ("p", "n", "b", "r", "q", "k"):
            return None
        letter = c1.upper() if c0 == "w" else c1
        return letter

    rows: List[str] = []
    # Ränge 8 .. 1
    for rank_idx in range(7, -1, -1):
        run = 0
        row_chars: List[str] = []
        for file_idx in range(8):
            sq = f"{files[file_idx]}{rank_idx + 1}"
            occ = bool(state.occupancy.get(sq, False))
            letter: Optional[str] = None
            if occ:
                letter = code_to_fen_char(state.pieces.get(sq))

            if letter is None:
                run += 1
            else:
                if run > 0:
                    row_chars.append(str(run))
                    run = 0
                row_chars.append(letter)
        if run > 0:
            row_chars.append(str(run))
        rows.append("".join(row_chars))

    placement = "/".join(rows)
    # Platzhalter für Side-to-move, Castling, En passant, Halbzug, Zugnummer
    return f"{placement} w - - 0 1"


def _should_process_frame(
        frame_idx: int,
        video_fps: float | None,
        detector_fps: float | None,
) -> bool:
    """
    Entscheidet, ob ein Frame in die Offline Baseline eingespeist wird.

    Wenn video_fps und detector_fps vorhanden sind, wird ein einfaches
    Sampling verwendet, damit nur so viele Frames verarbeitet werden, wie
    der Detektor im Live Betrieb ungefähr schaffen würde.
    """
    if video_fps is None or detector_fps is None or detector_fps <= 0.0:
        return True

    ratio = video_fps / detector_fps if detector_fps > 0 else 1.0
    step = max(1, int(round(ratio)))
    return frame_idx % step == 0


def run_baseline(
        states: List[DetectionState],
        *,
        video_fps: float | None = None,
        detector_fps: float | None = None,
) -> PipelineResult:
    """
    Offline Single Frame Baseline mit Start Gating und Debouncing.

    - Verwendet dieselbe SingleFrameBaseline Logik wie die Live Pipeline
    - Nutzt Start Gating auf die Anfangsstellung
    - Nutzt Debouncing über MOVE_MIN_CONFIRM_FRAMES Frames
    - Optional werden bei vorhandenen FPS Metadaten nur die Frames
      verarbeitet, die der Detektor realistischerweise schaffen würde
    """
    runner = _BaselineRunner()

    frame_fens: List[str] = []
    moves_uci: List[str] = []
    moves_san: List[str] = []
    move_frames: List[int] = []

    total_frames = len(states)

    for idx, state in enumerate(states):
        # Optionales Live Stil Sampling
        if _should_process_frame(idx, video_fps, detector_fps):
            _log.debug(
                "[BASELINE OFFLINE] processing frame %d/%d",
                idx + 1,
                total_frames,
            )
            info = runner.process_state(state, idx)
            if info is not None:
                uci, san, fen_after, frame_idx = info
                moves_uci.append(uci)
                moves_san.append(san or "")
                move_frames.append(frame_idx)

        # Für jede Original Frame Position einen FEN-String aus der Detektion ableiten
        frame_fens.append(_state_to_fen(state))

    return PipelineResult(
        frame_fens=frame_fens,
        moves_uci=moves_uci,
        moves_san=moves_san,
        move_frames=move_frames,
    )
