from src.utils.geometry import assign_to_squares


def infer_move(cur_pieces, prev_pieces, board_w, board_h):
    if prev_pieces is None:
        return None

    prev_occ = assign_to_squares(prev_pieces, board_w, board_h)
    cur_occ = assign_to_squares(cur_pieces, board_w, board_h)

    lost = [
        sq for sq in prev_occ.keys() if sq not in cur_occ or len(cur_occ[sq]) < len(prev_occ[sq])
    ]
    gained = [
        sq for sq in cur_occ.keys() if sq not in prev_occ or len(cur_occ[sq]) > len(prev_occ[sq])
    ]

    if len(lost) == 1 and len(gained) == 1:
        return {"from": lost[0], "to": gained[0]}
    if lost and gained:
        return {"from": lost[0], "to": gained[0]}
    return None
