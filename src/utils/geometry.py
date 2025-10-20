def bbox_center(b):
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def build_grid(width, height, rows=8, cols=8):
    cw, ch = width / cols, height / rows
    squares = []
    for r in range(rows):
        for c in range(cols):
            squares.append((r, c, (c * cw, r * ch, (c + 1) * cw, (r + 1) * ch)))
    return squares


def rc_to_square(r, c):
    files = "abcdefgh"
    ranks = "87654321"
    return f"{files[c]}{ranks[r]}"


def assign_to_squares(pieces, board_w, board_h):
    grid = build_grid(board_w, board_h)
    occ = {}
    for p in pieces:
        cx, cy = bbox_center(p["bbox"])
        for r, c, bb in grid:
            x1, y1, x2, y2 = bb
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                sq = rc_to_square(r, c)
                occ.setdefault(sq, []).append(p)
                break
    return occ
