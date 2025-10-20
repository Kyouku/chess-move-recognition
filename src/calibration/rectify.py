import cv2
import numpy as np


def load_intrinsics(path="data/calibration/intrinsics.npz"):
    data = np.load(path)
    return data["K"], data["D"]


def warp_to_topdown(image, H, out_size=(800, 800)):
    return cv2.warpPerspective(image, H, out_size)


def rectify_board(image, H=None, board_size=(800, 800)):
    if H is None:
        return image
    return warp_to_topdown(image, H, out_size=board_size)
