# %% [markdown]
# # Chess board rectification batch generator
#
# - Detect calibration board on a "calib.*" image per folder
# - Compute homography once
# - Warp all images in that folder with this fixed homography
# - Save warped images to `data/processed_images`

# %%
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import cv2
import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

print("Python", sys.version)
print("OpenCV", cv2.__version__)
print("NumPy", np.__version__)

# Project root relative to this notebook (training/notebooks)
PROJECT_ROOT = Path("../..").resolve()
print("Project root:", PROJECT_ROOT)

# %%

BOARD_SIZE_PX = 1024
INPUT_TARGET_LONG_EDGE = 1600
BOARD_MARGIN_SQUARES = 1.7


@dataclass
class BoardProjection:
    """Lightweight representation of a warped board and the square polygons.

    Attributes:
        warped_board: The output board image in canonical top-down view of size BOARD_SIZE_PX×BOARD_SIZE_PX.
        squares_px:   Array of shape (8, 8, 4, 2) with the four corner points (x, y) of each original-image
                      square polygon (A1..H8) mapped back into the source image's pixel space.
    """
    warped_board: np.ndarray
    squares_px: np.ndarray


@dataclass
class _BoardRectification:
    """Full rectification result used internally.

    Attributes:
        mode:       Text label describing how rectification was obtained (for example "corners", "calib_h0").
        h:          3×3 homography mapping input image → canonical board image with margins.
        h_inv:      Inverse of `h`, mapping canonical board coords → input image.
        warped:     The warped board image of size BOARD_SIZE_PX×BOARD_SIZE_PX.
        squares_px: The 8×8 grid of square polygons in input-image coordinates.
        input_gray: Grayscale version of the (possibly resized) input used for detection.
    """
    mode: str
    h: np.ndarray
    h_inv: np.ndarray
    warped: np.ndarray
    squares_px: np.ndarray
    input_gray: np.ndarray


@dataclass
class CalibrationCache:
    """Data kept from the calibration frame.

    Only the calibration homography `h0` is stored. All subsequent images will be warped using
    exactly this matrix to ensure identical alignment across the dataset.

    Attributes:
        h0: 3×3 homography from input image to canonical board image.
    """
    h0: np.ndarray


# %%

def _resize_to_fixed_long_edge(
        bgr: np.ndarray,
        target_long: int = INPUT_TARGET_LONG_EDGE,
) -> Tuple[np.ndarray, float]:
    """Resize image so that its longer side equals `target_long` pixels.

    Uses area interpolation when downscaling and linear when upscaling.

    Returns:
        (resized_bgr, scale): The resized image and the scale factor applied to width/height.
    """
    if bgr is None or bgr.size == 0:
        raise ValueError("Input image is empty or failed to load.")
    h, w = bgr.shape[:2]
    long_edge = max(h, w)
    if long_edge == 0:
        raise ValueError("Invalid image with zero dimension.")
    if long_edge == target_long:
        return bgr, 1.0
    s = float(target_long) / float(long_edge)
    new_w, new_h = int(round(w * s)), int(round(h * s))
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(bgr, (new_w, new_h), interpolation=interp), s


def _px_step_and_offset(board_size_px: int, margin_squares: float) -> Tuple[float, float]:
    """Compute pixel step and offset for the canonical board image with margins."""
    step = board_size_px / (8.0 + 2.0 * margin_squares)
    offset = step * margin_squares
    return step, offset


def _S_margin_matrix(board_size_px: int, margin_squares: float) -> np.ndarray:
    """Build the similarity transform that maps board coordinates → pixel coords with margins."""
    step, offset = _px_step_and_offset(board_size_px, margin_squares)
    S = np.array(
        [
            [step, 0.0, offset],
            [0.0, step, offset],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return S


def _compute_square_polygons(h_inv: np.ndarray, margin_squares: float) -> np.ndarray:
    """Return polygons of all 64 squares back projected into the input image."""
    step, offs = _px_step_and_offset(BOARD_SIZE_PX, margin_squares)
    polys = np.zeros((8, 8, 4, 2), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            x0 = offs + c * step
            y0 = offs + r * step
            quad = np.array(
                [
                    [x0, y0],
                    [x0 + step, y0],
                    [x0 + step, y0 + step],
                    [x0, y0 + step],
                ],
                dtype=np.float32,
            )
            polys[r, c] = cv2.perspectiveTransform(
                quad[None, ...], h_inv.astype(np.float32)
            )[0]
    return polys


def _try_find_corners(gray: np.ndarray, pattern_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Detect inner chessboard corners using OpenCV robust or legacy methods."""
    if hasattr(cv2, "findChessboardCornersSB"):
        out = cv2.findChessboardCornersSB(
            gray,
            pattern_size,
            cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
        )
        if isinstance(out, tuple) and out[0] and out[1] is not None:
            return out[1].astype(np.float32)
        if not isinstance(out, tuple) and out is not None:
            return out.astype(np.float32)

    flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_FILTER_QUADS
    )
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    return corners.astype(np.float32) if ok and corners is not None else None


def _solve_homography_corners(
        bgr_pre: np.ndarray,
        pattern_size: Tuple[int, int],
) -> Optional[_BoardRectification]:
    """Estimate board homography from a single image by detecting chessboard corners."""
    gray = cv2.cvtColor(bgr_pre, cv2.COLOR_BGR2GRAY)
    corners = _try_find_corners(gray, pattern_size)
    if corners is None:
        return None

    cv2.cornerSubPix(
        gray,
        corners,
        winSize=(5, 5),
        zeroZone=(-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-4),
    )

    imgp = corners.reshape(-1, 2).astype(np.float32)
    nx, ny = pattern_size
    objp = np.array(
        [[x + 1, y + 1] for y in range(ny) for x in range(nx)],
        dtype=np.float32,
    )

    h_board, _ = cv2.findHomography(imgp, objp, method=cv2.RANSAC)
    if h_board is None or not np.isfinite(h_board).all():
        return None

    S_m = _S_margin_matrix(BOARD_SIZE_PX, BOARD_MARGIN_SQUARES)
    h_total = (S_m @ h_board).astype(np.float32)
    h_inv = np.linalg.inv(h_total)

    warped = cv2.warpPerspective(
        bgr_pre,
        h_total,
        (BOARD_SIZE_PX, BOARD_SIZE_PX),
        flags=cv2.INTER_LINEAR,
    )
    squares_px = _compute_square_polygons(h_inv, BOARD_MARGIN_SQUARES)

    return _BoardRectification(
        mode="corners",
        h=h_total,
        h_inv=h_inv,
        warped=warped,
        squares_px=squares_px,
        input_gray=gray,
    )


def detect_and_rectify_board(
        bgr: np.ndarray,
        pattern_size: Tuple[int, int] = (7, 7),
) -> Optional[_BoardRectification]:
    """Detect chessboard in a single image and return rectification artifacts."""
    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")
    bgr_pre, _ = _resize_to_fixed_long_edge(bgr)
    return _solve_homography_corners(bgr_pre, pattern_size)


def build_calibration_cache(rect: _BoardRectification) -> CalibrationCache:
    """Create calibration cache from a rectified calibration frame."""
    return CalibrationCache(h0=rect.h.copy().astype(np.float32))


def detect_and_rectify_board_with_cache(
        bgr: np.ndarray,
        cache: CalibrationCache,
) -> Optional[_BoardRectification]:
    """Apply the cached homography to an image to obtain its warped board view."""
    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")
    bgr_pre, _ = _resize_to_fixed_long_edge(bgr)
    h = cache.h0.astype(np.float32)
    h_inv = np.linalg.inv(h)
    warped = cv2.warpPerspective(
        bgr_pre,
        h,
        (BOARD_SIZE_PX, BOARD_SIZE_PX),
        flags=cv2.INTER_LINEAR,
    )
    return _BoardRectification(
        mode="calib_h0",
        h=h,
        h_inv=h_inv,
        warped=warped,
        squares_px=_compute_square_polygons(h_inv, BOARD_MARGIN_SQUARES),
        input_gray=cv2.cvtColor(bgr_pre, cv2.COLOR_BGR2GRAY),
    )


def get_board_projection_with_cache(
        bgr: np.ndarray,
        cache: CalibrationCache,
) -> Optional[BoardProjection]:
    """Return only the warped board image and square polygons for downstream code."""
    rect = detect_and_rectify_board_with_cache(bgr, cache)
    if rect is None:
        return None
    return BoardProjection(
        warped_board=rect.warped,
        squares_px=rect.squares_px.astype(np.float32),
    )


# %% [markdown]
# ## Dataset configuration

# %%
# Image roots and output location
INPUT_ROOTS = [
    PROJECT_ROOT / "data" / "raw_images" / "game6",  # adjust to your dataset
]

OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed_images"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PATTERN_SIZE = (7, 7)
MAX_LONG_EDGE = 0  # extra downscale for calibration only, 0 = disabled
EXCLUDE_CALIB_FROM_OUTPUT = True  # skip calib image when writing warped outputs


def discover_image_folders(roots: List[Path]) -> Dict[Path, List[Path]]:
    """Find all folders under given roots that contain at least one image."""
    result: Dict[Path, List[Path]] = {}
    for root in roots:
        root_p = Path(root)
        if not root_p.exists():
            print(f"[WARN] Root not found: {root_p}")
            continue
        for dirpath, _, filenames in os.walk(root_p):
            folder = Path(dirpath)
            imgs = [
                folder / f
                for f in filenames
                if Path(f).suffix.lower() in IMG_EXTS
            ]
            imgs.sort()
            if imgs:
                result[folder] = imgs
    return result


def imread_color(path: Path) -> Optional[np.ndarray]:
    """Read an image in BGR color, returning None on failure."""
    arr = cv2.imread(str(path))
    return arr if arr is not None and arr.size > 0 else None


def save_image(path: Path, img: np.ndarray) -> None:
    """Write an image to disk, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def rel_to_any_root(p: Path, roots: List[Path]) -> Path:
    """Return a path of `p` relative to any of the provided roots, if possible."""
    for r in roots:
        try:
            return p.relative_to(r)
        except Exception:
            continue
    return Path(p.name)


def find_calib_in_folder(folder: Path, images: List[Path]) -> Path:
    """Return the unique calibration image in a folder named exactly `calib.*`."""
    cands = [
        p
        for p in images
        if p.stem.lower() == "calib" and p.suffix.lower() in IMG_EXTS
    ]
    if len(cands) == 1:
        return cands[0]
    if len(cands) == 0:
        raise RuntimeError(f"No calibration image named 'calib.*' in {folder}")
    raise RuntimeError(f"Multiple 'calib.*' images in {folder}: {cands}")


# %%

def rectify_with_cache_folder(
        folder: Path,
        imgs: List[Path],
        out_root: Path,
        roots: List[Path],
) -> Dict:
    """Process one folder: calibrate once, warp all images with the same homography."""
    stats = {
        "folder": str(folder),
        "total": len(imgs),
        "calibrated": False,
        "ok": 0,
        "fail": 0,
        "by_mode": defaultdict(int),
        "samples": [],
        "calibration_used": "",
    }

    calib_path = find_calib_in_folder(folder, imgs)
    bgr_calib = imread_color(calib_path)
    if bgr_calib is None:
        raise RuntimeError(f"Cannot read calibration image: {calib_path}")

    if MAX_LONG_EDGE and max(bgr_calib.shape[:2]) > MAX_LONG_EDGE:
        s = MAX_LONG_EDGE / max(bgr_calib.shape[:2])
        bgr_calib = cv2.resize(
            bgr_calib,
            (int(bgr_calib.shape[1] * s), int(bgr_calib.shape[0] * s)),
            interpolation=cv2.INTER_AREA,
        )

    rect0 = detect_and_rectify_board(bgr_calib, pattern_size=PATTERN_SIZE)
    if rect0 is None:
        raise RuntimeError(f"Calibration failed on {calib_path}")

    cache = build_calibration_cache(rect0)
    stats["calibrated"] = True
    stats["calibration_used"] = str(calib_path)

    for img_path in tqdm(imgs, desc=f"Process {folder.name}"):
        if EXCLUDE_CALIB_FROM_OUTPUT and img_path.resolve() == calib_path.resolve():
            continue

        bgr = imread_color(img_path)
        if bgr is None:
            stats["fail"] += 1
            continue

        rect = detect_and_rectify_board_with_cache(bgr, cache)
        if rect is None:
            stats["fail"] += 1
            continue

        stats["ok"] += 1
        stats["by_mode"][rect.mode] += 1

        rel = rel_to_any_root(img_path.parent, roots)
        out_dir = out_root / rel
        out_name = img_path.stem + "_warp.png"
        out_path = out_dir / out_name
        save_image(out_path, rect.warped)

        if len(stats["samples"]) < 3:
            stats["samples"].append(str(out_path))

    return stats


# %% [markdown]
# ## Run batch rectification

# %%
if not INPUT_ROOTS:
    print("Please set INPUT_ROOTS to your raw image folders first.")
else:
    folders = discover_image_folders(INPUT_ROOTS)
    print(f"Discovered {len(folders)} folders with images.")
    for folder, imgs in folders.items():
        try:
            s = rectify_with_cache_folder(
                folder,
                imgs,
                OUTPUT_ROOT,
                INPUT_ROOTS,
            )
            print(
                f"[OK] {folder} | ok={s['ok']} fail={s['fail']} calib={s['calibration_used']}"
            )
        except Exception as e:
            print(f"[ERROR] {folder}: {e}")
    print("Done.")
