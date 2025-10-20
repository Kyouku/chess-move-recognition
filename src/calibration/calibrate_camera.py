import glob
import os

import cv2
import numpy as np


def calibrate(
    pattern_size=(9, 6),
    image_folder="data/calibration",
    save_path="data/calibration/intrinsics.npz",
):
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)

    objpoints, imgpoints = [], []
    images = glob.glob(os.path.join(image_folder, "*.jpg"))

    print(f"[INFO] Found {len(images)} calibration images.")
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(img, pattern_size, corners, ret)
            cv2.imshow("Calibration", img)
            cv2.waitKey(100)
    cv2.destroyAllWindows()

    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    np.savez(save_path, K=K, D=D)
    print(f"[DONE] Camera calibrated. Reprojection error: {ret:.4f}")
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    calibrate()
