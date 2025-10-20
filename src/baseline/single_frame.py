import argparse

import cv2

from src.detection.detect_pieces import PieceDetector
from src.utils.visualize import draw_detections


def run(image_path, model_path="models/yolov8n.pt", out=None):
    img = cv2.imread(image_path)
    det = PieceDetector(model_path)
    pieces = det.detect(img)
    vis = draw_detections(img, pieces)
    if out:
        cv2.imwrite(out, vis)
    else:
        cv2.imshow("detections", vis)
        cv2.waitKey(0)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--image", required=True)
    a.add_argument("--model", default="models/yolov8n.pt")
    a.add_argument("--out")
    args = a.parse_args()
    run(args.image, args.model, args.out)
