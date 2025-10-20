import os
from pathlib import Path

import cv2

from src.calibration.rectify import rectify_board
from src.detection.detect_pieces import PieceDetector
from src.inference.move_infer import infer_move
from src.utils.io import save_json
from src.utils.visualize import draw_detections


def extract_frames(video_path, step=1):
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            yield idx, frame
        idx += 1
    cap.release()


def run_pipeline(video_path, model_path="models/yolov8n.pt", output_dir="results/", save_vis=False):
    detector = PieceDetector(model_path)
    os.makedirs(output_dir, exist_ok=True)
    predictions = []
    prev_state = None
    frame_count = 0

    for idx, frame in extract_frames(video_path, step=5):
        rectified = rectify_board(frame)
        h, w = rectified.shape[:2]
        pieces = detector.detect(rectified)
        move = infer_move(pieces, prev_state, w, h)
        predictions.append({"frame": idx, "pieces": pieces, "move": move})

        if save_vis:
            vis = draw_detections(rectified, pieces)
            Path(output_dir).mkdir(exist_ok=True, parents=True)
            cv2.imwrite(f"{output_dir}/frame_{idx:06d}.jpg", vis)

        prev_state = pieces
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"[INFO] Processed {frame_count} frames...")

    save_json(predictions, f"{output_dir}/{Path(video_path).stem}_pred.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="models/yolov8n.pt")
    parser.add_argument("--output", default="results/")
    parser.add_argument("--save-vis", action="store_true")
    args = parser.parse_args()
    run_pipeline(args.video, args.model, args.output, save_vis=args.save_vis)
