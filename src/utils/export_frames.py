import argparse
from pathlib import Path

import cv2


def export_frames(video_path, output_dir, step=30):
    cap = cv2.VideoCapture(video_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            cv2.imwrite(f"{output_dir}/frame_{idx:06d}.jpg", frame)
        idx += 1
    cap.release()
    print(f"[DONE] Frames exported to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--step", type=int, default=30)
    args = parser.parse_args()
    export_frames(args.video, args.out, args.step)
