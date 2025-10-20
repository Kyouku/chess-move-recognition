"""
detect_pieces.py
----------------
Wrapper around YOLOv8 for chess piece detection.
It loads a pretrained or fine-tuned YOLO model and
returns piece bounding boxes and class names for each frame.

Usage example:
    from src.detection.detect_pieces import PieceDetector
    det = PieceDetector("models/yolov8n.pt")
    pieces = det.detect(frame)
"""

from ultralytics import YOLO


class PieceDetector:
    def __init__(self, model_path: str = "models/yolov8n.pt", imgsz: int = 640, conf: float = 0.5):
        """
        Initialize a YOLOv8 detector.

        Args:
            model_path (str): Path to the YOLO model weights (.pt file)
            imgsz (int): Inference image size
            conf (float): Confidence threshold for detections
        """
        print(f"[INFO] Loading YOLOv8 model from {model_path} ...")
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.names = self.model.names
        print(f"[READY] Model loaded with {len(self.names)} classes.")

    def detect(self, image):
        """
        Run inference on an input image and return structured detections.

        Args:
            image (np.ndarray): BGR image (OpenCV format)

        Returns:
            list[dict]: [
                {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float,
                    'class': str
                }, ...
            ]
        """
        results = self.model.predict(image, imgsz=self.imgsz, conf=self.conf, verbose=False)
        if not results or not results[0].boxes:
            return []

        dets = results[0].boxes.data.cpu().numpy()
        pieces = []
        for x1, y1, x2, y2, conf, cls in dets:
            pieces.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(conf),
                    "class": self.names[int(cls)],
                }
            )
        return pieces
