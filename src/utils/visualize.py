import cv2


def draw_detections(img, pieces, color=(0, 255, 0)):
    vis = img.copy()
    for p in pieces:
        x1, y1, x2, y2 = map(int, p["bbox"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{p['class']} {p['confidence']:.2f}"
        cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return vis
