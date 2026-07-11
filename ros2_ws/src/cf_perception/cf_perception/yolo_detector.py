"""YOLO object detector via the `ultralytics` package itself.

Earlier versions of this file hand-rolled ONNX inference (first cv2.dnn, then
onnxruntime with manually-written letterbox/NMS postprocessing) to avoid
pulling in `ultralytics` as a dependency. Both were dropped: cv2.dnn can't
import YOLO11's attention (C2PSA) blocks at all, and hand-written
pre/postprocessing is a second place a format-assumption mismatch (letterbox
padding, output tensor layout, class ordering, ...) can silently produce wrong
boxes -- exactly the class of bug this project's weights are unlikely to hit
if the model is just run through the same library it was trained/exported
with. `ultralytics.YOLO` loads .onnx weights directly (dispatching to
onnxruntime internally) and returns boxes already in original-frame pixel
coordinates with class names read straight from the model's own export
metadata, so there's no separate format contract to keep in sync here.
"""
from ultralytics import YOLO


class YoloDetector:
    def __init__(self, weights_path, class_names=None, confidence_threshold=0.5,
                 nms_threshold=0.45, input_size=640):
        self.model = YOLO(weights_path)
        # Optional override -- if not given, class names come from the model's
        # own export metadata (model.names / per-result r.names), which is
        # what you get for free when the .onnx was produced by your own
        # `model.export()` from a trained ultralytics run.
        self.class_names = class_names or None
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

    def detect_raw(self, frame_bgr):
        """Returns a list of {class_id, class_name, confidence, bbox (x, y, w, h)
        in original-frame pixel coords, ground_px (u, v)}. `ground_px` is the
        bbox's bottom-center pixel, used as the object's ground-contact point
        for altitude-based world projection (see real_perception_node.py's
        pixel_ray_to_world) -- more accurate than the box center for an
        obliquely-mounted camera, since the bottom edge is where the object
        actually touches the floor in the image."""
        results = self.model.predict(
            frame_bgr, conf=self.confidence_threshold, iou=self.nms_threshold,
            imgsz=self.input_size, verbose=False)
        r = results[0]

        detections = []
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            if self.class_names and class_id < len(self.class_names):
                name = self.class_names[class_id]
            else:
                name = r.names.get(class_id, str(class_id))
            bw, bh = x2 - x1, y2 - y1
            detections.append({
                'class_id': class_id,
                'class_name': name,
                'confidence': confidence,
                'bbox': (x1, y1, bw, bh),
                'ground_px': (x1 + bw / 2.0, y2),
            })
        return detections
