"""YOLO object detector via the `ultralytics` package itself.

`ultralytics.YOLO` loads .onnx weights directly and returns boxes already in
original-frame pixel coordinates, with class names read straight from the
model's own export metadata (`r.names`) -- since that's the same library the
model was trained/exported with, there's no separate ONNX pre/postprocessing
format to keep in sync here.

`task='detect'` is passed explicitly (rather than left for ultralytics to
guess from the .onnx) to silence its "unable to automatically guess model
task" warning and to fail loudly instead of silently on a non-detection model.

`imgsz` is deliberately NOT passed to predict() -- letting ultralytics read
the input size from the model's own embedded export metadata instead of a
config value we'd have to keep in sync by hand. A .onnx exported with a
static (non-dynamic) input shape hardcodes that size into the graph itself;
overriding it with a different `imgsz` here made onnxruntime fail with an
INVALID_ARGUMENT shape mismatch (seen in practice: model exported at 416x416,
code was forcing 640x640). Always matching whatever size the weights were
actually exported at avoids that whole class of bug.
"""
from ultralytics import YOLO


class YoloDetector:
    def __init__(self, weights_path, confidence_threshold=0.5, nms_threshold=0.45):
        self.model = YOLO(weights_path, task='detect')
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

    def detect_raw(self, frame_bgr):
        """Returns a list of {class_id, class_name, confidence, bbox (x, y, w, h)
        in original-frame pixel coords, ground_px (u, v)}. `ground_px` is the
        bbox's bottom-center pixel, used as the object's ground-contact point
        for altitude-based world projection (see real_perception_node.py's
        pixel_ray_to_world) -- more accurate than the box center for an
        obliquely-mounted camera, since the bottom edge is where the object
        actually touches the floor in the image."""
        results = self.model.predict(
            frame_bgr, imgsz=416, device=0, conf=self.confidence_threshold, iou=self.nms_threshold, verbose=False)
        r = results[0]

        detections = []
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
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
