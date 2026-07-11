"""YOLOv8-ONNX object detector via OpenCV's DNN module (no extra dependency --
opencv-python's cv2.dnn already ships with the rest of this package's cv2 use).

Assumes the .onnx weights come from an Ultralytics `model.export(format='onnx')`
with default settings: single output tensor of shape (1, 4 + num_classes, N)
(box xywh in letterboxed-input pixel space, followed by one score per class --
no separate objectness column, no NMS baked into the graph). If your export
used `nms=True` or a different output layout, this postprocessing will not
line up with it -- re-export without `nms=True`, or adjust `_postprocess`.
"""
import cv2
import numpy as np


class YoloDetector:
    def __init__(self, weights_path, class_names=None, confidence_threshold=0.5,
                 nms_threshold=0.45, input_size=640):
        self.net = cv2.dnn.readNetFromONNX(weights_path)
        self.class_names = class_names or []
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
        h, w = frame_bgr.shape[:2]
        scale = self.input_size / max(h, w)
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        resized = cv2.resize(frame_bgr, (nw, nh))
        canvas = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        canvas[:nh, :nw] = resized

        blob = cv2.dnn.blobFromImage(
            canvas, 1 / 255.0, (self.input_size, self.input_size), swapRB=True, crop=False)
        self.net.setInput(blob)
        raw = self.net.forward()

        return self._postprocess(raw, scale)

    def _postprocess(self, raw, scale):
        # raw: (1, 4+nc, N) -> (N, 4+nc): [cx, cy, w, h, score_0, ..., score_{nc-1}]
        # in letterboxed-canvas pixel coordinates.
        preds = raw[0].T
        if preds.shape[0] == 0:
            return []
        boxes_cxcywh = preds[:, :4]
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

        keep = confidences >= self.confidence_threshold
        if not np.any(keep):
            return []
        boxes_cxcywh, class_ids, confidences = (
            boxes_cxcywh[keep], class_ids[keep], confidences[keep])

        # Undo the letterbox scale to get back to original-frame pixels, and
        # cxcywh -> xywh (top-left corner) for cv2.dnn.NMSBoxes.
        boxes_xywh = []
        for cx, cy, bw, bh in boxes_cxcywh:
            x = (cx - bw / 2.0) / scale
            y = (cy - bh / 2.0) / scale
            boxes_xywh.append([x, y, bw / scale, bh / scale])

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh, confidences.tolist(), self.confidence_threshold, self.nms_threshold)
        if len(indices) == 0:
            return []

        results = []
        for i in np.array(indices).flatten():
            x, y, bw, bh = boxes_xywh[i]
            class_id = int(class_ids[i])
            name = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
            results.append({
                'class_id': class_id,
                'class_name': name,
                'confidence': float(confidences[i]),
                'bbox': (x, y, bw, bh),
                'ground_px': (x + bw / 2.0, y + bh),
            })
        return results
