from __future__ import annotations

import ast
import math
import threading
import time
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics.trackers.byte_tracker import BYTETracker

try:
    cv2.setNumThreads(1)
except Exception:
    pass


class _TrackerDetections:
    """Minimal Ultralytics Results-like object consumed by BYTETracker."""

    def __init__(
        self,
        xyxy: np.ndarray,
        confidence: np.ndarray,
        class_ids: np.ndarray,
    ) -> None:
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(confidence, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(class_ids, dtype=np.float32).reshape(-1)

        if len(self.xyxy):
            x1, y1, x2, y2 = self.xyxy.T
            self.xywh = np.column_stack(
                (
                    0.5 * (x1 + x2),
                    0.5 * (y1 + y2),
                    np.maximum(0.0, x2 - x1),
                    np.maximum(0.0, y2 - y1),
                )
            ).astype(np.float32, copy=False)
        else:
            self.xywh = np.empty((0, 4), dtype=np.float32)

    def __len__(self) -> int:
        return int(self.conf.shape[0])


def _letterbox(
    image: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, float, float, float]:
    source_h, source_w = image.shape[:2]
    scale = min(
        float(width) / max(1, source_w),
        float(height) / max(1, source_h),
    )
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))

    if resized_w != source_w or resized_h != source_h:
        resized = cv2.resize(
            image,
            (resized_w, resized_h),
            interpolation=(
                cv2.INTER_LINEAR
                if scale >= 1.0
                else cv2.INTER_AREA
            ),
        )
    else:
        resized = image

    pad_x = float(width - resized_w) * 0.5
    pad_y = float(height - resized_h) * 0.5
    left = int(math.floor(pad_x))
    right = int(math.ceil(pad_x))
    top = int(math.floor(pad_y))
    bottom = int(math.ceil(pad_y))

    output = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    if output.shape[1] != width or output.shape[0] != height:
        output = cv2.resize(
            output,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
    return output, scale, float(left), float(top)


def _box_iou_one_to_many(
    box: np.ndarray,
    boxes: np.ndarray,
) -> np.ndarray:
    if not len(boxes):
        return np.empty((0,), dtype=np.float32)

    x1 = np.maximum(float(box[0]), boxes[:, 0])
    y1 = np.maximum(float(box[1]), boxes[:, 1])
    x2 = np.minimum(float(box[2]), boxes[:, 2])
    y2 = np.minimum(float(box[3]), boxes[:, 3])

    intersection = (
        np.maximum(0.0, x2 - x1)
        * np.maximum(0.0, y2 - y1)
    )
    first_area = max(
        0.0,
        float(box[2] - box[0]),
    ) * max(
        0.0,
        float(box[3] - box[1]),
    )
    second_area = (
        np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    )
    union = np.maximum(
        first_area + second_area - intersection,
        1.0e-9,
    )
    return intersection / union


def _classwise_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
    maximum_detections: int,
) -> np.ndarray:
    keep: List[int] = []
    for class_id in np.unique(class_ids.astype(np.int64)):
        indices = np.flatnonzero(
            class_ids.astype(np.int64) == int(class_id)
        )
        indices = indices[
            np.argsort(scores[indices])[::-1]
        ]

        while len(indices):
            current = int(indices[0])
            keep.append(current)
            if len(keep) >= int(maximum_detections):
                break
            if len(indices) == 1:
                break

            remaining = indices[1:]
            ious = _box_iou_one_to_many(
                boxes[current],
                boxes[remaining],
            )
            indices = remaining[
                ious <= float(iou_threshold)
            ]

        if len(keep) >= int(maximum_detections):
            break

    if not keep:
        return np.empty((0,), dtype=np.int64)

    keep_array = np.asarray(keep, dtype=np.int64)
    order = np.argsort(scores[keep_array])[::-1]
    return keep_array[order[: int(maximum_detections)]]


class YoloDetector:
    """One shared ONNX session, adaptive ByteTrack start, and one tracker per drone."""

    def __init__(
        self,
        weights_path,
        confidence_threshold=0.35,
        nms_threshold=0.45,
        low_confidence_threshold=0.10,
        image_size=416,
        device=0,
        maximum_detections=20,
        force_grayscale=True,
        use_clahe=False,
        track_buffer=120,
        match_threshold=0.80,
        new_track_threshold=0.35,
        source_ids=(),
    ):
        self.weights_path = str(weights_path)
        self.confidence_threshold = float(
            confidence_threshold)
        self.low_confidence_threshold = min(
            float(low_confidence_threshold),
            self.confidence_threshold,
        )
        self.nms_threshold = float(nms_threshold)
        self.requested_image_size = int(image_size)
        self.maximum_detections = max(
            1, int(maximum_detections))
        self.force_grayscale = bool(force_grayscale)
        self.use_clahe = bool(use_clahe)
        self.device = int(device)

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_options.execution_mode = (
            ort.ExecutionMode.ORT_SEQUENTIAL
        )
        # Explicit values suppress ORT's failing automatic CPU affinity path.
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.enable_mem_pattern = True
        session_options.enable_cpu_mem_arena = True

        available = set(ort.get_available_providers())
        providers = []
        if 'CUDAExecutionProvider' in available:
            providers.append(
                (
                    'CUDAExecutionProvider',
                    {
                        'device_id': str(self.device),
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'cudnn_conv_algo_search': 'HEURISTIC',
                        'do_copy_in_default_stream': '1',
                    },
                )
            )
        providers.append('CPUExecutionProvider')

        self.session = ort.InferenceSession(
            self.weights_path,
            sess_options=session_options,
            providers=providers,
        )
        self.active_providers = tuple(
            self.session.get_providers())
        self.input_meta = self.session.get_inputs()[0]
        self.input_name = self.input_meta.name
        self.output_names = [
            output.name
            for output in self.session.get_outputs()
        ]

        input_shape = list(self.input_meta.shape)
        batch_dim = input_shape[0] if input_shape else 1
        channel_dim = (
            input_shape[1]
            if len(input_shape) >= 2
            and isinstance(input_shape[1], int)
            else 3
        )
        height_dim = (
            input_shape[2]
            if len(input_shape) >= 3
            and isinstance(input_shape[2], int)
            else self.requested_image_size
        )
        width_dim = (
            input_shape[3]
            if len(input_shape) >= 4
            and isinstance(input_shape[3], int)
            else self.requested_image_size
        )

        self.input_channels = int(channel_dim)
        self.input_height = int(height_dim)
        self.input_width = int(width_dim)
        self.image_size = self.requested_image_size

        if (
            self.input_height != self.requested_image_size
            or self.input_width != self.requested_image_size
        ):
            # Static ONNX input shapes cannot be changed at runtime.
            self.image_size = min(
                self.input_height,
                self.input_width,
            )

        self.dynamic_batch = (
            not isinstance(batch_dim, int)
            or int(batch_dim) >= 2
        )
        self.static_batch_size = (
            int(batch_dim)
            if isinstance(batch_dim, int)
            else None
        )
        self.input_dtype = (
            np.float16
            if 'float16' in str(self.input_meta.type)
            else np.float32
        )

        self.class_names = self._read_class_names()
        self.class_count = (
            len(self.class_names)
            if self.class_names
            else 0
        )

        # Adaptive ByteTrack birth gate.  The registry already requires a
        # coherent multi-frame ground estimate, so a separate 0.30-style birth
        # threshold only loses short fly-bys.  The effective gate is derived
        # from the detector low/high thresholds and is bounded at 0.18.
        adaptive_high_threshold = min(
            self.confidence_threshold,
            max(0.18, self.low_confidence_threshold + 0.10),
        )
        tracker_args = SimpleNamespace(
            track_high_thresh=adaptive_high_threshold,
            track_low_thresh=self.low_confidence_threshold,
            new_track_thresh=adaptive_high_threshold,
            track_buffer=max(30, int(track_buffer)),
            match_thresh=float(match_threshold),
            fuse_score=True,
        )
        self._trackers = {
            str(source_id): BYTETracker(
                args=tracker_args,
                frame_rate=8,
            )
            for source_id in source_ids
        }

        self._clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )
        self._session_lock = threading.Lock()
        self._overlay_lock = threading.Lock()
        self._latest_overlay: Dict[
            str, Tuple[float, List[dict]]
        ] = {
            str(source_id): (0.0, [])
            for source_id in source_ids
        }
        # DeepSORT-style gallery features are produced without an extra CNN.
        # A lightweight crop descriptor is exponentially averaged per local
        # ByteTrack ID and later stored in the world-landmark gallery.
        self._appearance_state: Dict[
            str, Dict[int, Tuple[np.ndarray, float]]
        ] = {
            str(source_id): {}
            for source_id in source_ids
        }
        self._batch_probe_failed = False

        # Warm up only the shape that the model definitely accepts.
        warm_batch = (
            self.static_batch_size
            if (
                self.static_batch_size is not None
                and self.static_batch_size > 0
            )
            else 1
        )
        warm_batch = 1 if warm_batch > 2 else warm_batch
        dummy = np.zeros(
            (
                warm_batch,
                self.input_channels,
                self.input_height,
                self.input_width,
            ),
            dtype=self.input_dtype,
        )
        try:
            self._run_session(dummy)
        except Exception:
            # Real first-frame inference still gets a chance to report the
            # exact model/runtime error.
            pass

    def _read_class_names(self) -> Dict[int, str]:
        metadata = (
            self.session.get_modelmeta()
            .custom_metadata_map
        )
        raw_names = metadata.get('names', '')
        if not raw_names:
            return {}
        try:
            parsed = ast.literal_eval(raw_names)
        except (ValueError, SyntaxError):
            return {}

        if isinstance(parsed, dict):
            return {
                int(key): str(value)
                for key, value in parsed.items()
            }
        if isinstance(parsed, (list, tuple)):
            return {
                index: str(value)
                for index, value in enumerate(parsed)
            }
        return {}

    def _prepare_image(
        self,
        frame_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        image = frame_bgr
        if self.force_grayscale:
            gray = cv2.cvtColor(
                image, cv2.COLOR_BGR2GRAY)
            if self.use_clahe:
                gray = self._clahe.apply(gray)
            if self.input_channels == 1:
                image = gray[:, :, None]
            else:
                image = cv2.cvtColor(
                    gray, cv2.COLOR_GRAY2BGR)
        elif self.use_clahe:
            lab = cv2.cvtColor(
                image, cv2.COLOR_BGR2LAB)
            lightness, channel_a, channel_b = cv2.split(lab)
            lightness = self._clahe.apply(lightness)
            image = cv2.cvtColor(
                cv2.merge(
                    (lightness, channel_a, channel_b)),
                cv2.COLOR_LAB2BGR,
            )

        letterboxed, scale, pad_x, pad_y = _letterbox(
            image,
            self.input_width,
            self.input_height,
        )
        if self.input_channels == 1:
            if letterboxed.ndim == 3:
                letterboxed = cv2.cvtColor(
                    letterboxed, cv2.COLOR_BGR2GRAY)
            tensor = letterboxed[None, :, :]
        else:
            rgb = cv2.cvtColor(
                letterboxed, cv2.COLOR_BGR2RGB)
            tensor = np.transpose(rgb, (2, 0, 1))

        tensor = np.ascontiguousarray(
            tensor,
            dtype=self.input_dtype,
        )
        tensor *= np.asarray(
            1.0 / 255.0,
            dtype=self.input_dtype,
        )
        return tensor, (scale, pad_x, pad_y)

    def _run_session(
        self,
        tensor: np.ndarray,
    ) -> List[np.ndarray]:
        """Run ORT with explicit CUDA output binding when supported."""
        with self._session_lock:
            try:
                io_binding = self.session.io_binding()
                io_binding.bind_cpu_input(
                    self.input_name, tensor)
                for output_name in self.output_names:
                    io_binding.bind_output(
                        output_name,
                        'cuda'
                        if 'CUDAExecutionProvider'
                        in self.active_providers
                        else 'cpu',
                        self.device
                        if 'CUDAExecutionProvider'
                        in self.active_providers
                        else 0,
                    )
                self.session.run_with_iobinding(
                    io_binding)
                return io_binding.copy_outputs_to_cpu()
            except Exception:
                return self.session.run(
                    self.output_names,
                    {self.input_name: tensor},
                )

    @staticmethod
    def _select_detection_output(
        outputs: Sequence[np.ndarray],
    ) -> np.ndarray:
        candidates = [
            np.asarray(output)
            for output in outputs
            if np.asarray(output).ndim >= 2
        ]
        if not candidates:
            raise RuntimeError(
                'ONNX model returned no detection tensor')
        return max(
            candidates,
            key=lambda array: array.size,
        )

    def _decode_one(
        self,
        raw: np.ndarray,
        transform: Tuple[float, float, float],
        original_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = np.asarray(raw)
        while rows.ndim > 2 and rows.shape[0] == 1:
            rows = rows[0]
        if rows.ndim != 2:
            rows = rows.reshape(
                rows.shape[-2], rows.shape[-1])

        # Standard Ultralytics ONNX output is [channels, anchors].
        if (
            rows.shape[0] <= 256
            and rows.shape[1] > rows.shape[0]
        ):
            rows = rows.T

        if rows.shape[1] < 5:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        # End-to-end NMS export: [x1, y1, x2, y2, score, class].
        looks_end_to_end = (
            rows.shape[1] in (6, 7)
            and np.nanmax(rows[:, 4]) <= 1.01
            and np.nanmin(rows[:, 4]) >= -0.01
            and np.mean(
                rows[:, 2] >= rows[:, 0]
            ) > 0.85
            and np.mean(
                rows[:, 3] >= rows[:, 1]
            ) > 0.85
        )
        if looks_end_to_end:
            if rows.shape[1] == 7:
                rows = rows[:, -6:]
            boxes = rows[:, :4].astype(
                np.float32, copy=False)
            scores = rows[:, 4].astype(
                np.float32, copy=False)
            class_ids = rows[:, 5].astype(
                np.float32, copy=False)
        else:
            boxes_xywh = rows[:, :4].astype(
                np.float32, copy=False)
            channel_count = rows.shape[1]

            has_objectness = (
                self.class_count > 0
                and channel_count
                == 5 + self.class_count
            )
            if has_objectness:
                objectness = rows[:, 4]
                class_scores = rows[:, 5:]
                class_ids = np.argmax(
                    class_scores, axis=1)
                scores = (
                    objectness
                    * class_scores[
                        np.arange(len(class_scores)),
                        class_ids,
                    ]
                )
            else:
                class_scores = rows[:, 4:]
                class_ids = np.argmax(
                    class_scores, axis=1)
                scores = class_scores[
                    np.arange(len(class_scores)),
                    class_ids,
                ]

            if np.nanmax(boxes_xywh) <= 2.0:
                boxes_xywh[:, [0, 2]] *= float(
                    self.input_width)
                boxes_xywh[:, [1, 3]] *= float(
                    self.input_height)

            center_x = boxes_xywh[:, 0]
            center_y = boxes_xywh[:, 1]
            width = boxes_xywh[:, 2]
            height = boxes_xywh[:, 3]
            boxes = np.column_stack(
                (
                    center_x - 0.5 * width,
                    center_y - 0.5 * height,
                    center_x + 0.5 * width,
                    center_y + 0.5 * height,
                )
            ).astype(np.float32, copy=False)
            class_ids = class_ids.astype(
                np.float32, copy=False)
            scores = scores.astype(
                np.float32, copy=False)

        finite = (
            np.isfinite(boxes).all(axis=1)
            & np.isfinite(scores)
            & (scores >= self.low_confidence_threshold)
        )
        boxes = boxes[finite]
        scores = scores[finite]
        class_ids = class_ids[finite]
        if not len(boxes):
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        scale, pad_x, pad_y = transform
        boxes[:, [0, 2]] = (
            boxes[:, [0, 2]] - float(pad_x)
        ) / max(float(scale), 1.0e-9)
        boxes[:, [1, 3]] = (
            boxes[:, [1, 3]] - float(pad_y)
        ) / max(float(scale), 1.0e-9)

        original_h, original_w = original_shape
        boxes[:, [0, 2]] = np.clip(
            boxes[:, [0, 2]],
            0.0,
            max(0.0, float(original_w - 1)),
        )
        boxes[:, [1, 3]] = np.clip(
            boxes[:, [1, 3]],
            0.0,
            max(0.0, float(original_h - 1)),
        )

        valid_size = (
            (boxes[:, 2] - boxes[:, 0] > 2.0)
            & (boxes[:, 3] - boxes[:, 1] > 2.0)
        )
        boxes = boxes[valid_size]
        scores = scores[valid_size]
        class_ids = class_ids[valid_size]
        if not len(boxes):
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        keep = _classwise_nms(
            boxes,
            scores,
            class_ids,
            self.nms_threshold,
            self.maximum_detections,
        )
        return (
            boxes[keep],
            scores[keep],
            class_ids[keep],
        )

    def _detect_batch(
        self,
        source_frames: Mapping[str, np.ndarray],
    ) -> Tuple[
        Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        dict,
    ]:
        source_ids = list(source_frames.keys())
        tensors = []
        transforms = []
        original_shapes = []
        preprocess_start = time.perf_counter()
        for source_id in source_ids:
            frame = source_frames[source_id]
            tensor, transform = self._prepare_image(
                frame)
            tensors.append(tensor)
            transforms.append(transform)
            original_shapes.append(frame.shape[:2])
        preprocess_ms = (
            time.perf_counter() - preprocess_start
        ) * 1000.0

        inference_start = time.perf_counter()
        batch_used = False
        outputs_per_source: List[np.ndarray] = []

        can_batch = (
            len(tensors) > 1
            and not self._batch_probe_failed
            and (
                self.dynamic_batch
                or (
                    self.static_batch_size is not None
                    and self.static_batch_size
                    == len(tensors)
                )
            )
        )

        if can_batch:
            try:
                batch_tensor = np.stack(
                    tensors, axis=0)
                outputs = self._run_session(
                    batch_tensor)
                detection_output = (
                    self._select_detection_output(
                        outputs)
                )
                if (
                    detection_output.ndim >= 3
                    and detection_output.shape[0]
                    == len(tensors)
                ):
                    outputs_per_source = [
                        detection_output[index]
                        for index in range(len(tensors))
                    ]
                    batch_used = True
                else:
                    raise RuntimeError(
                        'model did not return per-batch outputs')
            except Exception:
                self._batch_probe_failed = True
                outputs_per_source = []

        if not outputs_per_source:
            for tensor in tensors:
                outputs = self._run_session(
                    tensor[None, ...])
                detection_output = (
                    self._select_detection_output(
                        outputs)
                )
                if (
                    detection_output.ndim >= 3
                    and detection_output.shape[0] == 1
                ):
                    detection_output = (
                        detection_output[0])
                outputs_per_source.append(
                    detection_output)

        inference_ms = (
            time.perf_counter() - inference_start
        ) * 1000.0

        post_start = time.perf_counter()
        decoded = {}
        for index, source_id in enumerate(source_ids):
            decoded[source_id] = self._decode_one(
                outputs_per_source[index],
                transforms[index],
                original_shapes[index],
            )
        postprocess_ms = (
            time.perf_counter() - post_start
        ) * 1000.0

        return decoded, {
            'batch_size': len(source_ids),
            'batch_used': bool(batch_used),
            'preprocess_ms': preprocess_ms,
            'inference_ms': inference_ms,
            'postprocess_ms': postprocess_ms,
        }

    @staticmethod
    def _appearance_descriptor(
        frame_bgr: np.ndarray,
        bbox_xyxy: Tuple[float, float, float, float],
    ):
        """Return a fast L2-normalized appearance vector for one YOLO box.

        The online association follows DeepSORT's nearest-neighbor gallery
        principle, but avoids a second ReID network in the Jetson hot path.
        The descriptor combines:
        - HSV color histogram;
        - low-frequency grayscale DCT;
        - gradient-orientation histogram;
        - coarse spatial grayscale pooling.

        It is intentionally cheap for 324x244 AI-deck frames.
        """
        x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
        image_h, image_w = frame_bgr.shape[:2]

        # Remove a small border so changing background occupies less of the
        # descriptor when the bounding box jitters.
        inset_x = 0.08 * max(1.0, x2 - x1)
        inset_y = 0.08 * max(1.0, y2 - y1)
        left = max(0, min(image_w - 1, int(math.floor(x1 + inset_x))))
        top = max(0, min(image_h - 1, int(math.floor(y1 + inset_y))))
        right = max(left + 1, min(image_w, int(math.ceil(x2 - inset_x))))
        bottom = max(top + 1, min(image_h, int(math.ceil(y2 - inset_y))))
        crop = frame_bgr[top:bottom, left:right]
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return None

        resized = cv2.resize(
            crop,
            (48, 48),
            interpolation=(
                cv2.INTER_AREA
                if crop.shape[0] > 48 or crop.shape[1] > 48
                else cv2.INTER_LINEAR
            ),
        )

        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        color = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [8, 4, 4],
            [0, 180, 0, 256, 0, 256],
        ).reshape(-1).astype(np.float32)
        color_norm = float(np.linalg.norm(color))
        if color_norm > 1.0e-8:
            color /= color_norm

        gray_u8 = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = gray_u8.astype(np.float32) / 255.0

        dct = cv2.dct(gray)
        texture = dct[:8, :8].reshape(-1)[1:].astype(np.float32)
        texture_norm = float(np.linalg.norm(texture))
        if texture_norm > 1.0e-8:
            texture /= texture_norm

        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(
            grad_x,
            grad_y,
            angleInDegrees=False,
        )
        bins = np.floor(
            angle * (9.0 / (2.0 * math.pi))
        ).astype(np.int32) % 9
        gradient_hist = np.bincount(
            bins.reshape(-1),
            weights=magnitude.reshape(-1),
            minlength=9,
        ).astype(np.float32)
        gradient_norm = float(np.linalg.norm(gradient_hist))
        if gradient_norm > 1.0e-8:
            gradient_hist /= gradient_norm

        pooled = cv2.resize(
            gray,
            (4, 4),
            interpolation=cv2.INTER_AREA,
        ).reshape(-1).astype(np.float32)
        pooled -= float(np.mean(pooled))
        pooled_norm = float(np.linalg.norm(pooled))
        if pooled_norm > 1.0e-8:
            pooled /= pooled_norm

        # Rotation-insensitive local context around the target helps tell two
        # identical targets apart while preserving the same target across a
        # lawnmower revisit.  It is only small OpenCV histogram work, not a
        # second neural network.
        box_width = max(1.0, x2 - x1)
        box_height = max(1.0, y2 - y1)
        context_left = max(0, int(math.floor(x1 - 0.55 * box_width)))
        context_top = max(0, int(math.floor(y1 - 0.55 * box_height)))
        context_right = min(image_w, int(math.ceil(x2 + 0.55 * box_width)))
        context_bottom = min(image_h, int(math.ceil(y2 + 0.55 * box_height)))
        context = frame_bgr[
            context_top:context_bottom,
            context_left:context_right,
        ]
        context_features = np.zeros(25, dtype=np.float32)
        if context.size and context.shape[0] >= 8 and context.shape[1] >= 8:
            context_gray = cv2.cvtColor(context, cv2.COLOR_BGR2GRAY)
            context_gray = cv2.resize(
                context_gray,
                (48, 48),
                interpolation=cv2.INTER_AREA,
            )
            gray_hist = cv2.calcHist(
                [context_gray], [0], None, [16], [0, 256]
            ).reshape(-1).astype(np.float32)
            gray_hist /= max(float(np.linalg.norm(gray_hist)), 1.0e-8)
            cg = context_gray.astype(np.float32) / 255.0
            cgx = cv2.Sobel(cg, cv2.CV_32F, 1, 0, ksize=3)
            cgy = cv2.Sobel(cg, cv2.CV_32F, 0, 1, ksize=3)
            cmag, cangle = cv2.cartToPolar(cgx, cgy, angleInDegrees=False)
            cbins = np.floor(
                cangle * (9.0 / (2.0 * math.pi))
            ).astype(np.int32) % 9
            context_gradient = np.bincount(
                cbins.reshape(-1),
                weights=cmag.reshape(-1),
                minlength=9,
            ).astype(np.float32)
            context_gradient /= max(
                float(np.linalg.norm(context_gradient)),
                1.0e-8,
            )
            context_features = np.concatenate(
                (gray_hist, context_gradient),
                axis=0,
            ).astype(np.float32)

        descriptor = np.concatenate(
            (
                color,
                texture,
                gradient_hist,
                pooled,
                0.45 * context_features,
            ),
            axis=0,
        ).astype(np.float32)
        norm = float(np.linalg.norm(descriptor))
        if norm <= 1.0e-8:
            return None
        descriptor /= norm
        return descriptor

    def _update_appearance(
        self,
        source_id: str,
        local_track_id: int,
        frame_bgr: np.ndarray,
        bbox_xyxy: Tuple[float, float, float, float],
        timestamp: float,
    ):
        descriptor = self._appearance_descriptor(
            frame_bgr,
            bbox_xyxy,
        )
        state = self._appearance_state.setdefault(
            str(source_id), {})
        previous = state.get(int(local_track_id))
        if descriptor is None:
            return (
                None
                if previous is None
                else previous[0].copy()
            )

        if previous is not None:
            # Stable track-level signature, analogous to maintaining a feature
            # gallery/EMA while the local track is alive.
            descriptor = (
                0.78 * previous[0]
                + 0.22 * descriptor
            ).astype(np.float32)
            norm = float(np.linalg.norm(descriptor))
            if norm > 1.0e-8:
                descriptor /= norm

        state[int(local_track_id)] = (
            descriptor,
            float(timestamp),
        )

        # Bounded memory for long missions.
        if len(state) > 256:
            stale = sorted(
                state.items(),
                key=lambda item: item[1][1],
            )[:-192]
            for key, _value in stale:
                state.pop(key, None)

        return descriptor.copy()

    def track_batch(
        self,
        source_frames: Mapping[str, np.ndarray],
    ) -> Tuple[Dict[str, List[dict]], dict]:
        """Detect all newest camera frames, then update independent trackers."""
        decoded, metrics = self._detect_batch(
            source_frames)

        output: Dict[str, List[dict]] = {}
        overlay_time = time.monotonic()

        for source_id, frame in source_frames.items():
            if source_id not in self._trackers:
                raise KeyError(
                    f'no ByteTrack state configured for '
                    f'{source_id}')

            boxes, scores, class_ids = decoded[
                source_id]
            tracker_input = _TrackerDetections(
                boxes,
                scores,
                class_ids,
            )
            tracks = self._trackers[
                source_id].update(
                    tracker_input,
                    img=None,
                )

            detections: List[dict] = []
            for track in tracks:
                if len(track) < 7:
                    continue

                x1 = float(track[0])
                y1 = float(track[1])
                x2 = float(track[2])
                y2 = float(track[3])
                local_track_id = int(track[4])
                confidence = float(track[5])
                class_id = int(track[6])

                width = max(0.0, x2 - x1)
                height = max(0.0, y2 - y1)
                if width <= 2.0 or height <= 2.0:
                    continue

                center_u = 0.5 * (x1 + x2)
                center_v = 0.5 * (y1 + y2)
                appearance = self._update_appearance(
                    source_id,
                    local_track_id,
                    frame,
                    (x1, y1, x2, y2),
                    overlay_time,
                )
                detections.append({
                    'track_id': local_track_id,
                    'class_id': class_id,
                    'class_name': self.class_names.get(
                        class_id,
                        str(class_id),
                    ),
                    'confidence': confidence,
                    'bbox': (
                        x1,
                        y1,
                        width,
                        height,
                    ),
                    'ground_px': (
                        center_u,
                        center_v,
                    ),
                    'image_center': (
                        center_u,
                        center_v,
                    ),
                    'appearance': appearance,
                })

            output[source_id] = detections
            with self._overlay_lock:
                self._latest_overlay[source_id] = (
                    overlay_time,
                    [
                        dict(detection)
                        for detection in detections
                    ],
                )

        return output, metrics

    def track_raw(
        self,
        source_id,
        frame_bgr,
    ) -> List[dict]:
        """Compatibility wrapper for single-frame callers."""
        output, _metrics = self.track_batch({
            str(source_id): frame_bgr,
        })
        return output[str(source_id)]

    def draw_latest_overlay(
        self,
        source_id,
        frame_bgr,
        maximum_age_sec=1.2,
    ) -> int:
        """Draw the newest tracker snapshot on a live frame in-place."""
        source_id = str(source_id)
        with self._overlay_lock:
            timestamp, detections = (
                self._latest_overlay.get(
                    source_id,
                    (0.0, []),
                )
            )
            detections = [
                dict(detection)
                for detection in detections
            ]

        age = time.monotonic() - float(timestamp)
        if age > float(maximum_age_sec):
            return 0

        for detection in detections:
            x, y, width, height = [
                int(round(value))
                for value in detection['bbox']
            ]
            cv2.rectangle(
                frame_bgr,
                (x, y),
                (x + width, y + height),
                (0, 200, 255),
                2,
            )
            label = (
                f"{detection['class_name']} "
                f"{detection['confidence']:.2f} "
                f"T{detection['track_id']}"
            )
            cv2.putText(
                frame_bgr,
                label,
                (x, max(12, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 200, 255),
                1,
                cv2.LINE_AA,
            )
        return len(detections)