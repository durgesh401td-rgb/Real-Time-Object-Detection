"""
YOLO Object Detection Module.
Handles model loading, inference, filtering by class & confidence,
extracting bounding boxes, and drawing annotations.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import cv2
import numpy as np
from ultralytics import YOLO


# Standard COCO 80 Class Names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


class YOLODetector:
    """Encapsulates Ultralytics YOLO object detection and visualization."""

    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Initializes the YOLO detector.
        
        Args:
            model_path: Path or name of YOLO weights (default 'yolov8n.pt').
        """
        self.model_path = model_path
        self.model = YOLO(model_path)
        # Model class dictionary mapping ID -> class name
        self.classes_dict = self.model.names if hasattr(self.model, "names") else {i: name for i, name in enumerate(COCO_CLASSES)}

    def get_available_classes(self) -> List[str]:
        """Returns the list of all object class names supported by the model."""
        return list(self.classes_dict.values())

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.70,
        allowed_classes: Optional[List[str]] = None,
        draw_annotations: bool = True,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Performs object detection on a single frame.
        
        Args:
            frame: BGR image frame from OpenCV.
            confidence_threshold: Minimum confidence score to accept (0.0 to 1.0).
            allowed_classes: List of class names to detect (e.g. ['person', 'cell phone']).
                             If None or empty, all classes are accepted.
            draw_annotations: If True, draws bounding boxes and labels onto the frame.
            
        Returns:
            Tuple of (annotated_frame, list_of_detections)
            Each detection dict contains:
                - object_class: str
                - confidence: float
                - bbox: (bbox_x, bbox_y, bbox_w, bbox_h)
                - xyxy: (x1, y1, x2, y2)
        """
        annotated_frame = frame.copy() if draw_annotations else frame
        detections: List[Dict[str, Any]] = []

        allowed_set: Optional[Set[str]] = set(allowed_classes) if allowed_classes else None

        # Run inference (verbose=False keeps console clean)
        results = self.model(frame, conf=confidence_threshold, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                conf = float(box.conf[0].cpu().numpy()) if hasattr(box.conf[0], "cpu") else float(box.conf[0])
                cls_id = int(box.cls[0].cpu().numpy()) if hasattr(box.cls[0], "cpu") else int(box.cls[0])
                class_name = self.classes_dict.get(cls_id, str(cls_id))

                # Filter by confidence
                if conf < confidence_threshold:
                    continue

                # Filter by class configuration
                if allowed_set is not None and class_name not in allowed_set:
                    continue

                # Coordinates in xyxy format
                xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], "cpu") else box.xyxy[0]
                x1, y1, x2, y2 = map(int, xyxy)

                # Convert to (bbox_x, bbox_y, bbox_w, bbox_h) required by assignment schema
                bbox_x = max(0, x1)
                bbox_y = max(0, y1)
                bbox_w = max(1, x2 - x1)
                bbox_h = max(1, y2 - y1)

                detection_info = {
                    "object_class": class_name,
                    "confidence": round(conf, 4),
                    "bbox": (bbox_x, bbox_y, bbox_w, bbox_h),
                    "bbox_x": bbox_x,
                    "bbox_y": bbox_y,
                    "bbox_w": bbox_w,
                    "bbox_h": bbox_h,
                    "xyxy": (x1, y1, x2, y2),
                }
                detections.append(detection_info)

                # Draw bounding box and label if requested
                if draw_annotations:
                    self._draw_box(annotated_frame, class_name, conf, x1, y1, x2, y2)

        return annotated_frame, detections

    def _draw_box(
        self,
        image: np.ndarray,
        class_name: str,
        confidence: float,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> None:
        """Draws aesthetic bounding box and badge with class and confidence."""
        # Color palette by class name hash for distinct colors per object
        color_seed = abs(hash(class_name)) % 0xFFFFFF
        r = (color_seed & 0xFF0000) >> 16
        g = (color_seed & 0x00FF00) >> 8
        b = color_seed & 0x0000FF
        color = (b, g, r)

        # Draw main rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Label text
        label = f"{class_name.upper()} {confidence * 100:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Background badge above or inside box
        badge_y1 = max(0, y1 - text_h - baseline - 6)
        badge_y2 = y1
        badge_x2 = min(image.shape[1], x1 + text_w + 10)

        cv2.rectangle(image, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
        # White text label
        cv2.putText(
            image,
            label,
            (x1 + 5, badge_y2 - baseline - 2),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


if __name__ == "__main__":
    print("Initializing YOLODetector...")
    detector = YOLODetector()
    print("Available classes count:", len(detector.get_available_classes()))
    
    # Test on a blank dummy image
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated, dets = detector.detect(dummy_img, confidence_threshold=0.50)
    print("Detections on blank image:", len(dets))
    print("YOLODetector ready!")
