"""
Main Entry Point for Real-Time Object Detection & Logging Platform.
Supports dual modes:
1. Streamlit Dashboard Mode (Default UI): python main.py --mode ui
2. Direct OpenCV Desktop Mode: python main.py --mode opencv
"""

import sys
import os
import time
import argparse
import subprocess
import cv2

from detector import YOLODetector, COCO_CLASSES
from database import DatabaseManager


def run_opencv_mode(
    video_source: int = 0,
    confidence_threshold: float = 0.70,
    target_classes: list = None,
    enable_logging: bool = True,
    cooldown: float = 2.0,
):
    """Runs direct OpenCV desktop window with YOLO detection and MySQL logging."""
    print("=" * 60)
    print("🚀 Starting Real-Time Object Detection Platform (OpenCV Mode)")
    print("=" * 60)

    # Initialize components
    print("[1/3] Initializing Database Manager...")
    db = DatabaseManager(cooldown_seconds=cooldown)
    db_ok, db_msg = db.is_connected()
    if db_ok:
        print(f"✅ MySQL connected: {db_msg}")
    else:
        print(f"⚠️ MySQL warning: {db_msg}")
        print("    (Detection will continue, but logs will not be saved until MySQL is running.)")

    print("[2/3] Loading YOLO Model (yolov8n.pt)...")
    detector = YOLODetector("yolov8n.pt")
    print("✅ Model loaded successfully.")

    print(f"[3/3] Opening Video Source: {video_source}...")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source {video_source}.")
        return

    print("=" * 60)
    print("🎥 Live Feed Running! Controls:")
    print("   Press 'q' to quit.")
    print("   Press 's' to save a screenshot.")
    print(f"   Filtering: Confidence >= {confidence_threshold * 100:.0f}%")
    if target_classes:
        print(f"   Target Classes: {', '.join(target_classes)}")
    else:
        print("   Target Classes: All 80 COCO classes")
    print("=" * 60)

    prev_time = time.time()
    frame_count = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended or frame capture failed.")
            break

        # Calculate FPS
        curr_time = time.time()
        frame_count += 1
        elapsed = curr_time - prev_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            prev_time = curr_time

        # Run inference and get annotations
        annotated_frame, detections = detector.detect(
            frame=frame,
            confidence_threshold=confidence_threshold,
            allowed_classes=target_classes,
            draw_annotations=True,
        )

        # Log detections to MySQL
        if enable_logging and db_ok:
            for det in detections:
                logged = db.log_detection(
                    object_class=det["object_class"],
                    confidence=det["confidence"],
                    bbox=det["bbox"],
                    enforce_cooldown=True,
                )
                if logged:
                    print(
                        f"💾 Logged: {det['object_class']} | Conf: {det['confidence']*100:.1f}% "
                        f"| BBox: {det['bbox']}"
                    )

        # Draw Telemetry HUD on Frame
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f} | Detections: {len(detections)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        db_status_text = "DB: Connected" if db_ok else "DB: Offline"
        db_color = (0, 255, 0) if db_ok else (0, 0, 255)
        cv2.putText(
            annotated_frame,
            db_status_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            db_color,
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Vision Platform - Real-Time Object Detection", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Exiting...")
            break
        elif key == ord("s"):
            filename = f"detection_screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"📸 Screenshot saved to {filename}")

    cap.release()
    cv2.destroyAllWindows()


def run_ui_mode():
    """Launches the Streamlit Web Dashboard."""
    print("🚀 Launching Streamlit Web Dashboard...")
    ui_script = os.path.join(os.path.dirname(__file__), "ui.py")
    cmd = [sys.executable, "-m", "streamlit", "run", ui_script]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nUI closed by user.")


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Object Detection & Logging Platform (Assignment 5)"
    )
    parser.add_argument(
        "--mode",
        choices=["ui", "opencv"],
        default="ui",
        help="Run mode: 'ui' for Streamlit dashboard (default) or 'opencv' for desktop window",
    )
    parser.add_argument(
        "--source",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.70,
        help="Confidence threshold between 0.0 and 1.0 (default: 0.70)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["person", "cell phone", "laptop", "bottle"],
        help="Target object classes to detect (default: person cell phone laptop bottle)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=2.0,
        help="Logging cooldown in seconds per class (default: 2.0)",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable database logging",
    )

    args = parser.parse_args()

    if args.mode == "ui":
        run_ui_mode()
    else:
        run_opencv_mode(
            video_source=args.source,
            confidence_threshold=args.conf,
            target_classes=args.classes,
            enable_logging=not args.no_log,
            cooldown=args.cooldown,
        )


# Serverless WSGI / Vercel compatibility handler
def handler(request=None, *args, **kwargs):
    return {"statusCode": 200, "headers": {"Content-Type": "text/plain"}, "body": "Vision Platform Running"}

app = handler
application = handler


if __name__ == "__main__":
    main()
