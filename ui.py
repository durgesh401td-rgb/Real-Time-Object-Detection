"""
Streamlit Web Dashboard for Real-Time Object Detection & Logging Platform.
Displays real-time video feed with YOLO annotations, live detection metrics,
and dynamically updated MySQL event logs.
"""

import time
import os
from datetime import datetime
import cv2
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

from detector import YOLODetector, COCO_CLASSES
from database import DatabaseManager

# Page Configuration
st.set_page_config(
    page_title="Vision Platform | Real-Time Object Detection & Logging",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 14px;
        border-left: 4px solid #1E88E5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .status-badge-ok {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 6px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.88rem;
        display: inline-block;
    }
    .status-badge-err {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 6px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.88rem;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_detector():
    """Caches the YOLO detector so it loads only once into memory."""
    return YOLODetector("yolov8n.pt")


@st.cache_resource
def get_db_manager():
    """Initializes and caches the DatabaseManager instance."""
    return DatabaseManager()


def main():
    detector = load_detector()
    db = get_db_manager()

    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.title("⚙️ Control Panel")
    st.sidebar.markdown("Configure detection parameters & database settings.")

    # 1. Database Connection Status
    st.sidebar.subheader("🗄️ MySQL Database")
    db_ok, db_msg = db.is_connected()
    if db_ok:
        st.sidebar.markdown(f'<span class="status-badge-ok">🟢 {db.database} Connected</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<span class="status-badge-err">🔴 MySQL Disconnected</span>', unsafe_allow_html=True)
        st.sidebar.caption(f"Error: {db_msg}")
        if st.sidebar.button("🔄 Retry Connection"):
            db.init_database()
            st.rerun()

    enable_logging = st.sidebar.toggle("Enable Database Event Logging", value=True, help="Logs detected objects directly to MySQL")
    cooldown = st.sidebar.slider(
        "Logging Cooldown (seconds)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="Prevents repeated identical logs within this time window",
    )
    db.cooldown_seconds = cooldown

    st.sidebar.divider()

    # 2. Object Filtering & Confidence
    st.sidebar.subheader("🎯 Object Configuration")
    default_classes = ["person", "cell phone", "bottle", "laptop", "chair", "cup"]
    selected_classes = st.sidebar.multiselect(
        "Target Object Classes to Detect:",
        options=COCO_CLASSES,
        default=default_classes,
        help="Select which objects YOLO will detect and display",
    )

    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold:",
        min_value=0.20,
        max_value=1.00,
        value=0.70,
        step=0.05,
        help="Only objects with confidence score greater than this threshold will be processed and logged (Default: 0.70 / 70%)",
    )

    st.sidebar.divider()

    # 3. Video Stream Source
    st.sidebar.subheader("📹 Video Feed Settings")
    input_source = st.sidebar.radio("Select Input Source:", ["Webcam", "Sample Video File", "Upload Video"], index=0)

    webcam_index = 0
    uploaded_file = None
    if input_source == "Webcam":
        webcam_index = st.sidebar.number_input("Webcam Device Index", min_value=0, max_value=5, value=0, step=1)
    elif input_source == "Upload Video":
        uploaded_file = st.sidebar.file_uploader("Upload MP4/AVI Video", type=["mp4", "avi", "mov"])

    # --- MAIN VIEW HEADER ---
    st.markdown('<div class="main-header">🚀 Real-Time Object Detection & Logging Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Assignment 5 • YOLOv8 • OpenCV • MySQL Database • Streamlit</div>', unsafe_allow_html=True)

    if not db_ok:
        st.warning(
            "⚠️ **MySQL Server is not running or unreachable.** "
            "Please make sure your local MySQL server (XAMPP / WAMP / MySQL Workbench) is started on port 3306. "
            "Detection will continue in preview mode, and logs will resume once MySQL is connected."
        )

    # --- TOP METRICS ROW ---
    stats = db.get_statistics()
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric(label="Total Database Logs", value=stats["total_detections"])
    with m_col2:
        st.metric(label="Filter Confidence", value=f"{int(confidence_threshold * 100)}%")
    with m_col3:
        st.metric(label="Active Target Classes", value=len(selected_classes))
    with m_col4:
        db_badge = "Connected" if db_ok else "Offline"
        st.metric(label="MySQL Server", value=db_badge)

    st.divider()

    # --- MAIN WORKSPACE TABS ---
    tab_feed, tab_logs, tab_about = st.tabs(["🎥 Live Detection Feed", "📊 MySQL Event Logs", "📘 System Documentation"])

    with tab_feed:
        feed_col, info_col = st.columns([3, 1])

        with feed_col:
            st.subheader("Live Video Stream")
            run_feed = st.checkbox("▶️ Start Video Stream", value=False)
            video_placeholder = st.empty()

        with info_col:
            st.subheader("Real-Time Telemetry")
            fps_placeholder = st.empty()
            detections_placeholder = st.empty()
            last_logged_placeholder = st.empty()

        if run_feed:
            # Determine capture source
            video_path = None
            if input_source == "Webcam":
                video_path = int(webcam_index)
            elif input_source == "Upload Video" and uploaded_file is not None:
                # Save uploaded video to temp file
                temp_video_path = "temp_uploaded_video.mp4"
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_file.read())
                video_path = temp_video_path
            else:
                # Default webcam
                video_path = 0

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                st.error(f"Could not open video source: {video_path}. Please check your camera connection or file format.")
            else:
                prev_time = time.time()
                frame_count = 0
                fps = 0.0

                try:
                    while run_feed:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Video stream ended or frame could not be read.")
                            break

                        # FPS calculation
                        curr_time = time.time()
                        frame_count += 1
                        time_diff = curr_time - prev_time
                        if time_diff >= 1.0:
                            fps = frame_count / time_diff
                            frame_count = 0
                            prev_time = curr_time

                        # Run YOLO Detection
                        annotated_frame, detections = detector.detect(
                            frame=frame,
                            confidence_threshold=confidence_threshold,
                            allowed_classes=selected_classes if len(selected_classes) > 0 else None,
                            draw_annotations=True,
                        )

                        # Event Logging to MySQL
                        logged_in_this_frame = []
                        if enable_logging and db_ok:
                            for det in detections:
                                success = db.log_detection(
                                    object_class=det["object_class"],
                                    confidence=det["confidence"],
                                    bbox=det["bbox"],
                                    enforce_cooldown=True,
                                )
                                if success:
                                    logged_in_this_frame.append(det["object_class"])

                        # Convert BGR frame to RGB for Streamlit rendering
                        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                        # Telemetry Update
                        fps_placeholder.metric("Inference FPS", f"{fps:.1f}")
                        
                        det_summary = f"**Current Objects ({len(detections)}):**\n"
                        if detections:
                            for d in detections:
                                det_summary += f"- **{d['object_class']}**: {d['confidence']*100:.1f}%\n"
                        else:
                            det_summary += "_No target objects detected._"
                        detections_placeholder.markdown(det_summary)

                        if logged_in_this_frame:
                            last_logged_placeholder.success(f"💾 Logged to MySQL: {', '.join(logged_in_this_frame)}")

                        # Small yield sleep to keep UI responsive
                        time.sleep(0.01)

                finally:
                    cap.release()
                    video_placeholder.empty()

    with tab_logs:
        st.subheader("MySQL Live Detection Logs")
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])

        with col_ctrl1:
            log_limit = st.selectbox("Show Recent Rows:", options=[10, 25, 50, 100, 200], index=1)
        with col_ctrl2:
            refresh_logs = st.button("🔄 Refresh Table")
        with col_ctrl3:
            clear_btn = st.button("🗑️ Clear Logs", type="secondary")

        if clear_btn:
            if db.clear_logs():
                st.success("All logs cleared successfully.")
                st.rerun()
            else:
                st.error("Failed to clear logs.")

        logs = db.get_recent_logs(limit=log_limit)

        if logs:
            df = pd.DataFrame(logs)
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "log_id": st.column_config.NumberColumn("ID", width="small"),
                    "timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm:ss"),
                    "object_class": st.column_config.TextColumn("Object Class"),
                    "confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.2f"),
                    "bbox_x": st.column_config.NumberColumn("X"),
                    "bbox_y": st.column_config.NumberColumn("Y"),
                    "bbox_w": st.column_config.NumberColumn("W"),
                    "bbox_h": st.column_config.NumberColumn("H"),
                },
                hide_index=True,
            )

            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Logs as CSV",
                data=csv_data,
                file_name=f"vision_platform_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No detection logs found in MySQL database yet. Start the stream with target objects in view to record events.")

    with tab_about:
        st.subheader("📘 About Platform & Architecture")
        st.markdown(
            """
            ### Real-Time Object Detection & Logging Platform
            - **Computer Vision Model:** YOLOv8 Nano (`yolov8n.pt`) pre-trained on 80 COCO dataset categories.
            - **Frame Pipeline:** OpenCV captures raw video frames at native camera framerates.
            - **Event Logging:** Every detected object exceeding the confidence threshold is inserted dynamically into MySQL database `vision_platform`.
            - **Database Schema:**
              - `log_id`: Auto-increment primary key
              - `timestamp`: Detection date and time
              - `object_class`: Category name (e.g., person, cell phone, laptop)
              - `confidence`: Normalized model prediction score
              - `bbox_x, bbox_y, bbox_w, bbox_h`: Exact bounding box coordinate metrics
            - **Security:** MySQL credentials loaded exclusively via `.env` with `.gitignore` protection.
            """
        )


if __name__ == "__main__":
    main()
