# Building a Real-Time Object Detection & Event Logging Platform with YOLOv8, OpenCV, MySQL, and Streamlit

*A comprehensive guide to integrating state-of-the-art computer vision models with relational databases and reactive web dashboards.*

---

## 1. Introduction

Computer vision has evolved from an academic curiosity into the backbone of modern automation, security surveillance, and industrial robotics. However, deploying a vision model into the real world requires far more than simply running inference on static images—it demands a full-stack engineering workflow capable of capturing high-throughput video streams, applying intelligent filtering, logging events to a persistent database, and presenting actionable metrics through an intuitive user interface.

In this project (**Assignment 5: Real-Time Object Detection & Logging Platform**), I built an end-to-end, modular application combining **Ultralytics YOLOv8**, **OpenCV**, **MySQL**, and **Streamlit**. The platform detects objects in real time through a live webcam or video feed, draws aesthetic bounding boxes and confidence badges, and dynamically records detection events directly into a MySQL relational database.

---

## 2. Computer Vision & YOLO Learning

### Mastering OpenCV Fundamentals
The first phase of this journey began with OpenCV (`cv2`). Handling video in real time requires understanding how video is structured as a sequential continuum of image matrices (frames). Key concepts learned and applied:
- **Stream Capture:** Initializing high-speed frame capture with `cv2.VideoCapture()`.
- **Frame Color Space Conversions:** Translating frames from OpenCV's native BGR format to RGB for web browser rendering.
- **Dynamic Overlays:** Calculating label dimensions with `cv2.getTextSize()` and rendering clean background badges and bounding box geometries without obstructing video clarity.

### Ultralytics YOLOv8 Deep Dive
YOLO (You Only Look Once) is renowned for treating object detection as a single regression problem, passing the entire image through a deep convolutional network to predict bounding boxes and class probabilities simultaneously. 
- **Model Selection:** I utilized `yolov8n.pt` (Nano), the lightweight variant in the YOLOv8 family, providing an ideal balance between ultra-low inference latency and high mean Average Precision (mAP) across 80 COCO categories.
- **Bounding Box Coordinate Transformation:** YOLO provides bounding boxes in normalized and `(x1, y1, x2, y2)` formats. I engineered a coordinate conversion function to transform these values into `(bbox_x, bbox_y, bbox_w, bbox_h)`, standardizing the telemetry for relational database storage.

---

## 3. Application Development & Modular Architecture

To ensure clean code principles, scalability, and ease of testing, the system follows a strict **Separation of Concerns (SoC)** across four modular Python components:

```
├── detector.py   # YOLO model lifecycle, inference, filtering & visual annotation
├── database.py   # MySQL connection pooling, schema migrations & telemetry logging
├── ui.py         # Reactive Streamlit dashboard with real-time video player & analytics
└── main.py       # Unified CLI launcher supporting both Web UI and Desktop OpenCV modes
```

### Module Breakdown:
1. **`detector.py` (`YOLODetector`):** Encapsulates the Ultralytics model. Accepts an incoming BGR frame along with a confidence threshold (e.g., $0.70$) and an allowed class filter (e.g., `['person', 'cell phone']`), returning annotated frames and structured detection dictionaries.
2. **`database.py` (`DatabaseManager`):** Manages MySQL connections via `mysql-connector-python`, automatically provisions tables on startup, and provides thread-safe logging and metric query functions.
3. **`ui.py`:** Renders the interactive Streamlit dashboard, giving users granular control over model parameters, classes, and logging behavior.
4. **`main.py`:** Provides an entry point with argument parsing (`argparse`) to launch either the Streamlit web dashboard or a standalone OpenCV desktop window.

---

## 4. Database Integration & Event Logging

Logging detection events is critical for auditing, security monitoring, and downstream data analytics.

### Database Schema
I configured a dedicated database named `vision_platform` with the following schema:

```sql
CREATE DATABASE IF NOT EXISTS vision_platform;
USE vision_platform;

CREATE TABLE IF NOT EXISTS detection_logs (
    log_id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    object_class VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    bbox_x INT NOT NULL,
    bbox_y INT NOT NULL,
    bbox_w INT NOT NULL,
    bbox_h INT NOT NULL,
    INDEX idx_timestamp (timestamp),
    INDEX idx_object_class (object_class)
);
```

### Intelligent Debouncing & Rate Limiting
A naive implementation would log an object on every single frame, resulting in 30+ duplicate rows per second for a stationary object. To prevent database congestion, I designed a **per-class cooldown timer**:
- An object is only logged if the time elapsed since its last recording exceeds a user-defined threshold (default: 2.0 seconds).
- This drastically reduces redundant I/O while preserving complete temporal event accuracy.

### Security Best Practices
To follow security best practices, no database passwords or connection strings are hardcoded into Python source code. All credentials (`DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, `DB_PORT`) are loaded using `python-dotenv` from a local `.env` file, which is protected from version control using `.gitignore`.

---

## 5. User Interface (Streamlit Dashboard)

The user interface was built using **Streamlit**, creating a clean, modern surveillance control center:
- **Interactive Control Panel:** Real-time sliders for confidence threshold (default 70%) and logging cooldowns, plus a multi-select filter for target object classes.
- **Live Stream Player:** Displays the annotated video stream with bounding boxes, confidence badges, and live FPS telemetry.
- **Dynamic Database Inspector:** A paginated table showing recent MySQL detection records with a one-click CSV export button.
- **Diagnostic Badges:** Visual connection status indicators displaying whether MySQL is active or offline.

---

## 6. Challenges & Solutions

| Challenge | Root Cause | Solution |
| :--- | :--- | :--- |
| **High Database Write Latency** | Inserting rows on every frame slowed down the video loop. | Implemented per-class debounce cooldowns and optimized connection handling. |
| **Streamlit Frame Refreshing** | Web rendering can lag if full frames are reloaded repeatedly. | Used OpenCV `time.sleep(0.01)` yield throttling and `st.empty()` image container recycling. |
| **Database Server Dropouts** | If MySQL was stopped, the vision pipeline crashed. | Wrapped database calls in graceful `try/except` handlers so the video stream remains uninterrupted while retrying the database connection. |
| **Credential Exposure Risks** | Risk of leaking MySQL passwords on GitHub. | Implemented strict `.env` isolation accompanied by `.env.example` templates and `.gitignore`. |

---

## 7. Final Result & Project Links

The resulting platform is a robust, responsive, and secure Computer Vision and IoT logging platform.

- **🐙 GitHub Repository:** [https://github.com/your-username/Real-Time-Object-Detection](https://github.com/your-username/Real-Time-Object-Detection)
- **🎥 Demonstration Video:** [Watch the Demo on YouTube/Google Drive](https://youtu.be/your-demo-video-link)
- **📋 Submission Form:** [Assignment 5 Google Form](https://forms.gle/fUgKta3BSytbpGpv7)

---

## Conclusion
Building this platform provided practical experience in bridging machine learning, computer vision, database engineering, and web development. Understanding how to pipe YOLO predictions into structured relational databases opens the door to intelligent surveillance, automated retail tracking, and smart industrial analytics.
