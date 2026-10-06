# 🚀 Real-Time Object Detection & Logging Platform

[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12%20|%203.14-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-00FFFF.svg)](https://ultralytics.com/)
[![MySQL](https://img.shields.io/badge/MySQL-Event%20Logging-4479A1.svg)](https://www.mysql.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20UI-FF4B4B.svg)](https://streamlit.io/)

A modular, production-grade Computer Vision and Event Logging platform built with **Python**, **OpenCV**, and **Ultralytics YOLOv8**, coupled with a **MySQL** relational database for real-time detection telemetry and a **Streamlit** live dashboard.

Developed for **Assignment 5: Real-Time Object Detection & Logging Platform**.

---

## 🌟 Key Features

- **⚡ Real-Time YOLO Inference:** High-throughput object detection using pre-trained `yolov8n.pt` with bounding box and label rendering.
- **🎯 Dynamic Object Configuration:** User-selectable target classes (e.g., detect only `person`, `cell phone`, `laptop`, or custom subsets of all 80 COCO categories).
- **🎚️ Adjustable Confidence Threshold:** Interactive slider to filter detections (e.g., log only predictions with $\ge 70\%$ confidence).
- **🗄️ MySQL Event Logging:** Automatically logs detection timestamps, object classes, normalized confidences, and bounding box coordinates `(bbox_x, bbox_y, bbox_w, bbox_h)`.
- **⏱️ Intelligent Debounce/Cooldown:** Prevents database flooding by enforcing configurable per-object logging cooldowns.
- **📊 Interactive Streamlit Dashboard:** Live video feed, real-time FPS counter, database connectivity badges, and dynamic log inspection table with CSV export.
- **🖥️ Dual Execution Modes:** Run as a rich web dashboard or a standalone low-latency OpenCV desktop application.
- **🔒 Security by Design:** Zero hardcoded credentials. Database credentials managed securely via `.env` and `.gitignore`.

---

## 🏗️ System Architecture

```
                                  +-----------------------+
                                  |   Video Stream Input  |
                                  |  (Webcam / Video MP4) |
                                  +-----------+-----------+
                                              |
                                              v
+------------------------+        +-----------+-----------+        +------------------------+
|      detector.py       | -----> |    Frame Processing   | -----> |         ui.py          |
|  - Ultralytics YOLOv8  |        |  - Filter by Class    |        |  - Streamlit Dashboard |
|  - Coordinate mapping  |        |  - Filter Confidence  |        |  - Live Annotated Feed |
|  - Visual Annotations  |        |  - Calculate FPS      |        |  - Telemetry & Metrics |
+------------------------+        +-----------+-----------+        +------------------------+
                                              |
                                              v
                                  +-----------+-----------+
                                  |      database.py      |
                                  |  - Debounce/Cooldown  |
                                  |  - MySQL Connector    |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  |    MySQL Database     |
                                  |   (vision_platform)   |
                                  |    detection_logs     |
                                  +-----------------------+
```

---

## 📁 Project Structure

```
├── main.py                  # Application entry point (supports UI and OpenCV modes)
├── detector.py              # YOLOv8 detection, filtering, and annotation engine
├── database.py              # MySQL connection manager, schema init, and log queries
├── ui.py                    # Streamlit interactive web dashboard
├── database_schema.sql      # MySQL DDL script for database and table creation
├── requirements.txt         # Project dependencies
├── .env.example             # Safe template for environment variables
├── .env                     # Local credentials file (gitignored)
├── .gitignore               # Excludes secrets, weights, and caches from version control
├── DEMO_GUIDE.md            # Step-by-step recording guide for the 1-2 min video
├── MEDIUM_ARTICLE.md        # Comprehensive technical blog post draft
└── README.md                # Project documentation
```

---

## 🛠️ Prerequisites & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Real-Time-Object-Detection.git
cd Real-Time-Object-Detection
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🗄️ MySQL Database Setup

### Option A: Using XAMPP / WAMP (Easiest)
1. Open the **XAMPP Control Panel**.
2. Click **Start** next to **MySQL** (port 3306).
3. Open `http://localhost/phpmyadmin` in your browser.
4. Click on the **SQL** tab.
5. Copy and paste the contents of `database_schema.sql` and click **Go**.

### Option B: Using MySQL Workbench / Command Line
Run the provided SQL file:
```bash
mysql -u root -p < database_schema.sql
```

### Database Schema Details:
- **Database:** `vision_platform`
- **Table:** `detection_logs`
  - `log_id`: `INT AUTO_INCREMENT PRIMARY KEY`
  - `timestamp`: `DATETIME DEFAULT CURRENT_TIMESTAMP`
  - `object_class`: `VARCHAR(50) NOT NULL`
  - `confidence`: `FLOAT NOT NULL`
  - `bbox_x`: `INT NOT NULL`
  - `bbox_y`: `INT NOT NULL`
  - `bbox_w`: `INT NOT NULL`
  - `bbox_h`: `INT NOT NULL`

---

## ⚙️ Configuration (`.env`)

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Update your credentials in `.env`:
```ini
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_NAME=vision_platform

CONFIDENCE_THRESHOLD=0.70
LOG_COOLDOWN_SECONDS=2.0
VIDEO_SOURCE=0
```
*(If using default XAMPP, `DB_USER=root` and `DB_PASSWORD=` empty).*

---

## 🚀 Running the Application

### Mode 1: Streamlit Interactive Dashboard (Recommended)
Launch the web UI directly:
```bash
streamlit run ui.py
```
*Or via the unified launcher:*
```bash
python main.py --mode ui
```
Open your browser at `http://localhost:8501`.
- Check the **Start Video Stream** box to view live detections.
- Adjust confidence slider and object filters on the sidebar in real time.
- Switch to the **MySQL Event Logs** tab to see live records updating.

### Mode 2: Direct OpenCV Desktop Window
For high-performance desktop execution:
```bash
python main.py --mode opencv --conf 0.70 --classes person "cell phone" bottle
```
**Keyboard Shortcuts:**
- Press `q` to exit.
- Press `s` to save a high-resolution annotated screenshot.

---

## 🧪 Testing & Verification

1. **Test Database Connectivity:**
   ```bash
   python database.py
   ```
2. **Test YOLO Inference:**
   ```bash
   python detector.py
   ```
3. **Verify Database Entries in MySQL:**
   ```sql
   USE vision_platform;
   SELECT * FROM detection_logs ORDER BY log_id DESC LIMIT 10;
   ```

---

## 📄 License
This project is licensed under the MIT License.
