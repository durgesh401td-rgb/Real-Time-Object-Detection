# 🎥 Demonstration Video Guide (1–2 Minutes)

This guide provides a structured walkthrough and script for recording your working demonstration of the **Real-Time Object Detection & Logging Platform** for Assignment 5.

---

## 🛠️ Pre-Recording Setup

1. **Start Your MySQL Server:**
   - Launch XAMPP / WAMP and click **Start** on **MySQL** (or ensure the Windows MySQL service is running).
   - Open **phpMyAdmin** (`http://localhost/phpmyadmin`) or **MySQL Workbench** in your browser.
   - Navigate to the `vision_platform` database -> `detection_logs` table.

2. **Launch the Platform:**
   ```bash
   streamlit run ui.py
   ```
   Open `http://localhost:8501`.

3. **Screen Layout (Split-Screen Recommended):**
   - **Left Half of Screen:** Streamlit Dashboard (`http://localhost:8501`).
   - **Right Half of Screen:** phpMyAdmin / MySQL Workbench showing the `detection_logs` table.
   - *Why?* This allows the viewer to see bounding boxes drawn in real-time on the left while simultaneously seeing rows appear in MySQL on the right.

4. **Have Objects Ready:**
   - Yourself (Person)
   - Cell Phone
   - Water Bottle / Cup
   - Laptop / Book

---

## ⏱️ Video Recording Script & Timeline (1–2 Minutes)

### Phase 1: Introduction & Architecture (0:00 – 0:25)
- **Visual:** Show full screen of the Streamlit dashboard. Point to the title and metadata.
- **Talking Points:**
  > "Hello everyone! This is my demonstration for Assignment 5: Real-Time Object Detection & Logging Platform.
  > Built using Python, OpenCV, Ultralytics YOLOv8, MySQL, and Streamlit.
  > Notice in the sidebar that our MySQL database `vision_platform` is successfully connected and initialized."

### Phase 2: Configuration & Thresholds (0:25 – 0:45)
- **Action:**
  - Adjust the **Confidence Threshold** slider to **0.70 (70%)**.
  - Show the **Object Configuration** multi-select and filter for `person`, `cell phone`, `bottle`.
- **Talking Points:**
  > "In the control panel, we have modular configuration options. We can select specific target classes such as 'person' and 'cell phone' and adjust our confidence threshold to 70% to ensure only high-confidence detections are processed."

### Phase 3: Live Detection Feed & Bounding Boxes (0:45 – 1:15)
- **Action:**
  - Click **▶️ Start Video Stream**.
  - Show yourself on camera (detects `person`).
  - Hold up your smartphone to the camera (detects `cell phone` with bounding box and confidence score).
  - Hold up a bottle (detects `bottle`).
- **Talking Points:**
  > "As the webcam stream starts, OpenCV grabs frames and YOLOv8 performs real-time inference.
  > You can see accurate bounding boxes and classification labels drawn directly over the video stream with live FPS tracking."

### Phase 4: Real-Time MySQL Verification (1:15 – 1:45)
- **Action:**
  - Switch to the **MySQL Event Logs** tab in Streamlit, or refresh the **phpMyAdmin / MySQL Workbench** window on the right.
  - Highlight newly added rows showing `timestamp`, `object_class`, `confidence`, and `bbox_x, bbox_y, bbox_w, bbox_h`.
- **Talking Points:**
  > "Every time a target object is detected above the threshold, an event is logged in MySQL.
  > Here in our database table, we can see the timestamp, detected class name, confidence, and exact bounding box coordinates saved dynamically."

### Phase 5: Conclusion & Wrap-Up (1:45 – 2:00)
- **Action:**
  - Stop the video stream. Show the CSV download button or GitHub repo.
- **Talking Points:**
  > "All credentials are safe in .env, and the modular code is organized cleanly into detector, database, UI, and main modules. Thank you for watching!"

---

## 📤 Video Upload Instructions

1. Use OBS Studio, Windows Game Bar (`Win + G`), or Loom to record.
2. Upload the MP4 file to:
   - **YouTube (Unlisted):** Anyone with the link can view, but it's not public.
   - **Google Drive:** Make sure access is set to **"Anyone with the link can view"**.
3. Save the link for the assignment submission form.
