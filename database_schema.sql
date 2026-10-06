-- =========================================================
-- Real-Time Object Detection & Logging Platform
-- Database Schema for MySQL
-- Database: vision_platform
-- Table: detection_logs
-- =========================================================

-- Create the database if it doesn't already exist
CREATE DATABASE IF NOT EXISTS vision_platform;

-- Switch to the vision_platform database
USE vision_platform;

-- Create the detection_logs table
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
