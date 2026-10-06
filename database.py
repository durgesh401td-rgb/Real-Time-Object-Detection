"""
Database Management Module for Real-Time Object Detection Platform.
Handles MySQL connections, table initialization, and detection event logging.
"""

import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error

# Load environment variables from .env file
load_dotenv()


class DatabaseManager:
    """Manages MySQL database connections and operations for object detection logs."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        cooldown_seconds: float = 2.0,
    ):
        self.host = host or os.getenv("DB_HOST", "localhost")
        self.port = int(port or os.getenv("DB_PORT", 3306))
        self.user = user or os.getenv("DB_USER", "root")
        self.password = password if password is not None else os.getenv("DB_PASSWORD", "")
        self.database = database or os.getenv("DB_NAME", "vision_platform")
        self.cooldown_seconds = cooldown_seconds

        # Track last logged timestamp per object class to avoid flooding the database
        self._last_logged_time: Dict[str, float] = {}
        self._last_error: Optional[str] = None
        self._connected: bool = False

        # Attempt initial setup
        self.init_database()

    def get_server_connection(self):
        """Connects to the MySQL server without selecting a specific database."""
        return mysql.connector.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            connect_timeout=3,
        )

    def get_db_connection(self):
        """Connects directly to the target database."""
        return mysql.connector.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            connect_timeout=3,
        )

    def init_database(self) -> bool:
        """
        Creates the database and detection_logs table if they do not exist.
        Returns True if successful, False otherwise.
        """
        try:
            # Step 1: Ensure database exists
            server_conn = self.get_server_connection()
            cursor = server_conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{self.database}`;")
            cursor.close()
            server_conn.close()

            # Step 2: Ensure detection_logs table exists
            db_conn = self.get_db_connection()
            cursor = db_conn.cursor()
            create_table_query = """
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
            """
            cursor.execute(create_table_query)
            db_conn.commit()
            cursor.close()
            db_conn.close()

            self._connected = True
            self._last_error = None
            return True

        except Error as err:
            self._connected = False
            self._last_error = str(err)
            return False
        except Exception as ex:
            self._connected = False
            self._last_error = str(ex)
            return False

    def is_connected(self) -> Tuple[bool, Optional[str]]:
        """Checks if connection to MySQL is alive and healthy."""
        try:
            conn = self.get_db_connection()
            if conn.is_connected():
                conn.close()
                self._connected = True
                self._last_error = None
                return True, "Connected to MySQL successfully"
        except Exception as ex:
            self._connected = False
            self._last_error = str(ex)

        return False, self._last_error

    def log_detection(
        self,
        object_class: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],
        enforce_cooldown: bool = True,
    ) -> bool:
        """
        Inserts a detection event into the detection_logs table.
        
        Args:
            object_class: Name of the detected object (e.g., 'person', 'cell phone')
            confidence: Confidence score (0.0 to 1.0)
            bbox: Tuple of (x, y, w, h) bounding box coordinates
            enforce_cooldown: If True, prevents logging the same object class within cooldown_seconds
        
        Returns:
            True if logged successfully, False otherwise.
        """
        now = time.time()
        if enforce_cooldown:
            last_time = self._last_logged_time.get(object_class, 0.0)
            if (now - last_time) < self.cooldown_seconds:
                return False

        bbox_x, bbox_y, bbox_w, bbox_h = bbox

        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            query = """
            INSERT INTO detection_logs 
                (timestamp, object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            current_time = datetime.now()
            cursor.execute(
                query,
                (current_time, object_class, float(confidence), int(bbox_x), int(bbox_y), int(bbox_w), int(bbox_h)),
            )
            conn.commit()
            cursor.close()
            conn.close()

            self._last_logged_time[object_class] = now
            self._connected = True
            return True

        except Exception as ex:
            self._connected = False
            self._last_error = str(ex)
            return False

    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieves the most recent detection events from MySQL.
        
        Args:
            limit: Maximum number of records to retrieve (default: 50)
            
        Returns:
            List of dictionaries representing log rows.
        """
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor(dictionary=True)
            query = """
            SELECT log_id, timestamp, object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h
            FROM detection_logs
            ORDER BY log_id DESC
            LIMIT %s
            """
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            return rows
        except Exception as ex:
            self._last_error = str(ex)
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Returns summary statistics for detection logs."""
        stats = {
            "total_detections": 0,
            "unique_classes": 0,
            "top_classes": [],
        }
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()

            # Total detections
            cursor.execute("SELECT COUNT(*) FROM detection_logs")
            stats["total_detections"] = cursor.fetchone()[0]

            # Unique classes
            cursor.execute("SELECT COUNT(DISTINCT object_class) FROM detection_logs")
            stats["unique_classes"] = cursor.fetchone()[0]

            # Top classes
            cursor.execute("""
                SELECT object_class, COUNT(*) as count 
                FROM detection_logs 
                GROUP BY object_class 
                ORDER BY count DESC 
                LIMIT 5
            """)
            stats["top_classes"] = cursor.fetchall()

            cursor.close()
            conn.close()
        except Exception as ex:
            self._last_error = str(ex)

        return stats

    def clear_logs(self) -> bool:
        """Clears all records from the detection_logs table."""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            cursor.execute("TRUNCATE TABLE detection_logs")
            conn.commit()
            cursor.close()
            conn.close()
            self._last_logged_time.clear()
            return True
        except Exception as ex:
            self._last_error = str(ex)
            return False


if __name__ == "__main__":
    print("Testing DatabaseManager connection...")
    db = DatabaseManager()
    connected, message = db.is_connected()
    print(f"Status: {connected} | Details: {message}")
    if connected:
        print("Inserting test detection event...")
        success = db.log_detection("test_object", 0.95, (100, 100, 200, 200), enforce_cooldown=False)
        print(f"Logged successfully: {success}")
        logs = db.get_recent_logs(5)
        print(f"Recent logs fetched: {len(logs)}")
        for log in logs:
            print(log)
