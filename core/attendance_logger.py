"""
Enterprise Biometric Attendance & Audit Trail Subsystem
Logs real-time recognition events, enforces anti-duplicate cooldown windows,
provides audit analytics, and exports compliance reports.
"""

import csv
import datetime
import io
import sqlite3
import time
from typing import Dict, List, Optional, Union


class AttendanceLogger:
    """
    Manages biometric check-in event logs, cooldown windows, and reporting.
    """

    def __init__(self, db_path: str = "./face_registry.db", cooldown_seconds: int = 60):
        self.db_path = db_path
        self.cooldown_seconds = cooldown_seconds
        self._last_logged_time: Dict[str, float] = {}
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize attendance log table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    person_name TEXT NOT NULL,
                    employee_id TEXT,
                    department TEXT,
                    confidence_pct REAL,
                    distance_score REAL,
                    status TEXT NOT NULL,
                    camera_id TEXT DEFAULT 'Kiosk_Cam_01',
                    snapshot_b64 TEXT
                )
            """)
            conn.commit()

    def log_recognition_event(
        self,
        person_name: str,
        confidence: float,
        distance: float,
        employee_id: Optional[str] = None,
        department: Optional[str] = None,
        is_unknown: bool = False,
        camera_id: str = "Kiosk_Cam_01",
        snapshot_b64: Optional[str] = None,
        force: bool = False,
    ) -> Optional[Dict]:
        """
        Record a biometric attendance entry if outside the cooldown window.

        :return: Created log dict, or None if skipped due to active cooldown
        """
        now_ts = time.time()
        key = person_name.strip().lower()

        # Cooldown check for known subjects
        if not is_unknown and not force:
            last_time = self._last_logged_time.get(key, 0.0)
            if now_ts - last_time < self.cooldown_seconds:
                return None

        # Unknown impostors have a 10-second spam throttle
        if is_unknown and not force:
            last_unknown = self._last_logged_time.get("unknown_alert", 0.0)
            if now_ts - last_unknown < 10.0:
                return None
            self._last_logged_time["unknown_alert"] = now_ts
        else:
            self._last_logged_time[key] = now_ts

        now_iso = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "UNKNOWN_ALERT" if is_unknown else "VERIFIED"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO attendance_logs (
                    timestamp, person_name, employee_id, department,
                    confidence_pct, distance_score, status, camera_id, snapshot_b64
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now_iso,
                person_name,
                employee_id or "N/A",
                department or "General",
                round(confidence, 1),
                round(distance, 2),
                status,
                camera_id,
                snapshot_b64,
            ))
            conn.commit()
            log_id = cursor.lastrowid

            cursor.execute("SELECT * FROM attendance_logs WHERE id = ?", (log_id,))
            row = cursor.fetchone()
            return dict(row)

    def get_logs(
        self,
        limit: int = 100,
        status: Optional[str] = None,
        department: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[Dict]:
        """Retrieve filtered attendance logs."""
        query = "SELECT * FROM attendance_logs WHERE 1=1"
        params = []

        if status and status != "ALL":
            query += " AND status = ?"
            params.append(status)

        if department and department != "ALL":
            query += " AND department = ?"
            params.append(department)

        if search:
            query += " AND (person_name LIKE ? OR employee_id LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_summary_metrics(self) -> Dict:
        """Calculate summary attendance KPIs."""
        today_date = datetime.datetime.now().strftime("%Y-%m-%d")
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Today's total
            cursor.execute("SELECT COUNT(*) FROM attendance_logs WHERE timestamp LIKE ?", (f"{today_date}%",))
            total_today = cursor.fetchone()[0]

            # Verified today
            cursor.execute("SELECT COUNT(*) FROM attendance_logs WHERE timestamp LIKE ? AND status = 'VERIFIED'", (f"{today_date}%",))
            verified_today = cursor.fetchone()[0]

            # Unknown alerts today
            cursor.execute("SELECT COUNT(*) FROM attendance_logs WHERE timestamp LIKE ? AND status = 'UNKNOWN_ALERT'", (f"{today_date}%",))
            unknown_today = cursor.fetchone()[0]

            # Average confidence
            cursor.execute("SELECT AVG(confidence_pct) FROM attendance_logs WHERE status = 'VERIFIED'")
            avg_conf = cursor.fetchone()[0] or 0.0

            # Unique attendees today
            cursor.execute("SELECT COUNT(DISTINCT person_name) FROM attendance_logs WHERE timestamp LIKE ? AND status = 'VERIFIED'", (f"{today_date}%",))
            unique_today = cursor.fetchone()[0]

            return {
                "total_today": int(total_today),
                "verified_today": int(verified_today),
                "unknown_today": int(unknown_today),
                "avg_confidence": round(float(avg_conf), 1),
                "unique_attendees_today": int(unique_today),
            }

    def export_csv(self) -> str:
        """Export all logs to CSV format string."""
        logs = self.get_logs(limit=1000)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "ID", "Timestamp", "Subject Name", "Employee ID",
            "Department", "Confidence (%)", "Distance Score", "Status", "Camera ID"
        ])

        for log in logs:
            writer.writerow([
                log["id"],
                log["timestamp"],
                log["person_name"],
                log["employee_id"],
                log["department"],
                log["confidence_pct"],
                log["distance_score"],
                log["status"],
                log["camera_id"],
            ])

        return output.getvalue()

    def clear_logs(self):
        """Clear all historical logs."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM attendance_logs")
            conn.commit()
