"""
Telemetry module - asynchronous logging of detections to InfluxDB.

Uses batch writing to not slow down the main detection loop.
Handles errors gracefully - application doesn't crash when database is unresponsive.
"""

import logging
from datetime import datetime
from typing import Optional
from threading import Thread
from queue import Queue, Empty
import atexit

try:
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS, WriteOptions
    INFLUX_AVAILABLE = True
except ImportError:
    INFLUX_AVAILABLE = False
    print("[Telemetry] WARNING: influxdb-client library not found. Install: pip install influxdb-client")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telemetry")


class TelemetryClient:
    """Client for sending gesture detection metrics to InfluxDB."""

    def __init__(
        self,
        url: str = "http://localhost:8086",
        token: str = "my-super-secret-token",
        org: str = "gesture_org",
        bucket: str = "gesture_detections",
        batch_size: int = 10,
        flush_interval_ms: int = 5000,
        enabled: bool = True
    ):
        """Initializes telemetry client."""
        self.enabled = enabled and INFLUX_AVAILABLE
        self.bucket = bucket
        self.org = org
        self._client: Optional[InfluxDBClient] = None
        self._write_api = None
        self._connected = False

        if not self.enabled:
            logger.warning("Telemetry disabled or influxdb-client library missing")
            return

        try:
            self._client = InfluxDBClient(url=url, token=token, org=org)
            write_options = WriteOptions(
                batch_size=batch_size,
                flush_interval=flush_interval_ms,
                jitter_interval=1000,
                retry_interval=5000,
                max_retries=3,
                max_retry_delay=30000,
            )
            self._write_api = self._client.write_api(write_options=write_options)
            health = self._client.health()
            if health.status == "pass":
                self._connected = True
                logger.info(f"Connected to InfluxDB: {url}")
            else:
                logger.warning(f"InfluxDB health check failed: {health.message}")
        except Exception as e:
            logger.error(f"Cannot connect to InfluxDB: {e}")
            self._connected = False

        atexit.register(self.close)

    @property
    def is_connected(self) -> bool:
        """Returns True if database connection is active."""
        return self._connected

    def log_detection(self, gesture_name: str, confidence: float, hand_index: int = 0,
                      extra_tags: Optional[dict] = None, extra_fields: Optional[dict] = None) -> bool:
        """Logs a single gesture detection."""
        if not self.enabled or not self._connected:
            return False
        try:
            point = (Point("gesture_detection").tag("gesture", gesture_name)
                     .tag("hand_index", str(hand_index)).field("confidence", float(confidence))
                     .field("detected", 1))
            if extra_tags:
                for key, value in extra_tags.items():
                    point = point.tag(key, str(value))
            if extra_fields:
                for key, value in extra_fields.items():
                    point = point.field(key, value)
            self._write_api.write(bucket=self.bucket, org=self.org, record=point)
            return True
        except Exception as e:
            logger.debug(f"Error writing to InfluxDB: {e}")
            return False

    def log_session_stats(self, gesture_counts: dict, duration_seconds: float, avg_fps: float = 0.0) -> bool:
        """Logs detection session statistics."""
        if not self.enabled or not self._connected:
            return False
        try:
            point = Point("session_stats").field("duration_seconds", float(duration_seconds)).field("avg_fps", float(avg_fps))
            for gesture, count in gesture_counts.items():
                point = point.field(f"count_{gesture}", int(count))
            self._write_api.write(bucket=self.bucket, org=self.org, record=point)
            return True
        except Exception as e:
            logger.debug(f"Error writing statistics: {e}")
            return False

    def log_activity_duration(self, activity_name: str, duration_seconds: float) -> bool:
        """Logs activity duration (e.g., nail biting)."""
        if not self.enabled or not self._connected:
            return False
        try:
            point = Point("activity_duration").tag("activity", activity_name).field("duration_seconds", float(duration_seconds))
            self._write_api.write(bucket=self.bucket, org=self.org, record=point)
            return True
        except Exception as e:
            logger.debug(f"Error writing activity: {e}")
            return False

    def flush(self):
        """Forces immediate sending of buffered data."""
        if self._write_api:
            try:
                self._write_api.flush()
            except Exception as e:
                logger.debug(f"Flush error: {e}")

    def close(self):
        """Closes database connection (called automatically on exit)."""
        if self._write_api:
            try:
                self._write_api.close()
            except:
                pass
        if self._client:
            try:
                self._client.close()
            except:
                pass
        logger.info("Telemetry closed")


_telemetry_instance: Optional[TelemetryClient] = None


def get_telemetry(url: str = "http://localhost:8086", token: str = "my-super-secret-token",
                  org: str = "gesture_org", bucket: str = "gesture_detections", enabled: bool = True) -> TelemetryClient:
    """Returns global telemetry client instance (singleton)."""
    global _telemetry_instance
    if _telemetry_instance is None:
        _telemetry_instance = TelemetryClient(url=url, token=token, org=org, bucket=bucket, enabled=enabled)
    return _telemetry_instance