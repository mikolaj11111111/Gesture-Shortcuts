"""
Moduł telemetrii - asynchroniczne logowanie detekcji do InfluxDB.

Używa batch writing aby nie spowalniać głównej pętli detekcji.
Obsługuje błędy gracefully - aplikacja nie crashuje gdy baza nie odpowiada.
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
    print("[Telemetry] UWAGA: Brak biblioteki influxdb-client. Zainstaluj: pip install influxdb-client")

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telemetry")


class TelemetryClient:
    """
    Klient do wysyłania metryk detekcji gestów do InfluxDB.

    Cechy:
    - Batch writing (buforowanie przed wysłaniem)
    - Asynchroniczne wysyłanie w osobnym wątku
    - Graceful error handling (nie crashuje aplikacji)
    - Auto-flush przy zamknięciu aplikacji
    """

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
        """
        Inicjalizuje klienta telemetrii.

        Args:
            url: URL do InfluxDB
            token: Token autoryzacyjny
            org: Nazwa organizacji w InfluxDB
            bucket: Nazwa bucketu do zapisu
            batch_size: Ilość punktów przed automatycznym wysłaniem
            flush_interval_ms: Interwał flush w milisekundach
            enabled: Czy telemetria jest włączona
        """
        self.enabled = enabled and INFLUX_AVAILABLE
        self.bucket = bucket
        self.org = org
        self._client: Optional[InfluxDBClient] = None
        self._write_api = None
        self._connected = False

        if not self.enabled:
            logger.warning("Telemetria wyłączona lub brak biblioteki influxdb-client")
            return

        try:
            self._client = InfluxDBClient(url=url, token=token, org=org)

            # Konfiguracja batch writing
            write_options = WriteOptions(
                batch_size=batch_size,
                flush_interval=flush_interval_ms,
                jitter_interval=1000,
                retry_interval=5000,
                max_retries=3,
                max_retry_delay=30000,
            )
            self._write_api = self._client.write_api(write_options=write_options)

            # Test połączenia
            health = self._client.health()
            if health.status == "pass":
                self._connected = True
                logger.info(f"Połączono z InfluxDB: {url}")
            else:
                logger.warning(f"InfluxDB health check failed: {health.message}")

        except Exception as e:
            logger.error(f"Nie można połączyć z InfluxDB: {e}")
            self._connected = False

        # Auto-flush przy zamknięciu programu
        atexit.register(self.close)

    @property
    def is_connected(self) -> bool:
        """Zwraca True jeśli połączenie z bazą jest aktywne."""
        return self._connected

    def log_detection(
        self,
        gesture_name: str,
        confidence: float,
        hand_index: int = 0,
        extra_tags: Optional[dict] = None,
        extra_fields: Optional[dict] = None
    ) -> bool:
        """
        Loguje pojedynczą detekcję gestu.

        Args:
            gesture_name: Nazwa wykrytego gestu
            confidence: Pewność detekcji (0.0 - 1.0)
            hand_index: Indeks ręki (0 = pierwsza wykryta)
            extra_tags: Dodatkowe tagi (indeksowane)
            extra_fields: Dodatkowe pola (wartości)

        Returns:
            True jeśli zapis się powiódł, False w przeciwnym razie
        """
        if not self.enabled or not self._connected:
            return False

        try:
            point = (
                Point("gesture_detection")
                .tag("gesture", gesture_name)
                .tag("hand_index", str(hand_index))
                .field("confidence", float(confidence))
                .field("detected", 1)  # Licznik do agregacji
            )

            # Dodaj extra tagi
            if extra_tags:
                for key, value in extra_tags.items():
                    point = point.tag(key, str(value))

            # Dodaj extra pola
            if extra_fields:
                for key, value in extra_fields.items():
                    point = point.field(key, value)

            self._write_api.write(bucket=self.bucket, org=self.org, record=point)
            return True

        except Exception as e:
            logger.debug(f"Błąd zapisu do InfluxDB: {e}")
            return False

    def log_session_stats(
        self,
        gesture_counts: dict,
        duration_seconds: float,
        avg_fps: float = 0.0
    ) -> bool:
        """
        Loguje statystyki sesji detekcji.

        Args:
            gesture_counts: Słownik {gesture_name: count}
            duration_seconds: Czas trwania sesji w sekundach
            avg_fps: Średnia ilość klatek na sekundę
        """
        if not self.enabled or not self._connected:
            return False

        try:
            point = (
                Point("session_stats")
                .field("duration_seconds", float(duration_seconds))
                .field("avg_fps", float(avg_fps))
            )

            for gesture, count in gesture_counts.items():
                point = point.field(f"count_{gesture}", int(count))

            self._write_api.write(bucket=self.bucket, org=self.org, record=point)
            return True

        except Exception as e:
            logger.debug(f"Błąd zapisu statystyk: {e}")
            return False

    def log_activity_duration(
        self,
        activity_name: str,
        duration_seconds: float
    ) -> bool:
        """
        Loguje czas trwania aktywności (np. obgryzanie paznokci).

        Args:
            activity_name: Nazwa aktywności
            duration_seconds: Czas trwania w sekundach
        """
        if not self.enabled or not self._connected:
            return False

        try:
            point = (
                Point("activity_duration")
                .tag("activity", activity_name)
                .field("duration_seconds", float(duration_seconds))
            )

            self._write_api.write(bucket=self.bucket, org=self.org, record=point)
            return True

        except Exception as e:
            logger.debug(f"Błąd zapisu aktywności: {e}")
            return False

    def flush(self):
        """Wymusza natychmiastowe wysłanie zbuforowanych danych."""
        if self._write_api:
            try:
                self._write_api.flush()
            except Exception as e:
                logger.debug(f"Błąd flush: {e}")

    def close(self):
        """Zamyka połączenie z bazą (wywoływane automatycznie przy exit)."""
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
        logger.info("Telemetria zamknięta")


# Singleton - globalna instancja
_telemetry_instance: Optional[TelemetryClient] = None


def get_telemetry(
    url: str = "http://localhost:8086",
    token: str = "my-super-secret-token",
    org: str = "gesture_org",
    bucket: str = "gesture_detections",
    enabled: bool = True
) -> TelemetryClient:
    """
    Zwraca globalną instancję klienta telemetrii (singleton).

    Przy pierwszym wywołaniu tworzy nową instancję.
    Kolejne wywołania zwracają tę samą instancję.
    """
    global _telemetry_instance

    if _telemetry_instance is None:
        _telemetry_instance = TelemetryClient(
            url=url,
            token=token,
            org=org,
            bucket=bucket,
            enabled=enabled
        )

    return _telemetry_instance