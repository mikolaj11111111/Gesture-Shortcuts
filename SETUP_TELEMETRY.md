# Konfiguracja Telemetrii (InfluxDB + Grafana)

## Wymagania
- Docker Desktop zainstalowany i uruchomiony
- Python z biblioteką `influxdb-client`

```bash
pip install influxdb-client
```

---

## 1. Uruchomienie kontenerów

```bash
cd "C:\Projekty\Gesture Shortcuts"
docker-compose up -d
```

Po uruchomieniu:
- **InfluxDB**: http://localhost:8086
- **Grafana**: http://localhost:3000

---

## 2. Dane logowania

### InfluxDB
| Pole | Wartość |
|------|---------|
| Username | `admin` |
| Password | `adminpassword123` |
| Organization | `gesture_org` |
| Bucket | `gesture_detections` |
| Token | `my-super-secret-token` |

### Grafana
| Pole | Wartość |
|------|---------|
| Username | `admin` |
| Password | `admin` |

---

## 3. Weryfikacja połączenia

### Sprawdź czy InfluxDB działa:
```bash
curl http://localhost:8086/health
```

Oczekiwana odpowiedź: `{"status":"pass"}`

### Sprawdź czy Grafana widzi InfluxDB:
1. Otwórz http://localhost:3000
2. Zaloguj się (admin/admin)
3. Idź do: Configuration → Data Sources
4. Powinieneś widzieć "InfluxDB" jako skonfigurowany

---

## 4. Uruchomienie detekcji

```bash
python live_detection/detector.py
```

W konsoli zobaczysz:
```
Wczytywanie modelu...
Kategorie: ['continue_recording_sign', 'nail_biting_sign', 'other', 'stop_recording_sign']
Telemetria: POŁĄCZONO z InfluxDB
Uruchomiono live detection. Q - wyjscie
```

Na obrazie w lewym górnym rogu: `DB: ON` oznacza aktywne połączenie.

---

## 5. Tworzenie dashboardu w Grafanie

### Przykładowe zapytania Flux:

**Liczba detekcji w czasie (wykres):**
```flux
from(bucket: "gesture_detections")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "gesture_detection")
  |> filter(fn: (r) => r._field == "detected")
  |> aggregateWindow(every: 1m, fn: sum, createEmpty: false)
  |> group(columns: ["gesture"])
```

**Średnia pewność detekcji:**
```flux
from(bucket: "gesture_detections")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "gesture_detection")
  |> filter(fn: (r) => r._field == "confidence")
  |> mean()
  |> group(columns: ["gesture"])
```

**Czas trwania aktywności (np. nail_biting):**
```flux
from(bucket: "gesture_detections")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "activity_duration")
  |> filter(fn: (r) => r.activity == "nail_biting_sign")
  |> sum()
```

**Statystyki sesji:**
```flux
from(bucket: "gesture_detections")
  |> range(start: -7d)
  |> filter(fn: (r) => r._measurement == "session_stats")
```

---

## 6. Struktura danych w InfluxDB

### Measurement: `gesture_detection`
| Tag/Field | Typ | Opis |
|-----------|-----|------|
| gesture (tag) | string | Nazwa gestu |
| hand_index (tag) | string | Indeks ręki |
| confidence (field) | float | Pewność 0.0-1.0 |
| detected (field) | int | Zawsze 1 (do zliczania) |

### Measurement: `activity_duration`
| Tag/Field | Typ | Opis |
|-----------|-----|------|
| activity (tag) | string | Nazwa aktywności |
| duration_seconds (field) | float | Czas trwania |

### Measurement: `session_stats`
| Field | Typ | Opis |
|-------|-----|------|
| duration_seconds | float | Długość sesji |
| avg_fps | float | Średnie FPS |
| count_<gesture> | int | Liczba detekcji danego gestu |

---

## 7. Zatrzymanie kontenerów

```bash
docker-compose down
```

Dane są zachowane w volumes. Aby usunąć dane:
```bash
docker-compose down -v
```

---

## 8. Rozwiązywanie problemów

### "Telemetria: OFFLINE"
1. Sprawdź czy Docker działa: `docker ps`
2. Sprawdź czy kontenery są uruchomione: `docker-compose ps`
3. Sprawdź logi: `docker-compose logs influxdb`

### Brak danych w Grafanie
1. Upewnij się że detektor działa i wykrywa gesty
2. Sprawdź czy token jest poprawny
3. W Grafanie: Query Inspector → sprawdź błędy

### Błąd "influxdb-client not found"
```bash
pip install influxdb-client
```