# Zagrożenia bezpieczeństwa - Publiczne repo

> [!CAUTION]
> **Te pliki zawierają wrażliwe dane i NIE powinny być na publicznym repo!**

---

## 🔴 KRYTYCZNE

### 1. docker-compose.yml
**Lokalizacja**: `/docker-compose.yml`

| Linia | Zagrożenie |
|-------|-----------|
| 14 | `DOCKER_INFLUXDB_INIT_USERNAME=admin` |
| 15 | `DOCKER_INFLUXDB_INIT_PASSWORD=adminpassword123` |
| 18 | `DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=my-super-secret-token` |
| 31 | `GF_SECURITY_ADMIN_USER=admin` |
| 32 | `GF_SECURITY_ADMIN_PASSWORD=admin` |

**Rozwiązanie**: Użyj zmiennych środowiskowych lub pliku `.env`:
```yaml
environment:
  - DOCKER_INFLUXDB_INIT_PASSWORD=${INFLUX_PASSWORD}
```

---

### 2. telemetry/telemetry.py
**Lokalizacja**: `/telemetry/telemetry.py` (linia 42)

```python
token: str = "my-super-secret-token"
```

**Rozwiązanie**: Wczytuj z zmiennej środowiskowej:
```python
import os
token: str = os.getenv("INFLUX_TOKEN", "")
```

---

## 🟡 ŚREDNIE

### 3. Zdjęcia datasetu
**Lokalizacja**: `/creating_dataset/data/*/images/`

- Mogą zawierać Twoje zdjęcia/twarz
- Już dodane do `.gitignore` ✅

### 4. Wagi modelu (.pt)
- Mogą ujawnić na jakich danych trenowano
- Już dodane do `.gitignore` ✅

---

## ✅ Zalecane przed pushowaniem

1. [ ] Usuń hardcoded hasła z `docker-compose.yml`
2. [ ] Usuń hardcoded token z `telemetry.py`
3. [ ] Sprawdź czy `.gitignore` działa: `git status`
4. [ ] Stwórz `.env.example` z przykładowymi wartościami (bez prawdziwych haseł)
