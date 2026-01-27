"""
Moduł akcji - wykonywanie skrótów klawiszowych na podstawie wykrytych gestów.

Używa pynput do symulacji klawiszy.
Zawiera debouncing żeby unikać wielokrotnego wywołania tej samej akcji.
"""

import time
import logging
from typing import Callable, Optional
from dataclasses import dataclass

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("[Actions] UWAGA: Brak biblioteki psutil. Zainstaluj: pip install psutil")

try:
    from pynput.keyboard import Controller, Key
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("[Actions] UWAGA: Brak biblioteki pynput. Zainstaluj: pip install pynput")


def _is_teams_running() -> bool:
    """Sprawdza czy Microsoft Teams jest uruchomiony."""
    if not PSUTIL_AVAILABLE:
        # Jeśli brak psutil, zakładamy że Teams działa (fallback)
        return True
    
    teams_process_names = {"teams.exe", "ms-teams.exe"}
    
    try:
        print("Lista uruchomionych aplikacji:")
        print([proc.info['name'] for proc in psutil.process_iter(['name'])])
        for proc in psutil.process_iter(['name']):
            proc_name = proc.info['name']
            if proc_name and proc_name.lower() in teams_process_names:
                return True
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass
    
    return False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("actions")


@dataclass
class GestureAction:
    """Definicja akcji dla gestu."""
    name: str
    keys: tuple  # Klawisze do naciśnięcia
    description: str
    cooldown: float = 2.0  # Minimalny czas między wywołaniami (sekundy)


class GestureActionHandler:
    """
    Obsługuje wykonywanie akcji na podstawie wykrytych gestów.

    Cechy:
    - Debouncing (cooldown między akcjami)
    - Minimalna pewność do wywołania akcji
    - Logowanie wykonanych akcji
    """

    # Domyślne mapowanie gestów na skróty klawiszowe
    DEFAULT_ACTIONS = {
        # Stop recording → Wyciszenie mikrofonu (Ctrl+Shift+M - Teams/Zoom)
        "stop_recording_sign": GestureAction(
            name="Wycisz mikrofon",
            keys=(Key.ctrl, Key.shift, 'm'),
            description="Wyciszenie/włączenie mikrofonu (Teams/Zoom)",
            cooldown=2.0
        ),

        # Continue recording → Włączenie mikrofonu (ten sam skrót - toggle)
        "continue_recording_sign": GestureAction(
            name="Włącz mikrofon",
            keys=(Key.ctrl, Key.shift, 'm'),
            description="Włączenie mikrofonu (Teams/Zoom)",
            cooldown=2.0
        ),

        # Nail biting → Powiadomienie (Win+A otwiera centrum powiadomień)
        # Alternatywnie możemy użyć własnego powiadomienia
        "nail_biting_sign": GestureAction(
            name="Alert - obgryzanie paznokci",
            keys=None,  # Specjalna obsługa - powiadomienie zamiast klawiszy
            description="Powiadomienie o wykryciu obgryzania paznokci",
            cooldown=5.0  # Dłuższy cooldown żeby nie spamować
        ),
    }

    def __init__(
        self,
        min_confidence: float = 0.7,
        enabled: bool = True,
        custom_actions: Optional[dict] = None
    ):
        """
        Inicjalizuje handler akcji.

        Args:
            min_confidence: Minimalna pewność do wywołania akcji
            enabled: Czy akcje są włączone
            custom_actions: Własne mapowanie gestów (nadpisuje domyślne)
        """
        self.enabled = enabled and PYNPUT_AVAILABLE
        self.min_confidence = min_confidence
        self.actions = custom_actions or self.DEFAULT_ACTIONS.copy()

        # Czas ostatniego wywołania dla każdego gestu (debouncing)
        self._last_triggered: dict[str, float] = {}

        # Callback dla powiadomień (można nadpisać)
        self.on_notification: Optional[Callable[[str, str], None]] = None

        if self.enabled:
            self._keyboard = Controller()
            logger.info("GestureActionHandler zainicjalizowany")
        else:
            self._keyboard = None
            logger.warning("GestureActionHandler wyłączony (brak pynput lub enabled=False)")

    def _can_trigger(self, gesture: str) -> bool:
        """Sprawdza czy można wywołać akcję (debouncing)."""
        if gesture not in self.actions:
            return False

        action = self.actions[gesture]
        last_time = self._last_triggered.get(gesture, 0)

        return (time.time() - last_time) >= action.cooldown

    def _press_keys(self, keys: tuple):
        """Naciska kombinację klawiszy."""
        if not self._keyboard or not keys:
            return

        # Naciśnij wszystkie modyfikatory
        modifiers = [k for k in keys[:-1]]
        key = keys[-1]

        try:
            for mod in modifiers:
                self._keyboard.press(mod)

            self._keyboard.press(key)
            self._keyboard.release(key)

            for mod in reversed(modifiers):
                self._keyboard.release(mod)

        except Exception as e:
            logger.error(f"Błąd przy naciskaniu klawiszy: {e}")

    def _show_notification(self, title: str, message: str):
        """Pokazuje powiadomienie systemowe (Windows 10 toast)."""
        # Użyj callbacka jeśli ustawiony
        if self.on_notification:
            self.on_notification(title, message)
            return

        # Windows toast notification przez winotify (stabilniejsza niż win10toast)
        try:
            from winotify import Notification
            toast = Notification(
                app_id="Gesture Shortcuts",
                title=title,
                msg=message,
                duration="short"
            )
            toast.show()
        except ImportError:
            # Ostateczny fallback - print
            logger.warning(f"POWIADOMIENIE: {title} - {message}")
            print(f"\n🔔 {title}: {message}\n")

    def trigger(self, gesture: str, confidence: float) -> bool:
        """
        Próbuje wywołać akcję dla gestu.

        Args:
            gesture: Nazwa wykrytego gestu
            confidence: Pewność detekcji (0.0 - 1.0)

        Returns:
            True jeśli akcja została wywołana, False w przeciwnym razie
        """
        if not self.enabled:
            return False

        if confidence < self.min_confidence:
            return False

        if gesture not in self.actions:
            return False

        if not self._can_trigger(gesture):
            return False

        action = self.actions[gesture]

        # Specjalna obsługa dla nail_biting (powiadomienie)
        if gesture == "nail_biting_sign":
            self._last_triggered[gesture] = time.time()
            self._show_notification(
                "Wykryto obgryzanie paznokci!",
                "Przestań obgryzać paznokcie 😤"
            )
            logger.info(f"Akcja: {action.name}")
            return True

        # Gesty związane z mikrofonem - wymagają uruchomionego Teams
        if gesture in ("stop_recording_sign", "continue_recording_sign"):
            if not _is_teams_running():
                logger.debug(f"Pominięto akcję '{action.name}' - Teams nie jest uruchomiony")
                return False

        self._last_triggered[gesture] = time.time()

        # Standardowa obsługa - skrót klawiszowy
        if action.keys:
            self._press_keys(action.keys)
            logger.info(f"Akcja: {action.name} | Klawisze: {action.keys}")
            return True

        return False

    def get_action_info(self, gesture: str) -> Optional[str]:
        """Zwraca opis akcji dla gestu."""
        if gesture in self.actions:
            return self.actions[gesture].description
        return None