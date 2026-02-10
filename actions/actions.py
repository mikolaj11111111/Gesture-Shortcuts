"""
Actions module - executing keyboard shortcuts based on detected gestures.

Uses pynput to simulate keystrokes.
Contains debouncing to avoid triggering the same action multiple times.
"""

import time
import logging
import threading
from typing import Callable, Optional
from dataclasses import dataclass

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("[Actions] WARNING: psutil library not found. Install: pip install psutil")

try:
    from pynput.keyboard import Controller, Key
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("[Actions] WARNING: pynput library not found. Install: pip install pynput")

_have_printed = False  # Module-level variable (without global)

def _is_teams_running() -> bool:
    """Checks if Microsoft Teams is running."""
    global _have_printed  # Declaration that we're using a global variable
    
    if not PSUTIL_AVAILABLE:
        # If psutil is not available, assume Teams is running (fallback)
        return True
    
    teams_process_names = {"teams.exe", "ms-teams.exe"}
    
    try:
        if not _have_printed:
            print("List of running applications:")
            print([proc.info['name'] for proc in psutil.process_iter(['name'])])
            _have_printed = True  # Set to True after first print
        
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
    """Definition of an action for a gesture."""
    name: str
    keys: tuple  # Keys to press
    description: str
    cooldown: float = 2.0  # Minimum time between triggers (seconds)


class GestureActionHandler:
    """
    Handles executing actions based on detected gestures.

    Features:
    - Debouncing (cooldown between actions)
    - Minimum confidence to trigger action
    - Logging of executed actions
    """

    # Default mapping of gestures to keyboard shortcuts
    DEFAULT_ACTIONS = {
        # Stop recording → Mute microphone (Ctrl+Shift+M - Teams/Zoom)
        "stop_recording_sign": GestureAction(
            name="Mute microphone",
            keys=(Key.ctrl, Key.shift, 'm'),
            description="Mute/unmute microphone (Teams/Zoom)",
            cooldown=2.0
        ),

        # Continue recording → Enable microphone (same shortcut - toggle)
        "continue_recording_sign": GestureAction(
            name="Start recording",
            keys=(Key.cmd, 'h'),
            description="Enable recording (Teams/Zoom)",
            cooldown=2.0
        ),

        # Nail biting → Notification (Win+A opens notification center)
        # Alternatively we can use a custom notification
        "nail_biting_sign": GestureAction(
            name="Alert - nail biting",
            keys=None,  # Special handling - notification instead of keys
            description="Notification about nail biting detection",
            cooldown=5.0  # Longer cooldown to avoid spamming
        ),
    }

    def __init__(
        self,
        min_confidence: float = 0.7,
        enabled: bool = True,
        custom_actions: Optional[dict] = None
    ):
        """
        Initializes the action handler.

        Args:
            min_confidence: Minimum confidence to trigger action
            enabled: Whether actions are enabled
            custom_actions: Custom gesture mapping (overrides defaults)
        """
        self.enabled = enabled and PYNPUT_AVAILABLE
        self.min_confidence = min_confidence
        self.actions = custom_actions or self.DEFAULT_ACTIONS.copy()

        # Time of last trigger for each gesture (debouncing)
        self._last_triggered: dict[str, float] = {}

        # Callback for notifications (can be overridden)
        self.on_notification: Optional[Callable[[str, str], None]] = None

        if self.enabled:
            self._keyboard = Controller()
            logger.info("GestureActionHandler initialized")
        else:
            self._keyboard = None
            logger.warning("GestureActionHandler disabled (missing pynput or enabled=False)")

    def _can_trigger(self, gesture: str) -> bool:
        """Checks if action can be triggered (debouncing)."""
        if gesture not in self.actions:
            return False

        action = self.actions[gesture]
        last_time = self._last_triggered.get(gesture, 0)

        return (time.time() - last_time) >= action.cooldown

    def _press_keys(self, keys: tuple):
        """Presses a key combination."""
        if not self._keyboard or not keys:
            return

        # Press all modifiers
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
            logger.error(f"Error pressing keys: {e}")

    def _show_notification(self, title: str, message: str):
        """Shows a system notification (Windows 10 toast)."""
        # Use callback if set
        if self.on_notification:
            self.on_notification(title, message)
            return

        # Windows toast notification via winotify (more stable than win10toast)
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
            # Final fallback - print
            logger.warning(f"NOTIFICATION: {title} - {message}")
            print(f"\n🔔 {title}: {message}\n")

    def trigger(self, gesture: str, confidence: float) -> bool:
        """
        Attempts to trigger an action for a gesture.

        Args:
            gesture: Name of the detected gesture
            confidence: Detection confidence (0.0 - 1.0)

        Returns:
            True if action was triggered, False otherwise
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

        # Special handling for nail_biting (notification)
        if gesture == "nail_biting_sign":
            self._last_triggered[gesture] = time.time()
            self._show_notification(
                "Nail biting detected!",
                "Stop biting your nails 😤"
            )
            logger.info(f"Action: {action.name}")
            return True

        # Microphone-related gestures - require Teams to be running
        if gesture == "stop_recording_sign":
            if not _is_teams_running():
                logger.debug(f"Skipped action '{action.name}' - Teams is not running")
                return False

        self._last_triggered[gesture] = time.time()

        # Special handling for continue_recording_sign (Win+H) - give time to click in target window
        if gesture == "continue_recording_sign":
            logger.info("🎤 Recording in 2 seconds - click where you want to type!")
            print("\n🎤 Recording in 2 seconds - click where you want to type!\n")
            # Use Timer to not block the main thread
            timer = threading.Timer(2.0, self._press_keys, args=[action.keys])
            timer.start()
            return True

        # Standard handling - keyboard shortcut
        if action.keys:
            self._press_keys(action.keys)
            logger.info(f"Action: {action.name} | Keys: {action.keys}")
            return True

        return False

    def get_action_info(self, gesture: str) -> Optional[str]:
        """Returns description of action for a gesture."""
        if gesture in self.actions:
            return self.actions[gesture].description
        return None