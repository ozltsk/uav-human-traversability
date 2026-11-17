"""Аварійне керування флотом: спільний прапорець переривання, посадка всіх,
аварійна зупинка двигунів і клавіатура оператора.

Клавіші (у вікні консолі, де запущено місію):
  ПРОБІЛ або L -- посадка всіх дронів у повітрі;
  E двічі за 2 с -- аварійна зупинка двигунів усіх дронів (падають!);
  Q -- перервати місії, дрони сідають штатно.
Резерв на випадок, коли головний процес завис: в іншому вікні
`python app.py panic` (посадка) або `python app.py panic --emergency`.
"""
import threading
import time


class MissionAbort(RuntimeError):
    """Місію перервано оператором або монітором безпеки."""


class Safety:
    def __init__(self):
        self.abort = threading.Event()
        self.reason = ""
        self.drones = []
        self._kb_thread = None
        self._last_e = 0.0

    def register(self, drone):
        self.drones.append(drone)

    def trigger(self, reason):
        if not self.abort.is_set():
            self.reason = reason
            self.abort.set()
            print(f"\n!!! ПЕРЕРИВАННЯ: {reason}")

    def check(self):
        if self.abort.is_set():
            raise MissionAbort(self.reason)

    def land_all(self, reason="посадка всіх (оператор)"):
        self.trigger(reason)
        for d in self.drones:
            if d.airborne:
                d.send_nowait("stop")
        time.sleep(0.1)
        for d in self.drones:
            if d.airborne:
                d.send_nowait("land")

    def emergency_all(self):
        self.trigger("аварійна зупинка двигунів (оператор)")
        for d in self.drones:
            d.emergency()

    # ------------------------------------------------------------ клавіатура
    def start_keyboard(self):
        try:
            import msvcrt  # noqa: F401  (лише Windows)
        except ImportError:
            print("Клавіатурне керування недоступне (не Windows): аварійна посадка -- `python app.py panic`")
            return
        self._kb_thread = threading.Thread(target=self._kb_loop, daemon=True)
        self._kb_thread.start()
        print("Аварійні клавіші: ПРОБІЛ/L -- посадка всіх, E E -- зупинка двигунів, Q -- перервати місію")

    def _kb_loop(self):
        import msvcrt
        while True:
            try:
                if not msvcrt.kbhit():
                    time.sleep(0.05)
                    continue
                ch = msvcrt.getwch().lower()
            except Exception:
                return
            if ch in (" ", "l", "д"):
                self.land_all()
            elif ch in ("q", "й"):
                self.trigger("місію перервано оператором (Q)")
            elif ch in ("e", "у"):
                now = time.time()
                if now - self._last_e < 2.0:
                    self.emergency_all()
                else:
                    print("\nЩе раз E протягом 2 с -- аварійна зупинка двигунів")
                self._last_e = now
