from pynput import keyboard


class KeyboardController:
    def __init__(self, on_mode_change=None):
        self.mode = "manual"
        self.on_mode_change = on_mode_change

        self.move_state = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        self.stand_toggle_requested = False
        self.lidar_toggle_requested = False
        self.save_map_requested = False
        self._listener = None

    def _reset_movement(self):
        self.move_state["x"] = 0.0
        self.move_state["y"] = 0.0
        self.move_state["yaw"] = 0.0

    def _on_press(self, key):
        if hasattr(key, "char") and key.char:
            char = key.char.lower()

            # Switch between Manual and Voice mode with 'P'
            if char == "p":
                self._reset_movement()
                self.mode = "voice" if self.mode == "manual" else "manual"
                print(f"\n🔄 [Mode Switch] Active workflow: {self.mode.upper()}")
                if self.on_mode_change:
                    self.on_mode_change(self.mode)
                return

            # Only accept WASD driving in Manual mode
            if self.mode == "manual":
                if char == "w":
                    self.move_state["x"] = 0.5
                elif char == "s":
                    self.move_state["x"] = -0.5
                elif char == "a":
                    self.move_state["yaw"] = 0.5
                elif char == "d":
                    self.move_state["yaw"] = -0.5
                elif char == "l":
                    self.lidar_toggle_requested = True
                elif char == "m":
                    self.save_map_requested = True
        else:
            if key == keyboard.Key.space and self.mode == "manual":
                self.stand_toggle_requested = True

    def _on_release(self, key):
        if hasattr(key, "char") and key.char:
            char = key.char.lower()
            if char in ("w", "s"):
                self.move_state["x"] = 0.0
            elif char in ("a", "d"):
                self.move_state["yaw"] = 0.0

    def start(self):
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

    def stop(self):
        if self._listener:
            self._listener.stop()