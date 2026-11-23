import csv
import re
import time
import threading
import sherpa_onnx
import sounddevice as sd


WORD_TO_NUM = {
    "half": 0.5,
    "a": 1.0,
    "one": 1.0,
    "two": 2.0,
    "to": 2.0,    # Common ASR homophone for "two"
    "too": 2.0,
    "three": 3.0,
    "four": 4.0,
    "for": 4.0,   # Common ASR homophone for "four"
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
}


class VoiceController:
    def __init__(self, model_dir: str, csv_path: str = "commands.csv", max_distance: float = 5.0):
        print("[Voice] Loading commands from CSV...")
        self.commands = self._load_commands(csv_path)
        self.max_distance = max_distance  # Safety cap in meters

        print("[Voice] Initializing Sherpa-ONNX ASR...")
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=f"{model_dir}/tokens.txt",
            encoder=f"{model_dir}/encoder-epoch-99-avg-1.int8.onnx",
            decoder=f"{model_dir}/decoder-epoch-99-avg-1.int8.onnx",
            joiner=f"{model_dir}/joiner-epoch-99-avg-1.int8.onnx",
            num_threads=4,
            sample_rate=16000,
            feature_dim=80,
        )
        self.sample_rate = 16000
        self.samples_per_read = int(0.05 * self.sample_rate)

        self.mic_stream = sd.InputStream(
            channels=1,
            dtype="float32",
            samplerate=self.sample_rate,
        )
        self.mic_stream.start()

    def _load_commands(self, csv_path: str) -> list:
        commands = []
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                commands.append({
                    "phrase": row["phrase"].strip().lower(),
                    "action": row["action"].strip().lower(),
                    "x": float(row.get("x", 0.0)),
                    "y": float(row.get("y", 0.0)),
                    "z": float(row.get("z", 0.0)),
                    "duration": float(row.get("duration", 0.0)),
                })
        commands.sort(key=lambda c: len(c["phrase"]), reverse=True)
        print(f"[Voice] Loaded {len(commands)} commands from {csv_path}.")
        return commands

    def _parse_distance(self, text: str):
        """Extracts a float distance in meters from digits or spoken English words."""
        if not text:
            return None

        # 1. Check for numeric digits (e.g., "2", "1.5")
        digit_match = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
        if digit_match:
            return float(digit_match.group(1))

        words = re.findall(r"[a-z]+", text)

        # 2. Check for "X point Y" (e.g., "one point five" -> 1.5)
        if "point" in words:
            idx = words.index("point")
            if idx > 0 and idx + 1 < len(words):
                whole = WORD_TO_NUM.get(words[idx - 1])
                frac = WORD_TO_NUM.get(words[idx + 1])
                if whole is not None and frac is not None and frac < 10:
                    return whole + (frac / 10.0)

        # 3. Check for "X and a half" (e.g., "two and a half" -> 2.5)
        for i, w in enumerate(words):
            if w in WORD_TO_NUM and w != "a":
                val = WORD_TO_NUM[w]
                if i + 2 < len(words) and "half" in words[i + 1 : i + 4]:
                    val += 0.5
                return val

        # 4. Handle "half a meter" / "a meter"
        if "half" in words:
            return 0.5
        if "a" in words and ("meter" in words or "meters" in words):
            return 1.0

        return None

    def _build_move_command(self, base_cmd: dict, distance: float) -> dict:
        """Returns a copy of the command with duration scaled to the requested meters."""
        cmd = dict(base_cmd)
        if distance is not None and cmd["x"] != 0.0:
            clamped = min(max(distance, 0.2), self.max_distance)
            cmd["duration"] = round(clamped / abs(cmd["x"]), 2)
            cmd["distance"] = clamped
        else:
            cmd["distance"] = round(cmd["duration"] * abs(cmd["x"]), 2)
        return cmd

    def listen_for_command(self, active_event: threading.Event):
        stream = self.recognizer.create_stream()
        last_result = ""
        last_update_time = time.time()
        pending_move_cmd = None

        print("\n🟢 [Voice Mode] Listening... (Press 'P' for Manual Mode)")

        while active_event.is_set():
            samples, _ = self.mic_stream.read(self.samples_per_read)
            stream.accept_waveform(self.sample_rate, samples.reshape(-1))

            while self.recognizer.is_ready(stream):
                self.recognizer.decode_stream(stream)

            result = self.recognizer.get_result(stream).lower().strip()

            if result != last_result:
                last_result = result
                last_update_time = time.time()
                print(f"\r🗣️ You: {result}", end="", flush=True)

                for cmd in self.commands:
                    if cmd["phrase"] in result:
                        # Linear move: check for optional distance after the phrase
                        if cmd["action"] == "move" and cmd["x"] != 0.0:
                            pending_move_cmd = cmd
                            tail = result.split(cmd["phrase"], 1)[1].strip()
                            dist = self._parse_distance(tail)

                            # Fire immediately once the unit ("meter"/"meters") is spoken
                            if dist is not None and ("meter" in tail or "metre" in tail):
                                final_cmd = self._build_move_command(cmd, dist)
                                print(f"  ⚡ [Matched: {cmd['phrase']} | {final_cmd['distance']}m]")
                                return final_cmd
                            break
                        else:
                            # Instant trigger for non-linear commands (stand, damp, dance1, turn, etc.)
                            print(f"  ⚡ [Matched: {cmd['phrase']} -> {cmd['action']}]")
                            return dict(cmd)

            # If waiting for a number after "move forward/backward", fire after 0.7s pause
            if pending_move_cmd and (time.time() - last_update_time) > 0.7:
                tail = last_result.split(pending_move_cmd["phrase"], 1)[1].strip()
                dist = self._parse_distance(tail)
                final_cmd = self._build_move_command(pending_move_cmd, dist)
                print(f"  ⚡ [Matched: {pending_move_cmd['phrase']} | {final_cmd['distance']}m]")
                return final_cmd

            # Clear stale unrecognized speech after 1.0s of silence
            if result and not pending_move_cmd and (time.time() - last_update_time) > 1.0:
                print("\r[Cleared]                         ", end="", flush=True)
                stream = self.recognizer.create_stream()
                last_result = ""

        return None

    def close(self):
        if self.mic_stream:
            self.mic_stream.stop()
            self.mic_stream.close()