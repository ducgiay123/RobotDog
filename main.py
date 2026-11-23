import asyncio
import logging
import threading
import time
import cv2
import numpy as np

from lidar_mapper import LidarMapper
from robot_controller import Go2Controller
from keyboard_controller import KeyboardController
from voice_controller import VoiceController

logging.basicConfig(level=logging.FATAL)

# --- CONFIGURATION ---
MODEL_DIR = "./models/sherpa-onnx-streaming-zipformer-en-2023-06-21"
COMMANDS_CSV = "commands.csv"


def main():
    # 1. Initialize Controllers
    # 1. Initialize Controllers
    robot = Go2Controller()
    voice = VoiceController(MODEL_DIR, COMMANDS_CSV)
    lidar = LidarMapper(map_size_px=600, max_range_m=6.0)
    lidar_window_open = False

    voice_active_event = threading.Event()
    shutdown_event = threading.Event()

    def handle_mode_change(new_mode: str):
        if new_mode == "voice":
            voice_active_event.set()
        else:
            voice_active_event.clear()

    kb = KeyboardController(on_mode_change=handle_mode_change)
    kb.start()

    # 2. Workflow 1: Manual Keyboard Loop
    async def manual_control_loop():
        while not shutdown_event.is_set():
            if kb.mode != "manual":
                # Auto-disable LiDAR when leaving Manual Mode to save bandwidth
                if robot.lidar_active:
                    await robot.disable_lidar()
                await asyncio.sleep(0.1)
                continue

            try:
                # Handle 'L' key: Toggle LiDAR Map ON/OFF
                if kb.lidar_toggle_requested:
                    kb.lidar_toggle_requested = False
                    if robot.lidar_active:
                        await robot.disable_lidar()
                    else:
                        await robot.enable_lidar(lidar.update_from_message)

                # Handle 'M' key: Save LiDAR Map snapshot (.png + .ply)
                if kb.save_map_requested:
                    kb.save_map_requested = False
                    if robot.lidar_active:
                        lidar.save_map()
                    else:
                        print("⚠️ [LiDAR] Press 'L' to enable LiDAR before saving.")

                # Handle Spacebar Stand / Sit toggle
                if kb.stand_toggle_requested:
                    kb.stand_toggle_requested = False
                    if robot.is_standing:
                        await robot.sit_down()
                    else:
                        await robot.stand_up()
                    await asyncio.sleep(2.0)
                    continue

                # Handle WASD movement (only while standing)
                if robot.is_standing:
                    await robot.send_move(
                        kb.move_state["x"], kb.move_state["y"], kb.move_state["yaw"]
                    )

                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"[Manual Loop Error] {e}")
                break

    # 3. Workflow 2: Voice Recognition Loop (CSV-driven)
    async def voice_control_loop():
        while not shutdown_event.is_set():
            if not voice_active_event.is_set():
                await asyncio.sleep(0.1)
                continue

            # Ensure robot is stopped when entering voice listening state
            await robot.stop()

            cmd = await asyncio.to_thread(voice.listen_for_command, voice_active_event)
            if not cmd or not voice_active_event.is_set():
                continue

            action = cmd["action"]
            try:
                if action == "stand":
                    await robot.stand_up()
                elif action == "sit":
                    await robot.sit_down()
                elif action == "damp":
                    await robot.soft_damp()
                elif action == "stop":
                    await robot.stop()
                elif action == "dance1":
                    if not robot.is_standing:
                        print("⚠️ [Voice] Robot is not standing. Say 'stand up' first.")
                    else:
                        await robot.dance1()
                elif action == "move":
                    if not robot.is_standing:
                        print("⚠️ [Voice] Robot is not standing. Say 'stand up' first.")
                    else:
                        dist_info = f"{cmd['distance']}m, " if "distance" in cmd else ""
                        print(f"⚙️ [Voice] Moving ({dist_info}x={cmd['x']}, z={cmd['z']}) for {cmd['duration']}s")
                        await robot.move_for_duration(
                            cmd["x"],
                            cmd["y"],
                            cmd["z"],
                            cmd["duration"],
                            cancel_check=lambda: not voice_active_event.is_set(),
                        )
                elif action == "quit":
                    print("🛑 [Voice] Quit command received. Shutting down...")
                    shutdown_event.set()
                    break
            except Exception as e:
                print(f"[Voice Execution Error] {e}")

    # 4. Asyncio Background Thread
    def run_asyncio_loop(loop):
        asyncio.set_event_loop(loop)

        async def setup():
            connected = await robot.connect()
            if not connected:
                shutdown_event.set()
                return

            await robot.init_normal_mode()
            print("\n🎮 System Ready!")
            print(" -> P     : Switch between MANUAL and VOICE mode")
            print(" -> WASD  : Move (Manual Mode)")
            print(" -> SPACE : Stand / Sit smoothly (Manual Mode)")
            print(" -> Q     : Quit (Click Video window first)\n")

            loop.create_task(manual_control_loop())
            loop.create_task(voice_control_loop())

        loop.run_until_complete(setup())
        loop.run_forever()

    loop = asyncio.new_event_loop()
    asyncio_thread = threading.Thread(target=run_asyncio_loop, args=(loop,), daemon=True)
    asyncio_thread.start()

    # 5. Main Thread: OpenCV Video UI with Mode HUD
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.imshow("Go2 Control", img)
    cv2.waitKey(1)

    try:
        while not shutdown_event.is_set():
            if not robot.frame_queue.empty():
                frame = robot.frame_queue.get()

                # Draw current mode banner on video feed
                mode_text = f"MODE: {kb.mode.upper()}  |  Press 'P' to switch  |  'Q' to quit"
                color = (0, 255, 0) if kb.mode == "manual" else (0, 165, 255)
                cv2.putText(frame, mode_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                cv2.imshow("Go2 Control", frame)
                # Render or close the LiDAR Map window
                if robot.lidar_active and kb.mode == "manual":
                    cv2.imshow("Go2 LiDAR Map", lidar.render_2d_map())
                    lidar_window_open = True
                elif lidar_window_open:
                    try:
                        cv2.destroyWindow("Go2 LiDAR Map")
                    except cv2.error:
                        pass
                    lidar_window_open = False
                    
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Shutting down...")
                shutdown_event.set()
                break

            if robot.frame_queue.empty():
                time.sleep(0.01)
    finally:
        shutdown_event.set()
        voice_active_event.clear()
        kb.stop()
        voice.close()
        cv2.destroyAllWindows()

        # Clean WebRTC disconnect inside the running asyncio loop
        if loop.is_running():
            future = asyncio.run_coroutine_threadsafe(robot.disconnect(), loop)
            try:
                future.result(timeout=3)
            except Exception:
                pass
            loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join(timeout=3)


if __name__ == "__main__":
    main()