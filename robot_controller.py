import asyncio
import logging
from queue import Queue
from aiortc import MediaStreamTrack
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD


class Go2Controller:
    def __init__(self, method=WebRTCConnectionMethod.LocalAP):
        self.conn = UnitreeWebRTCConnection(method)
        self.frame_queue = Queue(maxsize=2)
        self.data_api = None
        self.is_standing = True
        self.lidar_active = False
    async def _recv_camera_stream(self, track: MediaStreamTrack):
        while True:
            try:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put(img)
            except Exception:
                break

    async def connect(self) -> bool:
        print("[Robot] Initiating WebRTC Handshake (AP Mode)...")
        await self.conn.connect()

        # Enable live video feed
        self.conn.video.switchVideoChannel(True)
        self.conn.video.add_track_callback(self._recv_camera_stream)

        # Wait for Data Channel initialization
        print("[Robot] Waiting for Data Channel...")
        for _ in range(20):
            if hasattr(self.conn, "datachannel") and self.conn.datachannel:
                self.data_api = self.conn.datachannel.pub_sub
                break
            await asyncio.sleep(0.5)

        if not self.data_api:
            print("❌ [Robot] Data channel failed to initialize.")
            return False

        print("✅ [Robot] Connected and Data Channel ready!")
        return True

    async def init_normal_mode(self):
        print("[Robot] Switching to normal mode and standing up...")
        await self.data_api.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"],
            {"api_id": 1002, "parameter": {"name": "normal"}},
        )
        await asyncio.sleep(4)
        self.is_standing = True

    async def stand_up(self):
        print("🧍 [Robot] Standing up...")
        await self.data_api.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"],
            {"api_id": 1002, "parameter": {"name": "normal"}},
        )
        await asyncio.sleep(1)
        await self.data_api.publish_request_new(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": SPORT_CMD["RecoveryStand"]},
        )
        self.is_standing = True

    async def sit_down(self):
        print("⬇️ [Robot] Sitting down smoothly (Motors ON)...")
        await self.data_api.publish_request_new(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": SPORT_CMD["StandDown"]},
        )
        self.is_standing = False

    async def soft_damp(self):
        print("🛑 [Robot] Executing Soft Damp (StandDown -> Damp)...")
        await self.data_api.publish_request_new(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": SPORT_CMD["StandDown"]},
        )
        await asyncio.sleep(2.5)
        await self.data_api.publish_request_new(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": SPORT_CMD.get("Damp", 13)},
        )
        self.is_standing = False

    async def dance1(self):
        print("💃 [Robot] Performing Dance1...")
        await self.data_api.publish_request_new(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": SPORT_CMD.get("Dance1", 1022)},
        )

    async def send_move(self, x: float, y: float, z: float):
        if not self.data_api:
            return
        await self.data_api.publish_request_new(
            RTC_TOPIC["SPORT_MOD"],
            {
                "api_id": SPORT_CMD["Move"],
                "parameter": {"x": x, "y": y, "z": z},
            },
        )

    async def stop(self):
        await self.send_move(0.0, 0.0, 0.0)

    async def move_for_duration(self, x: float, y: float, z: float, duration: float, cancel_check=None):
        """Pulses velocity commands at 10Hz for a set duration, with optional early cancel."""
        steps = max(1, int(duration / 0.1))
        for _ in range(steps):
            if cancel_check and cancel_check():
                break
            await self.send_move(x, y, z)
            await asyncio.sleep(0.1)
        await self.stop()
    async def enable_lidar(self, callback):
        if not self.data_api or self.lidar_active:
            return
        print("📡 [Robot] Enabling LiDAR stream...")
        # Required by Unitree WebRTC to allow high-bandwidth LiDAR packets
        if hasattr(self.conn.datachannel, "disableTrafficSaving"):
            await self.conn.datachannel.disableTrafficSaving(True)

        # Turn on the LiDAR sensor topic and subscribe to voxel map
        switch_topic = RTC_TOPIC.get("ULIDAR_SWITCH", "rt/utlidar/switch")
        array_topic = RTC_TOPIC.get("ULIDAR_ARRAY", "rt/utlidar/voxel_map_compressed")

        self.data_api.publish_without_callback(switch_topic, "on")
        self.data_api.subscribe(array_topic, callback)
        self.lidar_active = True
        print("✅ [Robot] LiDAR stream active!")

    async def disable_lidar(self):
        if not self.data_api or not self.lidar_active:
            return
        print("📴 [Robot] Disabling LiDAR stream...")
        switch_topic = RTC_TOPIC.get("ULIDAR_SWITCH", "rt/utlidar/switch")
        array_topic = RTC_TOPIC.get("ULIDAR_ARRAY", "rt/utlidar/voxel_map_compressed")

        if hasattr(self.data_api, "unsubscribe"):
            self.data_api.unsubscribe(array_topic)
        self.data_api.publish_without_callback(switch_topic, "off")
        self.lidar_active = False
        
    async def disconnect(self):
        try:
            await self.disable_lidar()
            await self.stop()
            if hasattr(self.conn, "disconnect"):
                await self.conn.disconnect()
            elif hasattr(self.conn, "pc") and self.conn.pc:
                await self.conn.pc.close()
        except Exception as e:
            logging.error(f"Disconnect error: {e}")