import os
import time
import threading
import cv2
import numpy as np


class LidarMapper:
    def __init__(self, map_size_px: int = 600, max_range_m: float = 6.0):
        self.map_size_px = map_size_px
        self.max_range_m = max_range_m
        self.scale = (map_size_px / 2.0) / max_range_m  # pixels per meter

        self._lock = threading.Lock()
        self.latest_points = np.empty((0, 3), dtype=np.float32)
        self.last_update = 0.0

    def update_from_message(self, message: dict):
        """Callback for RTC_TOPIC['ULIDAR_ARRAY'] messages."""
        try:
            payload = message.get("data", {})
            if not isinstance(payload, dict):
                return

            origin = np.array(payload.get("origin", [0.0, 0.0, 0.0]), dtype=np.float32)
            resolution = float(payload.get("resolution", 0.05))
            decoded = payload.get("data", {})

            if not isinstance(decoded, dict):
                return

            raw = None
            if "positions" in decoded and decoded["positions"] is not None:
                raw = np.asarray(decoded["positions"], dtype=np.float32)
            elif "points" in decoded and decoded["points"] is not None:
                raw = np.asarray(decoded["points"], dtype=np.float32)

            if raw is None or raw.size < 3:
                return

            # Ensure clean (N, 3) shape
            n_points = raw.size // 3
            points = raw[: n_points * 3].reshape(-1, 3)

            # LibVoxelDecoder outputs voxel grid indices; convert to meters if needed
            if "positions" in decoded or np.max(np.abs(points)) > 40.0:
                points = points * resolution + origin

            with self._lock:
                self.latest_points = points
                self.last_update = time.time()
        except Exception as e:
            print(f"[LiDAR Parse Error] {e}")

    def render_2d_map(self) -> np.ndarray:
        """Renders a 2D top-down occupancy view centered on the robot."""
        size = self.map_size_px
        center = size // 2
        canvas = np.zeros((size, size, 3), dtype=np.uint8)

        # Draw 1-meter range rings and crosshairs
        for r_m in range(1, int(self.max_range_m) + 1):
            radius_px = int(r_m * self.scale)
            cv2.circle(canvas, (center, center), radius_px, (40, 40, 40), 1)
            cv2.putText(
                canvas, f"{r_m}m", (center + 4, center - radius_px + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1
            )

        cv2.line(canvas, (center, 0), (center, size), (40, 40, 40), 1)
        cv2.line(canvas, (0, center), (size, center), (40, 40, 40), 1)

        with self._lock:
            pts = self.latest_points.copy()

        if pts.shape[0] > 0:
            # Filter out floor noise (< 0.10m) and high ceiling (> 1.8m)
            mask = (pts[:, 2] >= 0.10) & (pts[:, 2] <= 1.80)
            filtered = pts[mask] if np.any(mask) else pts

            # Center around median/origin so robot stays in view
            # Go2 frame: +X is forward (up on screen), +Y is left (left on screen)
            px = (center - filtered[:, 1] * self.scale).astype(np.int32)
            py = (center - filtered[:, 0] * self.scale).astype(np.int32)

            in_bounds = (px >= 0) & (px < size) & (py >= 0) & (py < size)
            px, py = px[in_bounds], py[in_bounds]

            # Color points bright cyan/green
            canvas[py, px] = (0, 255, 180)

        # Draw robot marker at center (triangle pointing forward/up)
        robot_tri = np.array([
            [center, center - 10],
            [center - 6, center + 6],
            [center + 6, center + 6],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [robot_tri], (0, 120, 255))

        # HUD info
        cv2.putText(
            canvas,
            f"LiDAR Points: {pts.shape[0]} | 'M': Save Map | 'L': Close",
            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        return canvas

    def save_map(self, output_dir: str = "maps") -> str:
        """Saves both the 2D PNG map and the 3D PLY point cloud."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        png_path = os.path.join(output_dir, f"lidar_map_{timestamp}.png")
        ply_path = os.path.join(output_dir, f"lidar_cloud_{timestamp}.ply")

        # Save 2D image
        cv2.imwrite(png_path, self.render_2d_map())

        # Save 3D point cloud (.ply can be opened in MeshLab / CloudCompare / Open3D)
        with self._lock:
            pts = self.latest_points.copy()

        with open(ply_path, "w", encoding="utf-8") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {pts.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("end_header\n")
            for p in pts:
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")

        print(f"💾 [LiDAR] Saved {pts.shape[0]} points to '{ply_path}' and '{png_path}'")
        return ply_path