import math
import numpy as np
from environment.configuration import MAX_VELOCITY

BACK_IGNORE_DEG = 80.0
MAX_DIST = 5.0          # clamp lidar distance [m]
MAX_ANGULAR = 6.0       # rad/s safety clamp


def _clip_idx(i: int, n: int) -> int:
    return max(0, min(n - 1, i))


def reactive_controller_logic(dist_values, direction: int = 1, fov_mode: str = 'full') -> tuple[float, float]:
    """
    TP2-style reactive controller (stable, NaN/overflow-safe):
      ω = dir*distP*(dist_min - wallDist) + angleP*(angle_min - dir*pi/2)

    - FOV 'left'  -> follow left
    - FOV 'right' -> follow right
    - FOV 'full'  -> keep direction passed (no switching)
    """

    maxSpeed = MAX_VELOCITY
    distP = 10.0
    angleP = 7.0
    wallDist = 0.3

    # --- Input sanitation ---
    if dist_values is None:
        return 0.0, 0.0
    scan = np.asarray(dist_values, dtype=float)
    if scan.size == 0:
        return 0.0, 0.0

    finite = np.nan_to_num(scan, nan=np.inf, posinf=np.inf, neginf=0.0)
    N = finite.size
    if N < 3:
        return 0.0, 0.0
    half = N // 2

    masked = finite.copy()

    # --- FOV masking ---
    if fov_mode == "left":
        masked[:half] = np.inf
        direction = 1
    elif fov_mode == "right":
        masked[half:] = np.inf
        direction = -1
    # if full: keep given direction

    # --- Rear ignore ---
    rear_clip = max(1, int(BACK_IGNORE_DEG / 360.0 * N))
    masked[:rear_clip] = np.inf
    masked[-rear_clip:] = np.inf

    # --- No visible wall → wander ---
    if not np.any(np.isfinite(masked)):
        return maxSpeed, float(np.random.normal(0.0, 1.0))

    # --- Find closest point ---
    min_index = int(np.nanargmin(masked))
    angle_increment = 2.0 * math.pi / max(N - 1, 1)
    angleMin = (half - min_index) * angle_increment
    distMin = masked[min_index]

    # Clamp to safe numeric ranges
    if not np.isfinite(distMin):
        distMin = wallDist
    distMin = float(np.clip(distMin, 0.0, MAX_DIST))
    angleMin = float(np.clip(angleMin, -math.pi, math.pi))

    # --- Get reference distances ---
    distFront = finite[half] if np.isfinite(finite[half]) else np.inf
    side_idx = _clip_idx(int(half - (math.pi / 2.0) / angle_increment), N) if direction == 1 \
        else _clip_idx(int(half + (math.pi / 2.0) / angle_increment), N)
    distSide = finite[side_idx] if np.isfinite(finite[side_idx]) else np.inf
    distBack = finite[0] if np.isfinite(finite[0]) else np.inf

    # --- Control law ---
    if distFront < 1.25 * wallDist and (distSide < 1.25 * wallDist or distBack < 1.25 * wallDist):
        angular_vel = -direction
    else:
        angular_vel = (
            direction * distP * (distMin - wallDist)
            + angleP * (angleMin - direction * math.pi / 2.0)
        )

    # Clamp and sanitize angular velocity
    if not np.isfinite(angular_vel):
        angular_vel = 0.0
    angular_vel = float(np.clip(angular_vel, -MAX_ANGULAR, MAX_ANGULAR))

    # --- Linear velocity control ---
    if distFront < wallDist:
        linear_vel = 0.0
    elif (distFront < 2.0 * wallDist) or (distMin < 0.75 * wallDist) or (distMin > 1.25 * wallDist):
        linear_vel = 0.5 * maxSpeed
    else:
        linear_vel = maxSpeed

    if not np.isfinite(linear_vel):
        linear_vel = 0.0

    return float(linear_vel), float(angular_vel)
