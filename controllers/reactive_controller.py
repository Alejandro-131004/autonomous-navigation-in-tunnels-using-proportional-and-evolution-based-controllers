import math
import numpy as np
from environment.configuration import MAX_VELOCITY

# Sector to ignore around the rear (prevents following the start barrier)
BACK_IGNORE_DEG = 80.0
SIDE_WINDOW_DEG = 30.0  # window used to assess left/right proximity when in full FOV


def _clip_idx(i: int, n: int) -> int:
    return max(0, min(n - 1, i))


def reactive_controller_logic(dist_values, direction: int = 1, fov_mode: str = 'full') -> tuple[float, float]:
    """
    Subsumption-style reactive controller (TP2 Ex.4):
      ω = dir*distP*(dist_min - wallDist) + angleP*(angle_min - dir*pi/2)
    Behaviours:
      - TURN / SLOW / CRUISE for wall-following (when something is seen)
      - WANDER when nothing is seen
    Side policy:
      - 'left'  → follow left wall (dir=+1)
      - 'right' → follow right wall (dir=-1)
      - 'full'  → choose side dynamically (closest side in a ±30° window around ±90°)
    Extra:
      - Ignores a rear sector so the robot doesn't try to follow the start barrier.
    """

    # Gains and target distance
    maxSpeed = MAX_VELOCITY
    distP = 10.0
    angleP = 7.0
    wallDist = 0.3

    # --- Safety & normalization
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

    # --- Build a masked copy according to FOV and rear-ignoring
    masked = finite.copy()

    # FOV mask
    if fov_mode == 'left':
        masked[:half] = np.inf
        direction = 1
    elif fov_mode == 'right':
        masked[half:] = np.inf
        direction = -1
    # else: full view → direction may change below

    # Rear mask (avoid start barrier)
    rear_clip = max(1, int(BACK_IGNORE_DEG / 360.0 * N))
    masked[:rear_clip] = np.inf
    masked[-rear_clip:] = np.inf

    # --- If no obstacle is visible → WANDER (exploration layer)
    if not np.any(np.isfinite(masked)):
        linear_vel = maxSpeed
        angular_vel = float(np.random.normal(loc=0.0, scale=1.0))
        return float(linear_vel), float(angular_vel)

    # --- Side decision (A layer). Only dynamic in 'full'
    angle_increment = 2.0 * math.pi / max(N - 1, 1)
    side_win = max(1, int(SIDE_WINDOW_DEG * math.pi / 180.0 / angle_increment))

    if fov_mode == 'full':
        # Evaluate proximity around ±90°
        left_center = _clip_idx(int(half - (math.pi/2.0) / angle_increment), N)
        right_center = _clip_idx(int(half + (math.pi/2.0) / angle_increment), N)

        left_slice = masked[_clip_idx(left_center - side_win, N): _clip_idx(left_center + side_win + 1, N)]
        right_slice = masked[_clip_idx(right_center - side_win, N): _clip_idx(right_center + side_win + 1, N)]

        left_min = np.nanmin(left_slice) if left_slice.size and np.any(np.isfinite(left_slice)) else np.inf
        right_min = np.nanmin(right_slice) if right_slice.size and np.any(np.isfinite(right_slice)) else np.inf

        direction = -1 if right_min < left_min else 1  # -1:right, +1:left

    # --- Find closest point in (masked) view
    min_index = int(np.nanargmin(masked))
    angleMin = (half - min_index) * angle_increment
    distMin = masked[min_index]

    # --- Distances for behaviour selection (use unmasked values)
    distFront = finite[half]
    # Side ray near ±90° from front (robust to indexing conventions)
    side_idx = _clip_idx(int(half - (math.pi/2.0) / angle_increment), N) if direction == 1 \
        else _clip_idx(int(half + (math.pi/2.0) / angle_increment), N)
    distSide = finite[side_idx]
    distBack = finite[0]

    # --- Optional unblock for tight corners (exercise 4b improvement)
    if distFront < 1.25 * wallDist and (distSide < 1.25 * wallDist or distBack < 1.25 * wallDist):
        angular_vel = -direction  # turn away from jam
    else:
        # Proportional control on distance + angle (exactly as in the sheet)
        angular_vel = direction * distP * (distMin - wallDist) + angleP * (angleMin - direction * math.pi / 2.0)

    # --- Linear velocity per table (TURN / SLOW / CRUISE)
    if distFront < wallDist:
        linear_vel = 0.0
    elif (distFront < 2.0 * wallDist) or (distMin < 0.75 * wallDist) or (distMin > 1.25 * wallDist):
        linear_vel = 0.5 * maxSpeed
    else:
        linear_vel = maxSpeed

    return float(linear_vel), float(angular_vel)
