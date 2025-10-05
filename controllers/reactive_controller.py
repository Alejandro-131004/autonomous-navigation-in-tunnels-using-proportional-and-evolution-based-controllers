import math
import numpy as np
from environment.configuration import MAX_VELOCITY

# Sector to ignore around the rear (prevents following the start barrier)
BACK_IGNORE_DEG = 80.0


def _clip_idx(i: int, n: int) -> int:
    return max(0, min(n - 1, i))


def reactive_controller_logic(dist_values, direction: int = 1, fov_mode: str = 'full') -> tuple[float, float]:
    """
    Reactive controller following TP2 sheet (wall following + wander):
      ω = dir*distP*(dist_min - wallDist) + angleP*(angle_min - dir*pi/2)

    Side policy:
      - 'left'  -> force left-wall following (dir=+1)
      - 'right' -> force right-wall following (dir=-1)
      - 'full'  -> keep whatever 'direction' is passed (default left)
    """

    # Gains and desired wall distance
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

    # FOV mask (full keeps caller’s direction)
    if fov_mode == 'left':
        masked[:half] = np.inf
        direction = 1
    elif fov_mode == 'right':
        masked[half:] = np.inf
        direction = -1
    # if fov_mode == 'full': do not override direction

    # Rear mask (avoid start barrier)
    rear_clip = max(1, int(BACK_IGNORE_DEG / 360.0 * N))
    masked[:rear_clip] = np.inf
    masked[-rear_clip:] = np.inf

    # --- If no obstacle is visible → WANDER
    if not np.any(np.isfinite(masked)):
        return maxSpeed, float(np.random.normal(loc=0.0, scale=1.0))

    # --- Find closest point
    min_index = int(np.nanargmin(masked))
    angle_increment = 2.0 * math.pi / max(N - 1, 1)
    angleMin = (half - min_index) * angle_increment
    distMin = masked[min_index]

    # --- Distances for behaviour selection (use unmasked)
    distFront = finite[half]
    side_idx = _clip_idx(int(half - (math.pi/2.0) / angle_increment), N) if direction == 1 \
        else _clip_idx(int(half + (math.pi/2.0) / angle_increment), N)
    distSide = finite[side_idx]
    distBack = finite[0]

    # --- Optional unblock for tight corners
    if distFront < 1.25 * wallDist and (distSide < 1.25 * wallDist or distBack < 1.25 * wallDist):
        angular_vel = -direction
    else:
        angular_vel = direction * distP * (distMin - wallDist) + angleP * (angleMin - direction * math.pi / 2.0)

    # --- Linear velocity: TURN / SLOW / CRUISE
    if distFront < wallDist:
        linear_vel = 0.0
    elif (distFront < 2.0 * wallDist) or (distMin < 0.75 * wallDist) or (distMin > 1.25 * wallDist):
        linear_vel = 0.5 * maxSpeed
    else:
        linear_vel = maxSpeed

    return float(linear_vel), float(angular_vel)
