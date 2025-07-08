"""
Classic reactive controller (wall-follower)
Adapted to follow the left wall using only left-side sensors.
"""
import math
import numpy as np

def reactive_controller_logic(dist_values: list) -> tuple[float, float]:
    """
    Reactive left wall-following logic using only left-side LIDAR data.
    """
    direction: int = -1  # -1 = follow left wall
    MAX_SPEED: float = 0.18
    distP: float = 10.0
    angleP: float = 7.0
    WALL_DIST: float = 0.1

    size: int = len(dist_values)
    if size == 0:
        return 0.0, 0.0

    dist_values = np.nan_to_num(dist_values, nan=np.inf)

    # Consider only the left half of the LIDAR (0 to size//2)
    relevant_indices = range(0, size // 2)
    min_index = None
    min_distance = float('inf')

    for i in relevant_indices:
        if 0 < dist_values[i] < min_distance:
            min_distance = dist_values[i]
            min_index = i

    if min_index is None:
        # No valid distance found on the left side
        return 0.05, 0.0  # move slowly forward

    angle_increment = (2 * math.pi) / (size - 1)
    angle_min = (size / 2 - min_index) * angle_increment
    dist_min = dist_values[min_index]
    dist_front = dist_values[size // 2]

    angular_vel = direction * distP * (dist_min - WALL_DIST) + angleP * (angle_min - direction * math.pi / 2)

    # No more 'Turn': just slow down if obstacle is too close in front
    if dist_front < WALL_DIST * 1.5:
        linear_vel = 0.05  # slow
    else:
        linear_vel = MAX_SPEED

    return (
        np.clip(linear_vel, -MAX_SPEED, MAX_SPEED),
        np.clip(angular_vel, -MAX_SPEED * 2, MAX_SPEED * 2)
    )
