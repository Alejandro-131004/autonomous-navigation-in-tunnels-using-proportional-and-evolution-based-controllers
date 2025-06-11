# controllers/reactive_controller.py
"""
Contains the logic of the classic reactive controller (wall-follower)
adapted from the original user script.
"""
import math
import numpy as np

def reactive_controller_logic(dist_values: list) -> tuple[float, float]:
    """
    Reactive wall-following control logic adapted from the original file.
    """
    direction: int = 1  # Follow the wall on the right
    maxSpeed: float = 0.1
    distP: float = 10.0
    angleP: float = 7.0
    wallDist: float = 0.1

    size: int = len(dist_values)
    if size == 0:
        return 0.0, 0.0

    # Ensure distance values are finite numbers
    dist_values = np.nan_to_num(dist_values, nan=np.inf)

    # Find index of the ray with minimum distance
    min_index: int = 0
    current_min_dist = float('inf')
    for i in range(size):
        dist = dist_values[i]
        if 0 < dist < current_min_dist:
            current_min_dist = dist
            min_index = i

    angle_increment: float = 2 * math.pi / (size - 1) if size > 1 else 0.0
    angleMin: float = (size / 2 - min_index) * angle_increment
    distMin: float = dist_values[min_index]
    distFront: float = dist_values[size // 2]

    linear_vel: float
    angular_vel: float

    # Robot behavior decision logic
    if math.isfinite(distMin):
        # Regular wall control
        angular_vel = direction * distP * (distMin - wallDist) + angleP * (angleMin - direction * math.pi / 2)

        if distFront < wallDist:
            # Turn if too close to front obstacle
            linear_vel = 0
        elif distFront < 2 * wallDist or distMin < wallDist * 0.75 or distMin > wallDist * 1.25:
            # Slow down if adjusting distance to the wall
            linear_vel = 0.5 * maxSpeed
        else:
            # Cruise speed
            linear_vel = maxSpeed
    else:
        # Navigate randomly if no walls detected
        angular_vel = np.random.normal(loc=0.0, scale=0.5) * maxSpeed
        linear_vel = maxSpeed

    return linear_vel, angular_vel
