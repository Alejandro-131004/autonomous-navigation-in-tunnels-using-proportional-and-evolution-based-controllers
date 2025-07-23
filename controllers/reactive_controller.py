# No ficheiro reactive_controller.py585
import math
import numpy as np
from environment.configuration import MAX_VELOCITY  # Importa a velocidade do seu ficheiro de configuração


def reactive_controller_logic(dist_values: list, direction: int = 1) -> tuple[
    float, float]:  # Added direction parameter with default
    """
        Controlador modificado para replicar o comportamento do main.py.
    """
    # --- Parâmetros de Controlo ---
    maxSpeed = MAX_VELOCITY
    distP = 10.0
    angleP = 7.0
    wallDist = 0.3

    # --- Processamento dos dados do LIDAR ---
    size = len(dist_values)
    if size == 0:
        return 0.0, 0.0

    dist_values_finite = np.nan_to_num(dist_values, nan=np.inf)

    # Find the angle of the ray that returned the minimum distance - REVISED based on main.py
    min_index = 0
    if direction == -1:
        min_index = size - 1
    for i in range(size):
        idx = i
        if direction == -1:
            idx = size - 1 - i
        if dist_values_finite[idx] < dist_values_finite[min_index] and dist_values_finite[idx] > 0.0:
            min_index = idx

    angle_increment = 2 * math.pi / (size - 1)
    angleMin = (size // 2 - min_index) * angle_increment
    distMin = dist_values_finite[min_index]
    distFront = dist_values_finite[size // 2]
    distSide = dist_values_finite[size // 4] if (direction == 1) else dist_values_finite[3 * size // 4]
    distBack = dist_values_finite[0]

    # Prepare message for the robot's motors
    linear_vel: float
    angular_vel: float

    # --- Decide the robot's behavior - REVISED based on main.py ---
    if np.isfinite(distMin):
        if distFront < 1.25 * wallDist and (distSide < 1.25 * wallDist or distBack < 1.25 * wallDist):
            # UNBLOCK
            angular_vel = direction * -1
        else:
            # REGULAR
            angular_vel = direction * distP * (distMin - wallDist) + angleP * (angleMin - direction * math.pi / 2)

        if distFront < wallDist:
            # TURN
            linear_vel = 0
        elif distFront < 2 * wallDist or distMin < wallDist * 0.75 or distMin > wallDist * 1.25:
            # SLOW
            linear_vel = 0.5 * maxSpeed
        else:
            # CRUISE
            linear_vel = maxSpeed
    else:
        # WANDER - REVISED to match main.py's random wander
        angular_vel = np.random.normal(loc=0.0, scale=1.0)
        linear_vel = maxSpeed

    return linear_vel, angular_vel