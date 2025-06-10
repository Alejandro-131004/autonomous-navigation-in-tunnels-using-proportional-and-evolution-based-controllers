import math
import random as pyrandom
import numpy as np

# --- General Configuration ---
ROBOT_NAME = "e-puck"
ROBOT_RADIUS = 0.035  # meters
TIMEOUT_DURATION = 60.0

# --- Wall & Segment Configuration ---
BASE_WALL_LENGTH = ROBOT_RADIUS
WALL_THICKNESS = 0.01
WALL_HEIGHT = 0.07
WALL_JOINT_GAP = 0.001
# MODIFICAÇÃO: Aumentado para garantir que não existem espaços nas curvas.
OVERLAP_FACTOR = 2

# --- Curriculum Learning Stages ---
MAX_DIFFICULTY_STAGE = 10

# --- Tunnel Structure Progression ---
MIN_STRAIGHT_LENGTH = ROBOT_RADIUS * 5.0
MAX_STRAIGHT_LENGTH = ROBOT_RADIUS * 15.0
MAX_NUM_CURVES = 4
# MODIFICAÇÃO: Define o comprimento ideal para os pequenos segmentos que compõem uma curva.
IDEAL_CURVE_SEGMENT_LENGTH = ROBOT_RADIUS * 1.5
MIN_CLEARANCE_FACTOR = 2.2
MAX_CLEARANCE_FACTOR = 3.5

# --- Obstacle Progression ---
MAX_NUM_OBSTACLES = 6
OBSTACLE_START_STAGE = 5
MIN_OBSTACLE_DISTANCE = ROBOT_RADIUS * 5.0
MIN_ROBOT_CLEARANCE = ROBOT_RADIUS * 2.1

# --- Map Boundaries ---
MAP_X_MIN, MAP_X_MAX = -2.5, 2.5
MAP_Y_MIN, MAP_Y_MAX = -2.5, 2.5


def get_stage_parameters(stage: int, total_stages: float = MAX_DIFFICULTY_STAGE):
    """
    Fornece parâmetros de geração de túnel determinísticos com base no estágio de treino atual.
    """
    stage = int(np.clip(stage, 1, total_stages))
    progress = (stage - 1) / (total_stages - 1)

    # 1. Número de Curvas
    if stage == 1:
        num_curves = 0
    elif stage <= 3:
        num_curves = 1
    elif stage <= 5:
        num_curves = 2
    elif stage <= 7:
        num_curves = 3
    else:
        num_curves = 4
    num_curves = min(num_curves, MAX_NUM_CURVES)

    # 2. Ângulo das Curvas
    if stage == 1:
        angle_range = (0.0, 0.0)
    else:
        upper_bound_deg = (stage - 1) * 10
        lower_bound_deg = upper_bound_deg - 10
        angle_range = (math.radians(lower_bound_deg), math.radians(upper_bound_deg))

    # 3. Número de Obstáculos
    if stage < OBSTACLE_START_STAGE:
        num_obstacles = 0
    else:
        num_obstacles = stage - (OBSTACLE_START_STAGE - 1)
    num_obstacles = min(num_obstacles, MAX_NUM_OBSTACLES)

    # 4. Largura do Túnel
    target_clearance = MAX_CLEARANCE_FACTOR - progress * (MAX_CLEARANCE_FACTOR - MIN_CLEARANCE_FACTOR)
    random_offset = (pyrandom.random() - 0.5) * 0.2
    clearance_factor = np.clip(target_clearance + random_offset, MIN_CLEARANCE_FACTOR, MAX_CLEARANCE_FACTOR)

    print(
        f"[GET_PARAMS] Stage {stage}: "
        f"{num_curves} curves, "
        f"angles {math.degrees(angle_range[0]):.0f}°-{math.degrees(angle_range[1]):.0f}°, "
        f"clearance {clearance_factor:.2f}, "
        f"{num_obstacles} obstacles"
    )

    return num_curves, angle_range, clearance_factor, num_obstacles
