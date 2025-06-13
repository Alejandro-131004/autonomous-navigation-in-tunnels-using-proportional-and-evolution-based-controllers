import math
import random as pyrandom
import numpy as np

# --- General Configuration ---
ROBOT_NAME = "e-puck"
ROBOT_RADIUS = 0.035
TIMEOUT_DURATION = 100.0

# --- Wall & Segment Configuration ---
WALL_THICKNESS = 0.01
WALL_HEIGHT = 0.07

# --- Curriculum Learning Stages ---
MAX_DIFFICULTY_STAGE = 20  # Aumentado para 20 fases

# --- Tunnel Structure Progression ---
MIN_STRAIGHT_LENGTH = ROBOT_RADIUS * 6.0
MAX_STRAIGHT_LENGTH = ROBOT_RADIUS * 18.0
MAX_NUM_CURVES = 4
IDEAL_CURVE_SEGMENT_LENGTH = ROBOT_RADIUS * 1.5
MIN_CLEARANCE_FACTOR = 2.2
MAX_CLEARANCE_FACTOR = 4.0

# --- Obstacle Progression ---
MAX_NUM_OBSTACLES = 6
MIN_OBSTACLE_DISTANCE = ROBOT_RADIUS * 5.0
MIN_ROBOT_CLEARANCE = ROBOT_RADIUS * 2.1

# --- Movement Timeout Configuration ---
MOVEMENT_TIMEOUT_DURATION = 30.0
MIN_MOVEMENT_THRESHOLD = ROBOT_RADIUS * 0.75

# --- Map Boundaries ---
MAP_X_MIN, MAP_X_MAX = -2.5, 2.5
MAP_Y_MIN, MAP_Y_MAX = -2.5, 2.5


def get_stage_parameters(stage: int, total_stages: float = MAX_DIFFICULTY_STAGE):
    """
    Fornece parâmetros de geração de túnel com base num currículo de 20 fases.
    """
    stage = int(np.clip(stage, 1, total_stages))

    num_curves = 0
    angle_range = (0.0, 0.0)
    num_obstacles = 0
    obstacle_types = []  # Lista de tipos de obstáculos permitidos

    # Fases 1-5: Foco em Curvas e Navegação
    if 1 <= stage <= 5:
        num_curves = min(stage - 1, MAX_NUM_CURVES)
        angle_deg_upper = stage * 18  # Aumenta gradualmente até 90 graus na fase 5
        angle_range = (math.radians(angle_deg_upper - 18), math.radians(angle_deg_upper))
        num_obstacles = 0

    # Fases 6-10: Introdução de obstáculos "wall"
    elif 6 <= stage <= 10:
        num_curves = MAX_NUM_CURVES
        angle_deg_upper = 90  # A curvatura permanece alta
        angle_range = (math.radians(70), math.radians(90))
        num_obstacles = (stage - 5)  # 1, 2, 3, 4, 5 obstáculos
        obstacle_types = ['wall']

    # Fases 11-15: Introdução de obstáculos "pillar"
    elif 11 <= stage <= 15:
        num_curves = MAX_NUM_CURVES
        angle_deg_upper = 90
        angle_range = (math.radians(70), math.radians(90))
        num_obstacles = (stage - 10)  # 1, 2, 3, 4, 5 obstáculos
        obstacle_types = ['pillar']

    # Fases 16-20: Mistura de obstáculos e desafios máximos
    elif 16 <= stage <= 20:
        num_curves = MAX_NUM_CURVES
        angle_deg_upper = 90
        angle_range = (math.radians(70), math.radians(90))
        num_obstacles = (stage - 15) + 1  # 2, 3, 4, 5, 6 obstáculos
        obstacle_types = ['wall', 'pillar']

    if stage == 1:
        num_curves = 0
        angle_range = (0.0, 0.0)

    num_obstacles = min(num_obstacles, MAX_NUM_OBSTACLES)

    # A largura do túnel diminui progressivamente ao longo das 20 fases
    progress = (stage - 1) / (total_stages - 1)
    target_clearance = MAX_CLEARANCE_FACTOR - progress * (MAX_CLEARANCE_FACTOR - MIN_CLEARANCE_FACTOR)
    clearance_factor = np.clip(target_clearance, MIN_CLEARANCE_FACTOR, MAX_CLEARANCE_FACTOR)

    print(
        f"[GET_PARAMS] Fase {stage}: "
        f"{num_curves} curvas, "
        f"ângulos {math.degrees(angle_range[0]):.0f}°-{math.degrees(angle_range[1]):.0f}°, "
        f"clearance {clearance_factor:.2f}, "
        f"{num_obstacles} obstáculos (Tipos: {obstacle_types or 'Nenhum'})"
    )

    # A função agora retorna também os tipos de obstáculos permitidos
    return num_curves, angle_range, clearance_factor, num_obstacles, obstacle_types
