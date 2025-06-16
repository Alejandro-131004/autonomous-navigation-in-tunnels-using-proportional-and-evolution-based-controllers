import math
import random as pyrandom
import numpy as np
import os

# --- General Configuration ---
ROBOT_NAME = "e-puck"
ROBOT_RADIUS = 0.035
TIMEOUT_DURATION = 100.0

# --- Robot Dynamics ---
MIN_VELOCITY = 0.05
MAX_VELOCITY = 0.12

# --- Wall and Segment Configuration ---
WALL_THICKNESS = 0.01
WALL_HEIGHT = 0.07

# --- Curriculum Learning Stages ---
MAX_DIFFICULTY_STAGE = 20

# --- Tunnel Structure Progression ---
MAX_NUM_CURVES = 4
MIN_STRAIGHT_LENGTH = ROBOT_RADIUS * 6.0
MAX_STRAIGHT_LENGTH = ROBOT_RADIUS * 18.0
IDEAL_CURVE_SEGMENT_LENGTH = ROBOT_RADIUS * 1.5
MIN_CLEARANCE_FACTOR = 2.2
MAX_CLEARANCE_FACTOR = 4.0

# --- Obstacle Progression ---
MAX_NUM_OBSTACLES = 10
MIN_OBSTACLE_DISTANCE = ROBOT_RADIUS * 5.0
MIN_ROBOT_CLEARANCE = ROBOT_RADIUS * 2.1

# --- Inactivity Timeout Configuration ---
MOVEMENT_TIMEOUT_DURATION = 30.0
MIN_MOVEMENT_THRESHOLD = ROBOT_RADIUS * 0.75

# --- Map Limits ---
MAP_X_MIN, MAP_X_MAX = -2.5, 2.5
MAP_Y_MIN, MAP_Y_MAX = -2.5, 2.5

# --- Curriculum Stage Definitions ---
STAGE_DEFINITIONS = {
    1: {'num_curves': 0, 'angle_range': (0, 0), 'num_obstacles': 0, 'obstacle_types': []},
    2: {'num_curves': 1, 'angle_range': (0, 10), 'num_obstacles': 0, 'obstacle_types': []},
    3: {'num_curves': 1, 'angle_range': (10, 20), 'num_obstacles': 0, 'obstacle_types': []},
    4: {'num_curves': 1, 'angle_range': (10, 20), 'num_obstacles': 1, 'obstacle_types': ['wall']},
    5: {'num_curves': 2, 'angle_range': (30, 40), 'num_obstacles': 0, 'obstacle_types': ['wall']},
    6: {'num_curves': 1, 'angle_range': (20, 30), 'num_obstacles': 1, 'obstacle_types': ['wall']},
    7: {'num_curves': 2, 'angle_range': (30, 40), 'num_obstacles': 0, 'obstacle_types': ['wall']},
    8: {'num_curves': 2, 'angle_range': (40, 50), 'num_obstacles': 1, 'obstacle_types': ['wall']},
    9: {'num_curves': 2, 'angle_range': (50, 60), 'num_obstacles': 1, 'obstacle_types': ['wall']},
    10: {'num_curves': 2, 'angle_range': (40, 50), 'num_obstacles': 2, 'obstacle_types': ['wall']},
    11: {'num_curves': 3, 'angle_range': (40, 50), 'num_obstacles': 2, 'obstacle_types': ['wall']},
    12: {'num_curves': 3, 'angle_range': (60, 70), 'num_obstacles': 2, 'obstacle_types': ['wall']},
    13: {'num_curves': 3, 'angle_range': (80, 90), 'num_obstacles': 2, 'obstacle_types': ['wall']},
    14: {'num_curves': 4, 'angle_range': (80, 90), 'num_obstacles': 2, 'obstacle_types': ['wall']},
    15: {'num_curves': 3, 'angle_range': (100, 110), 'num_obstacles': 3, 'obstacle_types': ['wall']},
    16: {'num_curves': 4, 'angle_range': (90, 100), 'num_obstacles': 3, 'obstacle_types': ['wall']},
    17: {'num_curves': 4, 'angle_range': (110, 120), 'num_obstacles': 3, 'obstacle_types': ['wall']},
    18: {'num_curves': 4, 'angle_range': (130, 150), 'num_obstacles': 3, 'obstacle_types': ['wall']},
    19: {'num_curves': 4, 'angle_range': (150, 170), 'num_obstacles': 4, 'obstacle_types': ['wall']},
    20: {'num_curves': 4, 'angle_range': (170, 180), 'num_obstacles': 4, 'obstacle_types': ['wall']},
}


def get_stage_parameters(stage: int):
    """
    Returns tunnel generation parameters for an unlimited number of stages.
    """
    if stage <= MAX_DIFFICULTY_STAGE:
        params = STAGE_DEFINITIONS.get(stage)
    else:
        params = STAGE_DEFINITIONS[MAX_DIFFICULTY_STAGE].copy()
        additional_obstacles = (stage - MAX_DIFFICULTY_STAGE + 1) // 2
        params['num_obstacles'] = min(params['num_obstacles'] + additional_obstacles, MAX_NUM_OBSTACLES)

    num_curves = params['num_curves']
    angle_range_deg = params['angle_range']
    num_obstacles = params['num_obstacles']
    obstacle_types = params['obstacle_types']

    angle_range_rad = (math.radians(angle_range_deg[0]), math.radians(angle_range_deg[1]))

    if stage <= MAX_DIFFICULTY_STAGE:
        progress = (stage - 1) / (MAX_DIFFICULTY_STAGE - 1)
    else:
        progress = 1.0 + (stage - MAX_DIFFICULTY_STAGE) * 0.05

    target_clearance = MAX_CLEARANCE_FACTOR - progress * (MAX_CLEARANCE_FACTOR - MIN_CLEARANCE_FACTOR)
    clearance_factor = np.clip(target_clearance, MIN_CLEARANCE_FACTOR, MAX_CLEARANCE_FACTOR)

    # Stage parameters are printed only in debug mode
    if os.environ.get('ROBOT_DEBUG_MODE') == '1':
        print(
            f"[DEBUG | GET_PARAMS] Stage {stage}: "
            f"{num_curves} curves, "
            f"angles {angle_range_deg[0]}°–{angle_range_deg[1]}°, "
            f"clearance {clearance_factor:.2f}, "
            f"{num_obstacles} obstacles (Types: {obstacle_types or 'None'})"
        )

    return num_curves, angle_range_rad, clearance_factor, num_obstacles, obstacle_types
