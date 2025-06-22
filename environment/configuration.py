import math
import random as pyrandom
import numpy as np
import os

# --- General Constants ---
ROBOT_NAME = "e-puck"
ROBOT_RADIUS = 0.035
TIMEOUT_DURATION = 100.0
MIN_VELOCITY = 0.05
MAX_VELOCITY = 0.12
WALL_THICKNESS = 0.01
WALL_HEIGHT = 0.07
# --- FIX: Updated the max difficulty to reflect the new curriculum ---
MAX_DIFFICULTY_STAGE = 12
MAX_NUM_CURVES = 4
MIN_STRAIGHT_LENGTH = ROBOT_RADIUS * 8.0
MAX_STRAIGHT_LENGTH = ROBOT_RADIUS * 20.0
MOVEMENT_TIMEOUT_DURATION = 30.0
MIN_MOVEMENT_THRESHOLD = ROBOT_RADIUS * 0.75
MAP_X_MIN, MAP_X_MAX = -2.5, 2.5
MAP_Y_MIN, MAP_Y_MAX = -2.5, 2.5
MAX_CURVE_STEP_ANGLE = math.radians(2.0)
MIN_CURVE_SEGMENT_LENGTH = 0.010
MAX_CURVE_SEGMENT_LENGTH = 0.150
MIN_OBSTACLE_DISTANCE = ROBOT_RADIUS * 4.0

# --- NEW CURRICULUM: Focused on progressively harder curves and tighter spaces ---
STAGE_DEFINITIONS = {
    # Block 1: Basic Curves (Stages 0-10 remain the same)
    0: {'num_curves': 0, 'main_angle_range': (0, 0), 'num_obstacles': 0},
    1: {'num_curves': 1, 'main_angle_range': (0, 9), 'num_obstacles': 0},
    2: {'num_curves': 1, 'main_angle_range': (9, 18), 'num_obstacles': 0},
    3: {'num_curves': 2, 'main_angle_range': (18, 27), 'num_obstacles': 0},
    4: {'num_curves': 2, 'main_angle_range': (27, 36), 'num_obstacles': 0},
    5: {'num_curves': 3, 'main_angle_range': (36, 45), 'num_obstacles': 0},
    6: {'num_curves': 3, 'main_angle_range': (45, 54), 'num_obstacles': 0},
    7: {'num_curves': 4, 'main_angle_range': (54, 63), 'num_obstacles': 0},
    8: {'num_curves': 4, 'main_angle_range': (63, 72), 'num_obstacles': 0},
    9: {'num_curves': 4, 'main_angle_range': (72, 81), 'num_obstacles': 0},
    10: {'num_curves': 4, 'main_angle_range': (81, 90), 'num_obstacles': 0},

    # --- NEW Block 2: Advanced Curves with Tight Spaces (Stages 11-15) ---
    # These stages have NO obstacles and focus on navigation skill.
    11: {'num_curves': 4, 'main_angle_range': (90, 100), 'num_obstacles': 0, 'clearance_factor': 3.8},
    12: {'num_curves': 4, 'main_angle_range': (100, 110), 'num_obstacles': 0, 'clearance_factor': 3.6},
    13: {'num_curves': 4, 'main_angle_range': (110, 120), 'num_obstacles': 0, 'clearance_factor': 3.4},
    14: {'num_curves': 4, 'main_angle_range': (120, 130), 'num_obstacles': 0, 'clearance_factor': 3.2},
    15: {'num_curves': 4, 'main_angle_range': (130, 140), 'num_obstacles': 0, 'clearance_factor': 3.0},
}


def get_stage_parameters(stage: int, custom_params=None):
    if custom_params:
        params = custom_params
    else:
        stage = int(np.clip(stage, 0, MAX_DIFFICULTY_STAGE))
        params = STAGE_DEFINITIONS.get(stage, {})

    num_curves = params.get('num_curves', 0)
    main_angle_range = params.get('main_angle_range', (0, 0))
    # Obstacles are always 0 in this curriculum, but we get the value for consistency.
    num_obstacles = params.get('num_obstacles', 0)
    obstacle_types = params.get('obstacle_types', [])

    # --- FIX: Simplified logic for clearance factor ---
    # Use the specific clearance factor from the stage definition, or a default value.
    clearance_factor = params.get('clearance_factor', 4.0)
    # Passageway width is no longer needed as there are no obstacles.
    passageway_width = None

    curve_angles_list = []
    if num_curves > 0:
        main_angle = pyrandom.uniform(main_angle_range[0], main_angle_range[1])
        curve_angles_list.append(main_angle)

        if num_curves == 2:
            curve_angles_list.append(main_angle * 0.5)
        elif num_curves == 3:
            curve_angles_list.extend([main_angle * 0.33, main_angle * 0.66])
        elif num_curves >= 4:
            # For 4 or more curves, distribute them more evenly
            for i in range(1, num_curves):
                curve_angles_list.append(pyrandom.uniform(main_angle_range[0], main_angle_range[1]) * 0.75)

        pyrandom.shuffle(curve_angles_list)

    straight_length_range = params.get('straight_length_range', (MIN_STRAIGHT_LENGTH, MAX_STRAIGHT_LENGTH))

    if os.environ.get('ROBOT_DEBUG_MODE') == '1':
        source = "SUB" if custom_params else "STD"
        print(
            f"[DEBUG | GET_PARAMS | {source}] Stage {stage}: "
            f"{num_curves} curves, "
            f"Clearance: {clearance_factor:.1f}x Radius, "
            f"{num_obstacles} obstacles"
        )

    return num_curves, curve_angles_list, clearance_factor, num_obstacles, obstacle_types, passageway_width, straight_length_range


def generate_intermediate_stage(stage_params, sub_index=0):
    """
    Generate an easier version of the current stage by progressively reducing
    the angle range of curves or increasing the clearance.
    """
    new_stage = stage_params.copy()

    # In this new curriculum, we only adjust curves and clearance
    reduce_curves = (sub_index % 2 == 0)
    increase_clearance = (sub_index % 2 == 1)

    if reduce_curves and 'main_angle_range' in new_stage:
        start, end = new_stage['main_angle_range']
        range_size = end - start
        if range_size > 1:
            chunks = 4
            chunk_size = range_size / chunks
            i = (sub_index // 2) % chunks
            new_start = start + i * chunk_size
            new_end = min(start + (i + 1) * chunk_size, end)
            new_stage['main_angle_range'] = (new_start, new_end)
            print(f" Curves adjusted to range: ({new_start:.1f}, {new_end:.1f})")

    elif increase_clearance and 'clearance_factor' in new_stage:
        current = new_stage['clearance_factor']
        # Increase clearance by a small amount to make it easier
        increment = ((sub_index + 1) // 2) * 0.2
        new_stage['clearance_factor'] = min(current + increment, 5.0)  # Cap at 5.0
        print(f" Clearance increased to: {new_stage['clearance_factor']:.2f}")

    return new_stage
