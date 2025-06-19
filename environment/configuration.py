import math
import random as pyrandom
import numpy as np
import os

# --- (Constantes Gerais Inalteradas) ---
ROBOT_NAME = "e-puck"
ROBOT_RADIUS = 0.035
TIMEOUT_DURATION = 100.0
MIN_VELOCITY = 0.05
MAX_VELOCITY = 0.12
WALL_THICKNESS = 0.01
WALL_HEIGHT = 0.07
MAX_DIFFICULTY_STAGE = 30
MAX_NUM_CURVES = 4
MIN_STRAIGHT_LENGTH = ROBOT_RADIUS * 6.0
MAX_STRAIGHT_LENGTH = ROBOT_RADIUS * 18.0
IDEAL_CURVE_SEGMENT_LENGTH = ROBOT_RADIUS * 1.0
MIN_CLEARANCE_FACTOR = 4.0
MAX_CLEARANCE_FACTOR = 8.0
MAX_NUM_OBSTACLES = 6
MIN_OBSTACLE_DISTANCE = ROBOT_RADIUS * 5.0
MOVEMENT_TIMEOUT_DURATION = 30.0
MIN_MOVEMENT_THRESHOLD = ROBOT_RADIUS * 0.75
MAP_X_MIN, MAP_X_MAX = -2.5, 2.5
MAP_Y_MIN, MAP_Y_MAX = -2.5, 2.5

# --- Definições do Currículo com Fase 0 ---
STAGE_DEFINITIONS = {
    0: {'num_curves': 0, 'main_angle_range': (0, 0), 'num_obstacles': 0},  # Fase 0: Túnel a direito e sem obstáculos.
    # Bloco 1: Curvas (1-10)
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

    # Bloco 2: Obstáculos (11-20)
    11: {'num_curves': 0, 'main_angle_range': (0, 0), 'num_obstacles': 1, 'passageway_width_factor': 6.0},
    12: {'num_curves': 0, 'main_angle_range': (0, 0), 'num_obstacles': 1, 'passageway_width_factor': 5.5},
    13: {'num_curves': 0, 'main_angle_range': (0, 0), 'num_obstacles': 1, 'passageway_width_factor': 5.0},
    14: {'num_curves': 0, 'main_angle_range': (0, 0), 'num_obstacles': 1, 'passageway_width_factor': 4.5},
    15: {'num_curves': 0, 'main_angle_range': (0, 0), 'num_obstacles': 1, 'passageway_width_factor': 4.0},
    16: {'num_curves': 1, 'main_angle_range': (0, 40), 'num_obstacles': 2, 'passageway_width_factor': (4.0, 6.0)},
    17: {'num_curves': 1, 'main_angle_range': (0, 40), 'num_obstacles': 3, 'passageway_width_factor': (4.0, 6.0)},
    18: {'num_curves': 2, 'main_angle_range': (0, 40), 'num_obstacles': 3, 'passageway_width_factor': (4.0, 6.0)},
    19: {'num_curves': 2, 'main_angle_range': (0, 40), 'num_obstacles': 4, 'passageway_width_factor': (4.0, 6.0)},
    20: {'num_curves': 2, 'main_angle_range': (0, 40), 'num_obstacles': 5, 'passageway_width_factor': (4.0, 6.0)},

    # Bloco 3: Combinação (21-30)
    21: {'num_curves': 3, 'main_angle_range': (45, 90), 'num_obstacles': 2, 'passageway_width_factor': (4.0, 5.0)},
    22: {'num_curves': 3, 'main_angle_range': (45, 90), 'num_obstacles': 3, 'passageway_width_factor': (4.0, 5.0)},
    23: {'num_curves': 4, 'main_angle_range': (45, 90), 'num_obstacles': 3, 'passageway_width_factor': (4.0, 5.0)},
    24: {'num_curves': 4, 'main_angle_range': (45, 90), 'num_obstacles': 4, 'passageway_width_factor': (4.0, 5.0)},
    25: {'num_curves': 4, 'main_angle_range': (45, 90), 'num_obstacles': 5, 'passageway_width_factor': (4.0, 5.0)},
    26: {'num_curves': 4, 'main_angle_range': (45, 90), 'num_obstacles': 6, 'passageway_width_factor': (4.0, 5.0)},
    27: {'num_curves': 4, 'main_angle_range': (45, 90), 'num_obstacles': 6, 'passageway_width_factor': (3.8, 4.8)},
    28: {'num_curves': 4, 'main_angle_range': (45, 90), 'num_obstacles': 6, 'passageway_width_factor': (3.6, 4.6)},
    29: {'num_curves': 4, 'main_angle_range': (45, 90), 'num_obstacles': 6, 'passageway_width_factor': (3.4, 4.4)},
    30: {'num_curves': 4, 'main_angle_range': (45, 90), 'num_obstacles': 6, 'passageway_width_factor': (3.2, 4.2)},
}


def get_stage_parameters(stage: int, custom_params=None):
    if custom_params:
        params = custom_params
    else:
        stage = int(np.clip(stage, 0, MAX_DIFFICULTY_STAGE))
        params = STAGE_DEFINITIONS.get(stage)

    num_curves = params.get('num_curves', 0)
    main_angle_range = params.get('main_angle_range', (0, 0))
    num_obstacles = params.get('num_obstacles', 0)
    obstacle_types = params.get('obstacle_types', ['wall', 'pillar'])

    passageway_width = None
    clearance_factor = 4.0

    if 'passageway_width_factor' in params:
        clearance_factor = 8.0
        factor = params['passageway_width_factor']
        if isinstance(factor, tuple):
            passageway_width = pyrandom.uniform(factor[0], factor[1]) * ROBOT_RADIUS
        else:
            passageway_width = factor * ROBOT_RADIUS

    curve_angles_list = []
    if num_curves > 0:
        main_angle = pyrandom.uniform(main_angle_range[0], main_angle_range[1])
        curve_angles_list.append(main_angle)

        if num_curves == 2:
            curve_angles_list.append(main_angle * 0.5)
        elif num_curves == 3:
            curve_angles_list.extend([main_angle * 0.33, main_angle * 0.66])
        elif num_curves == 4:
            curve_angles_list.extend([main_angle * 0.25, main_angle * 0.5, main_angle * 0.75])

        pyrandom.shuffle(curve_angles_list)

    if os.environ.get('ROBOT_DEBUG_MODE') == '1':
        source = "SUB" if custom_params else "STD"
        print(
            f"[DEBUG | GET_PARAMS | {source}] Fase {stage}: "
            f"{num_curves} curvas, "
            f"Passagem: {passageway_width / ROBOT_RADIUS if passageway_width else 'N/A'} raios, "
            f"{num_obstacles} obstáculos"
        )

    return num_curves, curve_angles_list, clearance_factor, num_obstacles, obstacle_types, passageway_width


def generate_intermediate_stage(stage_params, sub_index=0):
    """
    Generate an easier version of the current stage by progressively reducing
    either the angle range of curves or the number of obstacles.

    The reduction alternates:
    - On even sub_index (0, 2, 4...): reduce the angle range (curves).
    - On odd sub_index (1, 3, 5...): reduce the number of obstacles.

    This prevents both aspects from being reduced at the same time,
    promoting better learning generalization.

    :param stage_params: dictionary with original stage parameters
    :param sub_index: index indicating how many adjustments have been made
    :return: new dictionary with adjusted parameters
    """
    new_stage = stage_params.copy()

    reduce_curves = (sub_index % 2 == 0)
    reduce_obstacles = (sub_index % 2 == 1)

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
            print(f" Curves adjusted to range: ({new_start}, {new_end})")

    elif reduce_obstacles and 'num_obstacles' in new_stage:
        current = new_stage['num_obstacles']
        reduction = max(1, (sub_index + 1) // 2)
        new_stage['num_obstacles'] = max(0, current - reduction)
        print(f" Obstacles reduced to: {new_stage['num_obstacles']}")

    return new_stage