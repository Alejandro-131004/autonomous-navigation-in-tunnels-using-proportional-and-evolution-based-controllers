import math
import random as pyrandom
import numpy as np

# --- General Configuration ---
ROBOT_NAME = "e-puck"
ROBOT_RADIUS = 0.035  # meters
TIMEOUT_DURATION = 60.0  # Increased timeout for more complex stages

# --- Wall & Segment Configuration ---
BASE_WALL_LENGTH = ROBOT_RADIUS  # Base length unit for straight segments
WALL_THICKNESS = 0.01
WALL_HEIGHT = 0.07
WALL_JOINT_GAP = 0.001  # Small gap to prevent visual overlap
OVERLAP_FACTOR = 1.1  # Factor to ensure wall segments overlap slightly in curves

# --- Curriculum Learning Stages ---
MAX_DIFFICULTY_STAGE = 10

# --- Tunnel Structure Progression ---
# Straight Segments
MIN_STRAIGHT_LENGTH_FACTOR = 5.0  # Min length in robot radii
MAX_STRAIGHT_LENGTH_FACTOR = 15.0  # Max length in robot radii
MIN_STRAIGHT_LENGTH = ROBOT_RADIUS * MIN_STRAIGHT_LENGTH_FACTOR
MAX_STRAIGHT_LENGTH = ROBOT_RADIUS * MAX_STRAIGHT_LENGTH_FACTOR

# Curves
MAX_NUM_CURVES = 4  # The maximum number of curves in the most difficult stages
# The angle range will be determined by the stage, up to 90 degrees
FINAL_MAX_CURVE_ANGLE = math.radians(90)
CURVE_SUBDIVISIONS = 20  # Number of small straight segments to make a curve

# Clearance (Tunnel Width)
# The tunnel will get narrower as the difficulty increases.
MIN_CLEARANCE_FACTOR = 2.2  # Clearance factor for the hardest stages
MAX_CLEARANCE_FACTOR = 3.5  # Clearance factor for the easiest stages

# --- Obstacle Progression ---
MAX_NUM_OBSTACLES = 6  # Corresponds to stage 10 (stage - 4)
OBSTACLE_START_STAGE = 5  # Obstacles are introduced from this stage onwards
MIN_OBSTACLE_DISTANCE_FACTOR = 5.0  # Min distance between obstacles in robot radii
MIN_OBSTACLE_DISTANCE = ROBOT_RADIUS * MIN_OBSTACLE_DISTANCE_FACTOR
MIN_ROBOT_CLEARANCE = ROBOT_RADIUS * 2.1  # Minimum space robot needs to pass an obstacle

# --- Map Boundaries ---
MAP_X_MIN, MAP_X_MAX = -2.5, 2.5
MAP_Y_MIN, MAP_Y_MAX = -2.5, 2.5


def get_stage_parameters(stage: int, total_stages: float = MAX_DIFFICULTY_STAGE):
    """
    Provides deterministic tunnel generation parameters based on the current training stage.
    As the stage increases, the challenge and complexity increase in a structured way.

    Args:
        stage (int): The current training stage (from 1 to 10).
        total_stages (float): The maximum possible difficulty stage for normalization.

    Returns:
        tuple: (num_curves, angle_range, clearance_factor, num_obstacles)
    """
    # Ensure stage is within bounds [1, 10]
    stage = int(np.clip(stage, 1, total_stages))

    # Normalized progress from 0.0 (stage 1) to 1.0 (stage 10)
    progress = (stage - 1) / (total_stages - 1)

    # --- 1. Number of Curves ---
    # Progressively increases from 0 to MAX_NUM_CURVES.
    if stage == 1:
        num_curves = 0
    else:
        # A gentle, stepped increase in the number of curves
        if stage <= 3:
            num_curves = 1
        elif stage <= 5:
            num_curves = 2
        elif stage <= 7:
            num_curves = 3
        else:
            num_curves = 4
    num_curves = min(num_curves, MAX_NUM_CURVES)

    # --- 2. Curve Angle Range ---
    # Increases by 10 degrees at each stage.
    if stage == 1:
        # Stage 1 is always a straight line
        angle_range = (0.0, 0.0)
    else:
        # Stage 2: [0, 10], Stage 3: [10, 20], ..., Stage 10: [80, 90]
        upper_bound_deg = (stage - 1) * 10
        lower_bound_deg = upper_bound_deg - 10
        angle_range = (math.radians(lower_bound_deg), math.radians(upper_bound_deg))

    # --- 3. Number of Obstacles ---
    # Introduced at OBSTACLE_START_STAGE and increases by one each stage.
    if stage < OBSTACLE_START_STAGE:
        num_obstacles = 0
    else:
        num_obstacles = stage - (OBSTACLE_START_STAGE - 1)
    num_obstacles = min(num_obstacles, MAX_NUM_OBSTACLES)

    # --- 4. Clearance (Tunnel Width) ---
    # Linearly decreases from MAX_CLEARANCE_FACTOR to MIN_CLEARANCE_FACTOR.
    # A small amount of randomness is added.
    target_clearance = MAX_CLEARANCE_FACTOR - progress * (MAX_CLEARANCE_FACTOR - MIN_CLEARANCE_FACTOR)
    random_offset = (pyrandom.random() - 0.5) * 0.2  # small random variation
    clearance_factor = np.clip(target_clearance + random_offset, MIN_CLEARANCE_FACTOR, MAX_CLEARANCE_FACTOR)

    print(
        f"[GET_PARAMS] Stage {stage}: "
        f"{num_curves} curves, "
        f"angles {math.degrees(angle_range[0]):.0f}°-{math.degrees(angle_range[1]):.0f}°, "
        f"clearance {clearance_factor:.2f}, "
        f"{num_obstacles} obstacles"
    )

    return num_curves, angle_range, clearance_factor, num_obstacles
