import math
import random as pyrandom  # Import pyrandom for diversity

# --- Configuration ---
ROBOT_NAME = "e-puck"
ROBOT_RADIUS = 0.035  # meters

# Tunnel clearance relative to robot diameter (2 * ROBOT_RADIUS)
# Define ranges for clearance factor
MIN_CLEARANCE_FACTOR_RANGE = (1.8, 2.5)  # Tighter minimum (e.g., 1.8 * 0.07m = 0.126m width)
MAX_CLEARANCE_FACTOR_RANGE = (3.0, 4.0)  # Wider maximum (e.g., 4.0 * 0.07m = 0.28m width)

# Minimum required clearance for the robot to pass an obstacle
MIN_ROBOT_CLEARANCE_FACTOR = 2.1
MIN_ROBOT_CLEARANCE = ROBOT_RADIUS * MIN_ROBOT_CLEARANCE_FACTOR

# Minimum distance between obstacles
MIN_OBSTACLE_DISTANCE_FACTOR = 4.0
MIN_OBSTACLE_DISTANCE = ROBOT_RADIUS * MIN_OBSTACLE_DISTANCE_FACTOR

# Max number of obstacles
MAX_NUM_OBSTACLES = 4

BASE_SPEED = 0.1
# Changed BASE_WALL_LENGTH to ROBOT_RADIUS to ensure compatibility with straight segment length calculations
BASE_WALL_LENGTH = ROBOT_RADIUS  # Base length for segments, now equals robot radius

# New constants for variable straight segment lengths
MIN_STRAIGHT_LENGTH_FACTOR = 6.0  # Minimum straight length in robot radii
MAX_STRAIGHT_LENGTH_FACTOR = 20.0  # Maximum straight length in robot radii
MIN_STRAIGHT_LENGTH = ROBOT_RADIUS * MIN_STRAIGHT_LENGTH_FACTOR
MAX_STRAIGHT_LENGTH = ROBOT_RADIUS * MAX_STRAIGHT_LENGTH_FACTOR
MAX_WALL_PIECES_PER_STRAIGHT = 10  # Max number of BASE_WALL_LENGTH units a straight can be

WALL_THICKNESS = 0.01
WALL_HEIGHT = 0.07
# Define ranges for curve angles
MIN_CURVE_ANGLE_RANGE = (math.radians(10), math.radians(30))  # Smaller minimum angle
MAX_CURVE_ANGLE_RANGE = (math.radians(60), math.radians(90))  # Larger maximum angle
CURVE_SUBDIVISIONS = 30  # Increased subdivisions for smoother curves
TIMEOUT_DURATION = 45.0
MAX_NUM_CURVES = 4  # Max number of curves

# Map boundaries (example values, adjust to your world limits)
MAP_X_MIN, MAP_X_MAX = -2.0, 2.0
MAP_Y_MIN, MAP_Y_MAX = -2.0, 2.0

# Small overlap factor to ensure no gaps in curves
OVERLAP_FACTOR = 1.1  # Adjusted overlap factor based on user feedback

# Define a maximum stage value for normalization
MAX_DIFFICULTY_STAGE = 100.0

# Define phase breakpoints for difficulty progression
PHASE1_END = 0.25  # Straight tunnels, narrowing clearance
PHASE2_END = 0.50  # Introduce curves, continue narrowing clearance
PHASE3_END = 0.75  # Introduce obstacles, limited curves


# Phase 4 is 0.75 to 1.0 (full diversity)

# Function to get difficulty settings based on a progress stage
# This function will be called by the genetic algorithm's training loop
def get_stage_parameters(stage):
    """
    Provides tunnel generation parameters based on a continuous training stage.
    As the stage increases, new elements are introduced one at a time,
    and then the diversity of scenarios for those elements increases.

    Args:
        stage (float): The current training stage (0.0 for easiest, up to MAX_DIFFICULTY_STAGE for hardest).

    Returns:
        tuple: (num_curves, angle_range, clearance_factor, num_obstacles)
    """
    # Normalize stage to a 0-1 range
    normalized_stage = min(max(0.0, stage), MAX_DIFFICULTY_STAGE) / MAX_DIFFICULTY_STAGE

    # Initialize all parameters to their easiest state
    num_curves = 0
    angle_range = (0.0, 0.0)
    clearance_factor = MAX_CLEARANCE_FACTOR_RANGE[1]  # Start with widest clearance
    num_obstacles = 0

    if normalized_stage <= PHASE1_END:
        # Phase 1: Straight tunnels, narrowing clearance
        # Scale stage within this phase (0 to 1)
        phase_normalized_stage = normalized_stage / PHASE1_END

        # Clearance: Interpolate from widest to upper end of tighter range
        clearance_factor = MAX_CLEARANCE_FACTOR_RANGE[1] - \
                           phase_normalized_stage * (MAX_CLEARANCE_FACTOR_RANGE[1] - MIN_CLEARANCE_FACTOR_RANGE[1])
        clearance_factor = max(clearance_factor, MIN_CLEARANCE_FACTOR_RANGE[1])  # Clamp to min of this phase

        # Curves and Obstacles remain at 0
        num_curves = 0
        angle_range = (0.0, 0.0)
        num_obstacles = 0

    elif normalized_stage <= PHASE2_END:
        # Phase 2: Introduce Curves, continue narrowing clearance
        # Scale stage within this phase (0 to 1)
        phase_normalized_stage = (normalized_stage - PHASE1_END) / (PHASE2_END - PHASE1_END)

        # Clearance: Continue interpolating from current (MIN_CLEARANCE_FACTOR_RANGE[1]) to absolute minimum
        clearance_factor = MIN_CLEARANCE_FACTOR_RANGE[1] - \
                           phase_normalized_stage * (MIN_CLEARANCE_FACTOR_RANGE[1] - MIN_CLEARANCE_FACTOR_RANGE[0])
        clearance_factor = max(clearance_factor, MIN_CLEARANCE_FACTOR_RANGE[0])  # Clamp to absolute min

        # Curves: Randomly chosen from 0 up to a max that scales with phase progress
        max_curves_for_phase = math.floor(phase_normalized_stage * MAX_NUM_CURVES)
        num_curves = pyrandom.randint(0, min(max_curves_for_phase, MAX_NUM_CURVES))

        # Angle Range: Max angle for random choice increases with phase progress
        chosen_angle_min = pyrandom.uniform(0, phase_normalized_stage * MIN_CURVE_ANGLE_RANGE[0])
        chosen_angle_max = pyrandom.uniform(chosen_angle_min, phase_normalized_stage * MIN_CURVE_ANGLE_RANGE[1])
        angle_range = (chosen_angle_min, chosen_angle_max)

        # Obstacles remain at 0
        num_obstacles = 0

    elif normalized_stage <= PHASE3_END:
        # Phase 3: Introduce Obstacles, limited curves, maintain challenging clearance
        # Scale stage within this phase (0 to 1)
        phase_normalized_stage = (normalized_stage - PHASE2_END) / (PHASE3_END - PHASE2_END)

        # Clearance: Randomly chosen within the tighter range
        clearance_factor = pyrandom.uniform(MIN_CLEARANCE_FACTOR_RANGE[0], MIN_CLEARANCE_FACTOR_RANGE[1])

        # Curves: Limited to 0 or 1 to emphasize obstacles
        num_curves = pyrandom.randint(0, min(1, MAX_NUM_CURVES))  # Max 1 curve in this phase
        angle_range = (0.0, MIN_CURVE_ANGLE_RANGE[0])  # Smallest possible curve angle if any

        # Obstacles: Randomly chosen from 0 up to a max that scales with phase progress
        max_obstacles_for_phase = phase_normalized_stage * MAX_NUM_OBSTACLES
        num_obstacles_raw = pyrandom.uniform(0, max_obstacles_for_phase)
        num_obstacles = math.floor(num_obstacles_raw)

        # If the raw calculated number is > 0 but less than 1, set it to 1
        if num_obstacles_raw > 0 and num_obstacles == 0:
            num_obstacles = 1
        num_obstacles = min(num_obstacles, MAX_NUM_OBSTACLES)  # Clamp to absolute max

    else:  # normalized_stage > PHASE3_END
        # Phase 4: Full Diversity and Max Challenge
        # Scale stage within this phase (0 to 1) - though for full diversity, it's less about scaling and more about range
        # phase_normalized_stage = (normalized_stage - PHASE3_END) / (1.0 - PHASE3_END)

        # Clearance: Randomly chosen across the full range
        clearance_factor = pyrandom.uniform(MIN_CLEARANCE_FACTOR_RANGE[0], MAX_CLEARANCE_FACTOR_RANGE[1])

        # Curves: Randomly chosen across the full range
        num_curves = pyrandom.randint(0, MAX_NUM_CURVES)

        # Angle Range: Randomly chosen across the full range
        chosen_angle_min = pyrandom.uniform(0, MIN_CURVE_ANGLE_RANGE[0])
        chosen_angle_max = pyrandom.uniform(chosen_angle_min, MAX_CURVE_ANGLE_RANGE[1])
        angle_range = (chosen_angle_min, chosen_angle_max)

        # Obstacles: Randomly chosen across the full range
        num_obstacles_raw = pyrandom.uniform(0, MAX_NUM_OBSTACLES)
        num_obstacles = math.floor(num_obstacles_raw)
        if num_obstacles_raw > 0 and num_obstacles == 0:
            num_obstacles = 1
        num_obstacles = min(num_obstacles, MAX_NUM_OBSTACLES)  # Clamp to absolute max

    return num_curves, angle_range, clearance_factor, num_obstacles
