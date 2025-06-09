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

# Define a maximum stage value for normalization (10 levels, so stages 1 to 10)
MAX_DIFFICULTY_STAGE = 10.0  # Use float for calculations, can be converted to int for display

# Function to get difficulty settings based on a progress stage
# This function will be called by the genetic algorithm's training loop
def get_stage_parameters(stage: float, total_stages: float = MAX_DIFFICULTY_STAGE):
    """
    Provides tunnel generation parameters based on a continuous training stage.
    As the stage increases, new elements are introduced one at a time,
    and then the diversity of scenarios for those elements increases.

    Args:
        stage (float): The current training stage (e.g., from 1.0 to MAX_DIFFICULTY_STAGE).
        total_stages (float): The maximum possible difficulty stage for normalization.

    Returns:
        tuple: (num_curves, angle_range, clearance_factor, num_obstacles)
    """
    # Normalize stage to a 0-1 range
    # Ensure stage is at least 1.0 for proper normalization if MAX_DIFFICULTY_STAGE is 10.0
    normalized_stage = min(max(1.0, stage), total_stages) / total_stages

    # Initialize all parameters to their easiest state
    num_curves = 0
    angle_range = (0.0, 0.0)
    clearance_factor = MAX_CLEARANCE_FACTOR_RANGE[1]  # Start with widest clearance
    num_obstacles = 0

    # Define phase breakpoints based on normalized_stage
    # These are illustrative breakpoints; adjust as needed for your desired curriculum shape
    PHASE1_STRAIGHT_NARROW = 0.2  # Stage 1-2 (normalized)
    PHASE2_INTRODUCE_CURVES = 0.5  # Stage 3-5 (normalized)
    PHASE3_INTRODUCE_OBSTACLES = 0.8  # Stage 6-8 (normalized)
    # PHASE 4 (0.8 - 1.0) is full diversity

    if normalized_stage <= PHASE1_STRAIGHT_NARROW:
        # Phase 1: Straight tunnels, narrowing clearance
        # Scale stage within this phase (0 to 1) for interpolation specific to this phase
        phase_t = normalized_stage / PHASE1_STRAIGHT_NARROW

        # Clearance: Interpolate from widest to upper end of tighter range
        clearance_factor = MAX_CLEARANCE_FACTOR_RANGE[1] - \
                           phase_t * (MAX_CLEARANCE_FACTOR_RANGE[1] - MIN_CLEARANCE_FACTOR_RANGE[1])
        clearance_factor = max(clearance_factor, MIN_CLEARANCE_FACTOR_RANGE[1])  # Clamp to min of this phase

        num_curves = 0  # No curves
        angle_range = (0.0, 0.0)  # Always straight
        num_obstacles = 0  # No obstacles

    elif normalized_stage <= PHASE2_INTRODUCE_CURVES:
        # Phase 2: Introduce Curves, continue narrowing clearance
        phase_t = (normalized_stage - PHASE1_STRAIGHT_NARROW) / (PHASE2_INTRODUCE_CURVES - PHASE1_STRAIGHT_NARROW)

        # Clearance: Continue interpolating towards absolute minimum (full range of tighter values)
        clearance_factor = MIN_CLEARANCE_FACTOR_RANGE[1] - \
                           phase_t * (MIN_CLEARANCE_FACTOR_RANGE[1] - MIN_CLEARANCE_FACTOR_RANGE[0])
        clearance_factor = pyrandom.uniform(clearance_factor, MAX_CLEARANCE_FACTOR_RANGE[1])  # Introduce diversity
        clearance_factor = max(clearance_factor, MIN_CLEARANCE_FACTOR_RANGE[0])  # Clamp to absolute min

        # Curves: Randomly chosen from 0 up to a max that scales with phase progress
        max_curves_for_phase = math.floor(phase_t * MAX_NUM_CURVES)
        num_curves = pyrandom.randint(0, min(max_curves_for_phase, MAX_NUM_CURVES))

        # Angle Range: Max angle for random choice increases with phase progress
        chosen_angle_min = pyrandom.uniform(0, phase_t * MIN_CURVE_ANGLE_RANGE[0])
        chosen_angle_max = pyrandom.uniform(chosen_angle_min, phase_t * MAX_CURVE_ANGLE_RANGE[1])
        angle_range = (chosen_angle_min, chosen_angle_max)

        num_obstacles = 0  # No obstacles

    elif normalized_stage <= PHASE3_INTRODUCE_OBSTACLES:
        # Phase 3: Introduce Obstacles, potentially with limited curves, challenging clearance
        phase_t = (normalized_stage - PHASE2_INTRODUCE_CURVES) / (PHASE3_INTRODUCE_OBSTACLES - PHASE2_INTRODUCE_CURVES)

        # Clearance: Randomly chosen across the full range to maintain diversity
        clearance_factor = pyrandom.uniform(MIN_CLEARANCE_FACTOR_RANGE[0], MAX_CLEARANCE_FACTOR_RANGE[1])

        # Curves: Limited or set to straight to emphasize obstacle navigation initially, then slowly diversify
        # We can either fix curves to 0-1, or make them vary from 0 to a scaled max_curves_for_phase
        max_curves_for_phase = min(MAX_NUM_CURVES, math.floor(phase_t * MAX_NUM_CURVES * 0.5))  # Limited curves
        num_curves = pyrandom.randint(0, max_curves_for_phase)

        # Angle Range: Small angles only, or scaling up to MIN_CURVE_ANGLE_RANGE
        chosen_angle_min = pyrandom.uniform(0, phase_t * MIN_CURVE_ANGLE_RANGE[0])
        chosen_angle_max = pyrandom.uniform(chosen_angle_min, MIN_CURVE_ANGLE_RANGE[1])  # Max small angle
        angle_range = (chosen_angle_min, chosen_angle_max)


        # Obstacles: Randomly chosen from 0 up to a max that scales with phase progress
        max_obstacles_for_phase = phase_t * MAX_NUM_OBSTACLES
        num_obstacles_raw = pyrandom.uniform(0, max_obstacles_for_phase)
        num_obstacles = math.floor(num_obstacles_raw)

        # Rule: if raw calculated number is > 0 but less than 1, set it to 1
        if num_obstacles_raw > 0 and num_obstacles == 0:
            num_obstacles = 1
        num_obstacles = min(num_obstacles, MAX_NUM_OBSTACLES)  # Clamp to absolute max

    else:  # normalized_stage > PHASE3_INTRODUCE_OBSTACLES (Phase 4)
        # Phase 4: Full Diversity and Max Challenge
        # Parameters are chosen randomly across their full defined ranges
        clearance_factor = pyrandom.uniform(MIN_CLEARANCE_FACTOR_RANGE[0], MAX_CLEARANCE_FACTOR_RANGE[1])
        num_curves = pyrandom.randint(0, MAX_NUM_CURVES)
        chosen_angle_min = pyrandom.uniform(0, MIN_CURVE_ANGLE_RANGE[0])
        chosen_angle_max = pyrandom.uniform(chosen_angle_min, MAX_CURVE_ANGLE_RANGE[1])
        angle_range = (chosen_angle_min, chosen_angle_max)

        num_obstacles_raw = pyrandom.uniform(0, MAX_NUM_OBSTACLES)
        num_obstacles = math.floor(num_obstacles_raw)
        if num_obstacles_raw > 0 and num_obstacles == 0:
            num_obstacles = 1
        num_obstacles = min(num_obstacles, MAX_NUM_OBSTACLES)  # Clamp to absolute max

    return num_curves, angle_range, clearance_factor, num_obstacles
