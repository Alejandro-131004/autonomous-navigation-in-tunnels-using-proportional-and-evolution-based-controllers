import math
# --- Configuration ---
ROBOT_NAME = "e-puck"
ROBOT_RADIUS = 0.035  # meters

# Tunnel clearance relative to robot diameter (2 * ROBOT_RADIUS)
# Define ranges for clearance factor
MIN_CLEARANCE_FACTOR_RANGE = (1.8, 2.5) # Tighter minimum
MAX_CLEARANCE_FACTOR_RANGE = (3.0, 4.0) # Wider maximum

# Minimum required clearance for the robot to pass an obstacle
MIN_ROBOT_CLEARANCE_FACTOR = 2.1
MIN_ROBOT_CLEARANCE = ROBOT_RADIUS * MIN_ROBOT_CLEARANCE_FACTOR

# Minimum distance between obstacles
MIN_OBSTACLE_DISTANCE_FACTOR = 4.0
MIN_OBSTACLE_DISTANCE = ROBOT_RADIUS * MIN_OBSTACLE_DISTANCE_FACTOR

# Max number of obstacles
MAX_NUM_OBSTACLES = 4


BASE_SPEED = 0.1
BASE_WALL_LENGTH = 1.0 # Base length for segments
WALL_THICKNESS = 0.01
WALL_HEIGHT = 0.07
# Define ranges for curve angles
MIN_CURVE_ANGLE_RANGE = (math.radians(10), math.radians(30)) # Smaller minimum angle
MAX_CURVE_ANGLE_RANGE = (math.radians(60), math.radians(90)) # Larger maximum angle
CURVE_SUBDIVISIONS = 30 # Increased subdivisions for smoother curves
TIMEOUT_DURATION = 45.0
MAX_NUM_CURVES = 4 # Max number of curves

# Map boundaries (example values, adjust to your world limits)
MAP_X_MIN, MAP_X_MAX = -2.0, 2.0
MAP_Y_MIN, MAP_Y_MAX = -2.0, 2.0

# Small overlap factor to ensure no gaps in curves
OVERLAP_FACTOR = 1.1 # Adjusted overlap factor based on user feedback

# Function to get difficulty settings based on a progress stage
# This function will be called by the genetic algorithm's training loop
def get_stage_parameters(stage):
    """
    Provides tunnel generation parameters based on a training stage.
    Stage 0: Wide straight tunnel
    Stage 1: Narrower straight tunnel
    Stage 2: Wide tunnel, small curves
    Stage 3: Wide tunnel, increasing curve angle
    Stage 4: Wide tunnel, obstacles introduced
    Stage 5: Wide tunnel, increasing number of obstacles
    Stage 6: Narrower tunnel, curves, obstacles (mixed)
    ... and so on, allowing for more complex mixtures.

    Args:
        stage (int): The current training stage (or a metric representing progress).

    Returns:
        tuple: (num_curves, angle_range, clearance_factor, num_obstacles)
    """
    # Define parameters for different stages
    if stage == 0:
        # Stage 0: Wide straight tunnel
        return 0, (0, 0), MAX_CLEARANCE_FACTOR_RANGE[1], 0
    elif stage == 1:
        # Stage 1: Narrower straight tunnel
        # Linearly interpolate clearance factor between max and min over several stages
        max_stage_for_clearance_reduction = 3 # Example: reduce clearance over stages 1-3
        clearance_factor = MAX_CLEARANCE_FACTOR_RANGE[1] - (MAX_CLEARANCE_FACTOR_RANGE[1] - MIN_CLEARANCE_FACTOR_RANGE[0]) * (stage / max_stage_for_clearance_reduction)
        clearance_factor = max(clearance_factor, MIN_CLEARANCE_FACTOR_RANGE[0]) # Ensure it doesn't go below min
        return 0, (0, 0), clearance_factor, 0
    elif stage == 2:
         # Stage 2: Wide tunnel, small curves introduced
        return 1, MIN_CURVE_ANGLE_RANGE, MAX_CLEARANCE_FACTOR_RANGE[1], 0
    elif stage == 3:
        # Stage 3: Wide tunnel, increasing curve angle
        # Interpolate curve angle range between min and max over several stages
        max_stage_for_angle_increase = 5 # Example: increase angle over stages 3-5
        angle_min = MIN_CURVE_ANGLE_RANGE[0] + (MAX_CURVE_ANGLE_RANGE[0] - MIN_CURVE_ANGLE_RANGE[0]) * ((stage - 3) / (max_stage_for_angle_increase - 3))
        angle_max = MIN_CURVE_ANGLE_RANGE[1] + (MAX_CURVE_ANGLE_RANGE[1] - MIN_CURVE_ANGLE_RANGE[1]) * ((stage - 3) / (max_stage_for_angle_increase - 3))
        angle_min = min(angle_min, MAX_CURVE_ANGLE_RANGE[0])
        angle_max = min(angle_max, MAX_CURVE_ANGLE_RANGE[1])
        return 1, (angle_min, angle_max), MAX_CLEARANCE_FACTOR_RANGE[1], 0
    elif stage == 4:
        # Stage 4: Wide tunnel, obstacles introduced (no curves)
        return 0, (0, 0), MAX_CLEARANCE_FACTOR_RANGE[1], 1 # Start with 1 obstacle
    elif stage == 5:
        # Stage 5: Wide tunnel, increasing number of obstacles (no curves)
        # Interpolate number of obstacles over several stages
        max_stage_for_obstacle_increase = 7 # Example: increase obstacles over stages 5-7
        num_obstacles = math.floor(MAX_NUM_OBSTACLES * ((stage - 5) / (max_stage_for_obstacle_increase - 5)))
        num_obstacles = max(1, min(num_obstacles, MAX_NUM_OBSTACLES)) # Ensure between 1 and MAX_NUM_OBSTACLES
        return 0, (0, 0), MAX_CLEARANCE_FACTOR_RANGE[1], num_obstacles
    else:
        # Stage 6 and beyond: Mixed difficulty
        # Combine parameters from different stages, potentially interpolating
        # Example: For stages > 5, gradually introduce narrower clearance, curves, and obstacles
        max_stage_for_full_difficulty = 10 # Example: reach full difficulty by stage 10

        # Interpolate clearance factor towards minimum
        clearance_factor = MAX_CLEARANCE_FACTOR_RANGE[1] - (MAX_CLEARANCE_FACTOR_RANGE[1] - MIN_CLEARANCE_FACTOR_RANGE[0]) * ((stage - 6) / (max_stage_for_full_difficulty - 6))
        clearance_factor = max(clearance_factor, MIN_CLEARANCE_FACTOR_RANGE[0])

        # Interpolate number of curves towards maximum
        num_curves = math.floor(MAX_NUM_CURVES * ((stage - 6) / (max_stage_for_full_difficulty - 6)))
        num_curves = max(0, min(num_curves, MAX_NUM_CURVES))

        # Interpolate curve angle range towards maximum
        angle_min = MIN_CURVE_ANGLE_RANGE[0] + (MAX_CURVE_ANGLE_RANGE[0] - MIN_CURVE_ANGLE_RANGE[0]) * ((stage - 6) / (max_stage_for_full_difficulty - 6))
        angle_max = MIN_CURVE_ANGLE_RANGE[1] + (MAX_CURVE_ANGLE_RANGE[1] - MIN_CURVE_ANGLE_RANGE[1]) * ((stage - 6) / (max_stage_for_full_difficulty - 6))
        angle_min = min(angle_min, MAX_CURVE_ANGLE_RANGE[0])
        angle_max = min(angle_max, MAX_CURVE_ANGLE_RANGE[1])

        # Interpolate number of obstacles towards maximum
        num_obstacles = math.floor(MAX_NUM_OBSTACLES * ((stage - 6) / (max_stage_for_full_difficulty - 6)))
        num_obstacles = max(0, min(num_obstacles, MAX_NUM_OBSTACLES))

        return num_curves, (angle_min, angle_max), clearance_factor, num_obstacles