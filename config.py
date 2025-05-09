import math
# --- Configuration ---
ROBOT_NAME = "e-puck"
ROBOT_RADIUS = 0.035  # meters

# Tunnel clearance relative to robot diameter (2 * ROBOT_RADIUS)
MIN_CLEARANCE_FACTOR = 2.5
MAX_CLEARANCE_FACTOR = 4.0

# Minimum required clearance for the robot to pass an obstacle
MIN_ROBOT_CLEARANCE_FACTOR = 2.1
MIN_ROBOT_CLEARANCE = ROBOT_RADIUS * MIN_ROBOT_CLEARANCE_FACTOR

# Minimum distance between obstacles
MIN_OBSTACLE_DISTANCE_FACTOR = 4.0
MIN_OBSTACLE_DISTANCE = ROBOT_RADIUS * MIN_OBSTACLE_DISTANCE_FACTOR

# Number of obstacles to place in the tunnel
NUM_OBSTACLES = 2


BASE_SPEED = 0.1
BASE_WALL_LENGTH = 1.0
WALL_THICKNESS = 0.01
WALL_HEIGHT = 0.07
MIN_CURVE_ANGLE = math.radians(10)
MAX_CURVE_ANGLE = math.radians(90)
CURVE_SUBDIVISIONS = 30 # Increased subdivisions for smoother curves
TIMEOUT_DURATION = 45.0
NUM_CURVES = 3

# Map boundaries (example values, adjust to your world limits)
MAP_X_MIN, MAP_X_MAX = -2.0, 2.0
MAP_Y_MIN, MAP_Y_MAX = -2.0, 2.0

# Small overlap factor to ensure no gaps in curves
OVERLAP_FACTOR = 1.01 # Adjusted overlap factor based on user feedback
