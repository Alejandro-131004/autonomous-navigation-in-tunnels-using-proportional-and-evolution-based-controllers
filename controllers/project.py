import sys
import random
from controller import Supervisor

# --- Configuration ---
ROBOT_NAME = "e-puck"
ROBOT_RADIUS = 0.035
ROBOT_DIAMETER = 2 * ROBOT_RADIUS
BASE_SPEED = 3.0
# Base wall parameters
BASE_WALL_DISTANCE_FROM_CENTER = 0.1
BASE_WALL_LENGTH = 1.5
WALL_THICKNESS = 0.01
WALL_HEIGHT = 0.04

# Randomization factor (0 to 10%)
RANDOM_FACTOR = 0.2

# Collision thresholds
ROTATION_AXIS_X = 0.577351
ROTATION_AXIS_Y = 0.577351
ROTATION_AXIS_Z = 0.577351
ROTATION_ANGLE  = 2.0944  # in radians (about 120 degrees)

def create_wall(supervisor, y_position, length):
    """
    Create a single wall with the specified axis-angle rotation,
    length, and place it at x_position (with y=z=0).
    """
    wall_string = f"""
    Solid {{
      translation 0 {y_position} 0.02
      rotation {ROTATION_AXIS_X} {ROTATION_AXIS_Y} {ROTATION_AXIS_Z} {ROTATION_ANGLE}
      children [
        Shape {{
          appearance Appearance {{
            material Material {{
              diffuseColor 1 0 0
            }}
          }}
          geometry Box {{
            size {WALL_THICKNESS} {WALL_HEIGHT} {length}
          }}
        }}
      ]
    }}
    """
    root_node = supervisor.getRoot()
    children_field = root_node.getField("children")
    children_field.importMFNodeFromString(-1, wall_string)

def run_robot_simulation(supervisor, timestep):
    """Runs the main simulation loop for the robot."""
    print("hello")
    # Get the robot node using its DEF name
    robot_node = supervisor.getFromDef(ROBOT_NAME)
    if robot_node is None:
        print(f"Error: Robot node '{ROBOT_NAME}' not found. Check DEF name in Webots.")
        sys.exit(1)

    robot_translation_field = robot_node.getField("translation")

    # Get wheel motors
    try:
        left_motor = supervisor.getDevice("left wheel motor")
        right_motor = supervisor.getDevice("right wheel motor")
        left_motor.setPosition(float('inf'))
        right_motor.setPosition(float('inf'))
    except Exception as e:
        print(f"Error getting motors: {e}. Ensure 'left wheel motor' and 'right wheel motor' exist.")
        sys.exit(1)

    # Calculate randomized wall distance
    random_distance_factor = 1 + random.uniform(-RANDOM_FACTOR, RANDOM_FACTOR)
    wall_distance = BASE_WALL_DISTANCE_FROM_CENTER * random_distance_factor
    positive_wall_y = wall_distance
    negative_wall_y = -wall_distance
    COLLISION_THRESHOLD_POS_Y = positive_wall_y - ROBOT_RADIUS
    COLLISION_THRESHOLD_NEG_Y = negative_wall_y + ROBOT_RADIUS

    # Calculate randomized wall length
    random_length_factor = 1 + random.uniform(-RANDOM_FACTOR, RANDOM_FACTOR)
    wall_length = BASE_WALL_LENGTH * random_length_factor

    # Place the robot at the start of the meaning half its size along the x-axis
    meaning_half_size_x = wall_length / 2.0
    robot_start_x = -meaning_half_size_x + ROBOT_RADIUS
    initial_z = 0.0
    robot_translation_field.setSFVec3f([robot_start_x, 0, initial_z])
    robot_node.resetPhysics()

    # Create both walls with randomized distance and length
    create_wall(supervisor, positive_wall_y, wall_length)
    create_wall(supervisor, negative_wall_y, wall_length)

    # --- Main Simulation Loop ---
    while supervisor.step(timestep) != -1:
        # Get current robot position
        robot_position = robot_translation_field.getSFVec3f()
        x, y, z = robot_position  # note: y is vertical in Webots

        collided = False
        if y >= COLLISION_THRESHOLD_POS_Y:
            print(f"Collision detected with positive wall at y={y:.3f}")
            current_left_speed = -BASE_SPEED
            current_right_speed = -BASE_SPEED
            collided = True
        elif y <= COLLISION_THRESHOLD_NEG_Y:
            print(f"Collision detected with negative wall at y={y:.3f}")
            current_left_speed = BASE_SPEED
            current_right_speed = BASE_SPEED
            collided = True

        if collided:
            left_motor.setVelocity(current_left_speed)
            right_motor.setVelocity(current_right_speed)

    print("Simulation ended or interrupted.")

# --- Entry Point ---
if __name__ == "__main__":
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    run_robot_simulation(supervisor, timestep)