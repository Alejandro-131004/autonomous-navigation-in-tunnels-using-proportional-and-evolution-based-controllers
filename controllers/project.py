import sys
import random
from controller import Supervisor

# --- Configuration ---
ROBOT_NAME = "e-puck"
ROBOT_RADIUS = 0.035
BASE_SPEED = 3.0

# Base wall parameters
BASE_WALL_DISTANCE_FROM_CENTER = 0.1
BASE_WALL_LENGTH = 1.5
WALL_THICKNESS = 0.01
WALL_HEIGHT = 0.04

# Randomization factor (variation up to 20%)
RANDOM_FACTOR = 0.2

# Collision thresholds and wall orientation parameters
ROTATION_AXIS_X = 0.577351
ROTATION_AXIS_Y = 0.577351
ROTATION_AXIS_Z = 0.577351
ROTATION_ANGLE  = 2.0944  # about 120 degrees in radians

def create_wall(supervisor, y_position, length):
    """
    Create a single wall with the specified rotation, length and position.
    Returns a reference to the newly created node so it can be removed later.
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
    index = children_field.getCount()  # index where the wall will be added
    children_field.importMFNodeFromString(-1, wall_string)
    # Return the index so it can be removed later.
    return index

def remove_wall(supervisor, index):
    """
    Remove the wall node at the given index from the simulation.
    """
    root_node = supervisor.getRoot()
    children_field = root_node.getField("children")
    # Ensure the index is valid.
    if index < children_field.getCount():
        children_field.removeMF(index)

def run_single_simulation_run(supervisor, timestep, initial_children_count):
    """
    Runs a single simulation run.
    The run ends when the robot reaches the finish line along the x-axis.
    Returns the number of collisions recorded in this run, and the indices of the created walls.
    """
    collision_count = 0

    # Get the robot node using its DEF name.
    robot_node = supervisor.getFromDef(ROBOT_NAME)
    if robot_node is None:
        print(f"Error: Robot node '{ROBOT_NAME}' not found. Check DEF name in Webots.")
        sys.exit(1)

    robot_translation_field = robot_node.getField("translation")

    # Get wheel motors.
    try:
        left_motor = supervisor.getDevice("left wheel motor")
        right_motor = supervisor.getDevice("right wheel motor")
        left_motor.setPosition(float('inf'))
        right_motor.setPosition(float('inf'))
    except Exception as e:
        print(f"Error getting motors: {e}. Ensure the wheel motors exist.")
        sys.exit(1)

    # Randomize wall distance and length.
    random_distance_factor = 1 + random.uniform(-RANDOM_FACTOR, RANDOM_FACTOR)
    wall_distance = BASE_WALL_DISTANCE_FROM_CENTER * random_distance_factor
    positive_wall_y = wall_distance
    negative_wall_y = -wall_distance

    # Collision thresholds (using the robot radius for clearance).
    COLLISION_THRESHOLD_POS_Y = positive_wall_y - ROBOT_RADIUS
    COLLISION_THRESHOLD_NEG_Y = negative_wall_y + ROBOT_RADIUS

    random_length_factor = 1 + random.uniform(-RANDOM_FACTOR, RANDOM_FACTOR)
    wall_length = BASE_WALL_LENGTH * random_length_factor

    # Define start and finish positions.
    half_length = wall_length / 2.0
    robot_start_x = -half_length + ROBOT_RADIUS
    finish_x = half_length - ROBOT_RADIUS

    # Place the robot at the start position.
    robot_translation_field.setSFVec3f([robot_start_x, 0, 0.0])
    robot_node.resetPhysics()

    # Create the walls and store the indices for removal.
    wall1_index = create_wall(supervisor, positive_wall_y, wall_length)
    wall2_index = create_wall(supervisor, negative_wall_y, wall_length)

    # Set initial forward velocity along the x-axis.
    left_motor.setVelocity(BASE_SPEED)
    right_motor.setVelocity(BASE_SPEED)

    # --- Main Simulation Loop for a Single Run ---
    while supervisor.step(timestep) != -1:
        robot_position = robot_translation_field.getSFVec3f()
        x, y, z = robot_position  # note: here x is the forward coordinate

        # Check if the robot reached the finish line.
        if x >= finish_x:
            print(f"Run finished: robot reached x = {x:.3f} (finish_x = {finish_x:.3f})")
            break

        collided = False

        # Check for collision with positive wall.
        if y >= COLLISION_THRESHOLD_POS_Y:
            print(f"Collision with positive wall at y = {y:.3f}")
            left_motor.setVelocity(-BASE_SPEED)
            right_motor.setVelocity(-BASE_SPEED)
            collided = True

        # Check for collision with negative wall.
        elif y <= COLLISION_THRESHOLD_NEG_Y:
            print(f"Collision with negative wall at y = {y:.3f}")
            left_motor.setVelocity(BASE_SPEED)
            right_motor.setVelocity(BASE_SPEED)
            collided = True

        # Count collision if detected.
        if collided:
            collision_count += 1
            # Optionally, you might want to insert a brief delay (e.g., a few simulation steps)
            # to allow the robot to reverse away from the wall before resuming its course.
        else:
            # Ensure the robot continues with forward motion.
            left_motor.setVelocity(BASE_SPEED)
            right_motor.setVelocity(BASE_SPEED)

    return collision_count, (wall1_index, wall2_index)

# --- Main Entry Point ---
if __name__ == "__main__":
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())

    # Record the initial number of children nodes in the root.
    initial_children_count = supervisor.getRoot().getField("children").getCount()

    total_runs = 100
    total_collisions = 0
    max_collisions = 0

    for run in range(total_runs):
        print(f"\n--- Run {run+1} ---")
        run_collisions, wall_indices = run_single_simulation_run(supervisor, timestep, initial_children_count)
        print(f"Collisions in run {run+1}: {run_collisions}")
        total_collisions += run_collisions
        if run_collisions > max_collisions:
            max_collisions = run_collisions

        # Remove the two wall nodes that were added.
        # Note: The walls were appended to the children field,
        # so they are likely at the end. Remove the higher index first.
        wall1_index, wall2_index = wall_indices
        # It is possible that other additions might shift these indices,
        # so an alternate approach is to remove nodes until children count is back to the original.
        current_children_count = supervisor.getRoot().getField("children").getCount()
        while current_children_count > initial_children_count:
            supervisor.getRoot().getField("children").removeMF(current_children_count - 1)
            current_children_count = supervisor.getRoot().getField("children").getCount()

        # Instead of simulationReset(), we now simply proceed to the next run.
        # A slight pause can be added if needed.
        print("Resetting run environment...")

    print("\n--- Summary After 100 Runs ---")
    print(f"Total number of collisions: {total_collisions}")
    print(f"Highest number of collisions in a single run: {max_collisions}")
