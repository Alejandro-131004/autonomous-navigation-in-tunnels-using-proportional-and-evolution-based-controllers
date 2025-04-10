import sys
import math
import random as pyrandom

# Import supervisor and devices
from controller import Supervisor, Lidar, Robot
# Assuming the cmd_vel helper function is available in controllers.utils.
# It is typically a thin wrapper that sets left and right wheel motor speeds.
from controllers.utils import cmd_vel

# --- Configuration for environment ---
ROBOT_NAME = "e-puck"
ROBOT_RADIUS = 0.035
BASE_SPEED = 0.1  # max forward speed for the controller
# Wall parameters
BASE_WALL_DISTANCE_FROM_CENTER = 0.1
BASE_WALL_LENGTH = 1.5
WALL_THICKNESS = 0.01
WALL_HEIGHT = 0.04
RANDOM_FACTOR = 0.2

# Parameters for wall (used in supervisor wall creation)
ROTATION_AXIS_X = 0.577351
ROTATION_AXIS_Y = 0.577351
ROTATION_AXIS_Z = 0.577351
ROTATION_ANGLE  = 2.0944  # about 120° in radians

# --- Controller parameters used in the distance_handler ---
# These gains and values are used to compute motor commands from lidar distances.
def distance_handler(direction: int, dist_values: [float]) -> (float, float):
    maxSpeed: float = BASE_SPEED
    distP: float = 10.0
    angleP: float = 7.0
    wallDist: float = 0.1

    size: int = len(dist_values)
    min_index: int = 0
    if direction == -1:
        min_index = size - 1
    for i in range(size):
        idx: int = i if direction == 1 else (size - 1 - i)
        if dist_values[idx] < dist_values[min_index] and dist_values[idx] > 0.0:
            min_index = idx

    angle_increment: float = 2 * math.pi / (size - 1)
    angleMin: float = (size // 2 - min_index) * angle_increment
    distMin: float = dist_values[min_index]
    distFront: float = dist_values[size // 2]
    distSide: float = dist_values[size // 4] if (direction == 1) else dist_values[3 * size // 4]
    distBack: float = dist_values[0]

    # Debug prints to trace sensor values
    print("distMin:", distMin, "angleMin (deg):", angleMin * 180 / math.pi)

    if math.isfinite(distMin):
        # If obstacles are in front and on the side or back, prepare to unblock.
        if distFront < 1.25 * wallDist and (distSide < 1.25 * wallDist or distBack < 1.25 * wallDist):
            print("UNBLOCK")
            angular_vel = direction * -1  # small corrective spin
        else:
            print("REGULAR")
            angular_vel = direction * distP * (distMin - wallDist) + angleP * (angleMin - direction * math.pi / 2)
            print("angular_vel:", angular_vel)
        if distFront < wallDist:
            # Turn on the spot.
            print("TURN")
            linear_vel = 0
        elif distFront < 2 * wallDist or distMin < wallDist * 0.75 or distMin > wallDist * 1.25:
            # Slow down.
            print("SLOW")
            linear_vel = 0.5 * maxSpeed
        else:
            # Cruise.
            print("CRUISE")
            linear_vel = maxSpeed
    else:
        # Wander if no valid sensor reading.
        print("WANDER")
        angular_vel = pyrandom.normal(loc=0.0, scale=1.0)
        print("angular_vel:", angular_vel)
        linear_vel = maxSpeed

    return linear_vel, angular_vel


# --- Supervisor Methods for setting up and cleaning the environment ---
def create_wall(supervisor, y_position, length):
    """
    Creates a wall and returns its index in the children field so it can be removed later.
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
    # Return the index of the new node (assumed to be the last one)
    return children_field.getCount() - 1

def remove_walls(supervisor, initial_children_count):
    """
    Remove any nodes added beyond the initial children count.
    """
    root_children = supervisor.getRoot().getField("children")
    while root_children.getCount() > initial_children_count:
        root_children.removeMF(root_children.getCount() - 1)

# --- Main single-run simulation ---
def run_single_simulation_run(supervisor, timestep, initial_children_count):
    """
    Runs one simulation run. This run uses the lidar-based controller and stops when
    the robot reaches a set finish line (based on its x coordinate).
    Returns the collision count in this run.
    """
    collision_count = 0
    collision_flag = False  # to avoid counting the same collision repeatedly

    # Get the robot node using its DEF name.
    robot_node = supervisor.getFromDef(ROBOT_NAME)
    if robot_node is None:
        print(f"Error: Robot '{ROBOT_NAME}' not found.")
        sys.exit(1)
    translation_field = robot_node.getField("translation")

    # Get wheel motors (assuming cmd_vel uses them).
    left_motor = supervisor.getDevice("left wheel motor")
    right_motor = supervisor.getDevice("right wheel motor")
    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))

    # Get and enable the lidar device.
    lidar: Lidar = supervisor.getDevice("lidar")
    lidar.enable(timestep)
    lidar.enablePointCloud()

    # Randomize wall distance and length.
    random_distance_factor = 1 + pyrandom.uniform(-RANDOM_FACTOR, RANDOM_FACTOR)
    wall_distance = BASE_WALL_DISTANCE_FROM_CENTER * random_distance_factor
    positive_wall_y = wall_distance
    negative_wall_y = -wall_distance

    # Set collision thresholds based on robot radius.
    COLLISION_THRESHOLD_POS_Y = positive_wall_y - ROBOT_RADIUS
    COLLISION_THRESHOLD_NEG_Y = negative_wall_y + ROBOT_RADIUS

    random_length_factor = 1 + pyrandom.uniform(-RANDOM_FACTOR, RANDOM_FACTOR)
    wall_length = BASE_WALL_LENGTH * random_length_factor

    # Define start and finish positions (using x coordinate).
    half_length = wall_length / 2.0
    robot_start_x = -half_length + ROBOT_RADIUS
    finish_x = half_length - ROBOT_RADIUS

    # Place the robot at the start.
    translation_field.setSFVec3f([robot_start_x, 0, 0.0])
    robot_node.resetPhysics()

    # Create the walls.
    wall1_index = create_wall(supervisor, positive_wall_y, wall_length)
    wall2_index = create_wall(supervisor, negative_wall_y, wall_length)

    # --- Main simulation loop for this run ---
    while supervisor.step(timestep) != -1:
        # Read lidar values and compute commands via the sensor-based controller.
        lidar_values = lidar.getRangeImage()
        linear_vel, angular_vel = distance_handler(1, lidar_values)
        cmd_vel(supervisor, linear_vel, angular_vel)

        # Check robot position:
        pos = translation_field.getSFVec3f()
        x, y, _ = pos

        # Count a collision if the robot's y exceeds the thresholds.
        if (y >= COLLISION_THRESHOLD_POS_Y or y <= COLLISION_THRESHOLD_NEG_Y):
            if not collision_flag:
                collision_count += 1
                collision_flag = True
                print("Collision detected at y =", y)
        else:
            collision_flag = False

        # End run when the robot reaches or passes the finish line along the x-axis.
        if x >= finish_x:
            print(f"Run finished: robot reached x = {x:.3f} (finish_x = {finish_x:.3f})")
            break

    # Remove walls from this run.
    remove_walls(supervisor, initial_children_count)
    return collision_count

# --- Main entry point ---
if __name__ == "__main__":
    # Using Supervisor so we can add/remove nodes.
    supervisor: Supervisor = Supervisor()
    timestep: int = int(supervisor.getBasicTimeStep())

    # Record initial children count (used for cleaning up walls).
    initial_children_count = supervisor.getRoot().getField("children").getCount()

    total_runs = 100
    total_collisions = 0
    max_collisions = 0

    for run in range(total_runs):
        print(f"\n--- Run {run+1} ---")
        run_collisions = run_single_simulation_run(supervisor, timestep, initial_children_count)
        print(f"Collisions in run {run+1}: {run_collisions}")
        total_collisions += run_collisions
        if run_collisions > max_collisions:
            max_collisions = run_collisions
        print("Preparing for next run...")

    print("\n--- Summary After 100 Runs ---")
    print("Total number of collisions:", total_collisions)
    print("Highest number of collisions in a single run:", max_collisions)
