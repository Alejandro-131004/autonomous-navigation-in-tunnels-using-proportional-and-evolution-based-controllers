import sys
import math
import random as pyrandom
import numpy as np
from controller import Supervisor, Lidar
# Assuming controllers.utils is available and contains cmd_vel
# If not, you might need to implement cmd_vel or adjust imports
try:
    from controllers.utils import cmd_vel
except ImportError:
    print("Warning: controllers.utils.cmd_vel not found. Using dummy function.")
    def cmd_vel(supervisor, lv, av):
        # Dummy implementation for testing without the utility file
        left_motor = supervisor.getDevice("left wheel motor")
        right_motor = supervisor.getDevice("right wheel motor")
        wheel_radius = 0.02 # Assuming a standard wheel radius
        axle_track = 0.0565 # Assuming a standard axle track
        left_velocity = (lv - av * axle_track / 2.0) / wheel_radius
        right_velocity = (lv + av * axle_track / 2.0) / wheel_radius
        left_motor.setVelocity(left_velocity)
        right_motor.setVelocity(right_velocity)


# --- Configuration ---
ROBOT_NAME = "e-puck"
ROBOT_RADIUS = 0.035  # meters

# Tunnel clearance relative to robot diameter (2 * ROBOT_RADIUS)
MIN_CLEARANCE_FACTOR = 1.5  # Adjusted minimum for potentially tighter turns
MAX_CLEARANCE_FACTOR = 3.0
# Desired clearance factor (between MIN and MAX)
TUNNEL_CLEARANCE_FACTOR = 2.0
# Clamp to valid range
TUNNEL_CLEARANCE_FACTOR = max(MIN_CLEARANCE_FACTOR,
                              min(MAX_CLEARANCE_FACTOR,
                                  TUNNEL_CLEARANCE_FACTOR))
# Distance from centerline to each wall
BASE_WALL_DISTANCE = ROBOT_RADIUS * TUNNEL_CLEARANCE_FACTOR

BASE_SPEED = 0.1
BASE_WALL_LENGTH = 1.0
WALL_THICKNESS = 0.01
WALL_HEIGHT = 0.07
MIN_CURVE_ANGLE = math.radians(10)
MAX_CURVE_ANGLE = math.radians(90)
CURVE_SUBDIVISIONS = 50 # More subdivisions can help with smoother curves and less overlap risk
TIMEOUT_DURATION = 45.0
NUM_CURVES = 3

# Map boundaries (example values, adjust to your world limits)
MAP_X_MIN, MAP_X_MAX = -2.0, 2.0
MAP_Y_MIN, MAP_Y_MAX = -2.0, 2.0

class TunnelBuilder:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        # Get the root children field to add nodes
        self.root_children = supervisor.getRoot().getField("children")

    def create_wall(self, pos, rot, size):
        # Webots node string for a Solid wall
        # Corrected syntax for translation, rotation, size, and nested nodes
        wall = f"""Solid {{
            translation {pos[0]} {pos[1]} {pos[2]}
            rotation {rot[0]} {rot[1]} {rot[2]} {rot[3]}
            children [
                Shape {{
                    appearance Appearance {{
                        material Material {{
                            diffuseColor 1 0 0 # Red color for walls
                        }}
                    }}
                    geometry Box {{
                        size {size[0]} {size[1]} {size[2]}
                    }}
                }}
            ]
        }}"""
        # Import the node into the world
        self.root_children.importMFNodeFromString(-1, wall)

    def build_tunnel(self, num_curves):
        # Ensure the number of curves doesn't exceed the configured maximum
        num_curves = min(num_curves, NUM_CURVES)
        # Randomize the total tunnel length slightly
        length = BASE_WALL_LENGTH * (1 + pyrandom.uniform(-0.15, 0.15))
        # Calculate the length of each straight/curved segment
        # There are num_curves + 1 straight segments (initial, between curves, final)
        segment_length = length / (num_curves + 1)

        # Initialize the transformation matrix T (identity matrix)
        T = np.eye(4)
        # Store the starting position
        start_pos = T[:3, 3].copy()
        walls = [] # List to store wall details (not strictly needed for placement, but useful for tracking)

        # Add the initial straight segment
        if not self._within_bounds(T, segment_length):
            # If the initial segment goes out of bounds, return None
            return None, None, 0
        self._add_straight(T, segment_length, walls)

        # Add curves and straight segments
        for _ in range(num_curves):
            # Randomize curve angle and direction
            angle = pyrandom.uniform(MIN_CURVE_ANGLE, MAX_CURVE_ANGLE) * pyrandom.choice([1, -1])
            # Check if the curve goes out of bounds
            if not self._within_bounds_after_curve(T, angle, segment_length):
                # If it does, stop adding segments
                break
            # Add the curved segment
            self._add_curve(T, angle, segment_length, walls)
            # Check if the straight segment after the curve goes out of bounds
            if not self._within_bounds(T, segment_length):
                break
            # Add the straight segment after the curve
            self._add_straight(T, segment_length, walls)

        # Store the ending position
        end_pos = T[:3, 3].copy()
        # Return start position, end position, and number of walls added
        return start_pos, end_pos, len(walls)

    def _add_straight(self, T, length, walls):
        # Calculate the heading of the current segment
        heading = math.atan2(T[1, 0], T[0, 0])
        # Add walls on both sides of the centerline
        for side in [-1, 1]:
            # Calculate the wall's center position:
            # Start at current T's position, move half the length along T's x-axis (forward),
            # then move BASE_WALL_DISTANCE along T's y-axis (sideways), and half height up.
            pos = T[:3, 3] + T[:3, 0] * (length / 2) + T[:3, 1] * (side * BASE_WALL_DISTANCE) + np.array([0, 0, WALL_HEIGHT / 2])
            # Rotation is around the z-axis based on the heading
            rot = (0, 0, 1, heading)
            # Size of the wall (length, thickness, height)
            size = (length, WALL_THICKNESS, WALL_HEIGHT)
            # Create the wall in the Webots world
            self.create_wall(pos, rot, size)
            # Append wall details to the list (optional, for tracking)
            walls.append((pos, rot, size))
        # Update T to the end of the straight segment
        T[:3, 3] += T[:3, 0] * length

    def _add_curve(self, T, angle, segment_length, walls):
        # Calculate the angle step and length step for each subdivision
        step = angle / CURVE_SUBDIVISIONS
        step_length = segment_length / CURVE_SUBDIVISIONS
        # Add walls for each subdivision of the curve
        for _ in range(CURVE_SUBDIVISIONS):
            # Calculate the heading of the current sub-segment
            heading = math.atan2(T[1, 0], T[0, 0])
            # Add walls on both sides
            for side in [-1, 1]:
                # Calculate the wall's center position:
                # Start at current T's position, move half the step_length along T's x-axis (forward),
                # then move BASE_WALL_DISTANCE along T's y-axis (sideways), and half height up.
                pos = T[:3, 3] + T[:3, 0] * (step_length / 2) + T[:3, 1] * (side * BASE_WALL_DISTANCE) + np.array([0, 0, WALL_HEIGHT / 2])
                # Rotation is based on the heading of the current sub-segment
                rot = (0, 0, 1, heading)
                # Size of the wall (step_length, thickness, height)
                size = (step_length, WALL_THICKNESS, WALL_HEIGHT)
                # Create the wall
                self.create_wall(pos, rot, size)
                # Append wall details (optional)
                walls.append((pos, rot, size))
            # Update T for the next sub-segment:
            # First rotate T by the step angle
            T[:] = T @ self._rotation_z(step)
            # Then translate T by the step_length along its new x-axis (forward)
            T[:3, 3] += T[:3, 0] * step_length

    def _translation(self, x, y, z):
        # Helper function to create a translation matrix
        M = np.eye(4)
        M[:3, 3] = [x, y, z]
        return M

    def _rotation_z(self, angle):
        # Helper function to create a rotation matrix around the z-axis
        c, s = math.cos(angle), math.sin(angle)
        R = np.eye(4)
        R[0, 0], R[0, 1] = c, -s
        R[1, 0], R[1, 1] = s, c
        return R

    def _within_bounds(self, T, length):
        # Check if the end point of a straight segment is within map bounds
        end = T[:3, 3] + T[:3, 0] * length
        return MAP_X_MIN <= end[0] <= MAP_X_MAX and MAP_Y_MIN <= end[1] <= MAP_Y_MAX

    def _within_bounds_after_curve(self, T, angle, seg_len):
        # Check if the end point after a curved segment is within map bounds
        tempT = T.copy()
        step = angle / CURVE_SUBDIVISIONS
        step_length = seg_len / CURVE_SUBDIVISIONS
        for _ in range(CURVE_SUBDIVISIONS):
            tempT = tempT @ self._rotation_z(step)
            tempT[:3, 3] += tempT[:3, 0] * step_length
        end = tempT[:3, 3]
        return MAP_X_MIN <= end[0] <= MAP_X_MAX and MAP_Y_MIN <= end[1] <= MAP_Y_MAX

class SimulationManager:
    def __init__(self):
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        # Get robot node and its translation field
        self.robot = self.supervisor.getFromDef(ROBOT_NAME)
        self.translation = self.robot.getField("translation")
        # Get and enable the Lidar sensor
        self.lidar = self.supervisor.getDevice("lidar")
        self.lidar.enable(self.timestep)

        # Initialize statistics dictionary
        self.stats = {'total_collisions': 0, 'successful_runs': 0, 'failed_runs': 0}

    def run_experiment(self, num_runs):
        # Run the experiment for the specified number of runs
        for run in range(num_runs):
            print(f"--- Starting Run {run + 1} ---")
            # Build a new random tunnel
            start_pos, end_pos, walls_added = TunnelBuilder(self.supervisor).build_tunnel(NUM_CURVES)

            # If tunnel building failed (out of bounds), skip this run
            if start_pos is None:
                print("Tunnel out of bounds, skipping run.")
                continue

            # Place the robot at the start of the tunnel and reset physics
            self.translation.setSFVec3f([start_pos[0], start_pos[1], ROBOT_RADIUS])
            self.robot.resetPhysics()

            # Get wheel motors and set to infinite position control (velocity mode)
            left = self.supervisor.getDevice("left wheel motor")
            right = self.supervisor.getDevice("right wheel motor")
            left.setPosition(float('inf'))
            right.setPosition(float('inf'))
            left.setVelocity(0) # Start with zero velocity
            right.setVelocity(0)

            collision_count = 0
            # Flag to prevent counting multiple collisions for a single contact event
            flag = False
            start_time = self.supervisor.getTime() # Record start time for timeout

            # Simulation loop
            while self.supervisor.step(self.timestep) != -1:
                # Check for timeout
                if self.supervisor.getTime() - start_time > TIMEOUT_DURATION:
                    print("Timeout")
                    break

                # Get Lidar data
                data = self.lidar.getRangeImage()
                # Process Lidar data to get linear and angular velocity commands
                lv, av = self._process_lidar(data)
                # Apply velocity commands to the robot
                cmd_vel(self.supervisor, lv, av)

                # Get current robot position
                pos = np.array(self.translation.getSFVec3f())

                # Simple collision detection based on x-position (assuming tunnel is aligned with y-axis initially)
                # This collision detection might need refinement depending on the tunnel path
                current_distance_from_center = abs(pos[1]) # Use y-coordinate for distance from centerline
                if current_distance_from_center > BASE_WALL_DISTANCE - ROBOT_RADIUS and not flag:
                    collision_count += 1
                    flag = True # Set flag to true to indicate collision is ongoing
                elif current_distance_from_center <= BASE_WALL_DISTANCE - ROBOT_RADIUS:
                    flag = False # Reset flag when robot is back within the safe zone

                # Check if the robot has reached the end of the tunnel
                if end_pos is not None and np.linalg.norm(pos[:2] - end_pos[:2]) < 0.1:
                    print(f"Reached end in {self.supervisor.getTime()-start_time:.1f}s")
                    break

            # Remove the generated walls after the run
            self._remove_walls(walls_added)

            # Update statistics
            self.stats['total_collisions'] += collision_count
            if collision_count == 0 and end_pos is not None and np.linalg.norm(pos[:2] - end_pos[:2]) < 0.1:
                 self.stats['successful_runs'] += 1
            else:
                 self.stats['failed_runs'] += 1
            print(f"Run {run + 1} finished with {collision_count} collisions.")


        # Print the final summary after all runs
        self._print_summary()


    def _process_lidar(self, data):
        # Basic obstacle avoidance and wall following logic
        # Extracts range data from front, left, and right
        front = data[len(data) // 2]
        left = data[len(data) // 4]
        right = data[3 * len(data) // 4]

        target_dist = 0.08 # Desired distance to walls
        kp = 0.8 # Proportional gain for angular velocity
        safe_dist = 0.06 # Distance to stop if obstacle is too close

        # Stop if obstacle is too close in front
        if front < safe_dist:
            return 0, 0

        # Calculate error based on difference in left and right distances
        error = (left - right) * kp
        # Adjust linear speed based on proximity to walls
        linear_velocity = BASE_SPEED * (0.8 if min(left, right) < target_dist * 1.5 else 1.0)
        # Return linear and angular velocities
        return linear_velocity, error

    def _remove_walls(self, count):
        # Remove the last 'count' children from the root node
        children = self.supervisor.getRoot().getField("children")
        for _ in range(count):
            children.removeMF(-1)

    def _print_summary(self):
        # Print the final summary of the experiment
        print("\n=== Final Results ===")
        print(f"Successful runs: {self.stats['successful_runs']}")
        print(f"Failed runs: {self.stats['failed_runs']}")
        print(f"Total collisions: {self.stats['total_collisions']}")
        total_runs = self.stats['successful_runs'] + self.stats['failed_runs']
        if total_runs > 0:
             print(f"Success rate: {self.stats['successful_runs'] / total_runs * 100:.1f}%")
        else:
             print("No runs completed.")


if __name__ == "__main__":
    # Create a SimulationManager instance and run the experiment
    experiment = SimulationManager()
    experiment.run_experiment(10) # Reduced number of runs for quicker testing
