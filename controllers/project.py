import sys
import math
import random as pyrandom # Kept original import name
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
CURVE_SUBDIVISIONS = 20 # Increased subdivisions for smoother curves
TIMEOUT_DURATION = 45.0
NUM_CURVES = 3

# Map boundaries (example values, adjust to your world limits)
MAP_X_MIN, MAP_X_MAX = -2.0, 2.0
MAP_Y_MIN, MAP_Y_MAX = -2.0, 2.0

# Small overlap factor to ensure no gaps in curves
OVERLAP_FACTOR = 1.01 # Adjusted overlap factor based on user feedback

class TunnelBuilder:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        # Get the root children field to add nodes
        self.root_children = supervisor.getRoot().getField("children")
        # Initialize wall distance here, will be set in build_tunnel
        self.base_wall_distance = 0

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
        # Randomize the tunnel clearance factor for this tunnel
        tunnel_clearance_factor = pyrandom.uniform(MIN_CLEARANCE_FACTOR, MAX_CLEARANCE_FACTOR)
        # Calculate the base wall distance based on the randomized clearance
        self.base_wall_distance = ROBOT_RADIUS * tunnel_clearance_factor
        print(f"Building tunnel with clearance factor: {tunnel_clearance_factor:.2f}")


        # Ensure the number of curves doesn't exceed the configured maximum
        num_curves = min(num_curves, NUM_CURVES)
        # Randomize the total tunnel length slightly
        length = BASE_WALL_LENGTH * (1 + pyrandom.uniform(-0.15, 0.15))
        # Calculate the length of each straight/curved segment (centerline length)
        segment_length = length / (num_curves + 1)

        # Initialize the transformation matrix T (identity matrix)
        T = np.eye(4)
        # Store the starting position
        start_pos = T[:3, 3].copy()
        walls = [] # List to store wall details (not strictly needed for placement, but useful for tracking)
        straight_segments_data = [] # Store data for straight segments to place obstacles

        # Add the initial straight segment
        if not self._within_bounds(T, segment_length):
            # If the initial segment goes out of bounds, return None
            return None, None, 0
        segment_start_pos = T[:3, 3].copy()
        self._add_straight(T, segment_length, walls)
        segment_end_pos = T[:3, 3].copy()
        segment_heading = math.atan2(T[1, 0], T[0, 0]) # Heading after moving
        straight_segments_data.append((segment_start_pos, segment_end_pos, segment_heading, segment_length))


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

            # Add the straight segment after the curve
            if not self._within_bounds(T, segment_length):
                break
            segment_start_pos = T[:3, 3].copy()
            self._add_straight(T, segment_length, walls)
            segment_end_pos = T[:3, 3].copy()
            segment_heading = math.atan2(T[1, 0], T[0, 0]) # Heading after moving
            straight_segments_data.append((segment_start_pos, segment_end_pos, segment_heading, segment_length))


        end_pos = T[:3, 3].copy()

        # Add obstacles to the straight segments
        self._add_obstacles(straight_segments_data, walls) # Pass walls list to add obstacles

        # Return start position, end position, and number of walls added (including obstacles)
        return start_pos, end_pos, len(walls)

    def _add_straight(self, T, length, walls):
        # Calculate the heading of the current segment
        heading = math.atan2(T[1, 0], T[0, 0])
        # Add walls on both sides of the centerline
        for side in [-1, 1]:
            # Calculate the wall's center position:
            # Start at current T's position, move half the length along T's x-axis (forward),
            # then move self.base_wall_distance along T's y-axis (sideways), and half height up.
            pos = T[:3, 3] + T[:3, 0] * (length / 2) + T[:3, 1] * (side * self.base_wall_distance) + np.array([0, 0, WALL_HEIGHT / 2])
            # Rotation is around the z-axis based on the heading
            rot = (0, 0, 1, heading)
            # Size of the wall (length, thickness, height)
            size = (length, WALL_THICKNESS, WALL_HEIGHT)
            self.create_wall(pos, rot, size)
            walls.append((pos, rot, size)) # Append wall details to the walls list
        # Update T to the end of the straight segment
        T[:3, 3] += T[:3, 0] * length


    def _add_curve(self, T, angle, segment_length, walls):
        # Calculate the angle step for each subdivision
        step = angle / CURVE_SUBDIVISIONS
        # Calculate the centerline length step for each subdivision
        centerline_step_length = segment_length / CURVE_SUBDIVISIONS

        # Radii of the inner and outer edges of the tunnel centerline based on randomized wall distance
        r_centerline = self.base_wall_distance
        r_inner_edge = r_centerline - WALL_THICKNESS / 2.0
        r_outer_edge = r_centerline + WALL_THICKNESS / 2.0

        # Add walls for each subdivision of the curve
        for _ in range(CURVE_SUBDIVISIONS):
            # Calculate the pose of the centerline at the start of this step
            T_start_step = T.copy()

            # Calculate the pose of the centerline at the middle of this step
            # This is T_start_step rotated by step/2 and translated by centerline_step_length/2
            T_mid_step = T_start_step @ self._rotation_z(step / 2)
            T_mid_step[:3, 3] += T_mid_step[:3, 0] * (centerline_step_length / 2)

            # Calculate the heading at the middle of this step for rotation
            heading = math.atan2(T_mid_step[1, 0], T_mid_step[0, 0])
            rot = (0, 0, 1, heading)

            # Add walls on both sides
            for side in [-1, 1]:
                # Calculate the wall segment length based on the arc length at the wall's radius
                if side == -1: # Inner wall
                    wall_length = r_inner_edge * abs(step)
                else: # Outer wall
                    wall_length = r_outer_edge * abs(step)

                # Add a small overlap to the wall length
                wall_length += OVERLAP_FACTOR * WALL_THICKNESS

                # Calculate the wall's center position relative to the start of the step
                # Move sideways by self.base_wall_distance, half height up,
                # and forward by half the wall's *adjusted* arc length along the initial direction of the step
                pos = T_start_step[:3, 3] + T_start_step[:3, 1] * (side * self.base_wall_distance) + np.array([0, 0, WALL_HEIGHT / 2]) + T_start_step[:3, 0] * (wall_length / 2)


                # Size of the wall (calculated length, thickness, height)
                size = (wall_length, WALL_THICKNESS, WALL_HEIGHT)

                # Create the wall
                self.create_wall(pos, rot, size)
                walls.append((pos, rot, size)) # Append wall details to the walls list

            # Update T to the end of the current step (centerline movement) for the next iteration
            T[:] = T_start_step @ self._rotation_z(step)
            T[:3, 3] += T[:3, 0] * centerline_step_length

    def _add_obstacles(self, straight_segments_data, walls):
        """
        Place NUM_OBSTACLES perpendicular walls into the middle straight segments.
        straight_segments_data: list of (start_pos, end_pos, heading, length)
        walls: list to append (pos, rot, size) tuples for cleanup/tracking
        """
        # skip entrance & exit straights
        segments = straight_segments_data[1:-1]
        if not segments:
            print("Not enough segments for obstacles.")
            return

        used = set()
        placed_positions = []

        # half-width of tunnel from centerline
        tunnel_half = ROBOT_RADIUS * MIN_CLEARANCE_FACTOR
        # obstacle span across tunnel, leaving MIN_ROBOT_CLEARANCE on the other side
        obstacle_length = 2 * tunnel_half - MIN_ROBOT_CLEARANCE - WALL_THICKNESS

        for _ in range(NUM_OBSTACLES):
            # pick an unused segment index
            choices = [i for i in range(len(segments)) if i not in used]
            if not choices:
                break
            idx = pyrandom.choice(choices)
            used.add(idx)

            start, end, heading, seg_len = segments[idx]
            # choose a point 20–80% along the straight
            d = pyrandom.uniform(0.2 * seg_len, 0.8 * seg_len)
            dir_vec = np.array([math.cos(heading), math.sin(heading), 0.0])
            pos = np.array(start) + dir_vec * d

            # pick inner/outer side
            side = pyrandom.choice([-1, +1])
            perp = np.array([-dir_vec[1], dir_vec[0], 0.0])
            shift = side * (MIN_ROBOT_CLEARANCE / 2 + obstacle_length / 2)
            pos += perp * shift
            pos[2] = WALL_HEIGHT / 2.0

            # no extra rotation—box X-axis is already perpendicular to the corridor
            rot = (0.0, 0.0, 1.0, heading)
            # size: X = thickness along path, Y = span across path
            size = (WALL_THICKNESS, obstacle_length, WALL_HEIGHT)

            # avoid clustering
            if any(np.linalg.norm(pos[:2] - p) < MIN_OBSTACLE_DISTANCE
                   for p in placed_positions):
                print("Skipping obstacle—too close to another.")
                continue

            # create and record
            self.create_wall(pos, rot, size)
            walls.append((pos, rot, size))
            placed_positions.append(pos[:2].copy())

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
        centerline_step_length = seg_len / CURVE_SUBDIVISIONS
        for _ in range(CURVE_SUBDIVISIONS):
            tempT = tempT @ self._rotation_z(step)
            tempT[:3, 3] += tempT[:3, 0] * centerline_step_length
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
        # Removed lidar.enablePointCloud() as it's not used by the new control logic

        # Initialize statistics dictionary
        self.stats = {'total_collisions': 0, 'successful_runs': 0, 'failed_runs': 0}

    def run_experiment(self, num_runs):
        # Run the experiment for the specified number of runs
        for run in range(num_runs):
            print(f"--- Starting Run {run + 1} ---")
            # Build a new random tunnel
            # Pass the supervisor to the TunnelBuilder constructor
            tunnel_builder = TunnelBuilder(self.supervisor)
            start_pos, end_pos, walls_added = tunnel_builder.build_tunnel(NUM_CURVES)

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
                # Process Lidar data using the new logic
                lv, av = self._process_lidar(data)
                # Apply velocity commands to the robot
                cmd_vel(self.supervisor, lv, av)

                # Get current robot position
                pos = np.array(self.translation.getSFVec3f())

                # Simple collision detection based on distance from centerline
                # This collision detection might need refinement depending on the tunnel path
                # It assumes the robot should stay within BASE_WALL_DISTANCE from the centerline
                # A more robust collision detection would check for actual contact with wall nodes.
                # Need to get the current BASE_WALL_DISTANCE from the tunnel_builder instance
                current_base_wall_distance = tunnel_builder.base_wall_distance
                current_distance_from_center = abs(pos[1]) # Use y-coordinate for distance from centerline
                # The collision detection logic here is still based on the overall tunnel width.
                # Detecting collisions with the new smaller obstacles requires a different approach,
                # ideally using Webots' collision detection events or checking proximity to obstacle nodes.
                # For now, the existing collision check remains, but it won't accurately count collisions with the new obstacles.
                if current_distance_from_center > current_base_wall_distance - ROBOT_RADIUS and not flag:
                    # This condition might not be met by colliding with the new obstacles
                    # collision_count += 1 # Uncomment if you implement more robust collision detection
                    flag = True # Set flag to true to indicate collision is ongoing
                elif current_distance_from_center <= current_base_wall_distance - ROBOT_RADIUS:
                    flag = False # Reset flag when robot is back within the safe zone


                # Check if the robot has reached the end of the tunnel
                if end_pos is not None and np.linalg.norm(pos[:2] - end_pos[:2]) < 0.1:
                    print(f"Reached end in {self.supervisor.getTime()-start_time:.1f}s")
                    break

            # Remove the generated walls after the run
            self._remove_walls(walls_added)

            # Update statistics
            # The collision count here might not be accurate due to the simple collision detection
            self.stats['total_collisions'] += collision_count
            # A run is successful if there were no collisions (based on current detection) and the robot reached the end
            if collision_count == 0 and end_pos is not None and np.linalg.norm(pos[:2] - end_pos[:2]) < 0.1:
                 self.stats['successful_runs'] += 1
            else:
                 self.stats['failed_runs'] += 1
            print(f"Run {run + 1} finished with {collision_count} collisions.")


        # Print the final summary after all runs
        self._print_summary()

    def _process_lidar(self, dist_values: [float]) -> (float, float):
        """
        Robot control logic based on Lidar data.
        Adapted from IRI - TP2 - Ex 4 by Gonçalo Leão.
        """
        # Assuming direction 1 for wall following (e.g., right wall)
        # You might need to adjust this or add logic to determine the direction
        # based on the tunnel structure or robot's initial position/goal.
        direction: int = 1

        maxSpeed: float = 0.1
        distP: float = 10.0  # Proportional gain for distance error
        angleP: float = 7.0  # Proportional gain for angle error
        wallDist: float = 0.1 # Desired distance to the wall

        # Find the angle of the ray that returned the minimum distance
        size: int = len(dist_values)
        if size == 0:
            # Handle case where lidar data is empty
            return 0.0, 0.0

        min_index: int = 0
        if direction == -1:
            min_index = size - 1
        for i in range(size):
            idx: int = i
            if direction == -1:
                idx = size - 1 - i
            # Ensure distance is valid (greater than 0) before considering it minimum
            if dist_values[idx] < dist_values[min_index] and dist_values[idx] > 0.0:
                min_index = idx
            # If the current min_index has an invalid distance, find the next valid one
            elif dist_values[min_index] <= 0.0 and dist_values[idx] > 0.0:
                 min_index = idx


        angle_increment: float = 2*math.pi / (size - 1) if size > 1 else 0.0
        angleMin: float = (size // 2 - min_index) * angle_increment
        distMin: float = dist_values[min_index]

        # Get distances from specific directions
        distFront: float = dist_values[size // 2] if size > 0 else float('inf')
        distSide: float = dist_values[size // 4] if size > 0 and size // 4 < size else float('inf') if (direction == 1) else dist_values[3*size // 4] if size > 0 and 3*size // 4 < size else float('inf')
        distBack: float = dist_values[0] if size > 0 else float('inf')


        # Prepare message for the robot's motors
        linear_vel: float
        angular_vel: float

        # print("distMin", distMin) # Commented out to reduce console output
        # print("angleMin", angleMin*180/math.pi) # Commented out

        # Decide the robot's behavior
        if math.isfinite(distMin):
            # Check for potential unblocking scenario (stuck)
            if distFront < 1.25*wallDist and (distSide < 1.25*wallDist or distBack < 1.25*wallDist):
                # print("UNBLOCK") # Commented out
                # Turn away from the detected obstacles
                angular_vel = direction * -1 * maxSpeed # Use maxSpeed for turning velocity
                linear_vel = 0 # Stop linear movement while unblocking
            else:
                # print("REGULAR") # Commented out
                # Calculate angular velocity based on distance and angle errors
                # Error 1: Difference between minimum distance and desired wall distance
                # Error 2: Difference between angle of minimum distance and desired angle (pi/2 for side wall)
                angular_vel = direction * distP * (distMin - wallDist) + angleP * (angleMin - direction * math.pi / 2)
                # print("angular_vel", angular_vel, " wall comp = ", direction * distP * (distMin - wallDist), ", angle comp = ", angleP * (angleMin - direction * math.pi / 2)) # Commented out

            # Adjust linear velocity based on front distance
            if distFront < wallDist:
                # If obstacle is very close in front, stop and turn
                # print("TURN") # Commented out
                linear_vel = 0
                # Angular velocity is already calculated above
            elif distFront < 2 * wallDist or distMin < wallDist * 0.75 or distMin > wallDist * 1.25:
                # If obstacle is somewhat close or not at desired wall distance, slow down
                # print("SLOW") # Commented out
                linear_vel = 0.5 * maxSpeed
                # Angular velocity is already calculated above
            else:
                # Otherwise, cruise at max speed
                # print("CRUISE") # Commented out
                linear_vel = maxSpeed
                # Angular velocity is already calculated above
        else:
            # If no finite minimum distance (e.g., open space), wander
            # print("WANDER") # Commented out
            # Use numpy.random.normal for random angular velocity
            angular_vel = np.random.normal(loc=0.0, scale=1.0) * maxSpeed # Scale random value by maxSpeed
            # print("angular_vel", angular_vel) # Commented out
            linear_vel = maxSpeed # Continue moving forward while wandering

        # Clamp angular velocity to a reasonable range if needed (optional but good practice)
        # angular_vel = max(-maxSpeed, min(maxSpeed, angular_vel))


        return linear_vel, angular_vel

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
