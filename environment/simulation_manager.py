import numpy as np
import math
import pickle  # Import for saving/loading models
import os  # Import for path manipulation

# Import necessary constants and functions from configuration
from environment.configuration import ROBOT_NAME, ROBOT_RADIUS, TIMEOUT_DURATION, \
    get_stage_parameters, MAX_DIFFICULTY_STAGE

# Import TunnelBuilder for generating environments
from environment.tunnel import TunnelBuilder

# Assuming cmd_vel is defined in agent.controller
# If agent.controller is not a file, it might be part of your project.py or another utility.
# Adjust this import if needed.
try:
    from agent.controller import cmd_vel
except ImportError:
    print("Warning: agent.controller.cmd_vel not found. Using dummy function.")


    # Dummy implementation if agent.controller is not available
    # You might need to adjust wheel_radius and axle_track for your specific robot
    def cmd_vel(supervisor, lv, av):
        left_motor = supervisor.getDevice("left wheel motor")
        right_motor = supervisor.getDevice("right wheel motor")
        wheel_radius = 0.02  # Example value, adjust as per your robot
        axle_track = 0.0565  # Example value, adjust as per your robot
        left_velocity = (lv - av * axle_track / 2.0) / wheel_radius
        right_velocity = (lv + av * axle_track / 2.0) / wheel_radius
        left_motor.setVelocity(left_velocity)
        right_motor.setVelocity(right_velocity)

# Typing for clarity
from typing import Any, List, Tuple


class SimulationManager:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.timestep = int(self.supervisor.getBasicTimeStep())

        # Get robot node and its fields
        self.robot = self.supervisor.getFromDef(ROBOT_NAME)
        if self.robot is None:
            raise ValueError(f"Robot with DEF name '{ROBOT_NAME}' not found. Please check your .wbt file.")
        self.translation = self.robot.getField("translation")
        self.rotation = self.robot.getField("rotation")  # Get the rotation field

        # Get and enable Lidar sensor
        self.lidar = self.supervisor.getDevice("lidar")
        if self.lidar:
            self.lidar.enable(self.timestep)
        else:
            raise ValueError("Lidar sensor not found. Please ensure your robot has a Lidar device named 'lidar'.")

        # Get and enable Touch sensor for collision detection
        self.touch_sensor = self.supervisor.getDevice("touch sensor")
        if self.touch_sensor:
            self.touch_sensor.enable(self.timestep)
        else:
            print("[WARNING] Touch sensor not found. Collision detection via touch might be less precise.")

        # Initialize statistics dictionary for overall runs
        self.stats = {'total_collisions': 0, 'successful_runs': 0, 'failed_runs': 0}

    def save_model(self, individual, generation, save_dir="saved_models"):
        """
        Saves a given individual (neural network model) to a file using pickle.
        The model is saved in a directory named 'saved_models' by default.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")

        filename = os.path.join(save_dir, f"best_model_gen_{generation}.pkl")
        try:
            with open(filename, 'wb') as f:
                pickle.dump(individual, f)
            print(f"Model saved successfully to: {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to save model to {filename}: {e}")

    def load_model(self, filepath):
        """
        Loads a neural network model from a specified file using pickle.
        Returns the loaded individual object, or None if loading fails.
        """
        try:
            with open(filepath, 'rb') as f:
                individual = pickle.load(f)
            print(f"Model loaded successfully from: {filepath}")
            return individual
        except FileNotFoundError:
            print(f"[ERROR] Model file not found: {filepath}")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to load model from {filepath}: {e}")
            return None

    def run_experiment(self, num_runs: int):
        """
        Runs a series of simulations with a fixed difficulty stage, typically for initial testing.
        This method is based on your original `run_experiment` but updated to use
        the new `TunnelBuilder` and `run_experiment_with_network` for consistency.
        """
        print(f"Running fixed difficulty experiment for {num_runs} runs.")

        # Example: Test on a fixed difficulty stage (e.g., stage 1 for easy straight tunnels)
        # For GA training, `run_experiment_with_network` is called directly by `NeuralPopulation.evaluate`.
        fixed_stage = 1.0  # This can be changed for testing different fixed difficulties

        # Create a dummy individual (or use a simple hardcoded controller) for fixed runs
        class SimpleTestIndividual:
            def act(self, lidar_data: np.ndarray) -> Tuple[float, float]:
                # Simple wall-following behavior
                min_dist_idx = np.argmin(lidar_data)
                min_dist = lidar_data[min_dist_idx]

                linear_vel = 0.1  # Constant forward speed
                angular_vel = 0.0  # No turning by default

                # If an obstacle is very close, turn away
                if min_dist < 0.25:  # Threshold for too close
                    # Determine which side the obstacle is on
                    if min_dist_idx < len(lidar_data) / 2:  # Obstacle on left (Lidar values go from left to right)
                        angular_vel = -0.5  # Turn right
                    else:  # Obstacle on right
                        angular_vel = 0.5  # Turn left
                    linear_vel = 0.05  # Slow down while turning

                return linear_vel, angular_vel

        test_individual = SimpleTestIndividual()

        for run_idx in range(num_runs):
            print(f"\n--- Starting Fixed Run {run_idx + 1} / {num_runs} (Stage: {fixed_stage:.1f}) ---")

            # Call the more robust run_experiment_with_network to run a single episode
            fitness, success_status = self.run_experiment_with_network(
                test_individual,
                stage=fixed_stage,
                total_stages=MAX_DIFFICULTY_STAGE  # Pass total stages for get_stage_parameters
            )

            print(f"Fixed Run {run_idx + 1} completed. Fitness: {fitness:.2f}, Success: {success_status}")

            # Update overall simulation statistics
            if success_status:
                self.stats['successful_runs'] += 1
            else:
                self.stats['failed_runs'] += 1
            # Note: total_collisions is updated internally by run_experiment_with_network

        self._print_summary()  # Print summary after all fixed runs

    def _process_lidar_with_params(self, dist_values: np.ndarray, distP: float, angleP: float) -> Tuple[float, float]:
        """
        Robot control logic based on Lidar data, with tunable distP and angleP parameters.
        This method was part of your original file, likely for a parameter-optimization GA.
        It's distinct from the neural network's `individual.act()` method.
        """
        if dist_values is None or len(dist_values) == 0:
            return 0.0, 0.0

        direction: int = 1  # Assuming right wall following (adjust if needed)

        maxSpeed: float = 0.1
        wallDist: float = 0.1  # Desired distance to the wall (consider making this dynamic based on tunnel width)

        size: int = len(dist_values)
        if size == 0:
            return 0.0, 0.0

        # Filter out invalid lidar readings (NaN, Inf, <= 0) and find the minimum valid distance
        valid_indices = np.where(np.isfinite(dist_values) & (dist_values > 0))[0]
        if len(valid_indices) == 0:
            return maxSpeed, 0.0  # If no valid readings, move straight

        min_dist_val = np.min(dist_values[valid_indices])  # Get the minimum valid distance value
        min_index = np.where(dist_values == min_dist_val)[0][0]  # Get the original index of that minimum value

        angle_increment: float = 2 * math.pi / (size - 1) if size > 1 else 0.0
        angleMin: float = (size // 2 - min_index) * angle_increment

        distMin: float = min_dist_val  # This was defined previously but needs to be here.

        distFront: float = dist_values[size // 2] if size // 2 < size else float('inf')
        distSide: float = dist_values[size // 4] if (size // 4 < size) else float('inf') if (direction == 1) else \
        dist_values[3 * size // 4] if (3 * size // 4 < size) else float('inf')
        distBack: float = dist_values[0] if size > 0 else float('inf')

        linear_vel: float
        angular_vel: float

        if math.isfinite(distMin):
            # Collision avoidance / Unblocking
            if distFront < 1.25 * wallDist and (distSide < 1.25 * wallDist or distBack < 1.25 * wallDist):
                angular_vel = direction * -1 * maxSpeed
                linear_vel = 0
            else:
                # PID-like control
                angular_vel = direction * distP * (distMin - wallDist) + angleP * (angleMin - direction * math.pi / 2)

            # Adjust linear velocity based on front distance
            if distFront < wallDist:
                linear_vel = 0
            elif distFront < 2 * wallDist or distMin < wallDist * 0.75 or distMin > wallDist * 1.25:
                linear_vel = 0.5 * maxSpeed
            else:
                linear_vel = maxSpeed
        else:
            # If no finite minimum distance (e.g., open space or no obstacles), wander
            angular_vel = np.random.normal(loc=0.0, scale=1.0) * maxSpeed
            linear_vel = maxSpeed

        # Clamp velocities to valid range
        linear_vel = np.clip(linear_vel, -maxSpeed, maxSpeed)
        angular_vel = np.clip(angular_vel, -maxSpeed, maxSpeed)

        return linear_vel, angular_vel

    def run_experiment_with_network(
            self,
            individual: Any,  # Can be an IndividualNeural or SimpleTestIndividual
            stage: float,  # Current difficulty stage (float for continuous progression)
            total_stages: float = MAX_DIFFICULTY_STAGE  # Max difficulty for normalization
    ) -> Tuple[float, bool]:
        """
        Run one simulation episode with the given individual (neural network) on a tunnel
        generated according to the specified difficulty stage.
        Handles robot reset, simulation loop, collision detection, goal checking,
        and wall cleanup.

        Args:
            individual: An object with an 'act(lidar_data)' method (your neural network individual, or a test controller).
            stage (float): The current difficulty stage (e.g., from 1.0 to 10.0).
            total_stages (float): The maximum possible difficulty stage for normalization.

        Returns:
            fitness (float): The computed fitness score for this run.
            success (bool): True if the robot reached the goal without critical failure (timeout or collision).
        """
        # 1) Generate tunnel parameters for this difficulty stage
        # The `get_stage_parameters` function will handle the mapping from continuous stage to parameters
        num_curves, angle_range, clearance, num_obstacles = \
            get_stage_parameters(stage, total_stages)
        print(f"[BUILD] stage {stage:.2f}/{total_stages} → curves={num_curves}, "
              f"angles={math.degrees(angle_range[0]):.0f}-{math.degrees(angle_range[1]):.0f}deg, "  # Added degrees conversion for clearer print
              f"clearance={clearance:.2f}, obs={num_obstacles}")

        # 2) Build tunnel using a new TunnelBuilder instance for each run
        # This ensures a clean slate for each simulation, as TunnelBuilder handles its own cleanup.
        builder = TunnelBuilder(self.supervisor)
        start_pos, end_pos, walls_added_count, initial_tunnel_heading = builder.build_tunnel(
            num_curves=num_curves,
            angle_range=angle_range,
            clearance_factor=clearance,
            num_obstacles=num_obstacles
        )

        if start_pos is None:  # Tunnel generation failed (e.g., out of bounds, invalid parameters)
            print("[ERROR] Tunnel build failed. Returning penalty fitness.")
            # Ensure walls are cleared even if build fails partially
            builder._clear_walls()  # It's good practice to try clearing just in case
            return -5000.0, False  # Return a low fitness and False for success

        # 3) Reset robot pose and physics
        # Set robot's initial position and rotation to match the tunnel's start
        self.translation.setSFVec3f([start_pos[0], start_pos[1], ROBOT_RADIUS])  # ROBOT_RADIUS for Z height
        # Set robot's initial rotation to align with the initial tunnel heading
        # Assuming initial_tunnel_heading is the yaw angle around Z axis
        self.rotation.setSFRotation([0, 0, 1, initial_tunnel_heading])  # Axis-angle representation: [x,y,z,angle]
        self.robot.resetPhysics()  # Reset robot's physics to apply new position/rotation

        # 4) Initialize devices and tracking variables for the current simulation run
        left_motor = self.supervisor.getDevice("left wheel motor")
        right_motor = self.supervisor.getDevice("right wheel motor")
        left_motor.setPosition(float('inf'));  # Set motors to velocity control
        right_motor.setPosition(float('inf'))
        left_motor.setVelocity(0);  # Start with zero velocity
        right_motor.setVelocity(0)

        # Ensure touch sensor is enabled for this run
        if self.touch_sensor:
            self.touch_sensor.enable(self.timestep)

        t0 = self.supervisor.getTime()  # Record start time of the simulation
        timeout_occurred = False
        goal_reached = False  # True if robot successfully reaches the end

        off_tunnel_events = 0  # Counter for times robot leaves the main path
        flag_off_path = False  # Flag to prevent multiple counts for a single continuous off-path event

        distance_traveled_inside = 0.0  # Distance traveled while being "inside" the tunnel geometry
        previous_pos = np.array(self.translation.getSFVec3f())  # For calculating distance step
        total_distance_traveled = 0.0  # Total absolute distance traveled by the robot

        # 5) Main Simulation Loop
        while self.supervisor.step(self.timestep) != -1:
            elapsed_time = self.supervisor.getTime() - t0

            # --- A. Timeout Check ---
            if elapsed_time > TIMEOUT_DURATION:
                timeout_occurred = True
                print("[TIMEOUT] Simulation exceeded time limit.")
                break

            # --- B. Collision Detection (via Touch Sensor) ---
            if self.touch_sensor and self.touch_sensor.getValue() > 0:
                print("[COLLISION] Touch sensor triggered. Episode ends.")
                self.stats['total_collisions'] += 1  # Update overall collision statistics
                break  # End simulation immediately on collision

            # --- C. Read Sensors and Control Robot ---
            # np.nan_to_num handles NaN, Inf values from Lidar, replacing them for network input
            lidar_data = np.nan_to_num(self.lidar.getRangeImage(), nan=0.0, posinf=self.lidar.getMaxRange(), neginf=0.0)

            # Get linear and angular velocities from the individual's neural network (or test controller)
            lv, av = individual.act(lidar_data)
            cmd_vel(self.supervisor, lv, av)  # Apply velocities to robot motors

            # --- D. Update Tracking Variables ---
            current_pos = np.array(self.translation.getSFVec3f())
            # Calculate distance traveled in this step
            step_distance = np.linalg.norm(current_pos[:2] - previous_pos[:2])
            total_distance_traveled += step_distance
            previous_pos = current_pos.copy()  # Update previous position for next step

            # Get current robot's orientation (heading) for 'is_robot_inside_tunnel' check
            current_rotation_sfr = self.rotation.getSFRotation()
            # Extract yaw/heading from axis-angle representation (assuming rotation mostly around Z)
            current_heading = current_rotation_sfr[3]  # The angle component is the yaw for Z-axis rotation

            # --- E. Check if Robot is Inside Tunnel Path ---
            # Combines centerline proximity and wall proximity for a robust "inside" check
            is_currently_near_centerline = builder.is_robot_near_centerline(current_pos)
            is_currently_within_walls = builder.is_robot_inside_tunnel(current_pos, current_heading)

            # Robot is considered on path if it's both near the centerline AND within the walls.
            is_robot_on_path = is_currently_near_centerline and is_currently_within_walls

            if is_robot_on_path:
                distance_traveled_inside += step_distance
                flag_off_path = False  # Reset flag if robot is back on path
            else:
                # If robot is off path AND grace period is over, count an "off_tunnel_event"
                if not flag_off_path and elapsed_time > 2.0:  # Grace period of 2 seconds to allow for minor corrections
                    off_tunnel_events += 1
                    flag_off_path = True  # Set flag to indicate robot is currently off path
                    print(f"[WARNING] Robot left tunnel path at {current_pos[:2]}. Episode ends (Stage {stage}).")
                    break  # End simulation immediately if robot leaves the path (for harsh penalty)

            # --- F. Goal Check ---
            # Project robot's position onto the tunnel's final direction (defined by `initial_tunnel_heading`)
            # If the robot's leading edge (center + radius) crosses the end point, goal is reached.
            if end_pos is not None and initial_tunnel_heading is not None:
                final_direction_vector = np.array([math.cos(initial_tunnel_heading), math.sin(initial_tunnel_heading)])
                vector_from_end_to_robot = current_pos[:2] - end_pos[:2]
                projection_on_final_direction = np.dot(vector_from_end_to_robot, final_direction_vector)

                # The robot reaches the goal if its position along the final direction,
                # considering its radius, is past the end_pos.
                if projection_on_final_direction + ROBOT_RADIUS >= 0:
                    goal_reached = True
                    print(f"[SUCCESS] Goal reached in {elapsed_time:.2f} seconds.")
                    break  # End simulation, goal achieved

        # 6) ALWAYS Clear tunnel walls at the end of EACH simulation episode
        # This is the crucial fix for the "walls not disappearing" problem.
        # Calling _clear_walls() from the builder instance ensures all walls created by it are removed.
        builder._clear_walls()
        print(f"Simulation episode for Individual {getattr(individual, 'id', 'N/A')} finished. Walls cleared.")

        # 7) Compute Fitness
        # Fitness calculation should reflect success, distance, and penalties.
        fitness = 0.0

        # Calculate total length of the tunnel path for normalization
        total_tunnel_length = sum(
            [seg[3] for seg in builder.segments]) if builder.segments else 1.0  # Avoid division by zero

        # Percentage of total tunnel length covered
        percent_traveled = total_distance_traveled / total_tunnel_length if total_tunnel_length > 0 else 0.0

        # Reward for covering a significant portion of the tunnel
        if percent_traveled >= 0.8:  # Example threshold for partial success
            fitness += 500.0

        # Large reward for reaching the goal without timeout or physical collision (touch sensor)
        # Note: `touch_sensor.getValue() > 0` indicates a collision happened right before break.
        success_condition = goal_reached and not timeout_occurred and \
                            (
                                        self.touch_sensor is None or self.touch_sensor.getValue() == 0)  # Check touch sensor last value

        if success_condition:
            fitness += 1000.0  # Big reward for full success

        # Penalties for undesirable events
        fitness -= 2000.0 * off_tunnel_events  # Penalty for leaving the main path (each event)

        # Reward for distance traveled *while inside* the tunnel path (more robust)
        fitness += 100.0 * distance_traveled_inside

        # Penalty for timeout
        if timeout_occurred:
            fitness -= 500.0  # Moderate penalty for not finishing on time

        # Additional severe penalty if simulation ended due to touch sensor collision
        # This is only applied if the loop broke due to touch sensor.
        if self.touch_sensor and self.touch_sensor.getValue() > 0:
            fitness -= 1500.0  # Severe penalty for direct physical collision

        print(f"[FITNESS RESULT] Stage {stage:.2f} | Fitness: {fitness:.2f} | Success: {success_condition}")

        # Return fitness and success status (boolean)
        return fitness, success_condition

    def run_on_existing_tunnel(self, distP, angleP, builder, start_pos, end_pos, final_heading):
        """
        Runs a simulation on a pre-built tunnel.
        Useful for re-evaluating individuals on specific test cases.
        Accepts final_heading for goal check.
        """
        print("\n--- Running on Existing Tunnel ---")
        # Reposition the robot at the start position with Z = 0
        self.translation.setSFVec3f([start_pos[0], start_pos[1], 0])
        self.rotation.setSFRotation([0, 0, 1, 0])  # Set rotation to face forward (along positive X)
        self.robot.resetPhysics()

        left = self.supervisor.getDevice("left wheel motor")
        right = self.supervisor.getDevice("right wheel motor")
        left.setPosition(float('inf'))
        right.setPosition(float('inf'))
        left.setVelocity(0)
        right.setVelocity(0)

        grace_period = 2.0
        off_tunnel_events = 0
        flag_off_tunnel = False
        start_time = self.supervisor.getTime()
        timeout_occurred = False
        goal_reached = False
        distance_traveled_inside = 0.0
        previous_pos = np.array(self.translation.getSFVec3f())
        distance_traveled = 0.0
        last_pos = previous_pos.copy()

        # Assuming the builder object used to create the tunnel is passed in
        # and contains the segments data and base_wall_distance
        if not hasattr(builder, 'segments') or not hasattr(builder, 'base_wall_distance'):
            print("Error: Provided builder object is missing segments or base_wall_distance.")
            return -6000.0  # Return a very low fitness

        while self.supervisor.step(self.timestep) != -1:
            if self.supervisor.getTime() - start_time > TIMEOUT_DURATION:
                timeout_occurred = True
                print("TIME OUT (Existing Tunnel)")
                break

            data = self.lidar.getRangeImage()
            lv, av = self._process_lidar_with_params(data, distP, angleP)
            cmd_vel(self.supervisor, lv, av)

            pos = np.array(self.translation.getSFVec3f())
            distance_traveled += np.linalg.norm(pos[:2] - last_pos[:2])
            last_pos = pos.copy()
            rot = self.robot.getField("rotation").getSFRotation()
            heading = rot[3] if rot[2] >= 0 else -rot[3]

            # Use the builder's centerline check
            inside_tunnel = builder.is_robot_near_centerline(pos)

            if inside_tunnel:
                distance_step = np.linalg.norm(pos[:2] - previous_pos[:2])
                distance_traveled_inside += distance_step
                flag_off_tunnel = False
            else:
                if not flag_off_tunnel and self.supervisor.getTime() - start_time > grace_period:
                    off_tunnel_events += 1
                    flag_off_tunnel = True
                    print(f"[WARNING] Robot left the tunnel path (Existing Tunnel): {pos[:2]}")

            previous_pos = pos

            # Check for goal condition: robot crosses the line at the end of the tunnel
            if end_pos is not None and final_heading is not None:
                final_direction_vector = np.array([math.cos(final_heading), math.sin(final_heading)])
                vector_from_end_to_robot = pos[:2] - end_pos[:2]
                projection_on_final_direction = np.dot(vector_from_end_to_robot, final_direction_vector)

                if projection_on_final_direction + ROBOT_RADIUS >= 0:
                    goal_reached = True
                    print(f"Reached end in {self.supervisor.getTime() - start_time:.1f}s (Existing Tunnel)")
                    break

        # Walls are NOT called here, as this method assumes the tunnel is already built and managed externally.
        # print("slay") # Removed slay print as walls are not removed here

        fitness = 0.0
        total_tunnel_length = sum([seg[3] for seg in builder.segments])
        percent_traveled = distance_traveled / total_tunnel_length if total_tunnel_length > 0 else 0

        if percent_traveled >= 0.8:
            fitness += 500.0
        if goal_reached and not timeout_occurred:
            fitness += 1000.0
        fitness -= 2000.0 * off_tunnel_events
        fitness += 100.0 * distance_traveled_inside
        if timeout_occurred:
            fitness -= 500.0

        print(f"Fitness on Existing Tunnel: {fitness:.2f}")
        return fitness

    def _print_summary(self, builder=None):
        """
        Prints a summary of the overall experiment statistics.
        """
        print("\n=== Overall Experiment Results Summary ===")
        print(f"Total Successful Runs: {self.stats['successful_runs']}")
        print(f"Total Failed Runs: {self.stats['failed_runs']}")
        print(f"Total Collisions Detected (via touch sensor): {self.stats['total_collisions']}")
        total_runs_completed = self.stats['successful_runs'] + self.stats['failed_runs']
        if total_runs_completed > 0:
            overall_success_rate = (self.stats['successful_runs'] / total_runs_completed) * 100
            print(f"Overall Success Rate: {overall_success_rate:.1f}%")
        else:
            print("No runs completed to calculate overall success rate.")
        # Optionally print total tunnel length if builder is provided and has segments
        if builder and hasattr(builder, 'segments'):
            total_tunnel_length_sum = sum([seg[3] for seg in builder.segments])
            print(f"Total tunnel length for last built tunnel: {total_tunnel_length_sum:.2f} meters")

    # The _process_lidar method is kept for compatibility with your original file,
    # but run_experiment_with_network now explicitly takes an 'individual' with an 'act' method.
    def _process_lidar(self, dist_values: np.ndarray) -> Tuple[float, float]:
        """
        Robot control logic based on Lidar data using fixed, default parameters.
        This is a pre-defined behavior, not the neural network's output.
        It's included for compatibility with your original `run_experiment` method.
        """
        if dist_values is None or len(dist_values) == 0:
            return 0.0, 0.0

        direction: int = 1  # Assuming right wall following

        maxSpeed: float = 0.1
        distP: float = 10.0  # Default Proportional gain for distance error
        angleP: float = 7.0  # Default Proportional gain for angle error
        wallDist: float = 0.1  # Desired distance to the wall

        size: int = len(dist_values)
        if size == 0:
            return 0.0, 0.0

        valid_indices = np.where(np.isfinite(dist_values) & (dist_values > 0))[0]
        if len(valid_indices) == 0:
            return maxSpeed, 0.0

        min_index = valid_indices[np.argmin(dist_values[valid_indices])]
        distMin = dist_values[min_index]

        angle_increment: float = 2 * math.pi / (size - 1) if size > 1 else 0.0
        angleMin: float = (size // 2 - min_index) * angle_increment

        distFront: float = dist_values[size // 2] if size // 2 < size else float('inf')
        distSide: float = dist_values[size // 4] if (size // 4 < size) else float('inf') if (direction == 1) else \
        dist_values[3 * size // 4] if (3 * size // 4 < size) else float('inf')
        distBack: float = dist_values[0] if size > 0 else float('inf')

        linear_vel: float
        angular_vel: float

        if math.isfinite(distMin):
            if distFront < 1.25 * wallDist and (distSide < 1.25 * wallDist or distBack < 1.25 * wallDist):
                angular_vel = direction * -1 * maxSpeed
                linear_vel = 0
            else:
                angular_vel = direction * distP * (distMin - wallDist) + angleP * (angleMin - direction * math.pi / 2)

            if distFront < wallDist:
                linear_vel = 0
            elif distFront < 2 * wallDist or distMin < wallDist * 0.75 or distMin > wallDist * 1.25:
                linear_vel = 0.5 * maxSpeed
            else:
                linear_vel = maxSpeed
        else:
            angular_vel = np.random.normal(loc=0.0, scale=1.0) * maxSpeed
            linear_vel = maxSpeed

        linear_vel = np.clip(linear_vel, -maxSpeed, maxSpeed)
        angular_vel = np.clip(angular_vel, -maxSpeed, maxSpeed)

        return linear_vel, angular_vel
