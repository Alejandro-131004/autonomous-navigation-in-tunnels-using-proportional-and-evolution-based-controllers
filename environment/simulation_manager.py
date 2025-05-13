from environment.configuration import ROBOT_NAME, ROBOT_RADIUS, TIMEOUT_DURATION, MAX_NUM_CURVES, get_stage_parameters
from environment.tunnel import TunnelBuilder
import numpy as np
import math
from agent.controller import cmd_vel # Assuming cmd_vel is defined elsewhere

class SimulationManager:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.robot = self.supervisor.getFromDef(ROBOT_NAME)
        if self.robot is None:
            raise ValueError(f"Robot with DEF name '{ROBOT_NAME}' not found.")
        self.translation = self.robot.getField("translation")
        self.lidar = self.supervisor.getDevice("lidar")
        self.lidar.enable(self.timestep)
        self.stats = {'total_collisions': 0, 'successful_runs': 0, 'failed_runs': 0}

    def run_experiment(self, num_runs):
        # This method is for fixed difficulty runs, potentially for initial testing
        # For GA training, run_experiment_with_params should be used.
        print("Running fixed difficulty experiment (consider using run_experiment_with_params for GA).")
        # Example usage with a fixed difficulty stage (e.g., stage 0 for easy)
        num_curves, angle_range, clearance_factor, num_obstacles = get_stage_parameters(0) # Example: Stage 0 (Easy)

        for run in range(num_runs):
            print(f"--- Starting Run {run + 1} ---")

            # Build a new tunnel with parameters from the stage
            tunnel_builder = TunnelBuilder(self.supervisor)
            start_pos, end_pos, walls_added = tunnel_builder.build_tunnel(
                num_curves=num_curves,
                angle_range=angle_range,
                clearance=clearance_factor,
                num_obstacles=num_obstacles
            )

            if start_pos is None:
                print("Tunnel out of bounds, skipping run.")
                continue

            self.translation.setSFVec3f([start_pos[0], start_pos[1], ROBOT_RADIUS])
            self.robot.resetPhysics()

            left = self.supervisor.getDevice("left wheel motor")
            right = self.supervisor.getDevice("right wheel motor")
            left.setPosition(float('inf'))
            right.setPosition(float('inf'))
            left.setVelocity(0)
            right.setVelocity(0)

            collision_count = 0
            flag = False
            start_time = self.supervisor.getTime()

            while self.supervisor.step(self.timestep) != -1:
                if self.supervisor.getTime() - start_time > TIMEOUT_DURATION:
                    print("Timeout")
                    break

                data = self.lidar.getRangeImage()
                # Use the general _process_lidar if no specific params are being optimized
                lv, av = self._process_lidar(data)
                cmd_vel(self.supervisor, lv, av)

                pos = np.array(self.translation.getSFVec3f())

                # Basic collision detection: deviation from centerline
                # Note: This check is a simplification and might not perfectly map to physical collisions.
                # Physical collision detection in Webots is usually handled by the simulation engine itself.
                # You might want to rely more on the robot's collision sensors if available,
                # or check for proximity to wall nodes more directly.
                current_base_wall_distance = tunnel_builder.base_wall_distance
                current_distance_from_center = abs(pos[1])

                # print(
                #     f"[DEBUG] Distance from center: {current_distance_from_center:.3f} | Allowed limit: {current_base_wall_distance - ROBOT_RADIUS:.3f}")

                if current_distance_from_center > current_base_wall_distance - ROBOT_RADIUS:
                    if not flag:
                        print(f"[WARNING] Robot left the tunnel! Position: {pos[:2]}")
                        # Increment collision count here if leaving the tunnel counts as a collision
                        collision_count += 1
                        flag = True
                else:
                    flag = False


                if end_pos is not None and np.linalg.norm(pos[:2] - end_pos[:2]) < 0.1:
                    print(f"Reached end in {self.supervisor.getTime() - start_time:.1f}s")
                    break

            self._remove_walls(walls_added)
            print("slay")

            self.stats['total_collisions'] += collision_count
            if collision_count == 0 and end_pos is not None and np.linalg.norm(pos[:2] - end_pos[:2]) < 0.1:
                self.stats['successful_runs'] += 1
            else:
                self.stats['failed_runs'] += 1

            print(f"Run {run + 1} finished with {collision_count} collisions.")

        self._print_summary()

    def _process_lidar_with_params(self, dist_values: [float], distP: float, angleP: float) -> (float, float):

        """
        Robot control logic based on Lidar data, with tunable distP and angleP parameters.
        Adapted for use with Genetic Algorithm optimization.
        """
        direction: int = 1  # Assuming right wall following

        maxSpeed: float = 0.1
        wallDist: float = 0.1  # Desired distance to the wall

        size: int = len(dist_values)
        if size == 0:
            return 0.0, 0.0  # No data available, stay still

        min_index: int = 0
        if direction == -1:
            min_index = size - 1
        for i in range(size):
            idx: int = i
            if direction == -1:
                idx = size - 1 - i
            if dist_values[idx] < dist_values[min_index] and dist_values[idx] > 0.0:
                min_index = idx
            elif dist_values[min_index] <= 0.0 and dist_values[idx] > 0.0:
                min_index = idx

        angle_increment: float = 2 * math.pi / (size - 1) if size > 1 else 0.0
        angleMin: float = (size // 2 - min_index) * angle_increment
        distMin: float = dist_values[min_index]

        distFront: float = dist_values[size // 2] if size > 0 else float('inf')
        distSide: float = dist_values[size // 4] if (size > 0 and size // 4 < size) else float('inf') if (
                    direction == 1) else dist_values[3 * size // 4] if (size > 0 and 3 * size // 4 < size) else float(
            'inf')
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

        return linear_vel, angular_vel

    # Modified run_experiment_with_params to accept difficulty parameters
    def run_experiment_with_params(self, distP: float, angleP: float, stage: int) -> float:
        """
        Runs a simulation experiment with given controller parameters (distP, angleP)
        and a specific difficulty stage.

        Args:
            distP (float): Proportional gain for distance error.
            angleP (float): Proportional gain for angle error.
            stage (int): The current training stage to determine tunnel parameters.

        Returns:
            float: The fitness score for this parameter pair on the given stage.
        """
        # Get tunnel parameters based on the stage
        num_curves, angle_range, clearance_factor, num_obstacles = get_stage_parameters(stage)

        print(f"\n--- Simulation with Stage: {stage} (Curves: {num_curves}, Angle Range: {angle_range}, Clearance: {clearance_factor:.2f}, Obstacles: {num_obstacles}) ---")

        # Create the tunnel with stage-specific parameters
        builder = TunnelBuilder(self.supervisor)
        start_pos, end_pos, walls_added = builder.build_tunnel(
            num_curves=num_curves,
            angle_range=angle_range,
            clearance=clearance_factor,
            num_obstacles=num_obstacles
        )

        # Handle tunnel generation failure
        if start_pos is None:
            print(f"Tunnel generation failed for Stage {stage}. Returning low fitness.")
            # Return a low fitness score to penalize parameters that fail on this stage
            return -5000.0

        # Reposition the robot
        self.translation.setSFVec3f([start_pos[0], start_pos[1], ROBOT_RADIUS])
        self.robot.resetPhysics()

        # Initialize motors
        left = self.supervisor.getDevice("left wheel motor")
        right = self.supervisor.getDevice("right wheel motor")
        left.setPosition(float('inf'))
        right.setPosition(float('inf'))
        left.setVelocity(0)
        right.setVelocity(0)

        # Initialize simulation variables for fitness calculation
        grace_period = 2.0 # Time at the start to allow robot to settle
        off_tunnel_events = 0 # Count times robot leaves the main tunnel path
        flag_off_tunnel = False # Flag to prevent multiple counts for one continuous off-path event
        start_time = self.supervisor.getTime()
        timeout_occurred = False
        goal_reached = False
        distance_traveled_inside = 0.0 # Distance traveled while inside the tunnel path
        previous_pos = np.array(self.translation.getSFVec3f()) # For calculating distance step
        distance_traveled = 0.0 # Total distance traveled
        last_pos = previous_pos.copy() # For calculating total distance

        # Simulation loop
        while self.supervisor.step(self.timestep) != -1:
            # Timeout check
            if self.supervisor.getTime() - start_time > TIMEOUT_DURATION:
                timeout_occurred = True
                print("TIME OUT")
                break

            # Get Lidar data
            data = self.lidar.getRangeImage()
            # Process Lidar data using the provided controller parameters
            lv, av = self._process_lidar_with_params(data, distP, angleP)
            # Send velocity commands to the robot
            cmd_vel(self.supervisor, lv, av)

            # Get current robot position and orientation
            pos = np.array(self.translation.getSFVec3f())
            distance_traveled += np.linalg.norm(pos[:2] - last_pos[:2])
            last_pos = pos.copy()
            rot = self.robot.getField("rotation").getSFRotation()
            # Extract heading from rotation quaternion/axis-angle (assuming Z-axis rotation)
            heading = rot[3] if rot[2] >= 0 else -rot[3] # Simplified heading extraction

            # Check if the robot is inside the tunnel path
            # Using the centerline check as the primary indicator of being "inside"
            inside_tunnel = builder.is_robot_near_centerline(pos)

            if inside_tunnel:
                distance_step = np.linalg.norm(pos[:2] - previous_pos[:2])
                distance_traveled_inside += distance_step
                flag_off_tunnel = False # Reset flag when back inside
            else:
                # If outside the tunnel path and grace period is over
                if not flag_off_tunnel and self.supervisor.getTime() - start_time > grace_period:
                    off_tunnel_events += 1
                    flag_off_tunnel = True # Set flag to indicate currently off path
                    print(f"[WARNING] Robot left the tunnel path (Stage {stage}): {pos[:2]}")

            # Update previous position for next distance calculation
            previous_pos = pos

            # Check if the robot reached the end of the tunnel
            if end_pos is not None and np.linalg.norm(pos[:2] - end_pos[:2]) < 0.1:
                goal_reached = True
                print(f"Reached end in {self.supervisor.getTime() - start_time:.1f}s (Stage {stage})")
                break

        # Remove tunnel walls after the simulation run for this stage
        self._remove_walls(walls_added)
        print("slay")

        # --- Compute Fitness ---
        fitness = 0.0
        total_tunnel_length = sum([seg[3] for seg in builder.segments]) # Sum of segment lengths
        percent_traveled = distance_traveled / total_tunnel_length if total_tunnel_length > 0 else 0

        # Reward for reaching a significant portion of the tunnel
        if percent_traveled >= 0.8:
            fitness += 500.0

        # Large reward for reaching the goal without timeout
        if goal_reached and not timeout_occurred:
            fitness += 1000.0

        # Penalty for leaving the tunnel path
        fitness -= 2000.0 * off_tunnel_events

        # Reward for distance traveled *inside* the tunnel path
        fitness += 100.0 * distance_traveled_inside

        # Small penalty for timeout
        if timeout_occurred:
             fitness -= 500.0 # Penalize timeout

        print(f"Fitness for Stage {stage}: {fitness:.2f}")

        return fitness


    # The run_on_existing_tunnel method is kept, but might be less relevant
    # if run_experiment_with_params is used for all GA evaluations.
    # It could be useful for re-evaluating the best individual on a specific
    # pre-generated tunnel if needed.
    def run_on_existing_tunnel(self, distP, angleP, builder, start_pos, end_pos):
        """
        Runs a simulation on a pre-built tunnel.
        Useful for re-evaluating individuals on specific test cases.
        """
        print("\n--- Running on Existing Tunnel ---")
        self.translation.setSFVec3f([start_pos[0], start_pos[1], ROBOT_RADIUS])
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
            return -6000.0 # Return a very low fitness

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

            if end_pos is not None and np.linalg.norm(pos[:2] - end_pos[:2]) < 0.1:
                goal_reached = True
                print(f"Reached end in {self.supervisor.getTime() - start_time:.1f}s (Existing Tunnel)")
                break

        # Remove walls are NOT called here, as this method assumes the tunnel is already built and managed externally.
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


    def _remove_walls(self, count):
        # Remove the last 'count' children from the root node
        children = self.supervisor.getRoot().getField("children")
        # Iterate backwards to safely remove elements
        for i in range(count):
             # Ensure there are enough children to remove
             if children.getCount() > 0:
                 children.removeMF(children.getCount() - 1)
             else:
                 print("Warning: Attempted to remove more children than exist.")
                 break

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

    # The _process_lidar method is kept for compatibility but _process_lidar_with_params is used in GA runs
    def _process_lidar(self, dist_values: [float]) -> (float, float):

        """
        Robot control logic based on Lidar data (using default parameters).
        """
        direction: int = 1  # Assuming right wall following

        maxSpeed: float = 0.1
        distP: float = 10.0  # Default Proportional gain for distance error
        angleP: float = 7.0  # Default Proportional gain for angle error
        wallDist: float = 0.1  # Desired distance to the wall

        size: int = len(dist_values)
        if size == 0:
            return 0.0, 0.0

        min_index: int = 0
        if direction == -1:
            min_index = size - 1
        for i in range(size):
            idx: int = i
            if direction == -1:
                idx = size - 1 - i
            if dist_values[idx] < dist_values[min_index] and dist_values[idx] > 0.0:
                min_index = idx
            elif dist_values[min_index] <= 0.0 and dist_values[idx] > 0.0:
                min_index = idx

        angle_increment: float = 2 * math.pi / (size - 1) if size > 1 else 0.0
        angleMin: float = (size // 2 - min_index) * angle_increment
        distMin: float = dist_values[min_index]

        distFront: float = dist_values[size // 2] if size > 0 else float('inf')
        distSide: float = dist_values[size // 4] if (size > 0 and size // 4 < size) else float('inf') if (
                    direction == 1) else dist_values[3 * size // 4] if (size > 0 and 3 * size // 4 < size) else float(
            'inf')
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

        return linear_vel, angular_vel
