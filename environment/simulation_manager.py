from environment.configuration import ROBOT_NAME, ROBOT_RADIUS, TIMEOUT_DURATION, NUM_CURVES, get_difficulty_settings
from environment.tunnel import TunnelBuilder
import numpy as np
import math
from agent.controller import cmd_vel

class SimulationManager:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.robot = self.supervisor.getFromDef(ROBOT_NAME)  # cuidado, ROBOT_NAME precisa existir
        if self.robot is None:
            raise ValueError(f"Robot with DEF name '{ROBOT_NAME}' not found.")
        self.translation = self.robot.getField("translation")
        self.lidar = self.supervisor.getDevice("lidar")
        self.lidar.enable(self.timestep)
        self.stats = {'total_collisions': 0, 'successful_runs': 0, 'failed_runs': 0}

    def run_experiment(self, num_runs):
        # Run the experiment for the specified number of runs
        for run in range(num_runs):
            print(f"--- Starting Run {run + 1} ---")

            # Build a new random tunnel
            tunnel_builder = TunnelBuilder(self.supervisor)
            start_pos, end_pos, walls_added = tunnel_builder.build_tunnel(NUM_CURVES)

            # If tunnel building failed (out of bounds), skip this run
            if start_pos is None:
                print("Tunnel out of bounds, skipping run.")
                continue

            # Place the robot at the start of the tunnel and reset physics
            self.translation.setSFVec3f([start_pos[0], start_pos[1], ROBOT_RADIUS])
            self.robot.resetPhysics()

            # Get wheel motors and set them to velocity control mode (infinite position)
            left = self.supervisor.getDevice("left wheel motor")
            right = self.supervisor.getDevice("right wheel motor")
            left.setPosition(float('inf'))
            right.setPosition(float('inf'))
            left.setVelocity(0)  # Start with zero velocity
            right.setVelocity(0)

            collision_count = 0
            # Flag to avoid counting multiple collisions for the same event
            flag = False
            start_time = self.supervisor.getTime()  # Record simulation start time

            # Simulation loop
            while self.supervisor.step(self.timestep) != -1:
                # Timeout check
                if self.supervisor.getTime() - start_time > TIMEOUT_DURATION:
                    print("Timeout")
                    break

                # Get Lidar data
                data = self.lidar.getRangeImage()

                # Process Lidar data using the control logic
                lv, av = self._process_lidar(data)

                # Send velocity commands to the robot
                cmd_vel(self.supervisor, lv, av)

                # Get current robot position
                pos = np.array(self.translation.getSFVec3f())

                # Basic collision detection: deviation from centerline
                current_base_wall_distance = tunnel_builder.base_wall_distance
                current_distance_from_center = abs(pos[1])  # y-axis is considered as lateral deviation

                print(
                    f"[DEBUG] Distance from center: {current_distance_from_center:.3f} | Allowed limit: {current_base_wall_distance - ROBOT_RADIUS:.3f}")

                if current_distance_from_center > current_base_wall_distance - ROBOT_RADIUS:
                    if not flag:
                        print(f"[WARNING] Robot left the tunnel! Position: {pos[:2]}")
                        flag = True
                else:
                    flag = False

                # Check if the robot reached the end of the tunnel
                if end_pos is not None and np.linalg.norm(pos[:2] - end_pos[:2]) < 0.1:
                    print(f"Reached end in {self.supervisor.getTime() - start_time:.1f}s")
                    break

            # Remove tunnel walls after the run
            self._remove_walls(walls_added)
            print("slay")

            # Update statistics
            self.stats['total_collisions'] += collision_count
            if collision_count == 0 and end_pos is not None and np.linalg.norm(pos[:2] - end_pos[:2]) < 0.1:
                self.stats['successful_runs'] += 1
            else:
                self.stats['failed_runs'] += 1

            print(f"Run {run + 1} finished with {collision_count} collisions.")

        # Print final results
        self._print_summary()

    def _process_lidar_with_params(self, dist_values: [float], distP=10.0, angleP=7.0) -> (float, float):

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
        wallDist: float = 0.1  # Desired distance to the wall

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

        angle_increment: float = 2 * math.pi / (size - 1) if size > 1 else 0.0
        angleMin: float = (size // 2 - min_index) * angle_increment
        distMin: float = dist_values[min_index]

        # Get distances from specific directions
        distFront: float = dist_values[size // 2] if size > 0 else float('inf')
        distSide: float = dist_values[size // 4] if size > 0 and size // 4 < size else float('inf') if (
                    direction == 1) else dist_values[3 * size // 4] if size > 0 and 3 * size // 4 < size else float(
            'inf')
        distBack: float = dist_values[0] if size > 0 else float('inf')

        # Prepare message for the robot's motors
        linear_vel: float
        angular_vel: float

        # print("distMin", distMin) # Commented out to reduce console output
        # print("angleMin", angleMin*180/math.pi) # Commented out

        # Decide the robot's behavior
        if math.isfinite(distMin):
            # Check for potential unblocking scenario (stuck)
            if distFront < 1.25 * wallDist and (distSide < 1.25 * wallDist or distBack < 1.25 * wallDist):
                # print("UNBLOCK") # Commented out
                # Turn away from the detected obstacles
                angular_vel = direction * -1 * maxSpeed  # Use maxSpeed for turning velocity
                linear_vel = 0  # Stop linear movement while unblocking
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
            angular_vel = np.random.normal(loc=0.0, scale=1.0) * maxSpeed  # Scale random value by maxSpeed
            # print("angular_vel", angular_vel) # Commented out
            linear_vel = maxSpeed  # Continue moving forward while wandering

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

    def _process_lidar(self, dist_values: [float], distP=10.0, angleP=7.0) -> (float, float):

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

        # Behavior of a random parameter pair, not using predefined easy/medium/hard levels
    def run_experiment_with_params(self, distP: float, angleP: float) -> float:
        total_fitness = 0.0
        difficulties = ['easy', 'medium', 'hard']

        for difficulty in difficulties:
            print(f"\n--- Simulation with difficulty: {difficulty.upper()} ---")

            # Get fixed parameters for the selected difficulty
            num_curves, angle_range, clearance = get_difficulty_settings(difficulty)

            # Create the tunnel
            builder = TunnelBuilder(self.supervisor)
            start_pos, end_pos, walls_added = builder.build_tunnel(
                num_curves=num_curves,
                angle_range=angle_range,
                clearance=clearance
            )

            if start_pos is None:
                print(f"Tunnel out of bounds ({difficulty}). Partial fitness: -1000.")
                total_fitness += -1000.0
                continue

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

            # Initialize simulation variables
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

            while self.supervisor.step(self.timestep) != -1:
                if self.supervisor.getTime() - start_time > TIMEOUT_DURATION:
                    timeout_occurred = True
                    print("TIME OUT")
                    break

                data = self.lidar.getRangeImage()
                lv, av = self._process_lidar_with_params(data, distP, angleP)
                cmd_vel(self.supervisor, lv, av)

                pos = np.array(self.translation.getSFVec3f())
                distance_traveled += np.linalg.norm(pos[:2] - last_pos[:2])
                last_pos = pos.copy()
                rot = self.robot.getField("rotation").getSFRotation()
                heading = rot[3] if rot[2] >= 0 else -rot[3]

                inside_by_geometry = builder.is_robot_near_centerline(pos)
                inside_by_lidar = builder.is_robot_inside_tunnel(pos, heading)
                inside_tunnel = inside_by_geometry or inside_by_lidar

                if inside_tunnel:
                    distance_step = np.linalg.norm(pos[:2] - previous_pos[:2])
                    distance_traveled_inside += distance_step
                    flag_off_tunnel = False
                else:
                    if not flag_off_tunnel and self.supervisor.getTime() - start_time > grace_period:
                        off_tunnel_events += 1
                        flag_off_tunnel = True
                        print(f"[WARNING] Robot left the tunnel ({difficulty}): {pos[:2]}")

                previous_pos = pos

                if end_pos is not None and np.linalg.norm(pos[:2] - end_pos[:2]) < 0.1:
                    goal_reached = True
                    break

            # Remove tunnel walls before next difficulty
            self._remove_walls(walls_added)
            print("slay")

            # Compute fitness for this difficulty
            fitness = 0.0
            total_tunnel_length = sum([seg[3] for seg in builder.segments])
            percent_traveled = distance_traveled / total_tunnel_length if total_tunnel_length > 0 else 0

            if percent_traveled >= 0.8:
                fitness += 500
            if goal_reached and not timeout_occurred:
                fitness += 1000

            fitness -= 2000 * off_tunnel_events
            fitness += 100 * distance_traveled_inside

            print(f"Fitness for {difficulty}: {fitness:.2f}")
            total_fitness += fitness

        # Average of the 3 difficulty levels
        average_fitness = total_fitness / len(difficulties)
        print(f"\nFinal average fitness: {average_fitness:.2f}")
        return average_fitness

    def run_on_existing_tunnel(self, distP, angleP, builder, start_pos, end_pos):
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

        while self.supervisor.step(self.timestep) != -1:
            if self.supervisor.getTime() - start_time > TIMEOUT_DURATION:
                timeout_occurred = True
                break

            data = self.lidar.getRangeImage()
            lv, av = self._process_lidar_with_params(data, distP, angleP)
            cmd_vel(self.supervisor, lv, av)

            pos = np.array(self.translation.getSFVec3f())
            distance_traveled += np.linalg.norm(pos[:2] - last_pos[:2])
            last_pos = pos.copy()
            rot = self.robot.getField("rotation").getSFRotation()
            heading = rot[3] if rot[2] >= 0 else -rot[3]

            inside_by_geometry = builder.is_robot_near_centerline(pos)
            inside_by_lidar = builder.is_robot_inside_tunnel(pos, heading)
            inside_tunnel = inside_by_geometry or inside_by_lidar

            if inside_tunnel:
                distance_step = np.linalg.norm(pos[:2] - previous_pos[:2])
                distance_traveled_inside += distance_step
                flag_off_tunnel = False
            else:
                if not flag_off_tunnel and self.supervisor.getTime() - start_time > grace_period:
                    off_tunnel_events += 1
                    flag_off_tunnel = True

            previous_pos = pos

            if end_pos is not None and np.linalg.norm(pos[:2] - end_pos[:2]) < 0.1:
                goal_reached = True
                break

        total_tunnel_length = sum([seg[3] for seg in builder.segments])
        percent_traveled = distance_traveled / total_tunnel_length if total_tunnel_length > 0 else 0

        fitness = 0.0
        if percent_traveled >= 0.8:
            fitness += 500
        if goal_reached and not timeout_occurred:
            fitness += 1000
        fitness -= 2000 * off_tunnel_events
        fitness += 100 * distance_traveled_inside

        return fitness