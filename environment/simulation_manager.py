import numpy as np
import math
import pickle  # Import for saving/loading models
import os  # Import for path manipulation

from environment.configuration import ROBOT_NAME, ROBOT_RADIUS, TIMEOUT_DURATION, MAX_NUM_CURVES, get_stage_parameters
from environment.tunnel import TunnelBuilder
from agent.controller import cmd_vel  # Assuming cmd_vel is defined elsewhere
from optimizer.neuralpopulation import NeuralPopulation  # Assuming NeuralPopulation is defined elsewhere


class SimulationManager:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.robot = self.supervisor.getFromDef(ROBOT_NAME)
        if self.robot is None:
            raise ValueError(f"Robot with DEF name '{ROBOT_NAME}' not found.")
        self.translation = self.robot.getField("translation")
        self.rotation = self.robot.getField("rotation")  # Get the rotation field
        self.lidar = self.supervisor.getDevice("lidar")
        self.lidar.enable(self.timestep)
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

    def run_experiment(self, num_runs):
        # This method is for fixed difficulty runs, potentially for initial testing
        # For GA training, run_experiment_with_params should be used.
        print("Running fixed difficulty experiment (consider using run_experiment_with_params for GA).")
        # Example usage with a fixed difficulty stage (e.e., stage 0 for easy)
        num_curves, angle_range, clearance_factor, num_obstacles = get_stage_parameters(0)  # Example: Stage 0 (Easy)

        for run in range(num_runs):
            print(f"--- Starting Run {run + 1} ---")

            # Build a new tunnel with parameters from the stage
            tunnel_builder = TunnelBuilder(self.supervisor)
            # Capture final_heading from build_tunnel
            start_pos, end_pos, walls_added, final_heading = tunnel_builder.build_tunnel(
                num_curves=num_curves,
                angle_range=angle_range,
                clearance=clearance_factor,
                num_obstacles=num_obstacles
            )

            if start_pos is None:
                print("Tunnel out of bounds, skipping run.")
                continue

            # Reposition the robot at the start position with Z = 0
            self.translation.setSFVec3f([start_pos[0], start_pos[1], 0.0])
            self.rotation.setSFRotation([0, 0, 1, 0])  # Set rotation to face forward (along positive X)
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

                # Check for goal condition: robot crosses the line at the end of the tunnel
                if end_pos is not None and final_heading is not None:
                    # Define the end line using a point (end_pos) and a normal vector
                    # The normal vector is perpendicular to the final heading
                    normal_vector = np.array([-math.sin(final_heading), math.cos(final_heading)])

                    # Vector from the end position to the robot's current position (2D)
                    vector_to_robot = pos[:2] - end_pos[:2]

                    # Project the vector_to_robot onto the normal vector
                    # This gives the signed distance from the end line to the robot's center
                    signed_distance = np.dot(vector_to_robot, normal_vector)

                    # If the signed distance is greater than or equal to the robot's radius,
                    # it means at least part of the robot has crossed the line.
                    if signed_distance >= -ROBOT_RADIUS:  # Use -ROBOT_RADIUS to check if any part crosses
                        goal_reached = True
                        print(f"Reached end in {self.supervisor.getTime() - start_time:.1f}s")
                        break

            self._remove_walls(walls_added)

            self.stats['total_collisions'] += collision_count
            if collision_count == 0 and goal_reached:  # Check goal_reached flag
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
        # Check if dist_values is None
        if dist_values is None:
            print("[WARNING] Lidar data is None in _process_lidar_with_params. Returning zero velocities.")
            return 0.0, 0.0

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

        print(
            f"\n--- Simulation with Stage: {stage} (Curves: {num_curves}, Angle Range: {angle_range}, Clearance: {clearance_factor:.2f}, Obstacles: {num_obstacles}) ---")

        # Create the tunnel with stage-specific parameters
        builder = TunnelBuilder(self.supervisor)
        # Capture final_heading from build_tunnel
        start_pos, end_pos, walls_added, final_heading = builder.build_tunnel(
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
        self.translation.setSFVec3f([start_pos[0], start_pos[1], 0])
        self.rotation.setSFRotation([0, 0, 1, 0])  # Set rotation to face forward (along positive X)
        self.robot.resetPhysics()

        # Initialize motors
        left = self.supervisor.getDevice("left wheel motor")
        right = self.supervisor.getDevice("right wheel motor")
        left.setPosition(float('inf'))
        right.setPosition(float('inf'))
        left.setVelocity(0)
        right.setVelocity(0)

        # Initialize simulation variables for fitness calculation
        grace_period = 2.0  # Time at the start to allow robot to settle
        off_tunnel_events = 0  # Count times robot leaves the main tunnel path
        flag_off_tunnel = False  # Flag to prevent multiple counts for one continuous off-path event
        start_time = self.supervisor.getTime()
        timeout_occurred = False
        goal_reached = False
        distance_traveled_inside = 0.0  # Distance traveled while inside the tunnel path
        previous_pos = np.array(self.translation.getSFVec3f())  # For calculating distance step
        distance_traveled = 0.0  # Total distance traveled
        last_pos = previous_pos.copy()  # For calculating total distance

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
            heading = rot[3] if rot[2] >= 0 else -rot[3]  # Simplified heading extraction

            # Check if the robot is inside the tunnel path
            # Using the centerline check as the primary indicator of being "inside"
            inside_tunnel = builder.is_robot_near_centerline(pos)

            if inside_tunnel:
                distance_step = np.linalg.norm(pos[:2] - previous_pos[:2])
                distance_traveled_inside += distance_step
                flag_off_tunnel = False  # Reset flag when back inside
            else:
                # If outside the tunnel path and grace period is over
                if not flag_off_tunnel and self.supervisor.getTime() - start_time > grace_period:
                    off_tunnel_events += 1
                    flag_off_tunnel = True  # Set flag to indicate currently off path
                    print(f"[COLLISION] Robot came out of the tunnel. Episode ends (Stage {stage})")
                    break

            # Update previous position for next distance calculation
            previous_pos = pos

            # Check for goal condition: robot crosses the line at the end of the tunnel
            if end_pos is not None and final_heading is not None:
                # Define the end line using a point (end_pos) and a normal vector
                # The normal vector is perpendicular to the final heading direction
                # The direction the robot should be moving to exit is along the final heading.
                # The line is perpendicular to this, so the normal is rotated by +pi/2 or -pi/2.
                # Let's assume the robot should be moving generally "forward" along the tunnel's final direction.
                # The line is perpendicular to the final heading.
                # A point P is on the "goal side" of the line if (P - end_pos) dot normal_vector is positive.
                # The normal vector should point "backwards" into the tunnel from the end line.
                # If final_heading is the direction *of* the tunnel at the end,
                # the perpendicular direction is final_heading + pi/2 or final_heading - pi/2.
                # Let's assume the goal line is perpendicular to the final heading, positioned at end_pos.
                # A point P is past the line if its projection onto the final_heading vector is beyond end_pos.
                # Alternatively, consider the line perpendicular to final_heading at end_pos.
                # The normal vector pointing *out* of the tunnel is final_heading rotated by -pi/2.
                # normal_vector_out = np.array([math.cos(final_heading - math.pi/2), math.sin(final_heading - math.pi/2)])

                # Let's use a simpler check based on the robot's position projected onto the final heading.
                # The robot reaches the goal if its center, plus its radius in the direction of the final heading,
                # is beyond the end_pos projected onto the same direction.

                final_direction_vector = np.array([math.cos(final_heading), math.sin(final_heading)])
                vector_from_end_to_robot = pos[:2] - end_pos[:2]

                # Project the vector from end_pos to robot_pos onto the final direction of the tunnel
                projection_on_final_direction = np.dot(vector_from_end_to_robot, final_direction_vector)

                # The robot reaches the goal if its position along the final direction,
                # considering its radius, is past the end_pos.
                # Robot's leading edge position along the final direction = projection_on_final_direction + ROBOT_RADIUS
                # The goal is reached if this leading edge is >= 0 (meaning it's at or past the end_pos along the final direction)
                if projection_on_final_direction + ROBOT_RADIUS >= 0:
                    goal_reached = True
                    print(f"Reached end in {self.supervisor.getTime() - start_time:.1f}s (Stage {stage})")
                    break

        # Remove tunnel walls after the simulation run for this stage
        self._remove_walls(walls_added)
        print("slay")

        # --- Compute Fitness ---
        fitness = 0.0
        total_tunnel_length = sum([seg[3] for seg in builder.segments])  # Sum of segment lengths
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
            fitness -= 500.0  # Penalize timeout

        print(f"Fitness for Stage {stage}: {fitness:.2f}")

        return fitness

    # The run_on_existing_tunnel method is kept, but might be less relevant
    # if run_experiment_with_params is used for all GA evaluations.
    # It could be useful for re-evaluating the best individual on a specific
    # pre-generated tunnel if needed.
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

    def run_neuroevolution(self, generations=30, pop_size=20,
                           input_size=32, hidden_size=16, output_size=2,
                           mutation_rate=0.1, elitism=1, save_interval=5, save_dir="saved_models"):
        """
        Main evolution loop for neuroevolution.
        Evolves a population of neural networks to control a robot using LIDAR input.
        Includes functionality to save the best model periodically.

        Args:
            generations (int): Total number of generations to run the evolution.
            pop_size (int): Size of the neural network population.
            input_size (int): Number of input neurons for the neural network.
            hidden_size (int): Number of hidden neurons for the neural network.
            output_size (int): Number of output neurons for the neural network.
            mutation_rate (float): Probability of mutation for offspring.
            elitism (int): Number of top individuals to carry over to the next generation.
            save_interval (int): How often (in generations) to save the best model.
            save_dir (str): Directory where models will be saved.
        """
        population = NeuralPopulation(pop_size, input_size, hidden_size, output_size,
                                      mutation_rate=mutation_rate, elitism=elitism)

        fitness_history = []

        for gen in range(generations):
            print(f"\nGeneration {gen + 1}/{generations}")

            # Evaluate current generation
            population.evaluate(self)  # Pass self (SimulationManager instance) for run_experiment_with_network

            # Log fitnesses
            gen_fitness = [ind.fitness for ind in population.individuals]
            fitness_history.append(gen_fitness)

            avg_fitness = np.mean(gen_fitness)
            best = population.get_best_individual()

            print(f"Generation {gen + 1} average fitness: {avg_fitness:.2f}")
            print(f"Best fitness: {best.fitness:.2f}")

            # Save the best model periodically
            if (gen + 1) % save_interval == 0:
                self.save_model(best, gen + 1, save_dir)

            # Create next generation
            population.create_next_generation()

        print("\nEvolution completed!")
        return population.get_best_individual(), fitness_history

    '''def run_experiment_with_network(self, individual, stage: int) -> float:
        """
        Executes a simulation using an MLP-based individual on a tunnel with a specific stage difficulty.
        Computes the fitness based on progression, collisions, and goal reach.
        """
        print(f"\n--- Simulation for Individual in Stage {stage} ---")

        # Get tunnel parameters
        num_curves, angle_range, clearance, num_obstacles = get_stage_parameters(stage)

        builder = TunnelBuilder(self.supervisor)
        start_pos, end_pos, walls_added, final_heading = builder.build_tunnel(
            num_curves=num_curves,
            angle_range=angle_range,
            clearance=clearance,
            num_obstacles=num_obstacles
        )

        if start_pos is None:
            print(f"[ERROR] Tunnel out of bounds (Stage {stage}). Returning penalty fitness.")
            return -5000.0

        # Reset robot state
        self.translation.setSFVec3f([start_pos[0], start_pos[1], ROBOT_RADIUS])
        self.rotation.setSFRotation([0, 0, 1, 0])
        self.robot.resetPhysics()

        left = self.supervisor.getDevice("left wheel motor")
        right = self.supervisor.getDevice("right wheel motor")
        left.setPosition(float('inf'))
        right.setPosition(float('inf'))
        left.setVelocity(0)
        right.setVelocity(0)

        # Initialize variables
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
            current_time = self.supervisor.getTime()
            elapsed_time = current_time - start_time

            if elapsed_time > TIMEOUT_DURATION:
                timeout_occurred = True
                print("[TIMEOUT] Simulation exceeded time limit.")
                break

            lidar_data = self.lidar.getRangeImage()
            lidar_data = np.nan_to_num(lidar_data, nan=0.0)
            lv, av = individual.act(lidar_data)
            cmd_vel(self.supervisor, lv, av)

            pos = np.array(self.translation.getSFVec3f())
            distance_traveled += np.linalg.norm(pos[:2] - last_pos[:2])
            last_pos = pos.copy()
            rot = self.robot.getField("rotation").getSFRotation()
            heading = rot[3] if rot[2] >= 0 else -rot[3]

            inside_geometry = builder.is_robot_near_centerline(pos)
            inside_lidar = builder.is_robot_inside_tunnel(pos, heading)
            inside_tunnel = inside_geometry or inside_lidar

            if inside_tunnel:
                distance_step = np.linalg.norm(pos[:2] - previous_pos[:2])
                distance_traveled_inside += distance_step
                flag_off_tunnel = False
            else:
                if not flag_off_tunnel and elapsed_time > grace_period:
                    off_tunnel_events += 1
                    flag_off_tunnel = True
                    print(f"[COLLISION] Robot came out of the tunnel. Episode ends (Stage {stage})")
                    break  # MATA O EPISÓDIO!

            previous_pos = pos

            if end_pos is not None and final_heading is not None:
                final_vec = np.array([math.cos(final_heading), math.sin(final_heading)])
                to_robot = pos[:2] - end_pos[:2]
                projection = np.dot(to_robot, final_vec)
                if projection + ROBOT_RADIUS >= 0:
                    goal_reached = True
                    print(f"[SUCCESS] Goal reached in {elapsed_time:.2f} seconds.")
                    break

        self._remove_walls(walls_added)

        # --- Fitness Calculation ---
        fitness = 0.0
        total_tunnel_length = sum([seg[3] for seg in builder.segments])
        percent_traveled = distance_traveled / total_tunnel_length if total_tunnel_length > 0 else 0

        if percent_traveled >= 0.8:
            fitness += 500
        if goal_reached and not timeout_occurred:
            fitness += 1000
        fitness -= 2000 * off_tunnel_events
        fitness += 100 * distance_traveled_inside
        if timeout_occurred:
            fitness -= 500

        print(f"[IND {individual.id}] Stage {stage} → Fitness = {fitness:.2f}")

        return fitness'''

    def run_experiment_with_network(self, individual, stage: int) -> float:
        """
        Runs a simulation using an MLP-based individual on a tunnel with a given difficulty stage.
        Ends early if collision is detected via the touch sensor. Calculates fitness based on
        distance traveled, staying on path, and reaching the goal.
        """
        print(f"\n--- Simulation for Individual in Stage {stage} ---")

        # Get tunnel configuration for this stage
        num_curves, angle_range, clearance, num_obstacles = get_stage_parameters(stage)

        # Build the tunnel
        builder = TunnelBuilder(self.supervisor)
        start_pos, end_pos, walls_added, final_heading = builder.build_tunnel(
            num_curves=num_curves,
            angle_range=angle_range,
            clearance=clearance,
            num_obstacles=num_obstacles
        )

        if start_pos is None:
            print(f"[ERROR] Tunnel generation failed. Returning low fitness.")
            return -5000.0

        # Reset robot state
        self.translation.setSFVec3f([start_pos[0], start_pos[1], ROBOT_RADIUS])
        self.rotation.setSFRotation([0, 0, 1, 0])
        self.robot.resetPhysics()

        # Setup motors
        left = self.supervisor.getDevice("left wheel motor")
        right = self.supervisor.getDevice("right wheel motor")
        left.setPosition(float('inf'))
        right.setPosition(float('inf'))
        left.setVelocity(0)
        right.setVelocity(0)

        # Setup touch sensor
        touch_sensor = self.supervisor.getDevice("touch sensor")
        touch_sensor.enable(self.timestep)

        # Initialize tracking variables
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

        # Simulation loop
        while self.supervisor.step(self.timestep) != -1:
            current_time = self.supervisor.getTime()
            elapsed_time = current_time - start_time

            # Check timeout
            if elapsed_time > TIMEOUT_DURATION:
                timeout_occurred = True
                print("[TIMEOUT] Simulation exceeded time limit.")
                break

            # Check for collision via touch sensor
            if touch_sensor.getValue() > 0:
                print("[COLLISION] Touch sensor was triggered — ending episode.")
                break

            # Process LIDAR and act
            lidar_data = self.lidar.getRangeImage()
            lidar_data = np.nan_to_num(lidar_data, nan=0.0)
            lv, av = individual.act(lidar_data)
            cmd_vel(self.supervisor, lv, av)

            # Update position tracking
            pos = np.array(self.translation.getSFVec3f())
            distance_traveled += np.linalg.norm(pos[:2] - last_pos[:2])
            last_pos = pos.copy()

            rot = self.robot.getField("rotation").getSFRotation()
            heading = rot[3] if rot[2] >= 0 else -rot[3]

            # Check if robot is inside the tunnel path
            inside_geometry = builder.is_robot_near_centerline(pos)
            inside_lidar = builder.is_robot_inside_tunnel(pos, heading)
            inside_tunnel = inside_geometry or inside_lidar

            if inside_tunnel:
                distance_step = np.linalg.norm(pos[:2] - previous_pos[:2])
                distance_traveled_inside += distance_step
                flag_off_tunnel = False
            else:
                if not flag_off_tunnel and elapsed_time > grace_period:
                    off_tunnel_events += 1
                    flag_off_tunnel = True
                    print(f"[WARNING] Robot deviated from tunnel at position {pos[:2]}")
                    # Optional: break here if you want to end simulation on deviation

            previous_pos = pos

            # Check if the goal was reached
            if end_pos is not None and final_heading is not None:
                goal_vector = np.array([math.cos(final_heading), math.sin(final_heading)])
                to_robot = pos[:2] - end_pos[:2]
                projection = np.dot(to_robot, goal_vector)
                if projection + ROBOT_RADIUS >= 0:
                    goal_reached = True
                    print(f"[SUCCESS] Goal reached in {elapsed_time:.2f} seconds.")
                    break

        # Remove tunnel walls after simulation ends
        self._remove_walls(walls_added)

        # Fitness calculation
        fitness = 0.0
        total_tunnel_length = sum([seg[3] for seg in builder.segments])
        percent_traveled = distance_traveled / total_tunnel_length if total_tunnel_length > 0 else 0

        if percent_traveled >= 0.8:
            fitness += 500
        if goal_reached and not timeout_occurred:
            fitness += 1000
        fitness -= 2000 * off_tunnel_events
        fitness += 100 * distance_traveled_inside
        if timeout_occurred:
            fitness -= 500

        print(f"[IND {individual.id}] Stage {stage} → Fitness = {fitness:.2f}")
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

    def _print_summary(self, builder=None):
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
        # Optionally print tunnel length if builder is provided
        if builder and hasattr(builder, 'segments'):
            total_tunnel_length = sum([seg[3] for seg in builder.segments])
            print(f"Total tunnel length: {total_tunnel_length:.2f}")

    # The _process_lidar method is kept for compatibility but _process_lidar_with_params is used in GA runs
    def _process_lidar(self, dist_values: [float]) -> (float, float):

        """
        Robot control logic based on Lidar data (using default parameters).
        """
        # Check if dist_values is None
        if dist_values is None:
            print("[WARNING] Lidar data is None in _process_lidar. Returning zero velocities.")
            return 0.0, 0.0

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
