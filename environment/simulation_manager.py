import numpy as np
import math
import pickle
import os

from environment.configuration import ROBOT_NAME, ROBOT_RADIUS, TIMEOUT_DURATION, get_stage_parameters, \
    MAX_DIFFICULTY_STAGE, MIN_STRAIGHT_LENGTH, MOVEMENT_TIMEOUT_DURATION, MIN_MOVEMENT_THRESHOLD
from environment.tunnel import TunnelBuilder
from controllers.utils import cmd_vel


class SimulationManager:
    """
    Complete and final version of the simulation manager.
    Maintains all original functionalities and is compatible with:
    1. Neuroevolution (via `run_experiment_with_network`).
    2. Classic Genetic Algorithm (via `run_experiment_with_params`).
    3. Generic tests (via `run_experiment`).
    """

    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.robot = self.supervisor.getFromDef(ROBOT_NAME)
        if self.robot is None:
            raise ValueError(f"Robot with DEF name '{ROBOT_NAME}' not found.")

        self.translation = self.robot.getField("translation")
        self.rotation = self.robot.getField("rotation")

        # Enable devices
        self.lidar = self.supervisor.getDevice("lidar")
        self.lidar.enable(self.timestep)

        self.touch_sensor = self.supervisor.getDevice("touch sensor")
        self.touch_sensor.enable(self.timestep)

        self.left_motor = self.supervisor.getDevice("left wheel motor")
        self.right_motor = self.supervisor.getDevice("right wheel motor")
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

        # Statistics for generic runs
        self.stats = {'total_collisions': 0, 'successful_runs': 0, 'failed_runs': 0}

    def save_model(self, individual, filename="best_model.pkl", save_dir="saved_models"):
        """Saves an individual (model) to a file."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, filename)
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(individual, f)
            print(f"Model successfully saved at: {filepath}")
        except Exception as e:
            print(f"[ERROR] Failed to save model at {filepath}: {e}")

    def _calculate_fitness(self,
                       success: bool,
                       collided: bool,
                       timeout: bool,
                       no_movement_timeout: bool,
                       initial_dist: float,
                       final_dist: float,
                       total_dist: float,
                       elapsed_time: float,
                       obstacle_pass_count: int) -> float:
        """
        Centralized function to calculate fitness based on episode results:
        - success bonus
        - proportional progress
        - average speed
        - penalties: collision, timeout, no movement, very short travel
        - bonus per obstacle passed
        """
        # Constants
        SUCCESS_BONUS       = 10_000.0
        PROGRESS_WEIGHT     = 500.0
        SPEED_WEIGHT        = 100.0
        COLLISION_PENALTY   = 5_000.0
        TIMEOUT_PENALTY     = 6_000.0   # increased timeout penalty
        NO_MOVE_PENALTY     = 6_000.0
        SHORT_TRIP_PENALTY  = 2_000.0
        OBSTACLE_BONUS      = 1_500.0   # per obstacle passed

        # Core metrics
        progress  = initial_dist - final_dist
        rel_prog  = progress / (initial_dist + 1e-6)
        avg_speed = total_dist / (elapsed_time + 1e-6)

        # Base fitness
        fitness = 0.0
        fitness += SUCCESS_BONUS if success else 0.0
        fitness += PROGRESS_WEIGHT   * rel_prog
        fitness += SPEED_WEIGHT      * avg_speed
        fitness -= COLLISION_PENALTY if collided else 0.0
        fitness -= TIMEOUT_PENALTY   if timeout else 0.0
        fitness -= NO_MOVE_PENALTY   if no_movement_timeout else 0.0
        fitness -= SHORT_TRIP_PENALTY if (total_dist < ROBOT_RADIUS * 3 and not success) else 0.0

        # Obstacle bonus
        fitness += OBSTACLE_BONUS * obstacle_pass_count

        return fitness

    def _run_single_episode(self, controller_callable, stage, total_stages):
        """
        Runs a single simulation episode, retrying tunnel generation if it fails.
        """
        MAX_ATTEMPTS = 5
        for attempt in range(MAX_ATTEMPTS):
            num_curves, angle_range, clearance, num_obstacles = get_stage_parameters(stage, total_stages)
            builder = TunnelBuilder(self.supervisor)
            start_pos, end_pos, walls_added, final_heading = builder.build_tunnel(
                num_curves, angle_range, clearance, num_obstacles
            )

            if start_pos is not None:
                break  # Successfully generated a valid tunnel
            else:
                print(f"[RETRY] Attempt {attempt + 1}/{MAX_ATTEMPTS} to generate a valid tunnel failed.")

        if start_pos is None:
            print("[FAILURE] All attempts to generate the tunnel failed. Assigning fitness -10000.")
            return {'fitness': -10000.0, 'success': False, 'collided': False, 'timeout': True,
                    'no_movement_timeout': False}

        # 2. Reset robot physics and position
        self.robot.resetPhysics()
        self.translation.setSFVec3f([start_pos[0], start_pos[1], 0.0])
        self.rotation.setSFRotation([0, 0, 1, 0])
        self.supervisor.step(5)

        # Initialize obstacle-passing tracking
        obstacles = builder.obstacles
        passed_flags = [False] * len(obstacles)
        obstacle_pass_count = 0

        # 3. Initialize episode variables
        t0 = self.supervisor.getTime()
        timeout = collided = success = no_movement_timeout = False
        last_pos = np.array(self.translation.getSFVec3f())
        total_dist = 0.0
        initial_dist_to_goal = np.linalg.norm(last_pos[:2] - end_pos[:2])

        # Variables for inactivity timeout
        last_movement_time = t0
        last_checked_pos = last_pos.copy()

        # 4. Main simulation loop
        while self.supervisor.step(self.timestep) != -1:
            elapsed = self.supervisor.getTime() - t0

            # Overall episode timeout
            if elapsed > TIMEOUT_DURATION:
                timeout = True
                print("[TIMEOUT]")
                break

            # Collision detection
            if self.touch_sensor.getValue() > 0:
                collided = True
                print("[COLLISION]")
                break

            # Current robot position
            current_pos = np.array(self.translation.getSFVec3f())

            # Check for inactivity timeout
            distance_since_last_check = np.linalg.norm(current_pos[:2] - last_checked_pos[:2])
            if distance_since_last_check > MIN_MOVEMENT_THRESHOLD:
                last_movement_time = self.supervisor.getTime()
                last_checked_pos = current_pos.copy()

            if (self.supervisor.getTime() - last_movement_time) > MOVEMENT_TIMEOUT_DURATION:
                no_movement_timeout = True
                print(f"[NO MOVEMENT TIMEOUT] Robot did not move significantly for {MOVEMENT_TIMEOUT_DURATION}s.")
                break

            # Detect passed obstacles
            robot_xy = current_pos[:2]
            diameter = 2 * ROBOT_RADIUS
            for i, obs in enumerate(obstacles):
                if not passed_flags[i]:
                    obs_xy = np.array(obs.getPosition()[:2])
                    if np.linalg.norm(obs_xy - robot_xy) < diameter:
                        passed_flags[i] = True
                        obstacle_pass_count += 1
                        print(f"[OBSTACLE PASSED] Obstacle #{i+1} at {obs_xy} passed (total {obstacle_pass_count})")

            # Controller step
            scan = np.nan_to_num(self.lidar.getRangeImage(), nan=np.inf)
            lv, av = controller_callable(scan)
            cmd_vel(self.supervisor, lv, av)

            # Update distance traveled
            total_dist += np.linalg.norm(current_pos[:2] - last_pos[:2])
            last_pos = current_pos

            # Robust goal check
            goal_area_width = builder.base_wall_distance * 2.0
            goal_area_length = ROBOT_RADIUS * 4.0

            vec_to_robot = current_pos[:2] - end_pos[:2]
            c, s = np.cos(-final_heading), np.sin(-final_heading)
            local_robot_x = vec_to_robot[0] * c - vec_to_robot[1] * s
            local_robot_y = vec_to_robot[0] * s + vec_to_robot[1] * c

            if abs(local_robot_x) < goal_area_length / 2 and abs(local_robot_y) < goal_area_width / 2:
                success = True
                print(f"[SUCCESS] Robot entered the goal area. Time: {elapsed:.2f}s")
                break

        # 5. Clear tunnel walls
        builder._clear_walls()

        # 6. Calculate final fitness (now passing obstacle_pass_count)
        final_dist_to_goal = np.linalg.norm(last_pos[:2] - end_pos[:2])
        fitness = self._calculate_fitness(
            success,
            collided,
            timeout,
            no_movement_timeout,
            initial_dist_to_goal,
            final_dist_to_goal,
            total_dist,
            elapsed,
            obstacle_pass_count  
        )

        return {
            'fitness': fitness,
            'success': success,
            'collided': collided,
            'timeout': timeout,
            'no_movement_timeout': no_movement_timeout
        }

    # --- Interface Functions for Optimizers and Tests ---

    def run_experiment_with_network(self, individual, stage, total_stages=MAX_DIFFICULTY_STAGE):
        """Interface for NEUROEVOLUTION (used by `curriculum.py`)."""
        ind_id = getattr(individual, 'id', 'N/A')
        print(f"[RUN-NETWORK] Ind {ind_id} | Stage {stage}")
        results = self._run_single_episode(individual.act, stage, total_stages)
        print(
            f"[FITNESS] Ind {ind_id} | Fit: {results['fitness']:.2f} | Success: {results['success']} | No Movement Timeout: {results['no_movement_timeout']}")
        return results['fitness'], results['success']

    def run_experiment_with_params(self, distP, angleP, stage, total_stages=MAX_DIFFICULTY_STAGE):
        """Returns (fitness: float, success: bool)."""
        print(f"[RUN-PARAMS] distP={distP:.2f}, angleP={angleP:.2f} | Stage {stage}")

        def ga_controller(scan):
            return self._process_lidar_for_ga(scan, distP, angleP)

        results = self._run_single_episode(ga_controller, stage, total_stages)
        fitness = results['fitness']
        success = bool(results.get('success', False))
        print(f"[FITNESS] Params | Fit: {fitness:.2f} | Success: {success}")
        return fitness, success

    def run_experiment(self, num_runs):
        """Generic test function with a default controller."""
        print("Running generic experiment with default controller.")
        self.stats = {'total_collisions': 0, 'successful_runs': 0, 'failed_runs': 0}  # Reset stats

        for run in range(num_runs):
            print(f"\n--- Test Run {run + 1}/{num_runs} ---")
            stage = 1  # Use easiest stage for tests

            # Default controller for this experiment
            def default_controller(scan):
                return self._process_lidar_for_ga(scan, 10.0, 5.0)  # Use some default params

            results = self._run_single_episode(default_controller, stage, MAX_DIFFICULTY_STAGE)

            # Update stats
            if results['success']:
                self.stats['successful_runs'] += 1
            else:
                self.stats['failed_runs'] += 1
            if results['collided']:
                self.stats['total_collisions'] += 1

        self._print_summary()

    def _process_lidar_for_ga(self, dist_values, distP, angleP):
        """Classic wall-following control logic for Genetic Algorithm."""
        direction: int = 1
        max_speed: float = 0.12
        wall_dist: float = 0.1

        size: int = len(dist_values)
        if size == 0:
            return 0.0, 0.0

        min_index = np.argmin(dist_values) if np.any(np.isfinite(dist_values)) else -1
        if min_index == -1:
            return 0.0, max_speed

        dist_min = dist_values[min_index]
        angle_increment = (2 * math.pi) / size
        angle_min = (size / 2 - min_index) * angle_increment
        dist_front = dist_values[size // 2]

        angular_vel = direction * distP * (dist_min - wall_dist) + angleP * (angle_min - direction * math.pi / 2)

        linear_vel = max_speed
        if dist_front < wall_dist * 1.5:
            linear_vel = 0
        elif dist_front < wall_dist * 2.5:
            linear_vel = max_speed * 0.5

        return np.clip(linear_vel, -max_speed, max_speed), np.clip(angular_vel, -max_speed * 2, max_speed * 2)

    def _print_summary(self):
        """Prints a summary of the test run statistics."""
        print("\n=== Final Experiment Summary ===")
        print(f"Successful runs: {self.stats['successful_runs']}")
        print(f"Failed runs: {self.stats['failed_runs']}")
        print(f"Total collisions: {self.stats['total_collisions']}")
        total_runs = self.stats['successful_runs'] + self.stats['failed_runs']
        if total_runs > 0:
            success_rate = (self.stats['successful_runs'] / total_runs) * 100
            print(f"Success rate: {success_rate:.1f}%")
        else:
            print("No runs were completed.")
