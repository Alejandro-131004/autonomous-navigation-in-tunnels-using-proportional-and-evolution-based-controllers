import numpy as np
import math
import pickle
import os

from environment.configuration import get_stage_parameters, ROBOT_NAME, ROBOT_RADIUS, TIMEOUT_DURATION, \
    MOVEMENT_TIMEOUT_DURATION, MIN_MOVEMENT_THRESHOLD, MAX_VELOCITY, MIN_VELOCITY
from environment.tunnel import TunnelBuilder
from controllers.utils import cmd_vel


class SimulationManager:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.robot = self.supervisor.getFromDef(ROBOT_NAME)
        if self.robot is None:
            raise ValueError(f"Robot with DEF name '{ROBOT_NAME}' not found.")

        self.translation = self.robot.getField("translation")
        self.rotation = self.robot.getField("rotation")

        self.lidar = self.supervisor.getDevice("lidar")
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()
        self.touch_sensor = self.supervisor.getDevice("touch sensor")
        self.touch_sensor.enable(self.timestep)

        self.left_motor = self.supervisor.getDevice("left wheel motor")
        self.right_motor = self.supervisor.getDevice("right wheel motor")
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

        self.tunnel_builder = TunnelBuilder(self.supervisor)
        self.stats = {'total_collisions': 0, 'successful_runs': 0, 'failed_runs': 0}

    def save_model(self, individual, filename="best_model.pkl", save_dir="saved_models"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, filename)
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(individual, f)
            print(f"Model successfully saved to: {filepath}")
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
        SUCCESS_BONUS = 10_000.0
        PROGRESS_WEIGHT = 500.0
        SPEED_WEIGHT = 100.0
        COLLISION_PENALTY = 5_000.0
        TIMEOUT_PENALTY = 5_000.0
        NO_MOVE_PENALTY = 2_000.0
        SHORT_TRIP_PENALTY = 2_000.0
        OBSTACLE_BONUS = 1_500.0

        progress = initial_dist - final_dist
        rel_prog = progress / (initial_dist + 1e-6)
        avg_speed = total_dist / (elapsed_time + 1e-6)

        fitness = 0.0
        fitness += SUCCESS_BONUS if success else 0.0
        fitness += PROGRESS_WEIGHT * rel_prog
        fitness += SPEED_WEIGHT * avg_speed
        fitness -= COLLISION_PENALTY if collided else 0.0
        fitness -= TIMEOUT_PENALTY if timeout else 0.0
        fitness -= NO_MOVE_PENALTY if no_movement_timeout else 0.0
        fitness -= SHORT_TRIP_PENALTY if (total_dist < ROBOT_RADIUS * 3 and not success) else 0.0
        fitness += OBSTACLE_BONUS * obstacle_pass_count

        return fitness

    def _run_single_episode(self, controller_callable, stage):
        attempt_count = 0
        MAX_BUILD_ATTEMPTS = 10

        while attempt_count < MAX_BUILD_ATTEMPTS:
            attempt_count += 1

            # --- FIX: Unpack the new straight_length_range parameter ---
            num_curves, curve_angles_list, clearance, num_obstacles, obstacle_types, passageway_width, straight_length_range = get_stage_parameters(
                stage)

            # --- FIX: Pass the new parameter to the tunnel builder ---
            start_pos, end_pos, walls_added, final_heading = self.tunnel_builder.build_tunnel(
                num_curves, curve_angles_list, clearance, num_obstacles, obstacle_types, passageway_width,
                straight_length_range
            )

            if start_pos is not None:
                break
            else:
                if os.environ.get('ROBOT_DEBUG_MODE') == '1':
                    print(f"[DEBUG | RETRY] Attempt {attempt_count} for valid tunnel generation failed. Retrying...")

        if start_pos is None:
            if os.environ.get('ROBOT_DEBUG_MODE') == '1':
                print("[DEBUG | FATAL] Could not generate a valid map after multiple attempts. Returning zero fitness.")
            return {'fitness': 0.0, 'success': False, 'collided': False, 'timeout': True, 'no_movement_timeout': False}

        self.robot.resetPhysics()
        self.translation.setSFVec3f([start_pos[0], start_pos[1], 0.0])
        self.rotation.setSFRotation([0, 0, 1, 0])

        self.supervisor.step(self.timestep)

        obstacles = self.tunnel_builder.obstacles
        passed_flags = [False] * len(obstacles)
        obstacle_pass_count = 0

        t0 = self.supervisor.getTime()
        timeout = collided = success = no_movement_timeout = False
        last_pos = np.array(self.translation.getSFVec3f())
        total_dist = 0.0
        initial_dist_to_goal = np.linalg.norm(last_pos[:2] - end_pos[:2])

        last_movement_time = t0
        last_checked_pos = last_pos.copy()

        while self.supervisor.step(self.timestep) != -1:
            elapsed = self.supervisor.getTime() - t0

            if elapsed > TIMEOUT_DURATION:
                timeout = True
                if os.environ.get('ROBOT_DEBUG_MODE') == '1': print("[DEBUG | TIMEOUT]")
                break

            if self.touch_sensor.getValue() > 0:
                collided = True
                if os.environ.get('ROBOT_DEBUG_MODE') == '1': print("[DEBUG | COLLISION]")
                break

            current_pos = np.array(self.translation.getSFVec3f())

            distance_since_last_check = np.linalg.norm(current_pos[:2] - last_checked_pos[:2])
            if distance_since_last_check > MIN_MOVEMENT_THRESHOLD:
                last_movement_time = self.supervisor.getTime()
                last_checked_pos = current_pos.copy()

            if (self.supervisor.getTime() - last_movement_time) > MOVEMENT_TIMEOUT_DURATION:
                no_movement_timeout = True
                if os.environ.get('ROBOT_DEBUG_MODE') == '1': print("[DEBUG | NO MOVEMENT TIMEOUT]")
                break

            robot_xy = current_pos[:2]
            diameter = 2 * ROBOT_RADIUS
            for i, obs in enumerate(obstacles):
                if not passed_flags[i]:
                    obs_xy = np.array(obs.getPosition()[:2])
                    if np.linalg.norm(obs_xy - robot_xy) < diameter:
                        passed_flags[i] = True
                        obstacle_pass_count += 1
                        if os.environ.get('ROBOT_DEBUG_MODE') == '1':
                            print(f"[DEBUG | OBSTACLE PASSED] #{i + 1}")

            scan = np.nan_to_num(self.lidar.getRangeImage(), nan=np.inf)
            lv, av = controller_callable(scan)
            cmd_vel(self.supervisor, lv, av)

            total_dist += np.linalg.norm(current_pos[:2] - last_pos[:2])
            last_pos = current_pos

            goal_area_width = self.tunnel_builder.base_wall_distance * 2.0
            goal_area_length = ROBOT_RADIUS * 4.0

            vec_to_robot = current_pos[:2] - end_pos[:2]
            c, s = np.cos(-final_heading), np.sin(-final_heading)
            local_robot_x = vec_to_robot[0] * c - vec_to_robot[1] * s
            local_robot_y = vec_to_robot[0] * s + vec_to_robot[1] * c

            if abs(local_robot_x) < goal_area_length / 2 and abs(local_robot_y) < goal_area_width / 2:
                success = True
                if os.environ.get('ROBOT_DEBUG_MODE') == '1': print("[DEBUG | SUCCESS]")
                break

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
        elapsed_time = self.supervisor.getTime() - t0

        return {
            'fitness': fitness,
            'success': success,
            'collided': collided,
            'timeout': timeout,
            'no_movement_timeout': no_movement_timeout,
            'elapsed_time': elapsed_time,
            'start_pos': start_pos.copy(),
            'end_pos': end_pos.copy(),
            'final_pos': last_pos.copy(),
            'total_dist': total_dist,
            'elapsed_time': elapsed
        }

    def run_experiment_with_network(self, individual, stage, return_time=False):
        results = self._run_single_episode(individual.act, stage)
        if return_time:
            return results['fitness'], results['success'], results.get('elapsed_time', 0.0)
        return results['fitness'], results['success']

    def run_experiment_with_params(self, distP, angleP, stage, return_time=False):
        def ga_controller(scan):
            return self._process_lidar_for_ga(scan, distP, angleP)

        results = self._run_single_episode(ga_controller, stage)
        if return_time:
            return results['fitness'], bool(results.get('success', False)), results.get('elapsed_time', 0.0)
        return results['fitness'], bool(results.get('success', False))

    def _process_lidar_for_ga(self, dist_values, distP, angleP):
        direction: int = 1
        wall_dist: float = 0.1

        size: int = len(dist_values)
        if size == 0:
            return 0.0, 0.0

        min_index = np.argmin(dist_values) if np.any(np.isfinite(dist_values)) else -1
        if min_index == -1:
            return 0.0, MAX_VELOCITY

        dist_min = dist_values[min_index]
        angle_increment = (2 * math.pi) / size
        angle_min = (size / 2 - min_index) * angle_increment
        dist_front = dist_values[size // 2]

        angular_vel = direction * distP * (dist_min - wall_dist) + angleP * (angle_min - direction * math.pi / 2)

        linear_vel = MAX_VELOCITY
        if dist_front < wall_dist * 1.5:
            linear_vel = 0.0
        elif dist_front < wall_dist * 2.5:
            linear_vel = max(MIN_VELOCITY, MAX_VELOCITY * 0.5)

        return np.clip(linear_vel, -MAX_VELOCITY, MAX_VELOCITY), np.clip(angular_vel, -MAX_VELOCITY * 2,
                                                                         MAX_VELOCITY * 2)
