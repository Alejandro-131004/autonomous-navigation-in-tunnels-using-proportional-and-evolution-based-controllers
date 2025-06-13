from environment.configuration import WALL_THICKNESS, WALL_HEIGHT, ROBOT_RADIUS, MAX_NUM_CURVES, \
    MIN_STRAIGHT_LENGTH, MAX_STRAIGHT_LENGTH, IDEAL_CURVE_SEGMENT_LENGTH, MIN_OBSTACLE_DISTANCE, MAP_X_MIN, MAP_X_MAX, \
    MAP_Y_MIN, MAP_Y_MAX, MIN_ROBOT_CLEARANCE
import numpy as np
import math
import random as pyrandom
import time


class TunnelBuilder:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.root_children = supervisor.getRoot().getField("children")
        self.base_wall_distance = 0
        self.walls = []
        self.segments_info = []
        self.obstacles = []
        self.wall_count = 0

    def create_wall(self, pos, rot, size, wall_type=None, is_obstacle=False):
        type_str = str(wall_type).upper() if wall_type is not None else "NONE"
        wall_def_name = f"TUNNEL_WALL_{type_str}_{self.wall_count}"
        diffuse_color = '0.5 0.5 0.5' if is_obstacle else '1 0 0'

        physics_node_string = ""

        wall_string = f"""DEF {wall_def_name} Solid {{
            translation {pos[0]} {pos[1]} {pos[2]} rotation {rot[0]} {rot[1]} {rot[2]} {rot[3]}
            children [ Shape {{ appearance Appearance {{ material Material {{ diffuseColor {diffuse_color} }} }} geometry Box {{ size {size[0]} {size[1]} {size[2]} }} }} ]
            name "{wall_def_name}" boundingObject Box {{ size {size[0]} {size[1]} {size[2]} }} contactMaterial "wall" {physics_node_string}
        }}"""
        try:
            self.root_children.importMFNodeFromString(-1, wall_string)
            node = self.supervisor.getFromDef(wall_def_name)
            if node:
                self.walls.append(node)
                if is_obstacle:
                    self.obstacles.append(node)
                self.wall_count += 1
                return node
        except Exception as e:
            print(f"[CRITICAL ERROR] Exception during wall creation: {e}")

    def _clear_walls(self):
        for node in self.walls:
            if node:
                node.remove()
        self.walls.clear()
        self.segments_info.clear()
        self.obstacles.clear()
        self.wall_count = 0
        self.supervisor.step(1)

    def build_tunnel(self, num_curves, angle_range, clearance_factor, num_obstacles, obstacle_types):
        self._clear_walls()
        time.sleep(0.1)
        self.base_wall_distance = ROBOT_RADIUS * clearance_factor

        path = self._generate_path(num_curves, angle_range)
        if not path:
            print("[ERROR] Failed to generate a valid tunnel path.")
            return None, None, 0, None

        self._build_walls_from_path(path)

        robot_start_pos = path[0] + (path[1] - path[0]) / np.linalg.norm(path[1] - path[0]) * ROBOT_RADIUS * 2 if len(
            path) > 1 else np.array([0.0, 0.0, 0.0])

        added_obstacles_count = self._add_obstacles(num_obstacles, robot_start_pos, obstacle_types)

        if added_obstacles_count < num_obstacles:
            print(
                f"[WARNING] Only {added_obstacles_count}/{num_obstacles} obstacles were added. Generating a new map.")
            self._clear_walls()
            return None, None, 0, None

        self.supervisor.step(5)
        start_pos = path[0] + (path[1] - path[0]) / np.linalg.norm(path[1] - path[0]) * ROBOT_RADIUS * 2
        start_pos[2] = 0.0
        end_pos = path[-1]
        final_heading = math.atan2(path[-1][1] - path[-2][1], path[-1][0] - path[-2][0])
        print(f"Tunnel built with {len(self.walls)} walls and {added_obstacles_count} obstacles.")
        return start_pos, end_pos, len(self.walls), final_heading

    def _generate_path(self, num_curves, angle_range):
        T = np.eye(4)
        T[:3, 3] = np.array([MAP_X_MIN, 0.0, 0.0])
        path = [T[:3, 3].copy()]

        length = pyrandom.uniform(MIN_STRAIGHT_LENGTH, MAX_STRAIGHT_LENGTH)
        T[:3, 3] += T[:3, 0] * length
        if not self._within_bounds(T[:3, 3]):
            return None
        path.append(T[:3, 3].copy())
        self.segments_info.append(
            {'type': 'straight', 'start': path[-2], 'end': path[-1], 'length': length, 'heading': 0.0})

        for _ in range(num_curves):
            angle = pyrandom.uniform(angle_range[0], angle_range[1]) * pyrandom.choice([1, -1])
            arc_length = pyrandom.uniform(MIN_STRAIGHT_LENGTH, MAX_STRAIGHT_LENGTH)
            if abs(angle) < 1e-6:
                continue
            num_subdivisions = math.ceil(arc_length / IDEAL_CURVE_SEGMENT_LENGTH)
            step_angle = angle / num_subdivisions
            R_centerline = arc_length / abs(angle)
            centerline_step_length = 2 * R_centerline * math.sin(abs(step_angle) / 2.0)

            for _ in range(num_subdivisions):
                T[:3, 3] += T[:3, 0] * centerline_step_length
                T[:] = T @ self._rotation_z(step_angle)
                if not self._within_bounds(T[:3, 3]):
                    return None
                path.append(T[:3, 3].copy())

            length = pyrandom.uniform(MIN_STRAIGHT_LENGTH, MAX_STRAIGHT_LENGTH)
            heading_before_straight = math.atan2(T[1, 0], T[0, 0])
            segment_start = T[:3, 3].copy()
            T[:3, 3] += T[:3, 0] * length
            if not self._within_bounds(T[:3, 3]):
                return None
            path.append(T[:3, 3].copy())
            self.segments_info.append({'type': 'straight', 'start': segment_start, 'end': path[-1], 'length': length,
                                       'heading': heading_before_straight})

        return path

    def _build_walls_from_path(self, path):
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i + 1]
            segment_vec = p2 - p1
            segment_len = np.linalg.norm(segment_vec)
            if segment_len < 1e-6:
                continue

            heading = math.atan2(segment_vec[1], segment_vec[0])
            unit_vec = segment_vec / segment_len
            perp_vec = np.array([-unit_vec[1], unit_vec[0], 0])

            overlap = WALL_THICKNESS * 2
            mid_point = p1 + (unit_vec * (segment_len / 2.0))

            for side in [-1, 1]:
                wall_pos = mid_point + perp_vec * (side * self.base_wall_distance)
                wall_pos[2] = WALL_HEIGHT / 2.0
                wall_rot = (0, 0, 1, heading)
                wall_size = (segment_len + overlap, WALL_THICKNESS, WALL_HEIGHT)
                self.create_wall(wall_pos, wall_rot, wall_size, 'wall')

    def _add_obstacles(self, num_obstacles, robot_start_pos, obstacle_types):
        if num_obstacles <= 0 or not self.segments_info or not obstacle_types:
            return 0

        added_obstacles_count = 0
        max_attempts = num_obstacles * 25

        # --- ALTERAÇÃO AQUI ---
        # Exclui o primeiro segmento da lista de candidatos para obstáculos.
        # Isto garante que a primeira secção do túnel está sempre livre.
        straight_segments = [s for s in self.segments_info[1:] if
                             s['type'] == 'straight' and s['length'] > MIN_OBSTACLE_DISTANCE * 2]

        if not straight_segments:
            print("[AVISO] Não existem segmentos retos suficientes para adicionar obstáculos após o segmento inicial.")
            return 0

        # Esta verificação de distância torna-se uma segurança adicional, mas a principal
        # lógica é a exclusão do primeiro segmento. A margem foi aumentada para 10 raios.
        min_distance_from_robot_start = ROBOT_RADIUS * 10.0

        for _ in range(max_attempts):
            if added_obstacles_count >= num_obstacles:
                break
            segment = pyrandom.choice(straight_segments)
            dist_along = pyrandom.uniform(MIN_OBSTACLE_DISTANCE, segment['length'] - MIN_OBSTACLE_DISTANCE)
            direction_vec = np.array([math.cos(segment['heading']), math.sin(segment['heading']), 0.0])
            centerline_pos = segment['start'] + direction_vec * dist_along

            obstacle_type = pyrandom.choice(obstacle_types)

            if obstacle_type == 'wall':
                clearance_needed = 2 * (2 * ROBOT_RADIUS)
                tunnel_width = 2 * self.base_wall_distance
                obstacle_width = tunnel_width - clearance_needed

                if obstacle_width <= WALL_THICKNESS:
                    continue

                perp_vec = np.array([-direction_vec[1], direction_vec[0], 0.0])
                side = pyrandom.choice([-1, 1])
                shift_from_wall = obstacle_width / 2.0
                wall_pos = centerline_pos + perp_vec * side * self.base_wall_distance
                obstacle_pos = wall_pos - perp_vec * side * shift_from_wall

                obstacle_rot = (0, 0, 1, segment['heading'] + math.pi / 2.0)
                obstacle_size = (obstacle_width, WALL_THICKNESS, WALL_HEIGHT)

            else:  # Pillar
                perp_vec = np.array([-direction_vec[1], direction_vec[0], 0.0])
                side = pyrandom.choice([-1, 1])
                offset = perp_vec * side * ROBOT_RADIUS
                obstacle_pos = centerline_pos + offset
                obstacle_size = (0.01, 0.01, WALL_HEIGHT)
                obstacle_rot = (0, 0, 1, segment['heading'])

            obstacle_pos[2] = WALL_HEIGHT / 2.0

            if np.linalg.norm(
                    np.array(obstacle_pos[:2]) - np.array(robot_start_pos[:2])) < min_distance_from_robot_start:
                continue

            if any(np.linalg.norm(np.array(obstacle_pos) - o.getPosition()) < MIN_OBSTACLE_DISTANCE for o in
                   self.obstacles):
                continue

            if self.create_wall(obstacle_pos, obstacle_rot, obstacle_size, 'obstacle', True):
                added_obstacles_count += 1

        return added_obstacles_count

    def _within_bounds(self, point):
        margin = self.base_wall_distance + (WALL_THICKNESS / 2.0) + 0.05
        return ((MAP_X_MIN + margin) <= point[0] <= (MAP_X_MAX - margin) and
                (MAP_Y_MIN + margin) <= point[1] <= (MAP_Y_MAX - margin))

    def _rotation_z(self, angle):
        c, s = math.cos(angle), math.sin(angle)
        R = np.eye(4)
        R[0, 0], R[0, 1], R[1, 0], R[1, 1] = c, -s, s, c
        return R
