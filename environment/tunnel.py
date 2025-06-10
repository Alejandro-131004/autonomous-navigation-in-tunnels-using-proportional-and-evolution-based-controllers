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
        self.segments_info = []  # Armazena info para obstáculos
        self.obstacles = []
        self.wall_count = 0

    def create_wall(self, pos, rot, size, wall_type=None, is_obstacle=False):
        type_str = str(wall_type).upper() if wall_type is not None else "NONE"
        wall_def_name = f"TUNNEL_WALL_{type_str}_{self.wall_count}"
        diffuse_color = '0.5 0.5 0.5' if is_obstacle else '1 0 0'
        physics_node_string = "physics Physics { density 1000 dampingFactor 0.9 }" if is_obstacle else ""

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
                if is_obstacle: self.obstacles.append(node)
                self.wall_count += 1
                return node
        except Exception as e:
            print(f"[CRITICAL ERROR] Exception during wall creation: {e}")

    def _clear_walls(self):
        for node in self.walls:
            if node: node.remove()
        self.walls.clear();
        self.segments_info.clear();
        self.obstacles.clear();
        self.wall_count = 0
        self.supervisor.step(1)

    def build_tunnel(self, num_curves, angle_range, clearance_factor, num_obstacles):
        self._clear_walls()
        time.sleep(0.1)
        self.base_wall_distance = ROBOT_RADIUS * clearance_factor

        # 1. Gerar o esqueleto do túnel (lista de pontos da linha central)
        path = self._generate_path(num_curves, angle_range)
        if not path:
            print("[ERROR] Falha ao gerar um caminho válido para o túnel.")
            return None, None, 0, None

        # 2. Construir a geometria das paredes a partir do caminho
        self._build_walls_from_path(path)

        # 3. Adicionar obstáculos
        self._add_obstacles(num_obstacles)

        # 4. Finalizar
        self.supervisor.step(5)
        start_pos = path[0] + (path[1] - path[0]) / np.linalg.norm(path[1] - path[0]) * ROBOT_RADIUS * 2
        start_pos[2] = 0.0  # Garante que começa no chão
        end_pos = path[-1]
        final_heading = math.atan2(path[-1][1] - path[-2][1], path[-1][0] - path[-2][0])
        print(f"Túnel construído com {len(self.walls)} paredes.")
        return start_pos, end_pos, len(self.walls), final_heading

    def _generate_path(self, num_curves, angle_range):
        """Gera uma lista de pontos 3D que definem a linha central do túnel."""
        T = np.eye(4)
        T[:3, 3] = np.array([MAP_X_MIN, 0.0, 0.0])  # Começa na borda
        path = [T[:3, 3].copy()]

        # Primeiro segmento reto
        length = pyrandom.uniform(MIN_STRAIGHT_LENGTH, MAX_STRAIGHT_LENGTH)
        T[:3, 3] += T[:3, 0] * length
        if not self._within_bounds(T[:3, 3]): return None
        path.append(T[:3, 3].copy())
        self.segments_info.append(
            {'type': 'straight', 'start': path[-2], 'end': path[-1], 'length': length, 'heading': 0.0})

        # Curvas e segmentos retos subsequentes
        for _ in range(num_curves):
            angle = pyrandom.uniform(angle_range[0], angle_range[1]) * pyrandom.choice([1, -1])
            arc_length = pyrandom.uniform(MIN_STRAIGHT_LENGTH, MAX_STRAIGHT_LENGTH)
            num_subdivisions = math.ceil(arc_length / IDEAL_CURVE_SEGMENT_LENGTH)
            step_angle = angle / num_subdivisions
            R_centerline = arc_length / abs(angle)
            centerline_step_length = 2 * R_centerline * math.sin(abs(step_angle) / 2.0)

            # Adiciona os pontos da curva
            for _ in range(num_subdivisions):
                T[:3, 3] += T[:3, 0] * centerline_step_length
                T[:] = T @ self._rotation_z(step_angle)
                if not self._within_bounds(T[:3, 3]): return None
                path.append(T[:3, 3].copy())

            # Adiciona o segmento reto após a curva
            length = pyrandom.uniform(MIN_STRAIGHT_LENGTH, MAX_STRAIGHT_LENGTH)
            heading_before_straight = math.atan2(T[1, 0], T[0, 0])
            segment_start = T[:3, 3].copy()
            T[:3, 3] += T[:3, 0] * length
            if not self._within_bounds(T[:3, 3]): return None
            path.append(T[:3, 3].copy())
            self.segments_info.append({'type': 'straight', 'start': segment_start, 'end': path[-1], 'length': length,
                                       'heading': heading_before_straight})

        return path

    def _build_walls_from_path(self, path):
        """Constrói as paredes do túnel com juntas perfeitas a partir de uma lista de pontos."""
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]

            # Vetor e heading do segmento atual
            segment_vec = p2 - p1
            segment_len = np.linalg.norm(segment_vec)
            if segment_len < 1e-6: continue
            heading = math.atan2(segment_vec[1], segment_vec[0])

            # --- Cálculo da Extensão da Junta (Miter Joint) ---
            # Extensão no início do segmento
            if i > 0:
                prev_vec = p1 - path[i - 1]
                angle_change_start = self._angle_between(prev_vec, segment_vec)
                start_ext = (WALL_THICKNESS / 2.0) / math.tan(angle_change_start / 2.0) if math.tan(
                    angle_change_start / 2.0) != 0 else 0
            else:
                start_ext = 0

            # Extensão no final do segmento
            if i < len(path) - 2:
                next_vec = path[i + 2] - p2
                angle_change_end = self._angle_between(segment_vec, next_vec)
                end_ext = (WALL_THICKNESS / 2.0) / math.tan(angle_change_end / 2.0) if math.tan(
                    angle_change_end / 2.0) != 0 else 0
            else:
                end_ext = 0

            # Construir as duas paredes para este segmento
            extended_len = segment_len + abs(start_ext) + abs(end_ext)
            mid_point = p1 + segment_vec / 2.0
            perp_vec = np.array([-segment_vec[1], segment_vec[0], 0]) / segment_len

            for side in [-1, 1]:
                wall_pos = mid_point + perp_vec * (side * self.base_wall_distance)
                wall_pos[2] = WALL_HEIGHT / 2.0
                wall_rot = (0, 0, 1, heading)
                wall_size = (extended_len, WALL_THICKNESS, WALL_HEIGHT)
                self.create_wall(wall_pos, wall_rot, wall_size, 'wall')

    def _angle_between(self, v1, v2):
        """Calcula o ângulo entre dois vetores 2D."""
        v1_u = v1[:2] / np.linalg.norm(v1[:2])
        v2_u = v2[:2] / np.linalg.norm(v2[:2])
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def _add_obstacles(self, num_obstacles):
        """Adiciona obstáculos de diferentes tipos no túnel."""
        if num_obstacles <= 0 or not self.segments_info: return
        added_obstacles_count = 0
        max_attempts = num_obstacles * 25
        straight_segments = [s for s in self.segments_info if s['length'] > MIN_OBSTACLE_DISTANCE * 2]
        if not straight_segments: return

        for _ in range(max_attempts):
            if added_obstacles_count >= num_obstacles: break
            segment = pyrandom.choice(straight_segments)
            dist_along = pyrandom.uniform(MIN_OBSTACLE_DISTANCE, segment['length'] - MIN_OBSTACLE_DISTANCE)
            direction_vec = np.array([math.cos(segment['heading']), math.sin(segment['heading']), 0.0])
            centerline_pos = segment['start'] + direction_vec * dist_along

            obstacle_type = pyrandom.choice(['wall', 'pillar'])
            if obstacle_type == 'wall':
                perp_vec = np.array([-direction_vec[1], direction_vec[0], 0.0])
                side = pyrandom.choice([-1, 1])
                obstacle_width = self.base_wall_distance * 0.6
                shift_from_wall = obstacle_width / 2 + WALL_THICKNESS / 2
                wall_pos = centerline_pos + perp_vec * side * self.base_wall_distance
                obstacle_pos = wall_pos - perp_vec * side * shift_from_wall
                obstacle_size = (obstacle_width, WALL_THICKNESS, WALL_HEIGHT)
            else:  # Pillar
                obstacle_pos = np.copy(centerline_pos)
                obstacle_size = (ROBOT_RADIUS * 1.5, ROBOT_RADIUS * 1.5, WALL_HEIGHT)

            obstacle_pos[2] = WALL_HEIGHT / 2.0
            obstacle_rot = (0, 0, 1, segment['heading'])
            if any(np.linalg.norm(np.array(obstacle_pos) - o.getPosition()) < MIN_OBSTACLE_DISTANCE for o in
                   self.obstacles): continue
            if self.create_wall(obstacle_pos, obstacle_rot, obstacle_size, 'obstacle', True):
                added_obstacles_count += 1

    def _within_bounds(self, point):
        """Verifica se um ponto está dentro dos limites do mapa com uma margem de segurança."""
        margin = self.base_wall_distance + (WALL_THICKNESS / 2.0) + 0.05
        return ((MAP_X_MIN + margin) <= point[0] <= (MAP_X_MAX - margin) and
                (MAP_Y_MIN + margin) <= point[1] <= (MAP_Y_MAX - margin))

    def _rotation_z(self, angle):
        c, s = math.cos(angle), math.sin(angle)
        R = np.eye(4);
        R[0, 0], R[0, 1], R[1, 0], R[1, 1] = c, -s, s, c
        return R
