from environment.configuration import WALL_THICKNESS, WALL_HEIGHT, OVERLAP_FACTOR, ROBOT_RADIUS, MAX_NUM_CURVES, \
    MIN_STRAIGHT_LENGTH, MAX_STRAIGHT_LENGTH, CURVE_SUBDIVISIONS, MIN_OBSTACLE_DISTANCE, MAP_X_MIN, MAP_X_MAX, \
    MAP_Y_MIN, MAP_Y_MAX, WALL_JOINT_GAP, BASE_WALL_LENGTH, MIN_ROBOT_CLEARANCE
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
        self.segments = []
        self.obstacles = []
        self.wall_count = 0

    def create_wall(self, pos, rot, size, wall_type=None, is_obstacle=False):
        """Cria uma parede ou obstáculo no Webots, restaurando a lógica original."""
        type_str = str(wall_type).upper() if wall_type is not None else "NONE"
        wall_def_name = f"TUNNEL_WALL_{type_str}_{self.wall_count}"
        diffuse_color = '0.5 0.5 0.5' if is_obstacle else '1 0 0'

        physics_node_string = ""
        if is_obstacle:
            # Obstacles are dynamic and have physics
            physics_node_string = "physics Physics { density 1000 dampingFactor 0.9 }"

        wall_string = f"""DEF {wall_def_name} Solid {{
            translation {pos[0]} {pos[1]} {pos[2]}
            rotation {rot[0]} {rot[1]} {rot[2]} {rot[3]}
            children [
                Shape {{
                    appearance Appearance {{ material Material {{ diffuseColor {diffuse_color} }} }}
                    geometry Box {{ size {size[0]} {size[1]} {size[2]} }}
                }}
            ]
            name "{wall_def_name}"
            boundingObject Box {{ size {size[0]} {size[1]} {size[2]} }}
            contactMaterial "wall"
            {physics_node_string}
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
            return None
        except Exception as e:
            print(f"[CRITICAL ERROR] Exception during wall creation: {e}")
            return None

    def _clear_walls(self):
        """Remove todas as paredes da simulação."""
        for node in self.walls:
            if node: node.remove()
        self.walls.clear()
        self.segments.clear()
        self.obstacles.clear()
        self.wall_count = 0
        self.supervisor.step(1)

    def build_tunnel(self, num_curves, angle_range, clearance_factor, num_obstacles):
        """Constrói a estrutura completa do túnel."""
        self._clear_walls()
        time.sleep(0.1)

        self.base_wall_distance = ROBOT_RADIUS * clearance_factor

        # O túnel começa na borda do mapa.
        initial_pos = np.array([MAP_X_MIN, 0.0, 0.0])
        initial_heading = 0.0
        T = np.eye(4)
        T[:3, 3], T[:3, :3] = initial_pos, self._rotation_z(initial_heading)[:3, :3]

        # MODIFICAÇÃO: Posição inicial do robô é dentro do túnel para evitar colisão na borda.
        # O túnel ainda é construído a partir de T, mas a posição retornada para o robô é mais à frente.
        start_pos = T[:3, 3].copy() + np.array([ROBOT_RADIUS * 2, 0.0, 0.0])

        segment_length = pyrandom.uniform(MIN_STRAIGHT_LENGTH, MAX_STRAIGHT_LENGTH)
        if not self._add_straight(T, segment_length): return None, None, 0, None

        for i in range(num_curves):
            angle = pyrandom.uniform(angle_range[0], angle_range[1]) * pyrandom.choice([1, -1])
            if not self._add_curve(T, angle): break
            segment_length = pyrandom.uniform(MIN_STRAIGHT_LENGTH, MAX_STRAIGHT_LENGTH)
            if not self._add_straight(T, segment_length): break

        end_pos = T[:3, 3].copy()
        final_heading = math.atan2(T[1, 0], T[0, 0])

        self._add_obstacles(num_obstacles)
        # MODIFICAÇÃO: A chamada para criar as paredes da arena foi removida.
        # self._add_main_boundary_walls()

        self.supervisor.step(5)
        print(f"Túnel construído com {len(self.walls)} paredes.")
        return start_pos, end_pos, len(self.walls), final_heading

    def _add_straight(self, T, length):
        """Adiciona um segmento reto usando UMA parede longa por lado (Otimizado)."""
        heading = math.atan2(T[1, 0], T[0, 0])
        start_pos_segment = T[:3, 3].copy()

        if not self._within_bounds(T, length): return False

        mid_point = T[:3, 3] + T[:3, 0] * (length / 2.0)

        for side in [-1, 1]:
            wall_perp_offset = side * self.base_wall_distance
            wall_pos = mid_point + T[:3, 1] * wall_perp_offset + np.array([0, 0, WALL_HEIGHT / 2])
            wall_rot = (0, 0, 1, heading)
            wall_size = (length, WALL_THICKNESS, WALL_HEIGHT)
            self.create_wall(wall_pos, wall_rot, wall_size, 'straight')

        T[:3, 3] += T[:3, 0] * length
        self.segments.append({'type': 'straight', 'start': start_pos_segment, 'end': T[:3, 3].copy(), 'length': length,
                              'heading': heading})
        return True

    def _add_curve(self, T, angle):
        """Adiciona um segmento curvo (Corrigido)."""
        if abs(angle) < 1e-6: return True
        arc_length = pyrandom.uniform(MIN_STRAIGHT_LENGTH, MAX_STRAIGHT_LENGTH)
        R_centerline = arc_length / abs(angle)
        step_angle = angle / CURVE_SUBDIVISIONS
        step_length = 2 * R_centerline * math.sin(abs(step_angle) / 2)

        start_pos_segment = T[:3, 3].copy()

        tempT = T.copy()
        for _ in range(CURVE_SUBDIVISIONS):
            tempT[:3, 3] += tempT[:3, 0] * step_length
            tempT[:] = tempT @ self._rotation_z(step_angle)
            if not self._within_bounds(tempT, 0): return False

        for _ in range(CURVE_SUBDIVISIONS):
            heading = math.atan2(T[1, 0], T[0, 0])
            wall_piece_length = step_length * OVERLAP_FACTOR
            for side in [-1, 1]:
                wall_center_offset = step_length / 2.0
                wall_perp_offset = side * self.base_wall_distance
                wall_pos = T[:3, 3] + T[:3, 0] * wall_center_offset + T[:3, 1] * wall_perp_offset + np.array(
                    [0, 0, WALL_HEIGHT / 2])
                wall_rot = (0, 0, 1, heading + step_angle / 2)
                wall_size = (wall_piece_length, WALL_THICKNESS, WALL_HEIGHT)
                self.create_wall(wall_pos, wall_rot, wall_size, 'curve')
            T[:3, 3] += T[:3, 0] * step_length
            T[:] = T @ self._rotation_z(step_angle)

        self.segments.append(
            {'type': 'curve', 'start': start_pos_segment, 'end': T[:3, 3].copy(), 'length': arc_length, 'angle': angle})
        return True

    def _add_obstacles(self, num_obstacles):
        """Adiciona obstáculos de diferentes tipos no túnel."""
        if num_obstacles <= 0 or not self.segments: return
        added_obstacles_count = 0
        max_attempts = num_obstacles * 25
        straight_segments = [s for s in self.segments if
                             s['type'] == 'straight' and s['length'] > MIN_OBSTACLE_DISTANCE * 2]
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

    # MODIFICAÇÃO: A função para criar as paredes da arena foi removida.
    # def _add_main_boundary_walls(self): ...

    def _within_bounds(self, T, length):
        """Verifica se um ponto está dentro dos limites do mapa com uma margem de segurança."""
        point_to_check = T[:3, 3] + T[:3, 0] * length
        # A margem precisa de considerar a distância da parede mais a sua espessura.
        margin = self.base_wall_distance + (WALL_THICKNESS / 2.0)
        return ((MAP_X_MIN + margin) <= point_to_check[0] <= (MAP_X_MAX - margin) and
                (MAP_Y_MIN + margin) <= point_to_check[1] <= (MAP_Y_MAX - margin))

    def _rotation_z(self, angle):
        c, s = math.cos(angle), math.sin(angle)
        R = np.eye(4);
        R[0, 0], R[0, 1], R[1, 0], R[1, 1] = c, -s, s, c
        return R

    # --- Funções de Verificação do Robô (Restauradas) ---

    def is_robot_near_centerline(self, robot_position):
        robot_xy = np.array(robot_position[:2])
        threshold = self.base_wall_distance + ROBOT_RADIUS
        for segment in self.segments:
            start = segment['start'][:2]
            end = segment['end'][:2]
            dist = self.point_to_segment_distance(start, end, robot_xy)
            if dist <= threshold * 1.2:
                return True
        return False

    def point_to_segment_distance(self, A, B, P):
        if np.array_equal(A, B): return np.linalg.norm(P - A)
        AB, AP = B - A, P - A
        t = np.dot(AP, AB) / (np.dot(AB, AB) + 1e-9)
        t = np.clip(t, 0, 1)
        closest_point = A + t * AB
        return np.linalg.norm(P - closest_point)

    def is_robot_inside_tunnel(self, robot_position, heading):
        return self.is_robot_near_centerline(robot_position)
