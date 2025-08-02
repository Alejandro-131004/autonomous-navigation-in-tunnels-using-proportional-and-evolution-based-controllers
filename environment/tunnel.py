import os
import numpy as np
import math
import random as pyrandom
import time
# --- FIX: Removed unused IDEAL_CURVE_SEGMENT_LENGTH import ---
from environment.configuration import WALL_THICKNESS, WALL_HEIGHT, ROBOT_RADIUS, \
    MIN_STRAIGHT_LENGTH, MAX_STRAIGHT_LENGTH, MIN_OBSTACLE_DISTANCE, MAP_X_MIN, MAP_X_MAX, \
    MAP_Y_MIN, MAP_Y_MAX, MAX_CURVE_STEP_ANGLE, MIN_CURVE_SEGMENT_LENGTH, MAX_CURVE_SEGMENT_LENGTH


class TunnelBuilder:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.root_children = supervisor.getRoot().getField("children")
        self.base_wall_distance = 0
        self.walls = []
        self.segments_info = []
        self.obstacles = []
        self.wall_count = 0
        self.debug_mode = os.environ.get('ROBOT_DEBUG_MODE') == '1'

        self.tunnel_group = None
        self._create_tunnel_group()

    def _create_tunnel_group(self):
        """Cria um nó de grupo para conter todos os componentes do túnel para operações em lote."""
        self.root_children.importMFNodeFromString(-1, 'DEF TUNNEL_GROUP Group {}')
        self.tunnel_group = self.supervisor.getFromDef("TUNNEL_GROUP")
        if self.tunnel_group:
            self.tunnel_children = self.tunnel_group.getField("children")
        else:
            if self.debug_mode:
                print("[DEBUG | CRITICAL ERROR] Não foi possível criar TUNNEL_GROUP.")
            self.tunnel_children = self.root_children

    def create_wall(self, pos, rot, size, wall_type=None, is_obstacle=False):
        type_str = str(wall_type).upper() if wall_type is not None else "NONE"
        wall_def_name = f"WALL_{type_str}_{self.wall_count}"
        # Modificado para cor preta para a barreira, e cinza para obstáculos regulares
        diffuse_color = '0 0 0' if wall_type == 'barrier' else ('0.5 0.5 0.5' if is_obstacle else '0 0 0')

        wall_string = f"""
            DEF {wall_def_name} Solid {{
              translation {pos[0]} {pos[1]} {pos[2]}
              rotation {rot[0]} {rot[1]} {rot[2]} {rot[3]}
              children [
                Shape {{
                  appearance Appearance {{
                    material Material {{ diffuseColor {diffuse_color} }}
                  }}
                  geometry Box {{ size {size[0]} {size[1]} {size[2]} }}
                }}
              ]
              name "{wall_def_name}"
              boundingObject Box {{ size {size[0]} {size[1]} {size[2]} }}
              contactMaterial "wall"
            }}
        """
        self.tunnel_children.importMFNodeFromString(-1, wall_string)

        node = self.supervisor.getFromDef(wall_def_name)

        if node:
            self.walls.append(node)
            if is_obstacle:
                self.obstacles.append(node)
            self.wall_count += 1
            return node
        elif self.debug_mode:
            print(f"[DEBUG | CRITICAL ERROR] Falha ao criar o nó da parede: {wall_def_name}")
        return None

    def _clear_walls(self):
        """Clears all tunnel walls safely, handling node removal exceptions."""
        try:
            # Try to remove the tunnel group if it exists
            if self.tunnel_group and self.tunnel_group.getType() != Node.NO_NODE:
                try:
                    self.tunnel_group.remove()
                except Exception as e:
                    if self.debug_mode:
                        print(f"[DEBUG] Error removing tunnel group: {e}")
        except Exception as e:
            if self.debug_mode:
                print(f"[DEBUG] Exception accessing tunnel group: {e}")

        # Clear local references regardless
        self.walls.clear()
        self.segments_info.clear()
        self.obstacles.clear()
        self.wall_count = 0
        self.tunnel_group = None

        # Always create a new tunnel group
        self._create_tunnel_group()

    def build_tunnel(self, num_curves, curve_angles_list, clearance_factor, num_obstacles, obstacle_types,
                     passageway_width, straight_length_range=(MIN_STRAIGHT_LENGTH, MAX_STRAIGHT_LENGTH)):
        self._clear_walls()
        self.base_wall_distance = ROBOT_RADIUS * clearance_factor

        path = self._generate_path(num_curves, curve_angles_list, straight_length_range)
        if not path:
            if self.debug_mode:
                print("[DEBUG | ERRO] Falha ao gerar um caminho de túnel válido.")
            return None, None, 0, None

        self._build_walls_from_path(path)

        # Ajusta a posição inicial do robô para estar mais dentro do túnel
        # O robô começará 4 vezes o seu raio à frente do início do primeiro segmento
        robot_start_offset = ROBOT_RADIUS * 4.0
        start_pos_segment_vec = path[1] - path[0]
        start_pos_unit_vec = start_pos_segment_vec / np.linalg.norm(start_pos_segment_vec)
        robot_start_pos = path[0] + start_pos_unit_vec * robot_start_offset
        robot_start_pos[2] = 0.0 # Garante que a altura é zero

        # Adiciona uma barreira logo atrás da posição inicial do robô
        # Esta barreira impede o robô de voltar para trás no início
        barrier_pos_offset = ROBOT_RADIUS * 0.5 # Posição da barreira mais próxima do início do túnel
        barrier_pos = path[0] + start_pos_unit_vec * barrier_pos_offset
        barrier_pos[2] = WALL_HEIGHT / 2.0 # Altura da barreira
        barrier_rot = (0, 0, 1, math.atan2(start_pos_unit_vec[1], start_pos_unit_vec[0]) + math.pi / 2.0) # Rotaciona para ser perpendicular ao túnel
        barrier_size = (self.base_wall_distance * 2 + WALL_THICKNESS * 1.5, WALL_THICKNESS * 0.8, WALL_HEIGHT) # Largura total do túnel, espessura ligeiramente reduzida
        self.create_wall(barrier_pos, barrier_rot, barrier_size, 'barrier', True)


        added_obstacles_count = self._add_obstacles(num_obstacles, robot_start_pos, obstacle_types, passageway_width)

        if added_obstacles_count < num_obstacles:
            if self.debug_mode:
                print(
                    f"[DEBUG | AVISO] Apenas {added_obstacles_count}/{num_obstacles} obstáculos foram adicionados. A gerar um novo mapa.")
            return None, None, 0, None

        # A posição final do túnel é o último ponto do caminho
        end_pos = path[-1]
        # A orientação final é a da último segmento
        final_heading = math.atan2(path[-1][1] - path[-2][1], path[-1][0] - path[-2][0])

        if self.debug_mode:
            print(f"  [DEBUG] Túnel construído com {len(self.walls)} paredes e {added_obstacles_count} obstáculos.")

        return robot_start_pos, end_pos, len(self.walls), final_heading

    def _generate_path(self, num_curves, curve_angles_list, straight_length_range):
        T = np.eye(4)
        T[:3, 3] = np.array([0.0, 0.0, 0.0])  # Começa no centro do mapa
        path = [T[:3, 3].copy()]

        length = pyrandom.uniform(straight_length_range[0], straight_length_range[1])
        T[:3, 3] += T[:3, 0] * length
        if not self._within_bounds(T[:3, 3]):
            return None
        path.append(T[:3, 3].copy())
        self.segments_info.append({
            'type': 'straight',
            'start': path[-2],
            'end': path[-1],
            'length': length,
            'heading': 0.0
        })

        default_length_range = (MIN_STRAIGHT_LENGTH, MAX_STRAIGHT_LENGTH)
        for angle_deg in curve_angles_list:
            angle = math.radians(angle_deg) * pyrandom.choice([1, -1])
            arc_length = pyrandom.uniform(default_length_range[0], default_length_range[1])
            if abs(angle) < 1e-6:
                continue

            angle_ratio = min(abs(angle) / math.radians(90.0), 1.0)

            ideal_length = (
                    MAX_CURVE_SEGMENT_LENGTH
                    - angle_ratio * (MAX_CURVE_SEGMENT_LENGTH - MIN_CURVE_SEGMENT_LENGTH)
            )
            n_length = math.ceil(arc_length / ideal_length) if ideal_length > 0 else float('inf')
            n_angle = math.ceil(abs(angle) / MAX_CURVE_STEP_ANGLE)
            num_subdivisions = max(n_length, n_angle, 1)

            step_angle = angle / num_subdivisions
            R_centerline = arc_length / abs(angle) if abs(angle) > 1e-6 else 0
            centerline_step_length = 2 * R_centerline * math.sin(
                abs(step_angle) / 2.0) if R_centerline > 0 else arc_length / num_subdivisions

            for _ in range(num_subdivisions):
                T[:3, 3] += T[:3, 0] * centerline_step_length
                T[:] = T @ self._rotation_z(step_angle)
                if not self._within_bounds(T[:3, 3]):
                    return None
                path.append(T[:3, 3].copy())

            length = pyrandom.uniform(default_length_range[0], default_length_range[1])
            # --- FIX: Changed incorrect math.atand to math.atan2 ---
            heading_before = math.atan2(T[1, 0], T[0, 0])
            seg_start = T[:3, 3].copy()
            T[:3, 3] += T[:3, 0] * length
            if not self._within_bounds(T[:3, 3]):
                return None
            path.append(T[:3, 3].copy())
            self.segments_info.append({
                'type': 'straight',
                'start': seg_start,
                'end': path[-1],
                'length': length,
                'heading': heading_before
            })

        return path

    def _build_walls_from_path(self, path):
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i + 1]
            segment_vec = p2 - p1
            segment_len = np.linalg.norm(segment_vec)
            if segment_len < 1e-6:
                continue

            heading = math.atan2(segment_vec[1], segment_vec[0])

            if i + 2 < len(path):
                next_vec = path[i + 2] - p2
                next_heading = math.atan2(next_vec[1], next_vec[0])
                delta = (next_heading - heading + math.pi) % (2 * math.pi) - math.pi
                overlap = WALL_THICKNESS / max(math.cos(delta / 2.0), 0.1)
            else:
                overlap = WALL_THICKNESS * 2

            unit_vec = segment_vec / segment_len
            perp_vec = np.array([-unit_vec[1], unit_vec[0], 0.0])
            mid_point = p1 + unit_vec * (segment_len / 2.0)

            for side in (-1, 1):
                wall_pos = mid_point + perp_vec * (side * self.base_wall_distance)
                wall_pos[2] = WALL_HEIGHT / 2.0
                wall_rot = (0, 0, 1, heading)
                wall_size = (segment_len + overlap, WALL_THICKNESS, WALL_HEIGHT)
                self.create_wall(wall_pos, wall_rot, wall_size, 'wall')

    def _add_obstacles(self, num_obstacles, robot_start_pos, obstacle_types, passageway_width):
        if num_obstacles <= 0 or not self.segments_info or not obstacle_types:
            return 0

        added_obstacles_count = 0
        max_attempts = num_obstacles * 25

        # Podemos colocar obstáculos em qualquer segmento reto, incluindo o primeiro agora
        straight_segments = [s for s in self.segments_info if
                             s['type'] == 'straight' and s['length'] > MIN_OBSTACLE_DISTANCE * 2]

        if not straight_segments:
            if self.debug_mode:
                print(
                    "[DEBUG | AVISO] Não existem segmentos retos suficientes para adicionar obstáculos.")
            return 0

        total_available_length = sum(s['length'] for s in straight_segments)
        required_length = num_obstacles * (MIN_OBSTACLE_DISTANCE * 2)

        if required_length > total_available_length:
            return 0

        min_distance_from_robot_start = ROBOT_RADIUS * 10.0

        for _ in range(max_attempts):
            if added_obstacles_count >= num_obstacles:
                break
            segment = pyrandom.choice(straight_segments)
            dist_along = pyrandom.uniform(MIN_OBSTACLE_DISTANCE, segment['length'] - MIN_OBSTACLE_DISTANCE)
            direction_vec = np.array([math.cos(segment['heading']), math.sin(segment['heading']), 0.0])
            centerline_pos = segment['start'] + direction_vec * dist_along

            obstacle_type = pyrandom.choice(obstacle_types)

            if passageway_width is not None:
                tunnel_width = self.base_wall_distance * 2
                obstacle_width = tunnel_width - passageway_width

                if obstacle_width <= WALL_THICKNESS:
                    continue

                perp_vec = np.array([-direction_vec[1], direction_vec[0], 0.0])
                side = pyrandom.choice([-1, 1])
                shift_from_wall = obstacle_width / 2.0
                wall_pos = centerline_pos + perp_vec * side * self.base_wall_distance
                obstacle_pos = wall_pos - perp_vec * side * shift_from_wall

                obstacle_rot = (0, 0, 1, segment['heading'] + math.pi / 2.0)
                obstacle_size = (obstacle_width, WALL_THICKNESS, WALL_HEIGHT)
            else:
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

            too_close = False
            for obs_node in self.obstacles:
                if np.linalg.norm(
                        np.array(obstacle_pos[:2]) - np.array(obs_node.getPosition()[:2])) < MIN_OBSTACLE_DISTANCE:
                    too_close = True
                    break
            if too_close:
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
