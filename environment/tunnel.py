from environment.configuration import WALL_THICKNESS, WALL_HEIGHT, OVERLAP_FACTOR, ROBOT_RADIUS, NUM_CURVES, MIN_CLEARANCE_FACTOR, MAX_CLEARANCE_FACTOR, MIN_CURVE_ANGLE, MAX_CURVE_ANGLE, BASE_WALL_LENGTH, CURVE_SUBDIVISIONS, MIN_ROBOT_CLEARANCE, NUM_OBSTACLES, MIN_OBSTACLE_DISTANCE, MAP_X_MIN, MAP_X_MAX, MAP_Y_MIN, MAP_Y_MAX
import numpy as np
import math
import random as pyrandom

class TunnelBuilder:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.root_children = supervisor.getRoot().getField("children")
        self.base_wall_distance = 0
        self.walls = []
        self.segments = []
        self.right_walls = []
        self.left_walls = []
        self.obstacles = []

    def create_wall(self, pos, rot, size, wall_type=None):
        wall = f"""Solid {{
            translation {pos[0]} {pos[1]} {pos[2]}
            rotation {rot[0]} {rot[1]} {rot[2]} {rot[3]}
            children [
                Shape {{
                    appearance Appearance {{
                        material Material {{
                            diffuseColor 1 0 0
                        }}
                    }}
                    geometry Box {{
                        size {size[0]} {size[1]} {size[2]}
                    }}
                }}
            ]
        }}"""
        self.root_children.importMFNodeFromString(-1, wall)
        if wall_type == 'left':
            self.left_walls.append((pos, rot, size))
        elif wall_type == 'right':
            self.right_walls.append((pos, rot, size))
        elif wall_type == 'obstacle':
            self.obstacles.append((pos, rot, size))

    def build_tunnel(self, num_curves, angle_range=None, clearance=None):
        if clearance is not None:
            tunnel_clearance_factor = clearance
        else:
            tunnel_clearance_factor = pyrandom.uniform(MIN_CLEARANCE_FACTOR, MAX_CLEARANCE_FACTOR)

        self.base_wall_distance = ROBOT_RADIUS * tunnel_clearance_factor
        print(f"Building tunnel with clearance factor: {tunnel_clearance_factor:.2f}")

        angle_min, angle_max = angle_range if angle_range else (MIN_CURVE_ANGLE, MAX_CURVE_ANGLE)

        num_curves = min(num_curves, NUM_CURVES)
        length = BASE_WALL_LENGTH * (1 + pyrandom.uniform(-0.15, 0.15))
        segment_length = length / (num_curves + 1)

        T = np.eye(4)
        start_pos = T[:3, 3].copy()
        walls = []
        segments_data = []

        if not self._within_bounds(T, segment_length):
            return None, None, 0
        segment_start_pos = T[:3, 3].copy()
        self._add_straight(T, segment_length, walls)
        segment_end_pos = T[:3, 3].copy()
        segment_heading = math.atan2(T[1, 0], T[0, 0])
        segments_data.append((segment_start_pos, segment_end_pos, segment_heading, segment_length))

        for _ in range(num_curves):
            angle = pyrandom.uniform(angle_min, angle_max) * pyrandom.choice([1, -1])

            if not self._within_bounds_after_curve(T, angle, segment_length):
                break
            prev_end_pos = T[:3, 3].copy()
            self._add_curve(T, angle, segment_length, walls, segments_data)
            new_start = segments_data[-CURVE_SUBDIVISIONS][0]  # início da curva recém-adicionada
            gap = np.linalg.norm(prev_end_pos - new_start)
            if gap > 1e-3:
                print(f"[WARNING] Discontinuity between curve segments: gap = {gap:.3f}")
            if not self._within_bounds(T, segment_length):
                break
            segment_start_pos = T[:3, 3].copy()
            self._add_straight(T, segment_length, walls)
            segment_end_pos = T[:3, 3].copy()
            segment_heading = math.atan2(T[1, 0], T[0, 0])
            segments_data.append((segment_start_pos, segment_end_pos, segment_heading, segment_length))

        end_pos = T[:3, 3].copy()
        self._add_obstacles(segments_data, walls)
        self.walls = walls
        self.segments = segments_data
        return start_pos, end_pos, len(walls)

    def point_to_segment_distance(self, A, B, P):
        AP = P - A
        AB = B - A
        t = np.clip(np.dot(AP, AB) / np.dot(AB, AB), 0.0, 1.0)
        closest_point = A + t * AB
        return np.linalg.norm(P - closest_point)

    def is_robot_near_centerline(self, robot_position):
        """
        Checks if the robot is close enough to the center line of the tunnel,
        considering all segments (straight + curves).
        """
        robot_xy = np.array(robot_position[:2])
        threshold = self.base_wall_distance + ROBOT_RADIUS

        for start, end, _, _ in self.segments:
            dist = self.point_to_segment_distance(start[:2], end[:2], robot_xy)
            if dist <= threshold:
                return True
        return False  # vir aqui

        robot_xy = np.array(robot_position[:2])
        right_dir = np.array([math.cos(heading - math.pi / 2), math.sin(heading - math.pi / 2)])
        left_dir = np.array([math.cos(heading + math.pi / 2), math.sin(heading + math.pi / 2)])

        def check_walls(walls, direction):
            for pos, _, _ in walls:
                wall_xy = np.array(pos[:2])
                to_wall = wall_xy - robot_xy
                dist = np.linalg.norm(to_wall)
                dot = np.dot(to_wall / (dist + 1e-6), direction)
                if dot > 0.7:
                    expected = self.base_wall_distance
                    if abs(dist - expected) < 0.05:
                        return True
            return False

        def check_obstacles(obstacles):
            for pos, _, size in obstacles:
                wall_xy = np.array(pos[:2])
                dist = np.linalg.norm(wall_xy - robot_xy)
                # Tolerância baseada na largura do obstáculo
                if dist < size[1] / 2 + ROBOT_RADIUS + 0.02:
                    return True
            return False

        near_right = check_walls(self.right_walls, right_dir)
        near_left = check_walls(self.left_walls, left_dir)
        near_obstacle = check_obstacles(self.obstacles)

        inside = (near_right and near_left) or near_obstacle
        # print(f"[DEBUG] Inside tunnel: {inside} | Right: {near_right}, Left: {near_left}, Obstacle: {near_obstacle}")
        return inside

    def _add_straight(self, T, length, walls):
        # Calculate the heading of the current segment
        heading = math.atan2(T[1, 0], T[0, 0])
        # Add walls on both sides of the centerline
        for side in [-1, 1]:
            # Calculate the wall's center position:
            # Start at current T's position, move half the length along T's x-axis (forward),
            # then move self.base_wall_distance along T's y-axis (sideways), and half height up.
            pos = T[:3, 3] + T[:3, 0] * (length / 2) + T[:3, 1] * (side * self.base_wall_distance) + np.array(
                [0, 0, WALL_HEIGHT / 2])
            # Rotation is around the z-axis based on the heading
            rot = (0, 0, 1, heading)
            # Size of the wall (length, thickness, height)
            size = (length, WALL_THICKNESS, WALL_HEIGHT)
            wall_type = 'left' if side == -1 else 'right'
            self.create_wall(pos, rot, size, wall_type=wall_type)
            walls.append((pos, rot, size))  # Append wall details to the walls list
        # Update T to the end of the straight segment
        T[:3, 3] += T[:3, 0] * length

    def _add_curve(self, T, angle, segment_length, walls, segments_data):
        step = angle / CURVE_SUBDIVISIONS
        centerline_step_length = segment_length / CURVE_SUBDIVISIONS
        r_centerline = self.base_wall_distance
        r_inner_edge = r_centerline - WALL_THICKNESS / 2.0
        r_outer_edge = r_centerline + WALL_THICKNESS / 2.0

        for _ in range(CURVE_SUBDIVISIONS):
            T_start_step = T.copy()
            segment_start = T[:3, 3].copy()

            T_mid_step = T_start_step @ self._rotation_z(step / 2)
            T_mid_step[:3, 3] += T_mid_step[:3, 0] * (centerline_step_length / 2)
            heading = math.atan2(T_mid_step[1, 0], T_mid_step[0, 0])
            rot = (0, 0, 1, heading)

            for side in [-1, 1]:
                wall_length = r_inner_edge * abs(step) if side == -1 else r_outer_edge * abs(step)
                wall_length += OVERLAP_FACTOR * WALL_THICKNESS
                pos = T_start_step[:3, 3] + T_start_step[:3, 1] * (side * self.base_wall_distance) + np.array(
                    [0, 0, WALL_HEIGHT / 2]) + T_start_step[:3, 0] * (wall_length / 2)
                size = (wall_length, WALL_THICKNESS, WALL_HEIGHT)
                wall_type = 'left' if side == -1 else 'right'
                self.create_wall(pos, rot, size, wall_type=wall_type)
                walls.append((pos, rot, size))

            T[:] = T_start_step @ self._rotation_z(step)
            T[:3, 3] += T[:3, 0] * centerline_step_length

            segment_end = T[:3, 3].copy()
            segment_heading = math.atan2(T[1, 0], T[0, 0])
            segments_data.append((segment_start, segment_end, segment_heading, centerline_step_length))
            # print(f"[CURVE SEGMENT] start: {segment_start}, end: {segment_end}, heading: {math.degrees(segment_heading):.1f}°")

    def _add_obstacles(self, straight_segments_data, walls):
        """
        Place NUM_OBSTACLES perpendicular walls into the middle straight segments.
        straight_segments_data: list of (start_pos, end_pos, heading, length)
        walls: list to append (pos, rot, size) tuples for cleanup/tracking
        """
        # skip entrance & exit straights
        segments = straight_segments_data[1:-1]
        if not segments:
            print("Not enough segments for obstacles.")
            return

        used = set()
        placed_positions = []

        # half-width of tunnel from centerline
        tunnel_half = ROBOT_RADIUS * MIN_CLEARANCE_FACTOR
        # obstacle span across tunnel, leaving MIN_ROBOT_CLEARANCE on the other side
        obstacle_length = 2 * tunnel_half - MIN_ROBOT_CLEARANCE - WALL_THICKNESS

        for _ in range(NUM_OBSTACLES):
            # pick an unused segment index
            choices = [i for i in range(len(segments)) if i not in used]
            if not choices:
                break
            idx = pyrandom.choice(choices)
            used.add(idx)

            start, end, heading, seg_len = segments[idx]
            # choose a point 20–80% along the straight
            d = pyrandom.uniform(0.2 * seg_len, 0.8 * seg_len)
            dir_vec = np.array([math.cos(heading), math.sin(heading), 0.0])
            pos = np.array(start) + dir_vec * d

            # pick inner/outer side
            side = pyrandom.choice([-1, +1])
            perp = np.array([-dir_vec[1], dir_vec[0], 0.0])
            shift = side * (MIN_ROBOT_CLEARANCE / 2 + obstacle_length / 2)
            pos += perp * shift
            pos[2] = WALL_HEIGHT / 2.0

            # no extra rotation—box X-axis is already perpendicular to the corridor
            rot = (0.0, 0.0, 1.0, heading)
            # size: X = thickness along path, Y = span across path
            size = (WALL_THICKNESS, obstacle_length, WALL_HEIGHT)

            # avoid clustering
            if any(np.linalg.norm(pos[:2] - p) < MIN_OBSTACLE_DISTANCE
                   for p in placed_positions):
                print("Skipping obstacle—too close to another.")
                continue

            # create and record
            wall_type = 'left' if side == -1 else 'right'
            self.create_wall(pos, rot, size, wall_type='obstacle')

            walls.append((pos, rot, size))
            placed_positions.append(pos[:2].copy())

    def _translation(self, x, y, z):
        # Helper function to create a translation matrix
        M = np.eye(4)
        M[:3, 3] = [x, y, z]
        return M

    def _rotation_z(self, angle):
        # Helper function to create a rotation matrix around the z-axis
        c, s = math.cos(angle), math.sin(angle)
        R = np.eye(4)
        R[0, 0], R[0, 1] = c, -s
        R[1, 0], R[1, 1] = s, c
        return R

    def _within_bounds(self, T, length):
        # Check if the end point of a straight segment is within map bounds
        end = T[:3, 3] + T[:3, 0] * length
        return MAP_X_MIN <= end[0] <= MAP_X_MAX and MAP_Y_MIN <= end[1] <= MAP_Y_MAX

    def _within_bounds_after_curve(self, T, angle, seg_len):
        # Check if the end point after a curved segment is within map bounds
        tempT = T.copy()
        step = angle / CURVE_SUBDIVISIONS
        centerline_step_length = seg_len / CURVE_SUBDIVISIONS
        for _ in range(CURVE_SUBDIVISIONS):
            tempT = tempT @ self._rotation_z(step)
            tempT[:3, 3] += tempT[:3, 0] * centerline_step_length
        end = tempT[:3, 3]
        return MAP_X_MIN <= end[0] <= MAP_X_MAX and MAP_Y_MIN <= end[1] <= MAP_Y_MAX

    def is_robot_inside_tunnel(self, robot_position, heading):
        robot_xy = np.array(robot_position[:2])
        right_dir = np.array([math.cos(heading - math.pi / 2), math.sin(heading - math.pi / 2)])
        left_dir = np.array([math.cos(heading + math.pi / 2), math.sin(heading + math.pi / 2)])

        def check_walls(walls, direction):
            for pos, _, _ in walls:
                wall_xy = np.array(pos[:2])
                to_wall = wall_xy - robot_xy
                dist = np.linalg.norm(to_wall)
                dot = np.dot(to_wall / (dist + 1e-6), direction)
                if dot > 0.7:
                    expected = self.base_wall_distance
                    if abs(dist - expected) < 0.05:
                        return True
            return False

        def check_obstacles(obstacles):
            for pos, _, size in obstacles:
                wall_xy = np.array(pos[:2])
                dist = np.linalg.norm(wall_xy - robot_xy)
                # Tolerância baseada na largura do obstáculo
                if dist < size[1] / 2 + ROBOT_RADIUS + 0.02:
                    return True
            return False

        near_right = check_walls(self.right_walls, right_dir)
        near_left = check_walls(self.left_walls, left_dir)
        near_obstacle = check_obstacles(self.obstacles)

        inside = (near_right and near_left) or near_obstacle
        # print(f"[DEBUG] Inside tunnel: {inside} | Right: {near_right}, Left: {near_left}, Obstacle: {near_obstacle}")
        return inside































