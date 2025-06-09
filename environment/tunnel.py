from environment.configuration import WALL_THICKNESS, WALL_HEIGHT, OVERLAP_FACTOR, ROBOT_RADIUS, MAX_NUM_CURVES, \
    MIN_CLEARANCE_FACTOR_RANGE, MAX_CLEARANCE_FACTOR_RANGE, MIN_CURVE_ANGLE_RANGE, MAX_CURVE_ANGLE_RANGE, \
    BASE_WALL_LENGTH, CURVE_SUBDIVISIONS, MIN_ROBOT_CLEARANCE, MAX_NUM_OBSTACLES, MIN_OBSTACLE_DISTANCE, MAP_X_MIN, \
    MAP_X_MAX, MAP_Y_MIN, MAP_Y_MAX, MIN_STRAIGHT_LENGTH, MAX_STRAIGHT_LENGTH, MAX_WALL_PIECES_PER_STRAIGHT
import numpy as np
import math
import random as pyrandom
import time  # Import time for potential timestamping or timing

# Define a small delay after clearing walls before building new ones (in seconds)
CLEAR_BUILD_DELAY = 0.1  # 100 milliseconds delay

# New constant for obstacle extension length
MAX_OBSTACLE_EXTENSION_FACTOR = 0.7  # Obstacles extend up to 70% of the way to the centerline from the wall


class TunnelBuilder:
    def __init__(self, supervisor):
        self.supervisor = supervisor  # Store the supervisor reference
        self.root_children = supervisor.getRoot().getField("children")
        self.base_wall_distance = 0
        # Stores (pos, rot, size, wall_type, node_ref) tuples.
        # All walls will have a node_ref and a boundingObject.
        # Obstacle walls will NOT have a Physics node, only a boundingObject.
        self.walls = []  # List to store (pos, rot, size, wall_type, node_ref)
        self.segments = []
        self.right_walls = []
        self.left_walls = []
        self.obstacles = []
        self.wall_count = 0  # Initialize wall counter

    def create_wall(self, pos, rot, size, wall_type=None):
        """
        Creates a Solid wall node in Webots with a boundingObject for collision.
        Static walls (boundary, entrance, left, right) and obstacle walls
        will NOT have a Physics node, only a boundingObject.
        Returns the created node reference if successful, None otherwise.
        """
        type_str = str(wall_type).upper() if wall_type is not None else "NONE"
        # Increment wall_count before using it to ensure unique names
        wall_def_name = f"TUNNEL_WALL_{type_str}_{self.wall_count}"
        self.wall_count += 1

        # Start building the Solid node string
        # Use a slightly different color for obstacles if desired, or make them dynamic
        diffuse_color = '0 0 1' if wall_type in ['boundary', 'entrance'] else \
            ('1 0 0' if wall_type in ['left', 'right'] else '0.5 0.5 0.5')  # Gray for obstacles

        wall_string = f"""DEF {wall_def_name} Solid {{
        translation {pos[0]} {pos[1]} {pos[2]}
        rotation {rot[0]} {rot[1]} {rot[2]} {rot[3]}
        children [
            Shape {{
                appearance Appearance {{
                    material Material {{
                        diffuseColor {diffuse_color}
                    }}
                }}
                geometry Box {{
                    size {size[0]} {size[1]} {size[2]}
                }}
            }}
        ]
        name "{wall_def_name}"
        model "static"
        boundingObject Box {{
            size {size[0]} {size[1]} {size[2]}
        }}
        contactMaterial "wall"
    }}"""  # Corrected closing brace for Solid node

        try:
            # Import the wall into the scene
            self.root_children.importMFNodeFromString(-1, wall_string)
            self.supervisor.step(1)  # Let Webots process the node creation

            # IMPORTANT: Retrieve the node after import
            node = self.supervisor.getFromDef(wall_def_name)

            if node:
                wall_data = (pos, rot, size, wall_type, node)
                self.walls.append(wall_data)  # Store the node reference
                if wall_type == 'left':
                    self.left_walls.append(wall_data)
                elif wall_type == 'right':
                    self.right_walls.append(wall_data)
                elif wall_type == 'obstacle':
                    self.obstacles.append(wall_data)
                return node
            else:
                print(f"[ERROR] Failed to retrieve node reference for {wall_def_name} after import.")
                return None
        except Exception as e:
            print(f"[CRITICAL ERROR] Exception during wall creation for {wall_def_name}: {e}")
            return None

    def _clear_walls(self):
        """
        Removes all walls created by this builder instance from the simulation
        and clears internal lists.
        """
        print("Clearing ALL previous walls...")
        removed_count = 0
        nodes_to_remove = [wall_data[4] for wall_data in self.walls if wall_data[4] is not None]
        # print(f"DEBUG: Found {len(nodes_to_remove)} nodes to attempt removal.") # DEBUG print removed

        for node_to_remove in reversed(nodes_to_remove):  # Iterate over a reversed copy
            node_name = "N/A"  # Initialize with default values
            node_id = "N/A"  # Initialize with default values
            if node_to_remove:  # Ensure the node reference is still valid
                try:
                    # Prefer getName() as it's often more consistently available for logging purposes.
                    # Fallback to getDefName() and then to a generic name.
                    try:
                        node_name = node_to_remove.getName()
                    except AttributeError:
                        try:
                            node_name = node_to_remove.getDefName()
                        except AttributeError:
                            node_name = "Unnamed Node"  # Fallback if neither exists

                    try:
                        node_id = node_to_remove.getId()
                    except AttributeError:
                        node_id = "Unknown ID"  # Fallback if ID is not available

                    # print(f"DEBUG: Attempting to remove node Name:{node_name} ID:{node_id}") # DEBUG print removed

                    # --- CRITICAL CHANGE: Use node.remove() directly ---
                    # This is the most reliable way to remove a node by its reference.
                    node_to_remove.remove()
                    # --------------------------------------------------

                    removed_count += 1
                    # print(f"DEBUG: Successfully sent remove command for Name:{node_name}") # DEBUG print removed
                except Exception as e:
                    print(f"[ERROR] Failed to remove node Name:{node_name} ID:{node_id}: {e}")
            else:
                print("[WARNING] Encountered a None node reference in walls list during cleanup. Skipping.")

        # Clear all internal lists for a fresh start AFTER attempting all removals
        self.walls.clear()
        self.segments.clear()
        self.right_walls.clear()
        self.left_walls.clear()
        self.obstacles.clear()
        self.wall_count = 0  # Reset wall counter

        # Step the simulation to ensure Webots processes the removals
        self.supervisor.step(1)
        print(
            f"Removed {removed_count} walls from the scene and cleared internal lists. Current root children count: {self.root_children.getCount()}")

    # Modified build_tunnel to accept parameters and return final_heading
    def build_tunnel(self, num_curves, angle_range, clearance_factor, num_obstacles):
        """
        Builds the main tunnel structure (straight segments, curves, obstacles)
        starting at a map boundary.
        Includes checks to prevent crossing map boundaries during generation.

        Args:
            num_curves (int): The number of curved segments to include.
            angle_range (tuple): (min_angle, max_angle) for curves.
            clearance_factor (float): The clearance factor to determine tunnel width.
            num_obstacles (int): The number of obstacles to place.

        Returns:
            tuple: (start_pos, end_pos, total_walls_count, final_heading) if successful, otherwise None, None, 0, None.
        """
        # Clear previous walls and segments BEFORE building a new one
        self._clear_walls()

        # Add a small delay after clearing walls before building new ones
        if CLEAR_BUILD_DELAY > 0:
            time.sleep(CLEAR_BUILD_DELAY)
            self.supervisor.step(1)  # Ensure Webots processes the sleep

        self.base_wall_distance = ROBOT_RADIUS * clearance_factor
        print(f"Attempting to build tunnel with clearance factor: {clearance_factor:.2f}")

        angle_min, angle_max = angle_range
        num_curves = min(num_curves, MAX_NUM_CURVES)  # Cap number of curves
        num_obstacles = min(num_obstacles, MAX_NUM_OBSTACLES)  # Cap number of obstacles

        # --- Set Initial Transformation Matrix to start at a boundary ---
        initial_pos = np.array([MAP_X_MIN + 2 * ROBOT_RADIUS, (MAP_Y_MIN + MAP_Y_MAX) / 2.0, 0.0])
        initial_heading = 0.0  # Pointing towards positive X
        T = np.eye(4)
        T[:3, 3] = initial_pos
        T[:3, :3] = self._rotation_z(initial_heading)[:3, :3]  # Set initial rotation

        start_pos = T[:3, 3].copy()
        self.segments = []  # Clear segments for the new tunnel

        # --- Calculate segment_length for straight segments ---
        min_wall_pieces = math.ceil(MIN_STRAIGHT_LENGTH / BASE_WALL_LENGTH)
        max_possible_wall_pieces = math.floor(MAX_STRAIGHT_LENGTH / BASE_WALL_LENGTH)

        # Ensure the range for randint is valid
        if min_wall_pieces > max_possible_wall_pieces:
            print(
                "[ERROR] Invalid straight segment length range: min_wall_pieces > max_possible_wall_pieces. Adjust MIN/MAX_STRAIGHT_LENGTH or BASE_WALL_LENGTH.")
            # Fallback to a default if range is invalid, or return failure
            segment_length = MIN_STRAIGHT_LENGTH  # Use minimum as a fallback
        else:
            num_wall_pieces = pyrandom.randint(min_wall_pieces,
                                               min(MAX_WALL_PIECES_PER_STRAIGHT, max_possible_wall_pieces))
            segment_length = num_wall_pieces * BASE_WALL_LENGTH
        print(f"Calculated straight segment length: {segment_length:.2f} (from {num_wall_pieces} wall pieces).")

        # --- Build the Initial Straight Segment ---
        if not self._within_bounds(T, segment_length):
            print("[ERROR] Initial straight segment end point out of bounds. Cannot build tunnel.")
            return None, None, 0, None

        segment_start_pos = T[:3, 3].copy()
        if not self._add_straight(T, segment_length):
            print("[ERROR] Initial straight segment crosses boundary. Cannot build tunnel.")
            return None, None, 0, None  # Indicate failure

        segment_end_pos = T[:3, 3].copy()
        segment_heading = math.atan2(T[1, 0], T[0, 0])
        self.segments.append((segment_start_pos, segment_end_pos, segment_heading, segment_length))

        # --- Add Curves and Subsequent Straight Segments ---
        for i in range(num_curves):
            angle = pyrandom.uniform(angle_min, angle_max) * pyrandom.choice([1, -1])

            if not self._within_bounds_after_curve(T, angle, segment_length):
                print(f"[WARNING] Curve {i + 1} end point out of bounds, stopping tunnel generation.")
                break

            if not self._add_curve(T, angle, segment_length, self.segments):
                print(f"[ERROR] Curve {i + 1} crosses boundary. Cannot build tunnel.")
                return None, None, 0, None

            # If this is the last curve and no straight segment follows, we are done with path building
            if i == num_curves - 1:
                break

            # For subsequent straight segments, recalculate length
            if min_wall_pieces > max_possible_wall_pieces:
                current_straight_segment_length = MIN_STRAIGHT_LENGTH
            else:
                num_wall_pieces = pyrandom.randint(min_wall_pieces,
                                                   min(MAX_WALL_PIECES_PER_STRAIGHT, max_possible_wall_pieces))
                current_straight_segment_length = num_wall_pieces * BASE_WALL_LENGTH
            print(
                f"Calculated subsequent straight segment length: {current_straight_segment_length:.2f} (from {num_wall_pieces} wall pieces).")

            if not self._within_bounds(T, current_straight_segment_length):
                print(
                    f"[WARNING] Straight segment after curve {i + 1} end point out of bounds, stopping tunnel generation.")
                break

            segment_start_pos = T[:3, 3].copy()
            if not self._add_straight(T, current_straight_segment_length):
                print(f"[ERROR] Straight segment after curve {i + 1} crosses boundary. Cannot build tunnel.")
                return None, None, 0, None

            segment_end_pos = T[:3, 3].copy()
            segment_heading = math.atan2(T[1, 0], T[0, 0])
            self.segments.append((segment_start_pos, segment_end_pos, segment_heading, current_straight_segment_length))

        # Add a final straight segment if the last segment was a curve and num_curves > 0
        # This logic ensures there's a final straight segment after the last curve, if applicable.
        # It prevents immediately ending with a curve, which might make goal detection harder.
        if num_curves > 0 and (len(self.segments) == num_curves * 2 + 1):  # Check if last added was a curve segment
            if min_wall_pieces > max_possible_wall_pieces:
                final_straight_segment_length = MIN_STRAIGHT_LENGTH
            else:
                num_wall_pieces = pyrandom.randint(min_wall_pieces,
                                                   min(MAX_WALL_PIECES_PER_STRAIGHT, max_possible_wall_pieces))
                final_straight_segment_length = num_wall_pieces * BASE_WALL_LENGTH
            print(
                f"Calculated final straight segment length: {final_straight_segment_length:.2f} (from {num_wall_pieces} wall pieces).")

            if not self._within_bounds(T, final_straight_segment_length):
                print(f"[WARNING] Final straight segment end point out of bounds, stopping tunnel generation.")
            else:
                segment_start_pos = T[:3, 3].copy()
                if self._add_straight(T, final_straight_segment_length):
                    segment_end_pos = T[:3, 3].copy()
                    segment_heading = math.atan2(T[1, 0], T[0, 0])
                    self.segments.append(
                        (segment_start_pos, segment_end_pos, segment_heading, final_straight_segment_length))
                else:
                    print(f"[ERROR] Final straight segment crosses boundary. Cannot build tunnel.")
                    return None, None, 0, None
        elif num_curves == 0 and len(self.segments) == 1:
            # If it was just a straight tunnel, segments_data has 1 entry. No need for a final straight.
            pass
        elif num_curves > 0 and len(self.segments) < (num_curves * 2 + 1):
            # If the loop broke early due to out of bounds, and it was a curve,
            # we don't add a final straight if one wasn't explicitly planned.
            pass

        end_pos = T[:3, 3].copy()
        # Get the final heading from the last segment added
        final_heading = self.segments[-1][2] if self.segments else initial_heading

        # --- Add Obstacles ---
        self._add_obstacles(self.segments, num_obstacles)

        # --- Add Main Boundary Walls ---
        self._add_main_boundary_walls()

        print(f"Successfully built tunnel with {len(self.walls)} walls.")
        return start_pos, end_pos, len(self.walls), final_heading

    def _rotation_z(self, angle):
        """Helper to create a 4x4 rotation matrix around the Z axis."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return np.array([
            [cos_a, -sin_a, 0, 0],
            [sin_a, cos_a, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def _within_bounds(self, T, length):
        """Checks if a straight segment of given length starting from T's position
           and heading stays within map boundaries."""
        current_pos = T[:3, 3].copy()
        next_pos = current_pos + T[:3, 0] * length
        # Check if both endpoints are within the map bounds
        is_within = (MAP_X_MIN <= next_pos[0] <= MAP_X_MAX and
                     MAP_Y_MIN <= next_pos[1] <= MAP_Y_MAX)
        # Also check if the path segment itself crosses a boundary
        crosses_boundary = self._check_segment_intersection_with_boundaries(current_pos, next_pos)
        return is_within and not crosses_boundary

    def _within_bounds_after_curve(self, T, angle, segment_length):
        """Checks if the end point after a curve of given angle and segment length
           stays within map boundaries."""
        tempT = T.copy()
        step = angle / CURVE_SUBDIVISIONS
        # The length of the centerline arc for one subdivision
        centerline_sub_segment_length = (self.base_wall_distance) * abs(step)  # Arc length = radius * angle
        # The length of the straight line segment approximating the arc
        straight_sub_segment_length = 2 * (self.base_wall_distance) * math.sin(abs(step) / 2)

        approx_end_pos = tempT[:3, 3].copy()
        for i in range(CURVE_SUBDIVISIONS):
            tempT_rotated = tempT @ self._rotation_z(step)
            approx_end_pos += tempT_rotated[:3,
                              0] * straight_sub_segment_length  # Accumulate translation based on new heading
            tempT[:] = tempT_rotated  # Update rotation for next step

        end = approx_end_pos
        return MAP_X_MIN <= end[0] <= MAP_X_MAX and MAP_Y_MIN <= end[1] <= MAP_Y_MAX

    def _add_entrance_walls(self, tunnel_start_pos, initial_heading):
        """
        Adds walls connecting the start of the tunnel to the map boundaries.
        This method is no longer called if the tunnel starts on a boundary.
        """
        # This method's logic is now effectively unused if the tunnel starts on a boundary.
        # Keeping it here but it won't be called by build_tunnel in the new setup.
        boundary_wall_height = WALL_HEIGHT * 2  # Match height of main boundary walls

        perp_dir_right = np.array(
            [math.cos(initial_heading - math.pi / 2), math.sin(initial_heading - math.pi / 2), 0.0])
        perp_dir_left = np.array(
            [math.cos(initial_heading + math.pi / 2), math.sin(initial_heading + math.pi / 2), 0.0])

        start_left_wall = tunnel_start_pos + perp_dir_left * self.base_wall_distance
        start_right_wall = tunnel_start_pos + perp_dir_right * self.base_wall_distance

        def find_boundary_intersection(start_point, direction):
            """Finds the intersection of a ray from start_point in direction with map boundaries."""
            intersections = []
            if direction[0] != 0:
                t_xmin = (MAP_X_MIN - start_point[0]) / direction[0]
                t_xmax = (MAP_X_MAX - start_point[0]) / direction[0]
                if t_xmin > 1e-6: intersections.append((t_xmin, 'x_min'))
                if t_xmax > 1e-6: intersections.append((t_xmax, 'x_max'))
            if direction[1] != 0:
                t_ymin = (MAP_Y_MIN - start_point[1]) / direction[1]
                t_ymax = (MAP_Y_MAX - start_point[1]) / direction[1]
                if t_ymin > 1e-6: intersections.append((t_ymin, 'y_min'))
                if t_ymax > 1e-6: intersections.append((t_ymax, 'y_max'))

            valid_intersections = [(t, boundary) for t, boundary in intersections if
                                   (start_point + t * direction)[0] >= MAP_X_MIN and (start_point + t * direction)[
                                       0] <= MAP_X_MAX and (start_point + t * direction)[1] >= MAP_Y_MIN and
                                   (start_point + t * direction)[1] <= MAP_Y_MAX]

            if valid_intersections:
                closest_t, boundary_hit = min(valid_intersections, key=lambda item: item[0])
                return start_point + closest_t * direction, closest_t
            return None, 0

        intersection_left, dist_left = find_boundary_intersection(start_left_wall, perp_dir_left)
        if intersection_left is not None and dist_left > WALL_THICKNESS:  # Ensure minimum length
            pos_left = (start_left_wall + intersection_left) / 2
            pos_left[2] = boundary_wall_height / 2
            rot_left = (0, 0, 1, initial_heading + math.pi / 2)
            size_left = (dist_left, WALL_THICKNESS, boundary_wall_height)
            self.create_wall(pos_left, rot_left, size_left, wall_type='entrance')

        intersection_right, dist_right = find_boundary_intersection(start_right_wall, perp_dir_right)
        if intersection_right is not None and dist_right > WALL_THICKNESS:  # Ensure minimum length
            pos_right = (start_right_wall + intersection_right) / 2
            pos_right[2] = boundary_wall_height / 2
            rot_right = (0, 0, 1, initial_heading - math.pi / 2)
            size_right = (dist_right, WALL_THICKNESS, boundary_wall_height)
            self.create_wall(pos_right, rot_right, size_right, wall_type='entrance')

    def _add_main_boundary_walls(self):
        """
        Adds walls around the entire map area.
        These walls are static and do NOT have a Physics node, but DO have a boundingObject.
        """
        boundary_wall_height = WALL_HEIGHT * 2  # Make boundary walls taller

        # Wall along MAP_X_MIN
        pos_xmin = [MAP_X_MIN, (MAP_Y_MIN + MAP_Y_MAX) / 2, boundary_wall_height / 2]
        rot_xmin = [0, 0, 1, math.pi / 2]  # Rotate 90 degrees around Z for Y-axis alignment
        size_xmin = [MAP_Y_MAX - MAP_Y_MIN, WALL_THICKNESS, boundary_wall_height]
        self.create_wall(pos_xmin, rot_xmin, size_xmin, wall_type='boundary')

        # Wall along MAP_X_MAX
        pos_xmax = [MAP_X_MAX, (MAP_Y_MIN + MAP_Y_MAX) / 2, boundary_wall_height / 2]
        rot_xmax = [0, 0, 1, math.pi / 2]
        size_xmax = [MAP_Y_MAX - MAP_Y_MIN, WALL_THICKNESS, boundary_wall_height]  # Corrected size_xmax length
        self.create_wall(pos_xmax, rot_xmax, size_xmax, wall_type='boundary')

        # Wall along MAP_Y_MIN
        pos_ymin = [(MAP_X_MIN + MAP_X_MAX) / 2, MAP_Y_MIN, boundary_wall_height / 2]
        rot_ymin = [0, 0, 1, 0]  # Aligned with X-axis
        size_ymin = [MAP_X_MAX - MAP_X_MIN, WALL_THICKNESS, boundary_wall_height]
        self.create_wall(pos_ymin, rot_ymin, size_ymin, wall_type='boundary')

        # Wall along MAP_Y_MAX
        pos_ymax = [(MAP_X_MIN + MAP_X_MAX) / 2, MAP_Y_MAX, boundary_wall_height / 2]
        rot_ymax = [0, 0, 1, 0]
        size_ymax = [MAP_X_MAX - MAP_X_MIN, WALL_THICKNESS, boundary_wall_height]
        self.create_wall(pos_ymax, rot_ymax, size_ymax, wall_type='boundary')

    def point_to_segment_distance(self, A, B, P):
        """
        Calculates the shortest distance from point P to the line segment AB.
        A, B: start and end points of the segment (numpy arrays)
        P: the point (numpy array)
        """
        AP = P - A
        AB = B - A
        t = np.clip(np.dot(AP, AB) / (np.dot(AB, AB) + 1e-9), 0.0, 1.0)
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
        return False

    def is_robot_inside_tunnel(self, robot_position, heading):
        """
        Checks if the robot is "inside" the tunnel based on proximity to the walls.
        """
        robot_xy = np.array(robot_position[:2])
        right_dir = np.array([math.cos(heading - math.pi / 2), math.sin(heading - math.pi / 2)])
        left_dir = np.array([math.cos(heading + math.pi / 2), math.sin(heading + math.pi / 2)])

        def check_walls(walls, direction):
            for pos, _, size, wall_type, node in walls:
                if wall_type in ['boundary', 'entrance']:
                    continue

                wall_xy = np.array(pos[:2])
                to_wall = wall_xy - robot_xy
                dist = np.linalg.norm(to_wall)
                dot = np.dot(to_wall / (dist + 1e-6), direction)
                if dot > 0.7:
                    expected = self.base_wall_distance
                    if abs(dist - expected) < 0.1:
                        return True
            return False

        def check_obstacles(obstacles):
            for pos, _, size, wall_type, node in obstacles:
                obstacle_xy = np.array(pos[:2])
                dist = np.linalg.norm(obstacle_xy - robot_xy)
                obstacle_effective_radius = max(size[0], size[1]) / 2
                if dist < obstacle_effective_radius + ROBOT_RADIUS + 0.05:
                    return True
            return False

        tunnel_left_walls = [w for w in self.walls if w[3] == 'left']
        tunnel_right_walls = [w for w in self.walls if w[3] == 'right']
        tunnel_obstacles = [w for w in self.walls if w[3] == 'obstacle']

        near_right = check_walls(tunnel_right_walls, right_dir)
        near_left = check_walls(tunnel_left_walls, left_dir)
        near_obstacle = check_obstacles(tunnel_obstacles)

        inside = (near_right and near_left) or near_obstacle
        return inside

    def _add_straight(self, T, length):
        """
        Adds a straight segment of the tunnel after checking for boundary intersection.
        Returns True if successful, False if intersection is detected.
        """
        heading = math.atan2(T[1, 0], T[0, 0])
        current_pos = T[:3, 3].copy()
        next_T = T.copy()
        next_T[:3, 3] += next_T[:3, 0] * length
        next_pos = next_T[:3, 3].copy()

        if self._check_segment_intersection_with_boundaries(current_pos, next_pos):
            return False

        for side in [-1, 1]:
            pos = current_pos + T[:3, 0] * (length / 2) + T[:3, 1] * (side * self.base_wall_distance) + np.array(
                [0, 0, WALL_HEIGHT / 2])
            rot = (0, 0, 1, heading)
            size = (length, WALL_THICKNESS, WALL_HEIGHT)
            self.create_wall(pos, rot, size, wall_type='right' if side == -1 else 'left')
        T[:3, 3] = next_pos
        return True

    def _add_curve(self, T, angle, segment_length, segments_data):
        """
        Adds a curved segment of the tunnel, subdivided into smaller straight segments.
        Returns True if successful, False if intersection is detected.
        """
        initial_pos = T[:3, 3].copy()
        initial_heading = math.atan2(T[1, 0], T[0, 0])

        if abs(angle) < 1e-6:
            return self._add_straight(T, segment_length)

        R_center = segment_length / abs(angle)
        center_offset_dir = np.array([
            math.cos(initial_heading - math.copysign(math.pi / 2, angle)),
            math.sin(initial_heading - math.copysign(math.pi / 2, angle)),
            0.0
        ])
        center_of_rotation = initial_pos + center_offset_dir * R_center

        angle_per_subdivision = angle / CURVE_SUBDIVISIONS
        current_T = T.copy()

        for i in range(CURVE_SUBDIVISIONS):
            sub_segment_start_pos = current_T[:3, 3].copy()
            sub_segment_heading = math.atan2(current_T[1, 0], current_T[0, 0])

            current_T_translated_to_origin = current_T.copy()
            current_T_translated_to_origin[:3, 3] -= center_of_rotation

            rotation_matrix_sub = self._rotation_z(angle_per_subdivision)
            current_T_rotated = rotation_matrix_sub @ current_T_translated_to_origin

            current_T_rotated[:3, 3] += center_of_rotation
            sub_segment_end_pos = current_T_rotated[:3, 3].copy()

            if self._check_segment_intersection_with_boundaries(sub_segment_start_pos, sub_segment_end_pos):
                return False

            sub_segment_length_straight = np.linalg.norm(sub_segment_end_pos - sub_segment_start_pos)

            mid_pos = (sub_segment_start_pos + sub_segment_end_pos) / 2.0
            sub_segment_avg_heading = math.atan2(current_T_rotated[1, 0], current_T_rotated[0, 0])

            perp_dir_right = np.array(
                [math.cos(sub_segment_avg_heading - math.pi / 2), math.sin(sub_segment_avg_heading - math.pi / 2), 0.0])
            perp_dir_left = np.array(
                [math.cos(sub_segment_avg_heading + math.pi / 2), math.sin(sub_segment_avg_heading + math.pi / 2), 0.0])

            pos_left = mid_pos + perp_dir_left * self.base_wall_distance + np.array([0, 0, WALL_HEIGHT / 2])
            rot_left = (0, 0, 1, sub_segment_avg_heading)
            size_left = (sub_segment_length_straight, WALL_THICKNESS, WALL_HEIGHT)
            self.create_wall(pos_left, rot_left, size_left, wall_type='left')

            pos_right = mid_pos + perp_dir_right * self.base_wall_distance + np.array([0, 0, WALL_HEIGHT / 2])
            rot_right = (0, 0, 1, sub_segment_avg_heading)
            size_right = (sub_segment_length_straight, WALL_THICKNESS, WALL_HEIGHT)
            self.create_wall(pos_right, rot_right, size_right, wall_type='right')

            current_T[:] = current_T_rotated

            segments_data.append(
                (sub_segment_start_pos, sub_segment_end_pos, sub_segment_avg_heading, sub_segment_length_straight))

        T[:] = current_T
        return True

    def _check_segment_intersection_with_boundaries(self, p1, p2):
        """
        Checks if the line segment (p1, p2) intersects with any of the map boundaries.
        Returns True if an intersection is found, False otherwise.
        """
        boundaries = [
            ((MAP_X_MIN, MAP_Y_MIN), (MAP_X_MAX, MAP_Y_MIN)),  # Bottom
            ((MAP_X_MIN, MAP_Y_MAX), (MAP_X_MAX, MAP_Y_MAX)),  # Top
            ((MAP_X_MIN, MAP_Y_MIN), (MAP_X_MIN, MAP_Y_MAX)),  # Left
            ((MAP_X_MAX, MAP_Y_MIN), (MAP_X_MAX, MAP_Y_MAX))  # Right
        ]

        p1_2d = np.array([p1[0], p1[1]])
        p2_2d = np.array([p2[0], p2[1]])

        if not (MAP_X_MIN <= p1_2d[0] <= MAP_X_MAX and MAP_Y_MIN <= p1_2d[1] <= MAP_Y_MAX):
            return True
        if not (MAP_X_MIN <= p2_2d[0] <= MAP_X_MAX and MAP_Y_MIN <= p2_2d[1] <= MAP_Y_MAX):
            return True

        for b1, b2 in boundaries:
            b1_2d = np.array(b1)
            b2_2d = np.array(b2)

            r = p2_2d - p1_2d
            s = b2_2d - b1_2d

            cross_product = np.cross(r, s)

            if abs(cross_product) < 1e-9:
                continue

            t = np.cross(b1_2d - p1_2d, s) / cross_product
            u = np.cross(b1_2d - p1_2d, r) / cross_product

            if (0 < t < 1) and (0 < u < 1):
                return True

        return False

    def _add_obstacles(self, segments_data, num_obstacles):
        """
        Adds various types of obstacles along the tunnel segments.
        Obstacles can be pillars or extend from the left/right walls.
        Ensures obstacles are not too close to each other.
        """
        if not segments_data:
            print("[WARNING] No tunnel segments available to place obstacles.")
            return

        obstacles_placed = 0
        max_attempts = num_obstacles * 5

        while obstacles_placed < num_obstacles and max_attempts > 0:
            max_attempts -= 1

            segment_index = pyrandom.randint(0, len(segments_data) - 1)
            segment_start_pos, segment_end_pos, segment_heading, segment_length = segments_data[segment_index]

            min_offset = ROBOT_RADIUS * 2
            if segment_length < 2 * min_offset:
                continue

            t_along_segment = pyrandom.uniform(min_offset / segment_length, 1.0 - min_offset / segment_length)
            obstacle_centerline_pos = segment_start_pos + (segment_end_pos - segment_start_pos) * t_along_segment
            obstacle_centerline_pos[2] = WALL_HEIGHT / 2

            is_too_close_to_existing = False
            for existing_pos, _, existing_size, _, _ in self.obstacles:
                dist = np.linalg.norm(obstacle_centerline_pos[:2] - existing_pos[:2])
                existing_obstacle_effective_radius = max(existing_size[0], existing_size[1]) / 2
                if dist < MIN_OBSTACLE_DISTANCE + ROBOT_RADIUS + existing_obstacle_effective_radius:
                    is_too_close_to_existing = True
                    break
            if is_too_close_to_existing:
                continue

            obstacle_type_choice = pyrandom.choice([0, 1])

            if obstacle_type_choice == 0:
                obstacle_pos = obstacle_centerline_pos
                obstacle_rot = (0, 0, 1, pyrandom.uniform(0, 2 * math.pi))
                obstacle_size = (WALL_THICKNESS, WALL_THICKNESS, WALL_HEIGHT)
                print(f"Placing pillar obstacle at {obstacle_pos[:2]}.")

            else:
                extend_from_side = pyrandom.choice(['left', 'right'])

                if extend_from_side == 'left':
                    wall_dir = np.array(
                        [math.cos(segment_heading + math.pi / 2), math.sin(segment_heading + math.pi / 2), 0.0])
                    obstacle_pos = obstacle_centerline_pos + wall_dir * (
                            self.base_wall_distance * (1 - MAX_OBSTACLE_EXTENSION_FACTOR / 2))
                    obstacle_rot = (0, 0, 1, segment_heading + math.pi / 2)
                else:
                    wall_dir = np.array(
                        [math.cos(segment_heading - math.pi / 2), math.sin(segment_heading - math.pi / 2), 0.0])
                    obstacle_pos = obstacle_centerline_pos + wall_dir * (
                            self.base_wall_distance * (1 - MAX_OBSTACLE_EXTENSION_FACTOR / 2))
                    obstacle_rot = (0, 0, 1, segment_heading - math.pi / 2)

                obstacle_extension_length = self.base_wall_distance * MAX_OBSTACLE_EXTENSION_FACTOR
                obstacle_size = (
                    obstacle_extension_length, WALL_THICKNESS, WALL_HEIGHT)
                print(
                    f"Placing {extend_from_side} wall-extension obstacle at {obstacle_pos[:2]} with length {obstacle_extension_length:.2f}.")

            node = self.create_wall(obstacle_pos, obstacle_rot, obstacle_size, wall_type='obstacle')
            if node:
                obstacles_placed += 1
            else:
                print(f"[WARNING] Failed to place obstacle. Retrying...")

        print(f"Finished placing {obstacles_placed} obstacles.")
