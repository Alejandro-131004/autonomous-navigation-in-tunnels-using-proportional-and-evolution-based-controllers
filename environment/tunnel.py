from environment.configuration import WALL_THICKNESS, WALL_HEIGHT, OVERLAP_FACTOR, ROBOT_RADIUS, MAX_NUM_CURVES, \
    MIN_CLEARANCE_FACTOR_RANGE, MAX_CLEARANCE_FACTOR_RANGE, MIN_CURVE_ANGLE_RANGE, MAX_CURVE_ANGLE_RANGE, \
    BASE_WALL_LENGTH, CURVE_SUBDIVISIONS, MIN_ROBOT_CLEARANCE, MAX_NUM_OBSTACLES, MIN_OBSTACLE_DISTANCE, MAP_X_MIN, \
    MAP_X_MAX, MAP_Y_MIN, MAP_Y_MAX, MIN_STRAIGHT_LENGTH, MAX_STRAIGHT_LENGTH, MAX_WALL_PIECES_PER_STRAIGHT, \
    WALL_JOINT_GAP
import numpy as np
import math
import random as pyrandom
import time  # Import time for potential timestamping or timing

# Define a constant for how often to check physics (in seconds)
CHECK_PHYSICS_INTERVAL = 5.0  # Check physics every 5 seconds
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
        # Static walls will have Physics field NULL.
        # Obstacle walls will have a Physics node.
        self.walls = []
        self.segments = []
        self.right_walls = []
        self.left_walls = []
        self.obstacles = []
        self.wall_count = 0  # Initialize wall counter
        self.physics_check_timer = 0.0  # Timer for periodic physics checks (in seconds)
        self.last_physics_check_time = time.time()  # Timestamp of the last check

    def create_wall(self, pos, rot, size, wall_type=None):
        """
        Creates a Solid wall node in Webots with a boundingObject for collision.
        Static walls (boundary, entrance, left, right) will NOT have a Physics node.
        Obstacle walls WILL have a Physics node with basic properties.
        Returns the created node reference if successful, None otherwise.
        """
        type_str = str(wall_type).upper() if wall_type is not None else "NONE"
        wall_def_name = f"TUNNEL_WALL_{type_str}_{self.wall_count}"

        diffuse_color = '0 0 1' if wall_type in ['boundary', 'entrance'] else \
            ('1 0 0' if wall_type in ['left', 'right'] else '0.5 0.5 0.5')  # Red for tunnel walls, Gray for obstacles

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
    }}"""

        # Add Physics node only for obstacle walls, as per documentation for movable objects
        if wall_type == 'obstacle':
            physics_node_string = """
                physics Physics {
                    density 1000 # Example density for obstacles (or use mass field)
                    dampingFactor 0.9 # Damping for stability
                }
             """
            insert_index = wall_string.rfind('}')
            if insert_index != -1:
                wall_string = wall_string[:insert_index] + physics_node_string + wall_string[insert_index:]
            else:
                print(f"[ERROR] Could not find closing brace in Solid string for {wall_def_name}")

        try:
            self.root_children.importMFNodeFromString(-1, wall_string)
            self.supervisor.step(1)  # Step the simulation to allow Webots to process the node
            node = self.supervisor.getFromDef(wall_def_name)

            if node:
                wall_data = (pos, rot, size, wall_type, node)
                self.walls.append(wall_data)
                if wall_type == 'left':
                    self.left_walls.append(wall_data)
                elif wall_type == 'right':
                    self.right_walls.append(wall_data)
                elif wall_type == 'obstacle':
                    self.obstacles.append(wall_data)
                self.wall_count += 1
                return node
            else:
                print(
                    f"[ERROR] Failed to retrieve node reference for {wall_def_name}. Node might not have been created or DEF name is incorrect.")
                return None
        except Exception as e:
            print(f"[CRITICAL ERROR] Exception during wall creation for {wall_def_name}: {e}")
            return None

    def delete_walls_by_type(self, wall_type):
        """
        Removes all walls of a specific type from the simulation and updates
        internal lists.
        """
        print(f"Clearing walls of type: {wall_type}...")
        walls_to_keep = []
        removed_count = 0
        for pos, rot, size, current_type, node in self.walls[:]:
            if current_type == wall_type:
                if node:  # Check if the node reference is valid
                    try:
                        node.remove()  # Simplified removal with R2023b+
                        removed_count += 1
                    except Exception as e:
                        print(
                            f"[ERROR] Exception during removal of node {node.getDefName()} (type: {current_type}): {e}")
                else:
                    print(
                        f"[WARNING] Invalid node reference found in walls list for type {current_type}. Skipping removal.")
            else:
                walls_to_keep.append((pos, rot, size, current_type, node))

        self.walls = walls_to_keep
        self.right_walls = [wall for wall in self.walls if wall[3] == 'right']
        self.left_walls = [wall for wall in self.walls if wall[3] == 'left']
        self.obstacles = [wall for wall in self.walls if wall[3] == 'obstacle']
        print(f"Attempted to clear walls of type '{wall_type}'. Successfully removed {removed_count}.")

    def _clear_walls(self):
        """
        Removes all walls created by this builder instance from the simulation
        and clears internal lists.
        """
        print("Clearing ALL previous walls...")
        removed_count = 0
        nodes_to_remove = [wall_data[4] for wall_data in self.walls if wall_data[4] is not None]

        for node_to_remove in reversed(nodes_to_remove):
            if node_to_remove:
                try:
                    node_to_remove.remove()  # Simplified removal with R2023b+
                    removed_count += 1
                except Exception as e:
                    print(
                        f"[ERROR] Failed to remove node {node_to_remove.getDefName() if hasattr(node_to_remove, 'getDefName') else 'Unnamed'}: {e}")
            else:
                print("[WARNING] Encountered a None node reference in walls list during cleanup. Skipping.")

        self.walls.clear()
        self.segments.clear()
        self.right_walls.clear()
        self.left_walls.clear()
        self.obstacles.clear()
        self.wall_count = 0
        self.supervisor.step(1)  # Step simulation to ensure deletions are processed
        print(
            f"Removed {removed_count} walls from the scene and cleared internal lists. Current root children count: {self.root_children.getCount()}")

    def build_tunnel(self, num_curves, angle_range, clearance_factor, num_obstacles):
        """
        Builds the main tunnel structure (straight segments, curves, obstacles)
        starting at a map boundary.

        Args:
            num_curves (int): Number of curved segments.
            angle_range (tuple): (min_angle, max_angle) for curves.
            clearance_factor (float): Tunnel width factor.
            num_obstacles (int): Number of obstacles to place.

        Returns:
            tuple: (start_pos, end_pos, total_walls, final_heading)
        """
        self._clear_walls()

        if CLEAR_BUILD_DELAY > 0:
            time.sleep(CLEAR_BUILD_DELAY)
            self.supervisor.step(1)

        self.base_wall_distance = ROBOT_RADIUS * clearance_factor
        print(f"Attempting to build tunnel with clearance factor: {clearance_factor:.2f}")

        angle_min, angle_max = angle_range
        num_curves = min(num_curves, MAX_NUM_CURVES)
        num_obstacles = min(num_obstacles, MAX_NUM_OBSTACLES)

        initial_pos = np.array([MAP_X_MIN + 2 * ROBOT_RADIUS, (MAP_Y_MIN + MAP_Y_MAX) / 2.0, 0.0])
        initial_heading = 0.0
        T = np.eye(4)
        T[:3, 3] = initial_pos
        T[:3, :3] = self._rotation_z(initial_heading)[:3, :3]

        start_pos = T[:3, 3].copy()
        segments_data = []

        min_wall_pieces = math.ceil(MIN_STRAIGHT_LENGTH / BASE_WALL_LENGTH)
        max_possible_wall_pieces = math.floor(MAX_STRAIGHT_LENGTH / BASE_WALL_LENGTH)

        if min_wall_pieces > max_possible_wall_pieces:
            print(
                "[ERROR] Invalid straight segment length range: min_wall_pieces > max_possible_wall_pieces. Adjust MIN/MAX_STRAIGHT_LENGTH or BASE_WALL_LENGTH.")
            segment_length = MIN_STRAIGHT_LENGTH
        else:
            num_wall_pieces = pyrandom.randint(min_wall_pieces,
                                               min(MAX_WALL_PIECES_PER_STRAIGHT, max_possible_wall_pieces))
            segment_length = num_wall_pieces * BASE_WALL_LENGTH
        print(f"Calculated initial straight segment length: {segment_length:.2f} (from {num_wall_pieces} wall pieces).")

        if not self._within_bounds(T, segment_length):
            print("[ERROR] Initial straight segment end point out of bounds. Cannot build tunnel.")
            return None, None, 0, None

        segment_start_pos = T[:3, 3].copy()
        if not self._add_straight(T, segment_length):
            print("[ERROR] Initial straight segment crosses boundary. Retrying tunnel generation.")
            return None, None, 0, None

        segment_end_pos = T[:3, 3].copy()
        segment_heading = math.atan2(T[1, 0], T[0, 0])
        segments_data.append((segment_start_pos, segment_end_pos, segment_heading, segment_length))

        for i in range(num_curves):
            angle = pyrandom.uniform(angle_min, angle_max) * pyrandom.choice([1, -1])

            # Pass the segment_length as curve_arc_length to _within_bounds_after_curve
            if not self._within_bounds_after_curve(T, angle, segment_length):
                print(f"[WARNING] Curve {i + 1} end point out of bounds, stopping tunnel generation.")
                break

            # Pass the segment_length as curve_arc_length to _add_curve
            if not self._add_curve(T, angle, segment_length, segments_data):
                print(f"[ERROR] Curve {i + 1} crosses boundary. Retrying tunnel generation.")
                return None, None, 0, None

            if i == num_curves - 1:
                break

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
                print(f"[ERROR] Straight segment after curve {i + 1} crosses boundary. Retrying tunnel generation.")
                return None, None, 0, None

            segment_end_pos = T[:3, 3].copy()
            segment_heading = math.atan2(T[1, 0], T[0, 0])
            segments_data.append((segment_start_pos, segment_end_pos, segment_heading, current_straight_segment_length))

        if num_curves > 0 and len(segments_data) > (1 + num_curves):
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
                    segments_data.append(
                        (segment_start_pos, segment_end_pos, segment_heading, final_straight_segment_length))
                else:
                    print(f"[ERROR] Final straight segment crosses boundary. Retrying tunnel generation.")
                    return None, None, 0, None
        elif num_curves == 0 and len(segments_data) == 1:
            pass
        elif num_curves > 0 and len(segments_data) == (1 + num_curves):
            pass
        else:
            pass

        end_pos = T[:3, 3].copy()
        final_heading = segments_data[-1][2] if segments_data else initial_heading

        self._add_obstacles(segments_data, num_obstacles)
        self.segments = segments_data
        self._add_main_boundary_walls()

        print(f"Successfully built tunnel with {len(self.walls)} walls.")
        self.physics_check_timer = 0.0
        self.last_physics_check_time = time.time()
        self.check_and_restore_wall_physics()

        return start_pos, end_pos, len(self.walls), final_heading

    def _within_bounds(self, T, length):
        """Checks if a straight segment of given length starting from T's position
           and heading stays within map boundaries."""
        current_pos = T[:3, 3].copy()
        next_pos = current_pos + T[:3, 0] * length
        is_within = (MAP_X_MIN <= next_pos[0] <= MAP_X_MAX and
                     MAP_Y_MIN <= next_pos[1] <= MAP_Y_MAX)
        crosses_boundary = self._check_segment_intersection_with_boundaries(current_pos, next_pos)
        return is_within and not crosses_boundary

    def _within_bounds_after_curve(self, T, angle, curve_arc_length):
        """Checks if the end point after a curve of given angle and TOTAL arc length
           stays within map boundaries."""
        tempT = T.copy()
        step = angle / CURVE_SUBDIVISIONS

        # Calculate the centerline radius from the total arc length and total angle
        # If angle is very small (near straight), avoid division by zero or large radius
        R_centerline = curve_arc_length / abs(angle) if abs(
            angle) > 1e-9 else curve_arc_length / CURVE_SUBDIVISIONS  # Fallback for straight-like curves

        # The length of the straight line segment approximating the arc for ONE subdivision
        straight_sub_segment_length = 2 * R_centerline * math.sin(abs(step) / 2) if R_centerline > 1e-9 else (
                    curve_arc_length / CURVE_SUBDIVISIONS)

        for i in range(CURVE_SUBDIVISIONS):
            # Move along the current heading by the straight sub-segment length
            tempT[:3, 3] += tempT[:3, 0] * straight_sub_segment_length
            # Then rotate for the next sub-segment
            tempT = tempT @ self._rotation_z(step)

            current_segment_end_pos = tempT[:3, 3].copy()
            if not (MAP_X_MIN <= current_segment_end_pos[0] <= MAP_X_MAX and
                    MAP_Y_MIN <= current_segment_end_pos[1] <= MAP_Y_MAX):
                return False

        end = tempT[:3, 3].copy()
        return MAP_X_MIN <= end[0] <= MAP_X_MAX and MAP_Y_MIN <= end[1] <= MAP_Y_MAX

    def _add_entrance_walls(self, tunnel_start_pos, initial_heading):
        boundary_wall_height = WALL_HEIGHT * 2

        perp_dir_right = np.array(
            [math.cos(initial_heading - math.pi / 2), math.sin(initial_heading - math.pi / 2), 0.0])
        perp_dir_left = np.array(
            [math.cos(initial_heading + math.pi / 2), math.sin(initial_heading + math.pi / 2), 0.0])

        start_left_wall = tunnel_start_pos + perp_dir_left * self.base_wall_distance
        start_right_wall = tunnel_start_pos + perp_dir_right * self.base_wall_distance

        def find_boundary_intersection(start_point, direction):
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
        if intersection_left is not None and dist_left > WALL_THICKNESS:
            pos_left = (start_left_wall + intersection_left) / 2
            pos_left[2] = boundary_wall_height / 2
            rot_left = (0, 0, 1, initial_heading + math.pi / 2)
            size_left = (dist_left, WALL_THICKNESS, boundary_wall_height)
            self.create_wall(pos_left, rot_left, size_left, wall_type='entrance')

        intersection_right, dist_right = find_boundary_intersection(start_right_wall, perp_dir_right)
        if intersection_right is not None and dist_right > WALL_THICKNESS:
            pos_right = (start_right_wall + intersection_right) / 2
            pos_right[2] = boundary_wall_height / 2
            rot_right = (0, 0, 1, initial_heading - math.pi / 2)
            size_right = (dist_right, WALL_THICKNESS, boundary_wall_height)
            self.create_wall(pos_right, rot_right, size_right, wall_type='entrance')

    def _add_main_boundary_walls(self):
        boundary_wall_height = WALL_HEIGHT * 2

        pos_xmin = [MAP_X_MIN, (MAP_Y_MIN + MAP_Y_MAX) / 2, boundary_wall_height / 2]
        rot_xmin = [0, 0, 1, math.pi / 2]
        size_xmin = [MAP_Y_MAX - MAP_Y_MIN, WALL_THICKNESS, boundary_wall_height]
        self.create_wall(pos_xmin, rot_xmin, size_xmin, wall_type='boundary')

        pos_xmax = [MAP_X_MAX, (MAP_Y_MIN + MAP_Y_MAX) / 2, boundary_wall_height / 2]
        rot_xmax = [0, 0, 1, math.pi / 2]
        size_xmax = [MAP_Y_MAX - MAP_Y_MIN, WALL_THICKNESS, boundary_wall_height]
        self.create_wall(pos_xmax, rot_xmax, size_xmax, wall_type='boundary')

        pos_ymin = [(MAP_X_MIN + MAP_X_MAX) / 2, MAP_Y_MIN, boundary_wall_height / 2]
        rot_ymin = [0, 0, 1, 0]
        size_ymin = [MAP_X_MAX - MAP_X_MIN, WALL_THICKNESS, boundary_wall_height]
        self.create_wall(pos_ymin, rot_ymin, size_ymin, wall_type='boundary')

        pos_ymax = [(MAP_X_MIN + MAP_X_MAX) / 2, MAP_Y_MAX, boundary_wall_height / 2]
        rot_ymax = [0, 0, 1, 0]
        size_ymax = [MAP_X_MAX - MAP_X_MIN, WALL_THICKNESS, boundary_wall_height]
        self.create_wall(pos_ymax, rot_ymax, size_ymax, wall_type='boundary')

    def point_to_segment_distance(self, A, B, P):
        AP = P - A
        AB = B - A
        t = np.clip(np.dot(AP, AB) / (np.dot(AB, AB) + 1e-9), 0.0, 1.0)
        closest_point = A + t * AB
        return np.linalg.norm(P - closest_point)

    def is_robot_near_centerline(self, robot_position):
        robot_xy = np.array(robot_position[:2])
        threshold = self.base_wall_distance + ROBOT_RADIUS

        for start, end, _, _ in self.segments:
            dist = self.point_to_segment_distance(start[:2], end[:2], robot_xy)
            if dist <= threshold:
                return True
        return False

    def is_robot_inside_tunnel(self, robot_position, heading):
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

    def check_and_restore_wall_physics(self):
        """
        Periodically checks if wall nodes still exist and have the correct physics configuration
        (boundingObject for all, Physics node for obstacles).
        If a node is missing or misconfigured, attempts to remove and re-create it.
        This function should be called regularly (e.g., at the end of build_tunnel).
        """
        current_time = time.time()
        if current_time - self.last_physics_check_time < CHECK_PHYSICS_INTERVAL:
            return

        print("[INFO] Performing periodic physics check...")
        walls_to_recreate = []

        # Create a temporary list of existing walls to iterate over
        current_walls_in_memory = self.walls[:]

        # Clear main wall lists to rebuild them based on current scene state
        self.walls = []
        self.left_walls = []
        self.right_walls = []
        self.obstacles = []

        # Re-populate self.walls by checking actual nodes in Webots and their properties
        # This is more robust as it accounts for external changes or Webots quirks
        found_nodes_count = 0
        for wall_data in current_walls_in_memory:
            pos, rot, size, wall_type, node = wall_data
            if node:
                # Get the node again by its DEF name to ensure it's still in the scene
                actual_node_in_scene = self.supervisor.getFromDef(node.getDefName())

                if actual_node_in_scene:
                    # Node still exists, check its physics configuration
                    physics_field = actual_node_in_scene.getField("physics")
                    has_physics_node = (physics_field and physics_field.getSFNode() is not None)

                    needs_recreation = False
                    if wall_type == 'obstacle':
                        if not has_physics_node:
                            print(
                                f"[WARNING] Obstacle '{actual_node_in_scene.getDefName()}' missing Physics node. Marking for recreation.")
                            needs_recreation = True
                    else:  # Static wall (boundary, entrance, left, right)
                        if has_physics_node:
                            print(
                                f"[WARNING] Static wall '{actual_node_in_scene.getDefName()}' has unexpected Physics node. Marking for recreation.")
                            needs_recreation = True

                    if needs_recreation:
                        # Add to list for recreation and attempt to remove the problematic node immediately
                        walls_to_recreate.append(wall_data)
                        try:
                            actual_node_in_scene.remove()
                            self.supervisor.step(1)  # Allow Webots to process removal
                        except Exception as e:
                            print(f"[ERROR] Failed to remove problematic node {actual_node_in_scene.getDefName()}: {e}")
                    else:
                        # If good, add back to the main lists
                        self.walls.append(wall_data)
                        if wall_type == 'left':
                            self.left_walls.append(wall_data)
                        elif wall_type == 'right':
                            self.right_walls.append(wall_data)
                        elif wall_type == 'obstacle':
                            self.obstacles.append(wall_data)
                        found_nodes_count += 1
                else:
                    print(
                        f"[WARNING] Wall node '{node.getDefName()}' not found in scene. It may have been removed unexpectedly. Will not recreate if it's not an obstacle.")
            else:
                print("[WARNING] Invalid node reference found in wall data. Skipping check for this entry.")

        # Recreate walls that were marked for recreation (if they were originally obstacles or static walls that broke)
        recreated_count = 0
        for wall_data in walls_to_recreate:
            pos, rot, size, wall_type, _ = wall_data  # Discard old node ref
            # When recreating, call create_wall with the original parameters
            new_node = self.create_wall(pos, rot, size, wall_type)  # This re-adds to self.walls/etc.
            if new_node:
                recreated_count += 1
            else:
                print(f"[CRITICAL ERROR] Failed to recreate wall of type {wall_type} at {pos[:2]}.")

        print(
            f"[INFO] Physics check completed. Found {len(walls_to_recreate)} issues, recreated {recreated_count} walls.")
        self.last_physics_check_time = current_time  # Reset the timer

    '''def _add_straight(self, T, length):
        """
        Adds a straight segment of the tunnel after checking for boundary intersection.
        Returns True if successful, False if intersection is detected.
        """
        heading = math.atan2(T[1, 0], T[0, 0])
        current_pos = T[:3, 3].copy()
        next_T_pos = current_pos + T[:3, 0] * length

        if self._check_segment_intersection_with_boundaries(current_pos, next_T_pos):
            return False

        # Apply WALL_JOINT_GAP to the wall length
        wall_length = length - WALL_JOINT_GAP
        # Ensure wall_length is not negative or too small
        if wall_length <= 0:
            print(
                f"[WARNING] Calculated straight wall length is too small ({wall_length:.4f}). Skipping wall creation.")
            # Still update T to allow tunnel generation to continue, but don't add a wall
            T[:3, 3] = next_T_pos
            return True  # Consider this a "success" for path progression, just no visible wall

        for side in [-1, 1]:
            wall_center_offset_along_segment = (length / 2.0)  # Use original segment length for center calculation
            wall_perp_offset = side * self.base_wall_distance

            wall_pos = current_pos + T[:3, 0] * wall_center_offset_along_segment + \
                       T[:3, 1] * wall_perp_offset + np.array([0, 0, WALL_HEIGHT / 2])

            wall_rot = (0, 0, 1, heading)

            wall_size = (wall_length, WALL_THICKNESS, WALL_HEIGHT)

            self.create_wall(wall_pos, wall_rot, wall_size, wall_type='right' if side == -1 else 'left')

        T[:3, 3] = next_T_pos
        return True'''

    def _add_straight(self, T, length):
        """
        Adds a straight segment of the tunnel after checking boundary intersection.
        Returns False if it would cross a boundary.
        """
        # 1) Current heading (radians) and start/end points
        heading = math.atan2(T[1, 0], T[0, 0])
        start_pos = T[:3, 3].copy()
        end_pos = start_pos + T[:3, 0] * length

        # 2) Boundary check
        if self._check_segment_intersection_with_boundaries(start_pos, end_pos):
            return False

        # 3) Build walls with overlap
        # compute wall length (with joint gap overlap)
        wall_len = length + WALL_JOINT_GAP * OVERLAP_FACTOR
        # midpoint along segment
        mid_pos = start_pos + (end_pos - start_pos) * 0.5

        # compute unit forward and perpendicular vectors
        forward = np.array([math.cos(heading), math.sin(heading), 0.0])
        perp = np.array([-forward[1], forward[0], 0.0])  # rotate forward by +90°

        # place left and right
        for side in (+1, -1):
            offset = perp * (side * self.base_wall_distance)
            pos = mid_pos + offset
            pos[2] = WALL_HEIGHT / 2
            rot = (0, 0, 1, heading)
            size = (wall_len, WALL_THICKNESS, WALL_HEIGHT)
            wtype = 'left' if side > 0 else 'right'
            self.create_wall(pos, rot, size, wall_type=wtype)

        # 4) Advance the transform T for the next segment
        T[:3, 3] = end_pos
        return True

    '''def _add_curve(self, T, angle, arc_length, segments_data):
        if abs(angle) < 1e-6:
            return True

        step = angle / CURVE_SUBDIVISIONS
        R_centerline = arc_length / abs(angle)
        straight_len = 2 * R_centerline * math.sin(abs(step) / 2)

        segment_start_pos = T[:3, 3].copy()

        for _ in range(CURVE_SUBDIVISIONS):
            heading = math.atan2(T[1, 0], T[0, 0])
            current_pos = T[:3, 3].copy()
            next_pos = current_pos + T[:3, 0] * straight_len

            wall_length = straight_len + WALL_JOINT_GAP * OVERLAP_FACTOR
            wall_center_offset = straight_len / 2.0

            for side in [-1, 1]:
                offset = side * self.base_wall_distance
                wall_pos = current_pos + T[:3, 0] * wall_center_offset + T[:3, 1] * offset + np.array(
                    [0, 0, WALL_HEIGHT / 2])
                wall_rot = (0, 0, 1, heading)
                wall_size = (wall_length, WALL_THICKNESS, WALL_HEIGHT)
                self.create_wall(wall_pos, wall_rot, wall_size, wall_type='right' if side == -1 else 'left')

            T[:3, 3] = next_pos
            T = T @ self._rotation_z(step)

        segment_end_pos = T[:3, 3].copy()
        segment_heading = math.atan2(T[1, 0], T[0, 0])
        segments_data.append((segment_start_pos, segment_end_pos, segment_heading, arc_length))

        return True'''

    def _add_curve(self, T, angle, arc_length, segments_data):
        """
        Adds a curved segment by subdividing into small straight pieces.
        Returns False if any piece crosses a boundary.
        """
        # 1) Early out for near-zero angle
        if abs(angle) < 1e-6:
            return True

        # 2) Compute subdivision step and centerline radius
        step = angle / CURVE_SUBDIVISIONS
        R = arc_length / abs(angle)

        # 3) Loop subdivisions
        for _ in range(CURVE_SUBDIVISIONS):
            # --- a) Current heading & position
            heading   = math.atan2(T[1,0], T[0,0])
            start_pos = T[:3,3].copy()

            # --- b) Move forward along tangent by the chord length
            chord_len = 2 * R * math.sin(abs(step)/2)
            # forward direction vector
            forward = np.array([math.cos(heading), math.sin(heading), 0.0])
            end_pos = start_pos + forward * chord_len

            # --- c) Boundary check for this sub-segment
            if self._check_segment_intersection_with_boundaries(start_pos, end_pos):
                return False

            # --- d) Place walls halfway along the chord
            mid_pos = start_pos + forward * (chord_len/2.0)
            perp    = np.array([-forward[1], forward[0], 0.0])

            wall_len = chord_len + WALL_JOINT_GAP * OVERLAP_FACTOR

            for side in (+1, -1):
                offset = perp * (side * self.base_wall_distance)
                pos    = mid_pos + offset
                pos[2] = WALL_HEIGHT/2.0
                rot    = (0, 0, 1, heading)
                size   = (wall_len, WALL_THICKNESS, WALL_HEIGHT)
                wtype  = 'left' if side>0 else 'right'
                self.create_wall(pos, rot, size, wall_type=wtype)

            # --- e) Advance T: first translate, then rotate about Z
            T[:3,3] = end_pos
            T[:]     = T @ self._rotation_z(step)

            # --- f) Record this sub-segment in your segments_data
            segments_data.append((start_pos, end_pos, heading, chord_len))

        return True

    def _check_segment_intersection_with_boundaries(self, p1, p2):
        boundaries = [
            ((MAP_X_MIN, MAP_Y_MIN), (MAP_X_MAX, MAP_Y_MIN)),
            ((MAP_X_MIN, MAP_Y_MAX), (MAP_X_MAX, MAP_Y_MAX)),
            ((MAP_X_MIN, MAP_Y_MIN), (MAP_X_MIN, MAP_Y_MAX)),
            ((MAP_X_MAX, MAP_Y_MIN), (MAP_X_MAX, MAP_Y_MAX))
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
        Adds obstacles within the generated tunnel segments.
        Obstacles are placed randomly along the centerline of the tunnel segments.
        Obstacles are dynamic and WILL have a Physics node and a boundingObject.
        """
        if num_obstacles <= 0 or not segments_data:
            return

        print(f"Attempting to add {num_obstacles} obstacles.")
        added_obstacles_count = 0
        attempts = 0
        max_attempts = num_obstacles * 20  # Increased attempts

        internal_straight_segments = []
        for i, seg in enumerate(segments_data):
            start, end, heading, seg_len = seg
            # Check if it's a straight segment (heading close to 0 or pi)
            is_straight = abs(heading) < 1e-3 or abs(heading - math.pi) < 1e-3 or abs(heading + math.pi) < 1e-3
            is_not_first = i > 0
            is_not_last = i < len(segments_data) - 1

            if is_straight and is_not_first and is_not_last and seg_len > MIN_OBSTACLE_DISTANCE * 2:  # Ensure segment is long enough
                internal_straight_segments.append((i, seg))

        if not internal_straight_segments:
            print("Not enough suitable internal straight segments for obstacles.")
            return

        used_segment_indices = set()
        placed_positions = []

        tunnel_half_width = self.base_wall_distance
        obstacle_length = 2 * tunnel_half_width - MIN_ROBOT_CLEARANCE - WALL_THICKNESS
        if obstacle_length <= 0.1:
            print(f"Obstacle length is too small ({obstacle_length:.2f}). Cannot place obstacles.")
            return

        while added_obstacles_count < num_obstacles and attempts < max_attempts:
            attempts += 1
            available_choices = [idx for idx, seg in internal_straight_segments if idx not in used_segment_indices]
            if not available_choices:
                break

            chosen_segment_idx_in_segments_data = pyrandom.choice(available_choices)
            chosen_segment = segments_data[chosen_segment_idx_in_segments_data]
            start, end, heading, seg_len = chosen_segment

            min_dist_along_segment = MIN_OBSTACLE_DISTANCE / 2.0 + WALL_THICKNESS / 2.0
            max_dist_along_segment = seg_len - min_dist_along_segment

            if max_dist_along_segment <= min_dist_along_segment:
                used_segment_indices.add(chosen_segment_idx_in_segments_data)
                continue

            d = pyrandom.uniform(min_dist_along_segment, max_dist_along_segment)
            dir_vec = np.array([math.cos(heading), math.sin(heading), 0.0])
            pos = np.array(start) + dir_vec * d

            side = pyrandom.choice([-1, +1])
            perp = np.array([-dir_vec[1], dir_vec[0], 0.0])

            shift_distance_from_centerline = tunnel_half_width - MIN_ROBOT_CLEARANCE - obstacle_length / 2.0
            if shift_distance_from_centerline < 0:
                shift_distance_from_centerline = 0

            shift = side * shift_distance_from_centerline

            pos += perp * shift
            pos[2] = WALL_HEIGHT / 2.0

            obstacle_rot_heading = heading + math.pi / 2.0
            rot = (0.0, 0.0, 1.0, obstacle_rot_heading)

            size = (WALL_THICKNESS, obstacle_length, WALL_HEIGHT)

            too_close_to_placed = False
            for placed_pos in placed_positions:
                if np.linalg.norm(pos[:2] - placed_pos) < MIN_OBSTACLE_DISTANCE:
                    too_close_to_placed = True
                    break

            if too_close_to_placed:
                continue

            too_close_to_wall = False
            min_allowed_distance_to_wall_centerline = obstacle_length / 2.0 + WALL_THICKNESS / 2.0 + 0.05

            expected_left_wall_center = pos[:2] + perp[:2] * (tunnel_half_width - WALL_THICKNESS / 2.0)
            expected_right_wall_center = pos[:2] - perp[:2] * (tunnel_half_width - WALL_THICKNESS / 2.0)

            dist_to_left_wall_centerline = np.linalg.norm(pos[:2] - expected_left_wall_center)
            dist_to_right_wall_centerline = np.linalg.norm(pos[:2] - expected_right_wall_center)

            if dist_to_left_wall_centerline < min_allowed_distance_to_wall_centerline or \
                    dist_to_right_wall_centerline < min_allowed_distance_to_wall_centerline:
                too_close_to_wall = True

            if too_close_to_wall:
                continue

            obstacle_node = self.create_wall(pos, rot, size, wall_type='obstacle')

            if obstacle_node:
                added_obstacles_count += 1
                placed_positions.append(pos[:2].copy())
                used_segment_indices.add(chosen_segment_idx_in_segments_data)
            else:
                print(f"[ERROR] Failed to create obstacle node at {pos[:2]}.")

        if added_obstacles_count < num_obstacles:
            print(
                f"[WARNING] Only added {added_obstacles_count} out of {num_obstacles} requested obstacles after {attempts} attempts.")
        else:
            print(f"Successfully added {added_obstacles_count} obstacles.")

    def _translation(self, x, y, z):
        M = np.eye(4)
        M[:3, 3] = [x, y, z]
        return M

    def _rotation_z(self, angle):
        """
        Return a 4×4 homogeneous rotation matrix about the Z axis.
        """
        c, s = math.cos(angle), math.sin(angle)
        R = np.eye(4)
        R[0, 0], R[0, 1] = c, -s
        R[1, 0], R[1, 1] = s, c
        return R

