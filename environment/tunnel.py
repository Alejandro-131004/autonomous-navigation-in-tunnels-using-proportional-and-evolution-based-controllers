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
        self.walls = []
        self.segments = []
        self.right_walls = []
        self.left_walls = []
        self.obstacles = []
        self.wall_count = 0  # Initialize wall counter

    def create_wall(self, pos, rot, size, wall_type=None):
        """
        Creates a Solid wall node in Webots with a boundingObject for collision.
        Static walls (boundary, entrance, left, right) and obstacle walls
        will NOT have a Physics node.
        Returns the created node reference if successful, None otherwise.
        """
        type_str = str(wall_type).upper() if wall_type is not None else "NONE"
        wall_def_name = f"TUNNEL_WALL_{type_str}_{self.wall_count}"

        # Start building the Solid node string
        wall_string = f"""DEF {wall_def_name} Solid {{
        translation {pos[0]} {pos[1]} {pos[2]}
        rotation {rot[0]} {rot[1]} {rot[2]} {rot[3]}
        children [
            Shape {{
                appearance Appearance {{
                    material Material {{
                        diffuseColor {'0 0 1' if wall_type in ['boundary', 'entrance'] else ('1 0 0' if wall_type in ['left', 'right'] else '0.5 0.5 0.5')}
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
    """

        # Physics node is explicitly removed for all wall types, including obstacles, as per user request.
        # This means obstacles will be static and only provide collision geometry.

        # Close the Solid node
        wall_string += "\n}"

        try:
            # Import the wall into the scene
            self.root_children.importMFNodeFromString(-1, wall_string)
            self.supervisor.step(1)  # Let Webots process the node
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
                print(f"[ERROR] Failed to retrieve node reference for {wall_def_name}.")
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
        # Iterate over a copy of the walls list
        for pos, rot, size, current_type, node in self.walls[:]:
            if current_type == wall_type:
                if node:  # Check if the node reference is valid
                    try:
                        parent_field = node.getParentField()
                        if parent_field:
                            parent_field.removeMFNode(node)
                            removed_count += 1
                            # print(f"Successfully removed node: {node.getDefName()}") # Debugging removal
                        else:
                            print(
                                f"[WARNING] Could not get parent field for node {node.getDefName()} (type: {current_type}). Skipping removal.")
                    except Exception as e:
                        print(
                            f"[ERROR] Exception during removal of node {node.getDefName()} (type: {current_type}): {e}")
                else:
                    print(
                        f"[WARNING] Invalid node reference found in walls list for type {current_type}. Skipping removal.")
            else:
                # Keep walls that do not match the type
                walls_to_keep.append((pos, rot, size, current_type, node))

        # Update the main walls list
        self.walls = walls_to_keep

        # Rebuild the specific type lists (simpler than removing elements while iterating)
        self.right_walls = [wall for wall in self.walls if wall[3] == 'right']
        self.left_walls = [wall for wall in self.walls if wall[3] == 'left']
        self.obstacles = [wall for wall in self.walls if wall[3] == 'obstacle']
        # Assuming segments is handled elsewhere or should also be rebuilt if it stores wall data

        print(f"Attempted to clear walls of type '{wall_type}'. Successfully removed {removed_count}.")

    def _clear_walls(self):
        """
        Removes all walls created by this builder instance from the simulation
        and clears internal lists. (Kept for backward compatibility or full reset)
        """
        print("Clearing ALL previous walls...")
        # Iterate through the stored wall nodes and remove them from the scene
        # Iterate over a copy because we are modifying the list
        walls_to_remove = self.walls[:]
        removed_count = 0
        for pos, rot, size, wall_type, node in walls_to_remove:
            if node:  # Check if the node reference is valid
                # print(f"Attempting to remove node: {node.getDefName()}") # Debugging removal
                try:
                    # Get the parent field and remove the node
                    parent_field = node.getParentField()
                    if parent_field:
                        parent_field.removeMFNode(node)
                        removed_count += 1
                        # print(f"Successfully removed node: {node.getDefName()}") # Debugging removal
                    else:
                        # If no parent field, the node might already be removed or is not in a standard place
                        print(f"[WARNING] Could not get parent field for node {node.getDefName()}. Skipping removal.")
                except Exception as e:
                    print(f"[ERROR] Exception during removal of node {node.getDefName()}: {e}")

            else:
                # This case happens if create_wall failed to get the node reference
                print(f"[WARNING] Invalid node reference found in walls list. Skipping removal.")
                pass

        # Clear internal lists regardless of removal success, to start fresh
        self.walls = []
        self.segments = []  # Assuming this should also be cleared
        self.right_walls = []
        self.left_walls = []
        self.obstacles = []
        # Reset the wall counter when clearing all walls
        self.wall_count = 0
        print(f"Attempted to clear {len(walls_to_remove)} walls. Successfully removed {removed_count}.")

    # Modified build_tunnel to accept parameters and return final_heading
    def build_tunnel(self, num_curves, angle_range, clearance, num_obstacles):
        """
        Builds the main tunnel structure (straight segments, curves, obstacles)
        starting at a map boundary.
        Includes checks to prevent crossing map boundaries during generation.

        Args:
            num_curves (int): The number of curved segments to include.
            angle_range (tuple): (min_angle, max_angle) for curves.
            clearance (float): The clearance factor to determine tunnel width.
            num_obstacles (int): The number of obstacles to place.

        Returns:
            tuple: (start_pos, end_pos, total_walls, final_heading) if successful, otherwise None, None, 0, None.
        """
        # Clear previous walls and segments
        self._clear_walls()

        # Add a small delay after clearing walls before building new ones
        # This gives Webots time to process the deletion requests before adding new nodes.
        if CLEAR_BUILD_DELAY > 0:
            time.sleep(CLEAR_BUILD_DELAY)
            # Step simulation after delay to ensure Webots processes the time.sleep
            self.supervisor.step(1)

        self.base_wall_distance = ROBOT_RADIUS * clearance
        print(f"Attempting to build tunnel with clearance factor: {clearance:.2f}")

        angle_min, angle_max = angle_range
        num_curves = min(num_curves, MAX_NUM_CURVES)  # Cap number of curves
        num_obstacles = min(num_obstacles, MAX_NUM_OBSTACLES)  # Cap number of obstacles

        # --- Set Initial Transformation Matrix to start at a boundary ---
        # Starting at MAP_X_MIN, centered in Y, heading towards positive X (angle 0)
        # Adjusted initial_pos to start 2 * ROBOT_RADIUS inside the MAP_X_MIN boundary
        initial_pos = np.array([MAP_X_MIN + 2 * ROBOT_RADIUS, (MAP_Y_MIN + MAP_Y_MAX) / 2.0, 0.0])
        initial_heading = 0.0  # Pointing towards positive X
        T = np.eye(4)
        T[:3, 3] = initial_pos
        T[:3, :3] = self._rotation_z(initial_heading)[:3, :3]  # Set initial rotation

        start_pos = T[:3, 3].copy()
        segments_data = []

        # --- Calculate segment_length for straight segments ---
        # Calculate min and max number of BASE_WALL_LENGTH pieces
        min_wall_pieces = math.ceil(MIN_STRAIGHT_LENGTH / BASE_WALL_LENGTH)
        max_possible_wall_pieces = math.floor(MAX_STRAIGHT_LENGTH / BASE_WALL_LENGTH)

        # Choose a random number of wall pieces, capped by MAX_WALL_PIECES_PER_STRAIGHT
        num_wall_pieces = pyrandom.randint(min_wall_pieces, min(MAX_WALL_PIECES_PER_STRAIGHT, max_possible_wall_pieces))
        segment_length = num_wall_pieces * BASE_WALL_LENGTH
        print(f"Calculated straight segment length: {segment_length:.2f} (from {num_wall_pieces} wall pieces).")

        # --- Build the Initial Straight Segment ---
        # Check if the end point is within bounds first
        if not self._within_bounds(T, segment_length):
            print("[ERROR] Initial straight segment end point out of bounds. Cannot build tunnel.")
            return None, None, 0, None

        segment_start_pos = T[:3, 3].copy()
        # Check if the straight segment path crosses boundaries
        if not self._add_straight(T, segment_length):
            print("[ERROR] Initial straight segment crosses boundary. Retrying tunnel generation.")
            return None, None, 0, None  # Indicate failure

        segment_end_pos = T[:3, 3].copy()
        segment_heading = math.atan2(T[1, 0], T[0, 0])
        segments_data.append((segment_start_pos, segment_end_pos, segment_heading, segment_length))

        # --- Entrance walls are no longer needed as the tunnel starts on the boundary ---
        # self._add_entrance_walls(segment_start_pos, initial_heading) # REMOVED

        # --- Add Curves and Subsequent Straight Segments ---
        for i in range(num_curves):
            angle = pyrandom.uniform(angle_min, angle_max) * pyrandom.choice([1, -1])

            # Check if the end point after the curve is within bounds
            if not self._within_bounds_after_curve(T, angle, segment_length):
                print(f"[WARNING] Curve {i + 1} end point out of bounds, stopping tunnel generation.")
                break  # Stop adding segments for this tunnel attempt

            # Check if the curved segment path crosses boundaries
            if not self._add_curve(T, angle, segment_length, segments_data):
                print(f"[ERROR] Curve {i + 1} crosses boundary. Retrying tunnel generation.")
                return None, None, 0, None  # Indicate failure

            # If this is the last curve and no straight segment follows, we are done with path building
            if i == num_curves - 1:
                break

            # For subsequent straight segments, recalculate length
            num_wall_pieces = pyrandom.randint(min_wall_pieces,
                                               min(MAX_WALL_PIECES_PER_STRAIGHT, max_possible_wall_pieces))
            current_straight_segment_length = num_wall_pieces * BASE_WALL_LENGTH
            print(
                f"Calculated subsequent straight segment length: {current_straight_segment_length:.2f} (from {num_wall_pieces} wall pieces).")

            # Check if the end point of the subsequent straight is within bounds
            if not self._within_bounds(T, current_straight_segment_length):
                print(
                    f"[WARNING] Straight segment after curve {i + 1} end point out of bounds, stopping tunnel generation.")
                break  # Stop adding segments for this tunnel attempt

            segment_start_pos = T[:3, 3].copy()
            # Check if the straight segment path crosses boundaries
            if not self._add_straight(T, current_straight_segment_length):
                print(f"[ERROR] Straight segment after curve {i + 1} crosses boundary. Retrying tunnel generation.")
                return None, None, 0, None  # Indicate failure

            segment_end_pos = T[:3, 3].copy()
            segment_heading = math.atan2(T[1, 0], T[0, 0])
            segments_data.append((segment_start_pos, segment_end_pos, segment_heading, current_straight_segment_length))

        # Add a final straight segment if the last segment was a curve and num_curves > 0
        if num_curves > 0 and len(segments_data) > (1 + num_curves):
            # For the final straight segment, recalculate length
            num_wall_pieces = pyrandom.randint(min_wall_pieces,
                                               min(MAX_WALL_PIECES_PER_STRAIGHT, max_possible_wall_pieces))
            final_straight_segment_length = num_wall_pieces * BASE_WALL_LENGTH
            print(
                f"Calculated final straight segment length: {final_straight_segment_length:.2f} (from {num_wall_pieces} wall pieces).")

            # Check if the end point of the final straight is within bounds
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
            # If it was just a straight tunnel, segments_data has 1 entry. No need for a final straight.
            pass
        elif num_curves > 0 and len(segments_data) == (1 + num_curves):
            # If the last segment added was the last curve's final sub-segment, and no straight follows, we are done.
            pass
        else:
            pass

        end_pos = T[:3, 3].copy()
        # Get the final heading from the last segment added
        final_heading = segments_data[-1][2] if segments_data else initial_heading

        # --- Add Obstacles ---
        # _add_obstacles now adds walls directly to self.walls
        # Ensure num_obstacles is used here
        self._add_obstacles(segments_data, num_obstacles)

        # Store the generated segments
        self.segments = segments_data

        # --- Add Main Boundary Walls ---
        # These walls form a box around the entire map area.
        # They are positioned at the map boundaries.
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
        # This is an approximation - a more rigorous check would check sub-segments
        # But for a quick check of the final endpoint:
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
        # print("Warning: _add_entrance_walls called, but tunnel should start on boundary.") # Reduced print
        boundary_wall_height = WALL_HEIGHT * 2  # Match height of main boundary walls

        # Calculate the initial sideways direction vectors
        perp_dir_right = np.array(
            [math.cos(initial_heading - math.pi / 2), math.sin(initial_heading - math.pi / 2), 0.0])
        perp_dir_left = np.array(
            [math.cos(initial_heading + math.pi / 2), math.sin(initial_heading + math.pi / 2), 0.0])

        # Starting points of the first left and right tunnel walls
        start_left_wall = tunnel_start_pos + perp_dir_left * self.base_wall_distance
        start_right_wall = tunnel_start_pos + perp_dir_right * self.base_wall_distance

        # Calculate intersection points with map boundaries
        # We'll trace a line from the wall start point outwards perpendicular to the tunnel's initial heading
        # and find where it hits a map boundary.

        def find_boundary_intersection(start_point, direction):
            """Finds the intersection of a ray from start_point in direction with map boundaries."""
            intersections = []
            # Check intersection with X boundaries
            if direction[0] != 0:
                t_xmin = (MAP_X_MIN - start_point[0]) / direction[0]
                t_xmax = (MAP_X_MAX - start_point[0]) / direction[0]
                if t_xmin > 1e-6: intersections.append((t_xmin, 'x_min'))
                if t_xmax > 1e-6: intersections.append((t_xmax, 'x_max'))
            # Check intersection with Y boundaries
            if direction[1] != 0:
                t_ymin = (MAP_Y_MIN - start_point[1]) / direction[1]
                t_ymax = (MAP_Y_MAX - start_point[1]) / direction[1]
                if t_ymin > 1e-6: intersections.append((t_ymin, 'y_min'))
                if t_ymax > 1e-6: intersections.append((t_ymax, 'y_max'))

            # Find the closest valid intersection (t > 0)
            valid_intersections = [(t, boundary) for t, boundary in intersections if
                                   (start_point + t * direction)[0] >= MAP_X_MIN and (start_point + t * direction)[
                                       0] <= MAP_X_MAX and (start_point + t * direction)[1] >= MAP_Y_MIN and
                                   (start_point + t * direction)[1] <= MAP_Y_MAX]

            if valid_intersections:
                closest_t, boundary_hit = min(valid_intersections, key=lambda item: item[0])
                return start_point + closest_t * direction, closest_t
            return None, 0  # No intersection found within bounds

        # Add wall from start_left_wall to boundary
        intersection_left, dist_left = find_boundary_intersection(start_left_wall, perp_dir_left)
        if intersection_left is not None and dist_left > WALL_THICKNESS:  # Ensure minimum length
            # Position is midpoint between start_left_wall and intersection
            pos_left = (start_left_wall + intersection_left) / 2
            pos_left[2] = boundary_wall_height / 2
            # Rotation is perpendicular to initial heading
            rot_left = (0, 0, 1, initial_heading + math.pi / 2)
            # Size is distance to boundary, thickness, height
            size_left = (dist_left, WALL_THICKNESS, boundary_wall_height)
            self.create_wall(pos_left, rot_left, size_left, wall_type='entrance')
            # print(f"Added left entrance wall of length {dist_left:.2f}") # Reduced print

        # Add wall from start_right_wall to boundary
        intersection_right, dist_right = find_boundary_intersection(start_right_wall, perp_dir_right)
        if intersection_right is not None and dist_right > WALL_THICKNESS:  # Ensure minimum length
            # Position is midpoint between start_right_wall and intersection
            pos_right = (start_right_wall + intersection_right) / 2
            pos_right[2] = boundary_wall_height / 2
            # Rotation is perpendicular to initial heading
            rot_right = (0, 0, 1, initial_heading - math.pi / 2)
            # Size is distance to boundary, thickness, height
            size_right = (dist_right, WALL_THICKNESS, boundary_wall_height)
            self.create_wall(pos_right, rot_right, size_right, wall_type='entrance')
            # print(f"Added right entrance wall of length {dist_right:.2f}") # Reduced print

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
        # Project AP onto AB, but clamp the result to the segment's extent
        t = np.clip(np.dot(AP, AB) / (np.dot(AB, AB) + 1e-9), 0.0, 1.0)  # Add epsilon for stability
        closest_point = A + t * AB
        return np.linalg.norm(P - closest_point)

    def is_robot_near_centerline(self, robot_position):
        """
        Checks if the robot is close enough to the center line of the tunnel,
        considering all segments (straight + curves).
        """
        robot_xy = np.array(robot_position[:2])
        # Threshold should be the base wall distance (centerline to wall)
        # plus the robot radius, as the robot's edge needs to be within this distance
        # of the centerline for the robot to be 'near' the centerline path.
        threshold = self.base_wall_distance + ROBOT_RADIUS

        for start, end, _, _ in self.segments:
            # Use only X and Y coordinates for 2D distance check
            dist = self.point_to_segment_distance(start[:2], end[:2], robot_xy)
            if dist <= threshold:
                return True
        # If the robot is not near any segment of the centerline, it's outside the tunnel path
        return False

    def is_robot_inside_tunnel(self, robot_position, heading):
        """
        Checks if the robot is "inside" the tunnel based on proximity to the walls.
        """
        robot_xy = np.array(robot_position[:2])
        # Calculate perpendicular directions based on robot heading
        right_dir = np.array([math.cos(heading - math.pi / 2), math.sin(heading - math.pi / 2)])
        left_dir = np.array([math.cos(heading + math.pi / 2), math.sin(heading + math.pi / 2)])

        def check_walls(walls, direction):
            """Helper to check proximity to a set of walls in a given direction."""
            for pos, _, size, wall_type, node in walls:
                # Exclude boundary and entrance walls from this check if needed
                if wall_type in ['boundary', 'entrance']:
                    continue

                wall_xy = np.array(pos[:2])
                to_wall = wall_xy - robot_xy
                dist = np.linalg.norm(to_wall)
                # Check if the wall is roughly in the expected direction relative to the robot
                dot = np.dot(to_wall / (dist + 1e-6), direction)  # Add epsilon for stability
                # A dot product > 0.7 means the wall is generally in the direction we're checking (right or left)
                if dot > 0.7:
                    # For tunnel walls, check if distance is close to the expected distance from the centerline
                    expected = self.base_wall_distance
                    # Allow a tolerance for being near the expected wall position
                    if abs(dist - expected) < 0.1:
                        return True
            return False

        def check_obstacles(obstacles):
            """Helper to check proximity to obstacles."""
            for pos, _, size, wall_type, node in obstacles:
                # Obstacles are checked differently - just proximity
                obstacle_xy = np.array(pos[:2])
                dist = np.linalg.norm(obstacle_xy - robot_xy)
                # Check if robot is within the obstacle's influence area
                # Influence is half the obstacle width + robot radius + a small buffer
                # Obstacle size is (length, thickness, height) for pillar, (length, width, height) for extension
                # We need to consider the largest dimension that extends towards the robot
                obstacle_effective_radius = max(size[0], size[1]) / 2
                if dist < obstacle_effective_radius + ROBOT_RADIUS + 0.05:
                    return True
            return False

        # Filter walls by type for the checks
        tunnel_left_walls = [w for w in self.walls if w[3] == 'left']
        tunnel_right_walls = [w for w in self.walls if w[3] == 'right']
        tunnel_obstacles = [w for w in self.walls if w[3] == 'obstacle']

        near_right = check_walls(tunnel_right_walls, right_dir)
        near_left = check_walls(tunnel_left_walls, left_dir)
        near_obstacle = check_obstacles(tunnel_obstacles)

        # Define "inside" based on proximity to tunnel walls or obstacles
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

        # Check if the straight segment crosses any boundary
        if self._check_segment_intersection_with_boundaries(current_pos, next_pos):
            return False  # Intersection detected

        # If no intersection, add the walls
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

        if abs(angle) < 1e-6:  # Avoid division by zero for very small angles
            # If angle is negligible, treat as straight and return False if it crosses boundary
            return self._add_straight(T, segment_length)

        # Calculate the radius of the centerline arc
        R_center = segment_length / abs(angle)
        # Calculate the center of rotation for the curve
        center_offset_dir = np.array([
            math.cos(initial_heading - math.copysign(math.pi / 2, angle)),
            math.sin(initial_heading - math.copysign(math.pi / 2, angle)),
            0.0
        ])
        center_of_rotation = initial_pos + center_offset_dir * R_center

        angle_per_subdivision = angle / CURVE_SUBDIVISIONS
        current_T = T.copy()

        for i in range(CURVE_SUBDIVISIONS):
            # Calculate the start and end points of the current sub-segment
            sub_segment_start_pos = current_T[:3, 3].copy()
            sub_segment_heading = math.atan2(current_T[1, 0], current_T[0, 0])

            # Rotate around the center of rotation
            # Translate to origin, rotate, translate back
            current_T_translated_to_origin = current_T.copy()
            current_T_translated_to_origin[:3, 3] -= center_of_rotation

            rotation_matrix_sub = self._rotation_z(angle_per_subdivision)
            current_T_rotated = rotation_matrix_sub @ current_T_translated_to_origin

            current_T_rotated[:3, 3] += center_of_rotation
            sub_segment_end_pos = current_T_rotated[:3, 3].copy()

            # Check for boundary intersection for this sub-segment
            if self._check_segment_intersection_with_boundaries(sub_segment_start_pos, sub_segment_end_pos):
                return False  # Intersection detected, abort curve generation

            # Calculate positions for the left and right wall segments
            # The length of the straight line segment approximating the arc
            sub_segment_length_straight = np.linalg.norm(sub_segment_end_pos - sub_segment_start_pos)

            # Midpoint of the current sub-segment for wall placement
            mid_pos = (sub_segment_start_pos + sub_segment_end_pos) / 2.0
            # Heading of the sub-segment (average of start and end heading, or just end heading)
            sub_segment_avg_heading = math.atan2(current_T_rotated[1, 0], current_T_rotated[0, 0])

            # Calculate perpendicular direction for wall placement relative to the sub-segment's heading
            perp_dir_right = np.array(
                [math.cos(sub_segment_avg_heading - math.pi / 2), math.sin(sub_segment_avg_heading - math.pi / 2), 0.0])
            perp_dir_left = np.array(
                [math.cos(sub_segment_avg_heading + math.pi / 2), math.sin(sub_segment_avg_heading + math.pi / 2), 0.0])

            # Position for left wall
            pos_left = mid_pos + perp_dir_left * self.base_wall_distance + np.array([0, 0, WALL_HEIGHT / 2])
            rot_left = (0, 0, 1, sub_segment_avg_heading)
            size_left = (sub_segment_length_straight, WALL_THICKNESS, WALL_HEIGHT)
            self.create_wall(pos_left, rot_left, size_left, wall_type='left')

            # Position for right wall
            pos_right = mid_pos + perp_dir_right * self.base_wall_distance + np.array([0, 0, WALL_HEIGHT / 2])
            rot_right = (0, 0, 1, sub_segment_avg_heading)
            size_right = (sub_segment_length_straight, WALL_THICKNESS, WALL_HEIGHT)
            self.create_wall(pos_right, rot_right, size_right, wall_type='right')

            # Update the transformation matrix for the next subdivision
            current_T[:] = current_T_rotated

            # Add segment data for the current sub-segment
            segments_data.append(
                (sub_segment_start_pos, sub_segment_end_pos, sub_segment_avg_heading, sub_segment_length_straight))

        # Update the main transformation matrix T with the final state after the curve
        T[:] = current_T
        return True

    def _check_segment_intersection_with_boundaries(self, p1, p2):
        """
        Checks if the line segment (p1, p2) intersects with any of the map boundaries.
        Returns True if an intersection is found, False otherwise.
        """
        # Define the map boundaries as line segments
        boundaries = [
            ((MAP_X_MIN, MAP_Y_MIN), (MAP_X_MAX, MAP_Y_MIN)),  # Bottom
            ((MAP_X_MIN, MAP_Y_MAX), (MAP_X_MAX, MAP_Y_MAX)),  # Top
            ((MAP_X_MIN, MAP_Y_MIN), (MAP_X_MIN, MAP_Y_MAX)),  # Left
            ((MAP_X_MAX, MAP_Y_MIN), (MAP_X_MAX, MAP_Y_MAX))  # Right
        ]

        # Convert p1 and p2 to 2D for intersection calculation
        p1_2d = np.array([p1[0], p1[1]])
        p2_2d = np.array([p2[0], p2[1]])

        # Check if either endpoint is outside the map boundaries
        if not (MAP_X_MIN <= p1_2d[0] <= MAP_X_MAX and MAP_Y_MIN <= p1_2d[1] <= MAP_Y_MAX):
            return True  # Start point is out of bounds
        if not (MAP_X_MIN <= p2_2d[0] <= MAP_X_MAX and MAP_Y_MIN <= p2_2d[1] <= MAP_Y_MAX):
            return True  # End point is out of bounds

        # Check for intersection with each boundary segment
        for b1, b2 in boundaries:
            b1_2d = np.array(b1)
            b2_2d = np.array(b2)

            # Line segment intersection algorithm
            # r = p2 - p1
            # s = b2 - b1
            # t = (q - p) x s / (r x s)
            # u = (q - p) x r / (r x s)

            r = p2_2d - p1_2d
            s = b2_2d - b1_2d

            cross_product = np.cross(r, s)

            if abs(cross_product) < 1e-9:  # Lines are parallel or collinear
                continue  # No unique intersection point, or they are collinear (handled by endpoint check)

            t = np.cross(b1_2d - p1_2d, s) / cross_product
            u = np.cross(b1_2d - p1_2d, r) / cross_product

            # If 0 <= t <= 1 and 0 <= u <= 1, there is an intersection
            if (0 < t < 1) and (0 < u < 1):  # Strictly between endpoints, not at endpoints
                return True  # Intersection found

        return False  # No intersection

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
        max_attempts = num_obstacles * 5  # Prevent infinite loops if placement is difficult

        while obstacles_placed < num_obstacles and max_attempts > 0:
            max_attempts -= 1

            # Randomly select a segment
            segment_index = pyrandom.randint(0, len(segments_data) - 1)
            segment_start_pos, segment_end_pos, segment_heading, segment_length = segments_data[segment_index]

            # Randomly choose a position along the segment
            # Ensure obstacle is not too close to segment start/end
            min_offset = ROBOT_RADIUS * 2  # Minimum distance from segment ends
            if segment_length < 2 * min_offset:  # Segment too short for placement
                continue

            # Position along the segment (0 to 1)
            # Ensure it's not at the very beginning or end of the segment
            t_along_segment = pyrandom.uniform(min_offset / segment_length, 1.0 - min_offset / segment_length)
            obstacle_centerline_pos = segment_start_pos + (segment_end_pos - segment_start_pos) * t_along_segment
            obstacle_centerline_pos[2] = WALL_HEIGHT / 2  # Obstacles are at half wall height

            # Check distance to existing obstacles
            is_too_close_to_existing = False
            for existing_pos, _, existing_size, _, _ in self.obstacles:
                dist = np.linalg.norm(obstacle_centerline_pos[:2] - existing_pos[:2])
                # Consider the largest dimension of the existing obstacle for proximity check
                existing_obstacle_effective_radius = max(existing_size[0], existing_size[1]) / 2
                # We need to estimate the size of the new obstacle before placing it for a proper check
                # For now, let's use a conservative estimate (e.g., ROBOT_RADIUS + some buffer)
                # A more accurate check would involve the actual obstacle size.
                if dist < MIN_OBSTACLE_DISTANCE + ROBOT_RADIUS + existing_obstacle_effective_radius:
                    is_too_close_to_existing = True
                    break
            if is_too_close_to_existing:
                continue  # Try another position

            # Randomly choose obstacle type: 0 for pillar, 1 for wall extension
            obstacle_type_choice = pyrandom.choice([0, 1])

            if obstacle_type_choice == 0:  # Pillar obstacle (existing logic)
                obstacle_pos = obstacle_centerline_pos
                obstacle_rot = (0, 0, 1, pyrandom.uniform(0, 2 * math.pi))  # Random rotation for pillar
                obstacle_size = (WALL_THICKNESS, WALL_THICKNESS, WALL_HEIGHT)  # Pillar size
                print(f"Placing pillar obstacle at {obstacle_pos[:2]}.")

            else:  # Wall-extending obstacle
                # Randomly choose which wall it extends from
                extend_from_side = pyrandom.choice(['left', 'right'])

                # Calculate perpendicular direction from centerline to wall
                # Heading of the segment determines the orientation of the tunnel walls
                # For a left wall, it's heading + pi/2 from centerline.
                # For a right wall, it's heading - pi/2 from centerline.
                if extend_from_side == 'left':
                    wall_dir = np.array(
                        [math.cos(segment_heading + math.pi / 2), math.sin(segment_heading + math.pi / 2), 0.0])
                    # Obstacle extends from left wall towards centerline
                    # Its center will be between the left wall and centerline
                    # Wall position: centerline_pos + wall_dir * self.base_wall_distance
                    # Obstacle center: centerline_pos + wall_dir * (self.base_wall_distance * (1 - MAX_OBSTACLE_EXTENSION_FACTOR / 2))
                    obstacle_pos = obstacle_centerline_pos + wall_dir * (
                                self.base_wall_distance * (1 - MAX_OBSTACLE_EXTENSION_FACTOR / 2))
                    obstacle_rot = (0, 0, 1, segment_heading + math.pi / 2)  # Aligned with wall
                else:  # 'right'
                    wall_dir = np.array(
                        [math.cos(segment_heading - math.pi / 2), math.sin(segment_heading - math.pi / 2), 0.0])
                    # Obstacle extends from right wall towards centerline
                    obstacle_pos = obstacle_centerline_pos + wall_dir * (
                                self.base_wall_distance * (1 - MAX_OBSTACLE_EXTENSION_FACTOR / 2))
                    obstacle_rot = (0, 0, 1, segment_heading - math.pi / 2)  # Aligned with wall

                # Size: length (how much it extends), width (thickness), height
                obstacle_extension_length = self.base_wall_distance * MAX_OBSTACLE_EXTENSION_FACTOR
                obstacle_size = (
                obstacle_extension_length, WALL_THICKNESS, WALL_HEIGHT)  # Length along extension, thickness, height
                print(
                    f"Placing {extend_from_side} wall-extension obstacle at {obstacle_pos[:2]} with length {obstacle_extension_length:.2f}.")

            # Create the wall
            node = self.create_wall(obstacle_pos, obstacle_rot, obstacle_size, wall_type='obstacle')
            if node:
                obstacles_placed += 1
            else:
                print(f"[WARNING] Failed to place obstacle. Retrying...")

        print(f"Finished placing {obstacles_placed} obstacles.")
