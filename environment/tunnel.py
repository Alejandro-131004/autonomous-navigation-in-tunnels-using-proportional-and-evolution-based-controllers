from environment.configuration import WALL_THICKNESS, WALL_HEIGHT, OVERLAP_FACTOR, ROBOT_RADIUS, MAX_NUM_CURVES, MIN_CLEARANCE_FACTOR_RANGE, MAX_CLEARANCE_FACTOR_RANGE, MIN_CURVE_ANGLE_RANGE, MAX_CURVE_ANGLE_RANGE, BASE_WALL_LENGTH, CURVE_SUBDIVISIONS, MIN_ROBOT_CLEARANCE, MAX_NUM_OBSTACLES, MIN_OBSTACLE_DISTANCE, MAP_X_MIN, MAP_X_MAX, MAP_Y_MIN, MAP_Y_MAX
import numpy as np
import math
import random as pyrandom

class TunnelBuilder:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.root_children = supervisor.getRoot().getField("children")
        self.base_wall_distance = 0 # Will be set based on clearance factor
        self.walls = []
        self.segments = []
        self.right_walls = [] # Kept for potential separate tracking
        self.left_walls = []  # Kept for potential separate tracking
        self.obstacles = []   # Kept for potential separate tracking

    def create_wall(self, pos, rot, size, wall_type=None):
        """
        Creates a Solid wall node in Webots with visual geometry and physics.
        pos: list/tuple/numpy array [x, y, z] for translation
        rot: list/tuple/numpy array [ax, ay, az, angle] for rotation
        size: list/tuple/numpy array [sx, sy, sz] for Box geometry size
        wall_type: string ('left', 'right', 'obstacle', 'boundary', 'entrance') for tracking
        """
        wall = f"""Solid {{
            translation {pos[0]} {pos[1]} {pos[2]}
            rotation {rot[0]} {rot[1]} {rot[2]} {rot[3]}
            children [
                Shape {{
                    appearance Appearance {{
                        material Material {{
                            # Use different color for boundary and entrance walls
                            diffuseColor {'0 0 1' if wall_type in ['boundary', 'entrance'] else '1 0 0'}
                        }}
                    }}
                    geometry Box {{
                        size {size[0]} {size[1]} {size[2]}
                    }}
                }}
                # Add Physics node for physical simulation
                Physics {{
                    # The bounding object defines the shape for physics interactions
                    boundingObject Box {{
                        size {size[0]} {size[1]} {size[2]}
                    }}
                    # You can add other physics properties here if needed,
                    # like mass (for dynamic objects), friction, bounce etc.
                    # For static walls, default values are usually fine.
                }}
            ]
            # Add a name for easier identification in the scene tree
            name "wall_{wall_type}_{len(self.walls)}"
            # Set the wall to be static so it doesn't move
            model "static"
        }}"""
        self.root_children.importMFNodeFromString(-1, wall)
        # Append wall details for tracking/cleanup
        self.walls.append((pos, rot, size, wall_type))
        if wall_type == 'left':
            self.left_walls.append((pos, rot, size, wall_type))
        elif wall_type == 'right':
            self.right_walls.append((pos, rot, size, wall_type))
        elif wall_type == 'obstacle':
            self.obstacles.append((pos, rot, size, wall_type))


    # Modified build_tunnel to accept parameters
    def build_tunnel(self, num_curves, angle_range, clearance, num_obstacles):
        """
        Builds the main tunnel structure (straight segments, curves, obstacles)
        and then adds entrance walls and main boundary walls.
        Includes checks to prevent crossing map boundaries during generation.

        Args:
            num_curves (int): The number of curved segments to include.
            angle_range (tuple): (min_angle, max_angle) for curves.
            clearance (float): The clearance factor to determine tunnel width.
            num_obstacles (int): The number of obstacles to place.

        Returns:
            tuple: (start_pos, end_pos, total_walls) if successful, otherwise None, None, 0.
        """
        # Clear previous walls and segments
        self._clear_walls()

        self.base_wall_distance = ROBOT_RADIUS * clearance
        #print(f"Attempting to build tunnel with clearance factor: {clearance:.2f}")

        angle_min, angle_max = angle_range
        num_curves = min(num_curves, MAX_NUM_CURVES) # Cap number of curves
        num_obstacles = min(num_obstacles, MAX_NUM_OBSTACLES) # Cap number of obstacles

        # Determine total length based on number of curves
        # Each curve adds complexity, so total length might need adjustment
        # A simple approach: total length is proportional to number of segments (curves + straights)
        total_segments = 1 + num_curves + (1 if num_curves > 0 else 0) # Initial straight + curves + straight after last curve (if any)
        segment_length = BASE_WALL_LENGTH # Keep base segment length consistent

        T = np.eye(4)
        # Start position is always at the origin initially
        start_pos = T[:3, 3].copy()
        segments_data = []

        # --- Build the Initial Straight Segment ---
        # Check if the end point is within bounds first
        if not self._within_bounds(T, segment_length):
            print("[ERROR] Initial straight segment end point out of bounds. Cannot build tunnel.")
            return None, None, 0

        segment_start_pos = T[:3, 3].copy()
        # Check if the straight segment path crosses boundaries
        if not self._add_straight(T, segment_length):
             print("[ERROR] Initial straight segment crosses boundary. Retrying tunnel generation.")
             return None, None, 0 # Indicate failure

        segment_end_pos = T[:3, 3].copy()
        segment_heading = math.atan2(T[1, 0], T[0, 0])
        segments_data.append((segment_start_pos, segment_end_pos, segment_heading, segment_length))

        # --- Add Entrance Walls ---
        # These walls connect the start of the tunnel to the map boundary.
        self._add_entrance_walls(segment_start_pos, segment_heading)

        # --- Add Curves and Subsequent Straight Segments ---
        for i in range(num_curves):
            angle = pyrandom.uniform(angle_min, angle_max) * pyrandom.choice([1, -1])

            # Check if the end point after the curve is within bounds
            if not self._within_bounds_after_curve(T, angle, segment_length):
                print(f"[WARNING] Curve {i+1} end point out of bounds, stopping tunnel generation.")
                break # Stop adding segments for this tunnel attempt

            # Check if the curved segment path crosses boundaries
            if not self._add_curve(T, angle, segment_length, segments_data):
                 print(f"[ERROR] Curve {i+1} crosses boundary. Retrying tunnel generation.")
                 return None, None, 0 # Indicate failure

            # If this is the last curve and no straight segment follows, we are done with path building
            if i == num_curves - 1:
                break

            # Check if the end point of the subsequent straight is within bounds
            if not self._within_bounds(T, segment_length):
                print(f"[WARNING] Straight segment after curve {i+1} end point out of bounds, stopping tunnel generation.")
                break # Stop adding segments for this tunnel attempt

            segment_start_pos = T[:3, 3].copy()
            # Check if the straight segment path crosses boundaries
            if not self._add_straight(T, segment_length):
                 print(f"[ERROR] Straight segment after curve {i+1} crosses boundary. Retrying tunnel generation.")
                 return None, None, 0 # Indicate failure

            segment_end_pos = T[:3, 3].copy()
            segment_heading = math.atan2(T[1, 0], T[0, 0])
            segments_data.append((segment_start_pos, segment_end_pos, segment_heading, segment_length))

        # Add a final straight segment if the last segment was a curve and num_curves > 0
        if num_curves > 0 and len(segments_data) > 0 and len(segments_data) <= num_curves * 2: # Check if the last added segment was a curve sub-segment
             # Check if the end point of the final straight is within bounds
             if not self._within_bounds(T, segment_length):
                 print(f"[WARNING] Final straight segment end point out of bounds, stopping tunnel generation.")
             else:
                 segment_start_pos = T[:3, 3].copy()
                 if self._add_straight(T, segment_length):
                     segment_end_pos = T[:3, 3].copy()
                     segment_heading = math.atan2(T[1, 0], T[0, 0])
                     segments_data.append((segment_start_pos, segment_end_pos, segment_heading, segment_length))
                 else:
                      print(f"[ERROR] Final straight segment crosses boundary. Retrying tunnel generation.")
                      return None, None, 0

        end_pos = T[:3, 3].copy()

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

        return start_pos, end_pos, len(self.walls)

    def _clear_walls(self):
        """Removes all existing wall nodes from the simulation."""
        # This requires Webots API calls to remove nodes from the scene.
        # Assuming self.root_children is a SFNode field allowing removal.
        # Iterate backwards to safely remove elements while modifying the list.
        # We only remove walls created by THIS builder instance in the current run.
        # A better approach would be to group nodes under a parent node and remove the parent.
        # For now, let's clear the internal lists and assume an external mechanism
        # or a more robust Webots API call handles the actual node removal.
        print("Clearing previous walls (internal lists only). Webots node removal needed here.")
        self.walls = []
        self.segments = []
        self.right_walls = []
        self.left_walls = []
        self.obstacles = []
        # Example Webots API call (conceptual):
        # while self.root_children.getCount() > 0:
        #     node = self.root_children.getMFNode(self.root_children.getCount() - 1)
        #     # Check if the node is one of our generated walls before removing
        #     if node and node.getTypeName() == "Solid" and node.getField("name") and node.getField("name").getSFString().startswith("wall_"):
        #          self.root_children.removeMF(self.root_children.getCount() - 1)
        #     else:
        #          # Stop if we encounter nodes not created by the builder
        #          break


    def _check_segment_intersection_with_boundaries(self, p1, p2):
        """
        Checks if the 2D line segment from p1 to p2 intersects any of the map boundaries.
        p1, p2: numpy arrays [x, y, z] (z is ignored for 2D check)
        Returns True if intersects, False otherwise.
        """
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]

        # Check if either endpoint is outside the map boundaries
        if not (MAP_X_MIN <= x1 <= MAP_X_MAX and MAP_Y_MIN <= y1 <= MAP_Y_MAX) or \
           not (MAP_X_MIN <= x2 <= MAP_X_MAX and MAP_Y_MIN <= y2 <= MAP_Y_MAX):
            # If an endpoint is outside, the segment might cross the boundary
            # Further checks are needed to confirm it's not just starting/ending outside
            # but for simplicity and robustness, we can treat segments with endpoints outside as crossing.
            # print(f"Endpoint outside bounds: {p1[:2]} or {p2[:2]}")
            return True


        # Check for intersection with vertical boundaries (x = constant)
        for x_boundary in [MAP_X_MIN, MAP_X_MAX]:
            # Check if segment spans the boundary's x-value
            if (x1 <= x_boundary < x2) or (x2 <= x_boundary < x1):
                # Calculate y-coordinate at the intersection point
                if (x2 - x1) != 0: # Avoid division by zero for vertical segments
                    y_intersection = y1 + (y2 - y1) * (x_boundary - x1) / (x2 - x1)
                    # Check if the intersection point's y-coordinate is within the segment's y-range
                    if (min(y1, y2) <= y_intersection <= max(y1, y2)):
                        # Also check if the intersection point is within the map's y-bounds
                        if MAP_Y_MIN <= y_intersection <= MAP_Y_MAX:
                            # print(f"Segment from {p1[:2]} to {p2[:2]} intersects vertical boundary at x={x_boundary}, y={y_intersection:.2f}")
                            return True

        # Check for intersection with horizontal boundaries (y = constant)
        for y_boundary in [MAP_Y_MIN, MAP_Y_MAX]:
            # Check if segment spans the boundary's y-value
            if (y1 <= y_boundary < y2) or (y2 <= y_boundary < y1):
                 # Calculate x-coordinate at the intersection point
                if (y2 - y1) != 0: # Avoid division by zero for horizontal segments
                    x_intersection = x1 + (x2 - x1) * (y_boundary - y1) / (y2 - y1)
                    # Check if the intersection point's x-coordinate is within the segment's x-range
                    if (min(x1, x2) <= x_intersection <= max(x1, x2)):
                         # Also check if the intersection point is within the map's x-bounds
                        if MAP_X_MIN <= x_intersection <= MAP_X_MAX:
                            # print(f"Segment from {p1[:2]} to {p2[:2]} intersects horizontal boundary at y={y_boundary}, x={x_intersection:.2f}")
                            return True

        return False # No intersection found


    def _add_entrance_walls(self, tunnel_start_pos, initial_heading):
        """
        Adds walls connecting the start of the tunnel to the map boundaries.
        """
        boundary_wall_height = WALL_HEIGHT * 2 # Match height of main boundary walls

        # Calculate the initial sideways direction vectors
        perp_dir_right = np.array([math.cos(initial_heading - math.pi / 2), math.sin(initial_heading - math.pi / 2), 0.0])
        perp_dir_left = np.array([math.cos(initial_heading + math.pi / 2), math.sin(initial_heading + math.pi / 2), 0.0])

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
            valid_intersections = [(t, boundary) for t, boundary in intersections if (start_point + t * direction)[0] >= MAP_X_MIN and (start_point + t * direction)[0] <= MAP_X_MAX and (start_point + t * direction)[1] >= MAP_Y_MIN and (start_point + t * direction)[1] <= MAP_Y_MAX]

            if valid_intersections:
                closest_t, boundary_hit = min(valid_intersections, key=lambda item: item[0])
                return start_point + closest_t * direction, closest_t
            return None, 0 # No intersection found within bounds

        # Add wall from start_left_wall to boundary
        intersection_left, dist_left = find_boundary_intersection(start_left_wall, perp_dir_left)
        if intersection_left is not None and dist_left > WALL_THICKNESS: # Ensure minimum length
            # Position is midpoint between start_left_wall and intersection
            pos_left = (start_left_wall + intersection_left) / 2
            pos_left[2] = boundary_wall_height / 2
            # Rotation is perpendicular to initial heading
            rot_left = (0, 0, 1, initial_heading + math.pi / 2)
            # Size is distance to boundary, thickness, height
            size_left = (dist_left, WALL_THICKNESS, boundary_wall_height)
            self.create_wall(pos_left, rot_left, size_left, wall_type='entrance')
            #print(f"Added left entrance wall of length {dist_left:.2f}")

        # Add wall from start_right_wall to boundary
        intersection_right, dist_right = find_boundary_intersection(start_right_wall, perp_dir_right)
        if intersection_right is not None and dist_right > WALL_THICKNESS: # Ensure minimum length
            # Position is midpoint between start_right_wall and intersection
            pos_right = (start_right_wall + intersection_right) / 2
            pos_right[2] = boundary_wall_height / 2
            # Rotation is perpendicular to initial heading
            rot_right = (0, 0, 1, initial_heading - math.pi / 2)
            # Size is distance to boundary, thickness, height
            size_right = (dist_right, WALL_THICKNESS, boundary_wall_height)
            self.create_wall(pos_right, rot_right, size_right, wall_type='entrance')
            #print(f"Added right entrance wall of length {dist_right:.2f}")


    def _add_main_boundary_walls(self):
        """
        Adds walls around the entire map area.
        """
        boundary_wall_height = WALL_HEIGHT * 2 # Make boundary walls taller

        # Wall along MAP_X_MIN
        pos_xmin = [MAP_X_MIN, (MAP_Y_MIN + MAP_Y_MAX) / 2, boundary_wall_height / 2]
        rot_xmin = [0, 0, 1, math.pi/2] # Rotate 90 degrees around Z for Y-axis alignment
        size_xmin = [MAP_Y_MAX - MAP_Y_MIN, WALL_THICKNESS, boundary_wall_height]
        self.create_wall(pos_xmin, rot_xmin, size_xmin, wall_type='boundary')

        # Wall along MAP_X_MAX
        pos_xmax = [MAP_X_MAX, (MAP_Y_MIN + MAP_Y_MAX) / 2, boundary_wall_height / 2]
        rot_xmax = [0, 0, 1, math.pi/2]
        size_xmax = [MAP_Y_MAX - MAP_Y_MIN, WALL_THICKNESS, boundary_wall_height]
        self.create_wall(pos_xmax, rot_xmax, size_xmax, wall_type='boundary')

        # Wall along MAP_Y_MIN
        pos_ymin = [(MAP_X_MIN + MAP_X_MAX) / 2, MAP_Y_MIN, boundary_wall_height / 2]
        rot_ymin = [0, 0, 1, 0] # Aligned with X-axis
        size_ymin = [MAP_X_MAX - MAP_X_MIN, WALL_THICKNESS, boundary_wall_height]
        self.create_wall(pos_ymin, rot_ymin, size_ymin, wall_type='boundary')

        # Wall along MAP_Y_MAX
        pos_ymax = [(MAP_X_MIN + MAP_X_MAX) / 2, MAP_Y_MAX, boundary_wall_height / 2]
        rot_ymax = [0, 0, 1, 0]
        size_ymax = [MAP_X_MAX - MAP_X_MIN, WALL_THICKNESS, boundary_wall_height]
        self.create_wall(pos_ymax, rot_ymax, size_ymax, wall_type='boundary')

        #print(f"Added main boundary walls.")


    def point_to_segment_distance(self, A, B, P):
        """
        Calculates the shortest distance from point P to the line segment AB.
        A, B: start and end points of the segment (numpy arrays)
        P: the point (numpy array)
        """
        AP = P - A
        AB = B - A
        # Project AP onto AB, but clamp the result to the segment's extent
        t = np.clip(np.dot(AP, AB) / (np.dot(AB, AB) + 1e-9), 0.0, 1.0) # Add epsilon for stability
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
        This method's logic might be redundant if is_robot_near_centerline is used
        to enforce following the path.
        """
        # This method's implementation is kept as is from the previous version,
        # but its utility for the specific goal of preventing circumvention
        # by going *around* the tunnel is questionable compared to checking
        # proximity to the centerline or simply relying on the physical boundaries.
        # print("[INFO] is_robot_inside_tunnel method called. Consider using is_robot_near_centerline or physical collision detection.")
        robot_xy = np.array(robot_position[:2])
        # Calculate perpendicular directions based on robot heading
        right_dir = np.array([math.cos(heading - math.pi / 2), math.sin(heading - math.pi / 2)])
        left_dir = np.array([math.cos(heading + math.pi / 2), math.sin(heading + math.pi / 2)])

        def check_walls(walls, direction):
            """Helper to check proximity to a set of walls in a given direction."""
            for pos, _, size, wall_type in walls:
                 # Exclude boundary and entrance walls from this check if needed
                 if wall_type in ['boundary', 'entrance']:
                     continue # Skip boundary and entrance walls for this specific check

                 wall_xy = np.array(pos[:2])
                 to_wall = wall_xy - robot_xy
                 dist = np.linalg.norm(to_wall)
                 # Check if the wall is roughly in the expected direction relative to the robot
                 dot = np.dot(to_wall / (dist + 1e-6), direction) # Add epsilon for stability
                 # A dot product > 0.7 means the wall is generally in the direction we're checking (right or left)
                 if dot > 0.7:
                     # For tunnel walls, check if distance is close to the expected distance from the centerline
                     expected = self.base_wall_distance
                     # Allow a tolerance for being near the expected wall position
                     if abs(dist - expected) < 0.1: # Increased tolerance slightly
                         return True
            return False

        def check_obstacles(obstacles):
            """Helper to check proximity to obstacles."""
            for pos, _, size, wall_type in obstacles:
                # Obstacles are checked differently - just proximity
                obstacle_xy = np.array(pos[:2])
                dist = np.linalg.norm(obstacle_xy - robot_xy)
                # Check if robot is within the obstacle's influence area
                # Influence is half the obstacle width + robot radius + a small buffer
                # Obstacle size is (thickness, length, height) - length is the relevant dimension for width
                if dist < size[1] / 2 + ROBOT_RADIUS + 0.05: # Increased buffer slightly
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
        # print(f"[DEBUG] Inside tunnel (proximity check): {inside} | Right: {near_right}, Left: {near_left}, Obstacle: {near_obstacle}")
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
            return False # Intersection detected

        # If no intersection, add the walls
        for side in [-1, 1]:
            pos = current_pos + T[:3, 0] * (length / 2) + T[:3, 1] * (side * self.base_wall_distance) + np.array(
                [0, 0, WALL_HEIGHT / 2])
            rot = (0, 0, 1, heading)
            size = (length, WALL_THICKNESS, WALL_HEIGHT)
            wall_type = 'left' if side == -1 else 'right'
            self.create_wall(pos, rot, size, wall_type=wall_type)

        # Update the transformation matrix T
        T[:] = next_T

        return True # Successfully added straight segment


    def _add_curve(self, T, angle, segment_length, segments_data):
        """
        Adds a curved segment of the tunnel after checking for boundary intersection
        of each sub-segment.
        Returns True if successful, False if intersection is detected in any sub-segment.
        """
        step = angle / CURVE_SUBDIVISIONS
        # The length of the centerline arc for one subdivision
        centerline_sub_segment_length = (self.base_wall_distance) * abs(step) # Arc length = radius * angle
        # The length of the straight line segment approximating the arc
        straight_sub_segment_length = 2 * (self.base_wall_distance) * math.sin(abs(step) / 2)


        T_current = T.copy() # Use a copy to check sub-segments before committing to T

        for i in range(CURVE_SUBDIVISIONS):
            T_start_step = T_current.copy()
            segment_start = T_current[:3, 3].copy()

            # Calculate the end point of this sub-segment's centerline
            # Rotate first, then translate along the new heading
            T_rotated = T_current @ self._rotation_z(step)
            sub_segment_end = T_rotated[:3, 3] + T_rotated[:3, 0] * straight_sub_segment_length # Use straight line distance for endpoint check

            # Check if this sub-segment of the centerline crosses any boundary
            if self._check_segment_intersection_with_boundaries(segment_start, sub_segment_end):
                return False # Intersection detected in a sub-segment

            # If no intersection, proceed to add walls for this sub-segment
            # Position the wall at the midpoint of the straight line segment approximation
            mid_point = (segment_start + sub_segment_end) / 2.0
            mid_point[2] = WALL_HEIGHT / 2.0 # Set Z height

            # Calculate rotation for the wall - should be aligned with the direction of the straight sub-segment
            direction_vector = sub_segment_end - segment_start
            heading = math.atan2(direction_vector[1], direction_vector[0])
            rot = (0, 0, 1, heading)


            for side in [-1, 1]:
                # Wall length should approximate the arc length for this sub-segment
                wall_length = centerline_sub_segment_length + OVERLAP_FACTOR * WALL_THICKNESS

                # Calculate the position offset perpendicular to the segment heading
                perp_offset = np.array([-direction_vector[1], direction_vector[0], 0.0])
                perp_offset = perp_offset / (np.linalg.norm(perp_offset) + 1e-9) * (side * self.base_wall_distance) # Normalize and scale

                pos = mid_point + perp_offset

                size = (wall_length, WALL_THICKNESS, WALL_HEIGHT)
                wall_type = 'left' if side == -1 else 'right'
                self.create_wall(pos, rot, size, wall_type=wall_type)

            # Update T_current for the next iteration
            T_current[:] = T_rotated
            T_current[:3, 3] = sub_segment_end # Update position to the end of the straight approximation


            # Append segment data for the successfully added sub-segment
            # Store the straight line approximation segment data
            segments_data.append((segment_start, sub_segment_end, heading, straight_sub_segment_length))

        # If all sub-segments were added without intersection, update the main T
        T[:] = T_current

        return True # Successfully added curved segment


    # Modified _add_obstacles to accept num_obstacles
    def _add_obstacles(self, segments_data, num_obstacles):
        """
        Place num_obstacles perpendicular walls into the middle straight segments.
        segments_data: list of (start_pos, end_pos, heading, length) for all segments
        num_obstacles (int): The number of obstacles to place.
        """
        # Filter for straight segments that are not the very first or very last
        # This assumes segments_data contains alternating straight and curved segments (or just straights)
        # and the first/last entries correspond to the entrance/exit straights.
        straight_segments = [seg for i, seg in enumerate(segments_data) if i % 2 == 0 and i > 0 and i < len(segments_data) - 1]

        if not straight_segments:
            print("Not enough internal straight segments for obstacles.")
            return

        used_segment_indices = set()
        placed_positions = []

        tunnel_half_width = self.base_wall_distance
        # Obstacle length should be less than the tunnel width to allow passing
        # It spans from one side of the tunnel to the other, leaving MIN_ROBOT_CLEARANCE
        obstacle_length = 2 * tunnel_half_width - MIN_ROBOT_CLEARANCE - WALL_THICKNESS

        # Ensure obstacle length is positive
        if obstacle_length <= 0:
             print(f"Obstacle length is zero or negative ({obstacle_length:.2f}). Cannot place obstacles.")
             return


        for _ in range(num_obstacles): # Use the provided num_obstacles
            choices = [i for i in range(len(straight_segments)) if i not in used_segment_indices]
            if not choices:
                print("Ran out of available segments for obstacles.")
                break
            idx = pyrandom.choice(choices)
            used_segment_indices.add(idx)

            start, end, heading, seg_len = straight_segments[idx]
            # Place obstacle somewhere along the segment, avoiding ends
            d = pyrandom.uniform(0.2 * seg_len, 0.8 * seg_len)
            dir_vec = np.array([math.cos(heading), math.sin(heading), 0.0])
            pos = np.array(start) + dir_vec * d

            # Decide which side of the centerline the obstacle is placed on
            side = pyrandom.choice([-1, +1])
            perp = np.array([-dir_vec[1], dir_vec[0], 0.0]) # Perpendicular vector
            # Shift position based on tunnel half-width, obstacle length, and robot clearance
            # The obstacle is placed such that it leaves MIN_ROBOT_CLEARANCE on one side
            shift = side * (tunnel_half_width - obstacle_length / 2.0 - MIN_ROBOT_CLEARANCE / 2.0 - WALL_THICKNESS/2.0)
            pos += perp * shift
            pos[2] = WALL_HEIGHT / 2.0 # Set Z height

            # Obstacle rotation is perpendicular to the segment heading
            obstacle_rot_heading = heading + math.pi / 2.0 # Rotate 90 degrees from segment heading
            rot = (0.0, 0.0, 1.0, obstacle_rot_heading)

            # Obstacle size: thickness along segment, length across tunnel, height
            size = (WALL_THICKNESS, obstacle_length, WALL_HEIGHT)

            # Check for proximity to already placed obstacles
            if any(np.linalg.norm(pos[:2] - p) < MIN_OBSTACLE_DISTANCE
                   for p in placed_positions):
                print("Skipping obstacle—too close to another placed obstacle.")
                continue

            self.create_wall(pos, rot, size, wall_type='obstacle')
            placed_positions.append(pos[:2].copy())


    def _translation(self, x, y, z):
        # Helper function to create a translation matrix (not used in current add_ methods but kept)
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
        # Use the straight line approximation length for endpoint check
        straight_sub_segment_length = 2 * (self.base_wall_distance) * math.sin(abs(step) / 2)

        for _ in range(CURVE_SUBDIVISIONS):
            # Rotate first, then translate
            tempT = tempT @ self._rotation_z(step)
            tempT[:3, 3] += tempT[:3, 0] * straight_sub_segment_length

        end = tempT[:3, 3]
        return MAP_X_MIN <= end[0] <= MAP_X_MAX and MAP_Y_MIN <= end[1] <= MAP_Y_MAX

