from environment.configuration import WALL_THICKNESS, WALL_HEIGHT, OVERLAP_FACTOR, ROBOT_RADIUS, MAX_NUM_CURVES, MIN_CLEARANCE_FACTOR_RANGE, MAX_CLEARANCE_FACTOR_RANGE, MIN_CURVE_ANGLE_RANGE, MAX_CURVE_ANGLE_RANGE, BASE_WALL_LENGTH, CURVE_SUBDIVISIONS, MIN_ROBOT_CLEARANCE, MAX_NUM_OBSTACLES, MIN_OBSTACLE_DISTANCE, MAP_X_MIN, MAP_X_MAX, MAP_Y_MIN, MAP_Y_MAX
import numpy as np
import math
import random as pyrandom
import time # Import time for potential timestamping or timing

# Define a constant for how often to check physics (in seconds)
CHECK_PHYSICS_INTERVAL = 5.0 # Check physics every 5 seconds
# Define a small delay after clearing walls before building new ones (in seconds)
CLEAR_BUILD_DELAY = 0.1 # 100 milliseconds delay

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
        self.physics_check_timer = 0.0  # Timer for periodic physics checks (in seconds)
        self.last_physics_check_time = time.time() # Timestamp of the last check

    def create_wall(self, pos, rot, size, wall_type=None):
        """
        Creates a Solid wall node in Webots with a boundingObject for collision.
        Static walls (boundary, entrance, left, right) will NOT have a Physics node.
        Obstacle walls WILL have a Physics node with basic properties.
        Returns the created node reference if successful, None otherwise.
        """
        # Use a unique DEF name for each node
        # Use a more robust naming convention that Webots is less likely to modify
        # Ensure wall_type is a string for the DEF name
        type_str = str(wall_type).upper() if wall_type is not None else "NONE"
        # Generate incremental DEF name
        wall_def_name = f"TUNNEL_WALL_{type_str}_{self.wall_count}"

        # Define the basic Solid node structure
        # *** MODIFICATION: Added DEF keyword here ***
        wall_string = f"""DEF {wall_def_name} Solid {{
            translation {pos[0]} {pos[1]} {pos[2]}
            rotation {rot[0]} {rot[1]} {rot[2]} {rot[3]}
            children [
                Shape {{
                    appearance Appearance {{
                        material Material {{
                            diffuseColor {'0 0 1' if wall_type in ['boundary', 'entrance'] else '1 0 0'}
                        }}
                    }}
                    geometry Box {{
                        size {size[0]} {size[1]} {size[2]}
                    }}
                }}
            ]
            # The 'name' field is separate from the DEF name but often set to the same value for clarity
            name "{wall_def_name}"
            model "static" # Model is often used for visual/semantic grouping, doesn't enforce physics type
            # Add the boundingObject field directly to the Solid node
            boundingObject Box {{
                size {size[0]} {size[1]} {size[2]}
            }}
            # Note: contactMaterial should ideally be defined in the WorldInfo node's contactProperties field
            contactMaterial "wall" # Add contactMaterial here to link to ContactProperties
        }}"""

        # Add Physics node only for obstacle walls, as per documentation for movable objects
        if wall_type == 'obstacle':
             # Obstacles need Physics to be pushed/interact dynamically
             # Removed deprecated fields (bounceVelocity, coulombFriction, forceDependentSlip)
             physics_node_string = """
                physics Physics {
                    density 1000 # Example density for obstacles (or use mass field)
                    # bounceVelocity and friction properties should be in ContactProperties node
                    dampingFactor 0.9 # Damping for stability
                    # boundingObject is NOT inside Physics node in R2025a
                }
             """
             # Insert the physics node string before the closing brace of the Solid
             # Find the position before the last '}'
             insert_index = wall_string.rfind('}')
             if insert_index != -1:
                  wall_string = wall_string[:insert_index] + physics_node_string + wall_string[insert_index:]
             else:
                  print(f"[ERROR] Could not find closing brace in Solid string for {wall_def_name}")
        # For static walls, the physics field is implicitly NULL by not adding it.

        # --- Debugging: Print the PROTO string being imported ---
        # print(f"Attempting to import node string for {wall_def_name}:\n{wall_string}")

        try:
            # Import the node string into the scene tree
            # print(f"Importing node string for {wall_def_name}...")
            self.root_children.importMFNodeFromString(-1, wall_string)
            # print(f"Imported node string for {wall_def_name}. Stepping simulation...")

            # IMPORTANT: Step the simulation briefly to allow Webots to process the node creation
            # and make the DEF name available via getFromDef. A timestep of 1 is usually sufficient.
            self.supervisor.step(1)
            # print(f"Simulation stepped. Attempting to get node reference for {wall_def_name}...")

            # Get the reference to the newly created node using its DEF name
            node = self.supervisor.getFromDef(wall_def_name)

            if node:
                # print(f"Successfully retrieved node reference for {wall_def_name}.")
                # Append wall details including the node reference for tracking/cleanup/checks
                wall_data = (pos, rot, size, wall_type, node)
                self.walls.append(wall_data)
                if wall_type == 'left':
                    self.left_walls.append(wall_data)
                elif wall_type == 'right':
                    self.right_walls.append(wall_data)
                elif wall_type == 'obstacle':
                    self.obstacles.append(wall_data)
                # Increment the wall counter
                self.wall_count += 1
                # print(f"Created wall: {wall_def_name} (Bounding Object: Yes, Physics: {'Yes' if wall_type == 'obstacle' else 'No'})") # Reduced print
                return node
            else:
                print(f"[ERROR] Failed to get node reference after creation for: {wall_def_name}. Node might not have been created or DEF name is incorrect.")
                return None
        except Exception as e:
            print(f"[CRITICAL ERROR] Exception during wall creation for {wall_def_name}: {e}")
            # Consider printing the wall_string here if exceptions occur frequently
            # print(f"Problematic wall string:\n{wall_string}")
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
                if node: # Check if the node reference is valid
                    try:
                        parent_field = node.getParentField()
                        if parent_field:
                            parent_field.removeMFNode(node)
                            removed_count += 1
                            # print(f"Successfully removed node: {node.getDefName()}") # Debugging removal
                        else:
                            print(f"[WARNING] Could not get parent field for node {node.getDefName()} (type: {current_type}). Skipping removal.")
                    except Exception as e:
                        print(f"[ERROR] Exception during removal of node {node.getDefName()} (type: {current_type}): {e}")
                else:
                    print(f"[WARNING] Invalid node reference found in walls list for type {current_type}. Skipping removal.")
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
            if node: # Check if the node reference is valid
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
        self.segments = [] # Assuming this should also be cleared
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
            # print(f"Waiting for {CLEAR_BUILD_DELAY} seconds after clearing walls...") # Debugging delay
            time.sleep(CLEAR_BUILD_DELAY)
            # Step simulation after delay to ensure Webots processes the time.sleep
            self.supervisor.step(1)
            # print("Delay complete.") # Debugging delay


        self.base_wall_distance = ROBOT_RADIUS * clearance
        print(f"Attempting to build tunnel with clearance factor: {clearance:.2f}")

        angle_min, angle_max = angle_range
        num_curves = min(num_curves, MAX_NUM_CURVES) # Cap number of curves
        num_obstacles = min(num_obstacles, MAX_NUM_OBSTACLES) # Cap number of obstacles

        # Determine total length based on number of segments
        # A simple approach: total length is proportional to number of segments (curves + straights)
        total_segments = 1 + num_curves + (1 if num_curves > 0 else 0) # Initial straight + curves + straight after last curve (if any)
        segment_length = BASE_WALL_LENGTH # Keep base segment length consistent

        # --- Set Initial Transformation Matrix to start at a boundary ---
        # Starting at MAP_X_MIN, centered in Y, heading towards positive X (angle 0)
        # Adjusted initial_pos to start 2 * ROBOT_RADIUS inside the MAP_X_MIN boundary
        initial_pos = np.array([MAP_X_MIN + 2 * ROBOT_RADIUS, (MAP_Y_MIN + MAP_Y_MAX) / 2.0, 0.0])
        initial_heading = 0.0 # Pointing towards positive X
        T = np.eye(4)
        T[:3, 3] = initial_pos
        T[:3, :3] = self._rotation_z(initial_heading)[:3, :3] # Set initial rotation

        start_pos = T[:3, 3].copy()
        segments_data = []

        # --- Build the Initial Straight Segment ---
        # Check if the end point is within bounds first
        if not self._within_bounds(T, segment_length):
            print("[ERROR] Initial straight segment end point out of bounds. Cannot build tunnel.")
            return None, None, 0, None

        segment_start_pos = T[:3, 3].copy()
        # Check if the straight segment path crosses boundaries
        if not self._add_straight(T, segment_length):
             print("[ERROR] Initial straight segment crosses boundary. Retrying tunnel generation.")
             return None, None, 0, None # Indicate failure

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
                print(f"[WARNING] Curve {i+1} end point out of bounds, stopping tunnel generation.")
                break # Stop adding segments for this tunnel attempt

            # Check if the curved segment path crosses boundaries
            if not self._add_curve(T, angle, segment_length, segments_data):
                 print(f"[ERROR] Curve {i+1} crosses boundary. Retrying tunnel generation.")
                 return None, None, 0, None # Indicate failure

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
                 return None, None, 0, None # Indicate failure

            segment_end_pos = T[:3, 3].copy()
            segment_heading = math.atan2(T[1, 0], T[0, 0])
            segments_data.append((segment_start_pos, segment_end_pos, segment_heading, segment_length))

        # Add a final straight segment if the last segment was a curve and num_curves > 0
        # Check if the last added segment was part of a curve (segments_data length will be > total_segments if curves were added)
        # A simpler check: if the last segment added was the last curve's final sub-segment, and no straight follows, we are done.
        if num_curves > 0 and len(segments_data) > (1 + num_curves):
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
                      return None, None, 0, None
        elif num_curves == 0 and len(segments_data) == 1:
            # If it was just a straight tunnel, segments_data has 1 entry. No need for a final straight.
            pass
        elif num_curves > 0 and len(segments_data) == (1 + num_curves):
             # If the last segment added was the last curve's final sub-segment, and no straight follows, we are done.
             pass
        else:
             # This case might indicate an issue with segment tracking or generation flow
             # print(f"[WARNING] Unexpected segment count ({len(segments_data)}) after curve/straight generation (num_curves={num_curves}).") # Reduced print
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
        # Reset timer after building
        self.physics_check_timer = 0.0
        self.last_physics_check_time = time.time()

        # Ensure all walls exist after building
        # check_and_restore_wall_physics now handles the different wall types
        self.check_and_restore_wall_physics()

        return start_pos, end_pos, len(self.walls), final_heading


    # --- Helper methods (within_bounds, within_bounds_after_curve, rotation_z, etc.)
    #     should be included here as they were in your original code ---
    #     (Omitting them in this response for brevity, assuming they are present)

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
        centerline_sub_segment_length = (self.base_wall_distance) * abs(step) # Arc length = radius * angle
        # The length of the straight line segment approximating the arc
        straight_sub_segment_length = 2 * (self.base_wall_distance) * math.sin(abs(step) / 2)

        approx_end_pos = tempT[:3, 3].copy()
        for i in range(CURVE_SUBDIVISIONS):
             tempT_rotated = tempT @ self._rotation_z(step)
             approx_end_pos += tempT_rotated[:3, 0] * straight_sub_segment_length # Accumulate translation based on new heading
             tempT[:] = tempT_rotated # Update rotation for next step

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
            # print(f"Added left entrance wall of length {dist_left:.2f}") # Reduced print

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
            # print(f"Added right entrance wall of length {dist_right:.2f}") # Reduced print


    def _add_main_boundary_walls(self):
        """
        Adds walls around the entire map area.
        These walls are static and do NOT have a Physics node, but DO have a boundingObject.
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
        size_xmax = [MAP_Y_MAX - MAP_X_MIN, WALL_THICKNESS, boundary_wall_height] # Corrected size_xmax length
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

        # print(f"Added main boundary walls.") # Reduced print


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
        # print("[INFO] is_robot_inside_tunnel method called. Consider using is_robot_near_centerline or physical collision detection.") # Reduced print
        robot_xy = np.array(robot_position[:2])
        # Calculate perpendicular directions based on robot heading
        right_dir = np.array([math.cos(heading - math.pi / 2), math.sin(heading - math.pi / 2)])
        left_dir = np.array([math.cos(heading + math.pi / 2), math.sin(heading + math.pi / 2)])

        def check_walls(walls, direction):
            """Helper to check proximity to a set of walls in a given direction."""
            for pos, _, size, wall_type, node in walls: # Added node reference
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
            for pos, _, size, wall_type, node in obstacles: # Added node reference
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
        # print(f"[DEBUG] Inside tunnel (proximity check): {inside} | Right: {near_right}, Left: {near_left}, Obstacle: {near_obstacle}") # Reduced print
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
            # create_wall now returns the node reference
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
                # create_wall now returns the node reference
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


    def _add_obstacles(self, segments_data, num_obstacles):
        """
        Adds obstacles within the generated tunnel segments.
        Obstacles are placed randomly along the centerline of the tunnel segments.
        Obstacles are dynamic and WILL have a Physics node and a boundingObject.
        """
        if num_obstacles <= 0 or not segments_data:
            # print("No obstacles to add or no tunnel segments generated.") # Reduced print
            return

        print(f"Attempting to add {num_obstacles} obstacles.")
        added_obstacles_count = 0
        attempts = 0
        max_attempts = num_obstacles * 10 # Prevent infinite loops

        # Flatten all segments into a single list of points along the centerline for easier sampling
        centerline_points = []
        for start, end, _, _ in segments_data:
            # For straight segments, add start and end. For curves, add sub-segment points if needed.
            # A simple approach: just use the start and end points of the main segments
            centerline_points.append(start)
            centerline_points.append(end)
        # Remove duplicates and sort (optional, but can help)
        # centerline_points = sorted(list(set(tuple(p) for p in centerline_points)))
        # Convert back to numpy arrays
        # centerline_points = [np.array(p) for p in centerline_points]

        # A better approach for sampling points along the centerline:
        # For each segment, generate intermediate points.
        detailed_centerline_points = []
        for start, end, _, length in segments_data:
            num_points = max(2, int(length / (ROBOT_RADIUS * 2))) # Add points based on segment length
            for i in range(num_points):
                t = i / (num_points - 1.0) if num_points > 1 else 0.5
                point = start + t * (end - start)
                detailed_centerline_points.append(point)

        # Ensure we have points to place obstacles
        if not detailed_centerline_points:
            # print("[WARNING] No detailed centerline points generated for obstacle placement.") # Reduced print
            return

        # Obstacle size: fixed thickness, length is related to tunnel width, height is WALL_HEIGHT
        obstacle_thickness = WALL_THICKNESS * 2 # Make obstacles thicker than walls
        obstacle_length = self.base_wall_distance * 1.5 # Obstacle width relative to tunnel width
        obstacle_height = WALL_HEIGHT

        # Keep track of placed obstacle positions to ensure minimum distance
        placed_obstacle_positions = []

        while added_obstacles_count < num_obstacles and attempts < max_attempts:
            attempts += 1
            # Randomly select a point along the detailed centerline
            if not detailed_centerline_points:
                 # print("[WARNING] Ran out of detailed centerline points during obstacle placement.") # Reduced print
                 break # Exit if no points are available

            random_point_index = pyrandom.randint(0, len(detailed_centerline_points) - 1)
            obstacle_center_pos_2d = detailed_centerline_points[random_point_index][:2]
            obstacle_center_pos = np.array([obstacle_center_pos_2d[0], obstacle_center_pos_2d[1], obstacle_height / 2.0])


            # Check minimum distance to already placed obstacles
            too_close_to_existing = False
            for placed_pos in placed_obstacle_positions:
                if np.linalg.norm(obstacle_center_pos[:2] - placed_pos[:2]) < MIN_OBSTACLE_DISTANCE:
                    too_close_to_existing = True
                    break

            if too_close_to_existing:
                # print(f"Attempt {attempts}: Proposed obstacle at {obstacle_center_pos[:2]} is too close to an existing obstacle. Retrying.") # Reduced print
                continue # Try another random point


            # Determine obstacle rotation. For simplicity, align with the nearest segment's heading.
            # Find the segment closest to the proposed obstacle position
            closest_segment_index = -1
            min_dist_to_segment = float('inf')
            for i, (start, end, _, _) in enumerate(segments_data):
                 dist = self.point_to_segment_distance(start[:2], end[:2], obstacle_center_pos[:2])
                 if dist < min_dist_to_segment:
                     min_dist_to_segment = dist
                     closest_segment_index = i

            obstacle_heading = 0.0
            if closest_segment_index != -1:
                 # Get the heading of the closest segment
                 obstacle_heading = segments_data[closest_segment_index][2]
            else:
                 # Default to initial heading if no segments found (shouldn't happen if segments_data is not empty)
                 obstacle_heading = math.atan2(segments_data[0][1][1] - segments_data[0][0][1], segments_data[0][1][0] - segments_data[0][0][0])


            obstacle_rot = (0, 0, 1, obstacle_heading)
            obstacle_size = (obstacle_thickness, obstacle_length, obstacle_height)

            # Check if the obstacle bounding box intersects with any tunnel walls
            # This is a simplified check. A more robust check would involve
            # checking bounding box overlaps or using Webots collision detection API
            # if available and performant during generation.
            # For now, let's just check if the obstacle center is too close to any wall centerline.
            too_close_to_wall = False
            min_wall_distance = self.base_wall_distance - obstacle_length / 2.0 - WALL_THICKNESS / 2.0 - 0.05 # Allow a small buffer
            for pos, _, size, wall_type, node in self.walls:
                 if wall_type in ['left', 'right']:
                      wall_center_2d = pos[:2]
                      dist_to_wall_center = np.linalg.norm(obstacle_center_pos[:2] - wall_center_2d)
                      # Check if the distance is too close to the wall centerline distance
                      if abs(dist_to_wall_center - self.base_wall_distance) < (obstacle_length/2 + WALL_THICKNESS/2 + 0.05):
                           too_close_to_wall = True
                           # print(f"Attempt {attempts}: Proposed obstacle at {obstacle_center_pos[:2]} is too close to a tunnel wall. Retrying.") # Reduced print
                           break

            if too_close_to_wall:
                 continue # Try another random point


            # If checks pass, create the obstacle wall
            # create_wall now returns the node reference
            obstacle_node = self.create_wall(obstacle_center_pos, obstacle_rot, obstacle_size, wall_type='obstacle')

            if obstacle_node:
                added_obstacles_count += 1
                placed_obstacle_positions.append(obstacle_center_pos)
                # print(f"Added obstacle {added_obstacles_count}/{num_obstacles} at {obstacle_center_pos[:2]}.") # Reduced print
            else:
                 print(f"[ERROR] Failed to create obstacle node at {obstacle_center_pos[:2]}.")


        if added_obstacles_count < num_obstacles:
            print(f"[WARNING] Only added {added_obstacles_count} out of {num_obstacles} requested obstacles after {attempts} attempts.")
        else:
            print(f"Successfully added {added_obstacles_count} obstacles.")


    def _check_segment_intersection_with_boundaries(self, p1, p2):
        """
        Checks if the 2D line segment from p1 to p2 intersects any of the map boundaries.
        A segment is considered to cross a boundary if one endpoint is strictly on one side
        and the other endpoint is strictly on the other side.
        Segments starting or ending on a boundary are NOT considered to be crossing.

        p1, p2: numpy arrays [x, y, z] (z is ignored for 2D check)
        Returns True if intersects, False otherwise.
        """
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]

        # Check if either endpoint is strictly outside the map boundaries
        # This check is still valid: if an endpoint is *strictly* outside, it's a problem.
        if x1 < MAP_X_MIN or x1 > MAP_X_MAX or y1 < MAP_Y_MIN or y1 > MAP_Y_MAX or \
           x2 < MAP_X_MIN or x2 > MAP_X_MAX or y2 < MAP_Y_MIN or y2 > MAP_Y_MAX:
            # print(f"Endpoint strictly outside bounds: {p1[:2]} or {p2[:2]}") # Reduced print
            return True


        # Check for strict intersection with vertical boundaries (x = constant)
        for x_boundary in [MAP_X_MIN, MAP_X_MAX]:
            # A strict crossing occurs if one endpoint is strictly less than the boundary
            # and the other is strictly greater than the boundary.
            if (x1 < x_boundary and x2 > x_boundary) or (x2 < x_boundary and x1 > x_boundary):
                # Calculate y-coordinate at the intersection point
                if (x2 - x1) != 0: # Avoid division by zero for vertical segments
                    y_intersection = y1 + (y2 - y1) * (x_boundary - x1) / (x2 - x1)
                    # Check if the intersection point's y-coordinate is within the segment's y-range (inclusive)
                    if (min(y1, y2) <= y_intersection <= max(y1, y2)): # Corrected min/max to use y1, y2
                        # Also check if the intersection point is within the map's y-bounds
                        if MAP_X_MIN <= y_intersection <= MAP_Y_MAX:
                            # print(f"Segment from {p1[:2]} to {p2[:2]} strictly intersects vertical boundary at x={x_boundary}, y={y_intersection:.2f}") # Reduced print
                            return True

        # Check for strict intersection with horizontal boundaries (y = constant)
        for y_boundary in [MAP_Y_MIN, MAP_Y_MAX]:
            # A strict crossing occurs if one endpoint is strictly less than the boundary
            # and the other is strictly greater than the boundary.
            if (y1 < y_boundary and y2 > y_boundary) or (y2 < y_boundary and y1 > y_boundary):
                 # Calculate x-coordinate at the intersection point
                if (y2 - y1) != 0: # Avoid division by zero for horizontal segments
                    x_intersection = x1 + (x2 - x1) * (y_boundary - y1) / (y2 - y1)
                    # Check if the intersection point's x-coordinate is within the segment's x-range (inclusive)
                    if (min(x1, x2) <= x_intersection <= max(x1, x2)):
                         # Also check if the intersection point is within the map's x-bounds
                        if MAP_X_MIN <= x_intersection <= MAP_X_MAX:
                            # print(f"Segment from {p1[:2]} to {p2[:2]} strictly intersects horizontal boundary at y={y_boundary}, x={x_intersection:.2f}") # Reduced print
                            return True

        return False # No strict intersection found


    def check_and_restore_wall_physics(self):
        """
        Periodically checks if wall nodes still exist and have the correct physics configuration
        (boundingObject for all, Physics node for obstacles).
        If a node is missing or misconfigured, attempts to log a warning or potentially re-create it.
        """
        current_time = time.time()
        # Use supervisor.getTime() for simulation time if preferred, but wall creation is often
        # done based on real time or simulation step count. Using time.time() for simplicity
        # if the check interval is large relative to the simulation step.
        # If CHECK_PHYSICS_INTERVAL is small, use supervisor.getTime()
        # elapsed_sim_time = self.supervisor.getTime() - self.last_physics_check_time # Requires storing last check sim time
        # if elapsed_sim_time < CHECK_PHYSICS_INTERVAL:
        #     return # Not time to check yet

        # Using real time check for now
        if current_time - self.last_physics_check_time < CHECK_PHYSICS_INTERVAL:
             return # Not time to check yet


        print(f"[INFO] Performing periodic physics check ({current_time:.2f})...")
        issues_found = 0
        restored_count = 0

        # Iterate through the walls we *think* we created
        walls_to_recheck = self.walls[:] # Create a copy to iterate

        # Clear lists before repopulating based on current scene state
        self.walls = []
        self.left_walls = []
        self.right_walls = []
        self.obstacles = []

        # Re-populate lists by checking the scene tree for nodes we created
        # We can use getFromDef based on the stored DEF names, but iterating
        # the root children is more robust if DEF names were somehow lost or changed.
        # Let's stick to iterating root children and matching by name prefix.
        nodes_in_scene = {}
        # Get all nodes in the root children that start with "WALL_" (using the new DEF prefix)
        for i in range(self.root_children.getCount()):
             node = self.root_children.getMFNode(i)
             if node and node.getDefName().startswith("TUNNEL_WALL_"): # Use the new prefix
                  nodes_in_scene[node.getDefName()] = node

        # Now, compare our expected walls with what's in the scene
        # We need to map the old node references (from walls_to_recheck) to the nodes_in_scene by name.
        # This is tricky if the node was re-created with a new timestamp name.
        # A better approach is to just check the nodes currently in the scene that match our naming pattern.
        # We'll lose track of *which* specific wall was intended if it's missing, but we can check if walls
        # matching our pattern exist and have the correct properties.

        # Let's simplify the check: just iterate through nodes in the scene that look like our walls
        # and check their properties. We'll rely on the re-creation logic in create_wall if something is missing.

        # Rebuild internal lists based on current scene state
        for node_name, actual_node in nodes_in_scene.items():
             # Attempt to infer wall_type from the name
             wall_type = None
             if "WALL_BOUNDARY_" in node_name: wall_type = 'boundary'
             elif "WALL_ENTRANCE_" in node_name: wall_type = 'entrance'
             elif "WALL_LEFT_" in node_name: wall_type = 'left'
             elif "WALL_RIGHT_" in node_name: wall_type = 'right'
             elif "WALL_OBSTACLE_" in node_name: wall_type = 'obstacle'

             if wall_type:
                  # Check for BoundingObject (all our walls should have one)
                  bounding_object_field = actual_node.getField("boundingObject")
                  has_bounding_object = (bounding_object_field and bounding_object_field.getSFNode())

                  # Check for Physics node (only obstacles should have one)
                  physics_field = actual_node.getField("physics")
                  has_physics_node = (physics_field and physics_field.getSFNode())

                  is_okay = True
                  if not has_bounding_object:
                       issues_found += 1
                       is_okay = False
                       print(f"[WARNING] Wall node '{node_name}' missing BoundingObject.")

                  if wall_type == 'obstacle':
                      if not has_physics_node:
                           issues_found += 1
                           is_okay = False
                           print(f"[WARNING] Obstacle node '{node_name}' missing Physics node.")
                  else: # Static walls
                      if has_physics_node:
                           issues_found += 1
                           is_okay = False
                           print(f"[WARNING] Static wall node '{node_name}' has unexpected Physics node.")

                  if is_okay:
                       # Re-add to internal lists if it looks correct
                       # Note: We don't have the original pos, rot, size here.
                       # We could get them from the node, but for the purpose of just
                       # tracking which nodes exist, the node reference is enough.
                       # If full wall data is needed later, we might need a different storage approach.
                       # For now, let's just store the node reference and type.
                       wall_data = (None, None, None, wall_type, actual_node) # Store partial data
                       self.walls.append(wall_data)
                       if wall_type == 'left':
                           self.left_walls.append(wall_data)
                       elif wall_type == 'right':
                           self.right_walls.append(wall_data)
                       elif wall_type == 'obstacle':
                           self.obstacles.append(wall_data)
                  else:
                       # Problematic node found - consider re-creation (this is complex)
                       # For now, just log the warning. Automatic re-creation of a node
                       # with the exact same properties and position if it's just
                       # missing a field is difficult via API. It's better to ensure
                       # creation is correct initially.
                       pass # Just log warnings for misconfigured nodes found in scene

        # The check_and_restore_wall_physics method is primarily for verifying
        # that nodes *remain* in the scene and have their basic physics setup.
        # The more critical part for the "deletion" issue is ensuring _clear_walls
        # works and that create_wall successfully adds nodes and returns valid references.


        # Re-check if the number of walls in our internal list matches the number
        # of walls we expect based on the last build_tunnel call (approximate check)
        # This is hard to do reliably as the number of segments can vary.
        # Let's trust the iteration through scene nodes for now.


        if issues_found > 0:
             print(f"[INFO] Physics check completed. Found {issues_found} potential issues.")
             print("[INFO] All walls should have a BoundingObject for collision detection.")
             print("[INFO] Static walls (boundary, entrance, left, right) should NOT have Physics nodes.")
             print("[INFO] Obstacle walls SHOULD have Physics nodes.")
             print("[INFO] Contact properties (friction, bounce) should be defined in the WorldInfo node's contactProperties field.")
        else:
             print("[INFO] Physics check completed. No issues found.")

        self.last_physics_check_time = current_time # Update the timer


    # --- Add other helper methods like _add_obstacles here if they were in your original code ---
    #     (Omitting them in this response for brevity, assuming they are present)

    # Modified _add_obstacles to accept num_obstacles
    def _add_obstacles(self, segments_data, num_obstacles):
        """
        Place num_obstacles perpendicular walls into the middle straight segments.
        segments_data: list of (start_pos, end_pos, heading, length) for all segments
        num_obstacles (int): The number of obstacles to place.
        """
        if num_obstacles <= 0 or not segments_data:
            # print("No obstacles to add or no tunnel segments generated.") # Reduced print
            return

        print(f"Attempting to add {num_obstacles} obstacles.")
        added_obstacles_count = 0
        attempts = 0
        max_attempts = num_obstacles * 20 # Increased attempts

        # Filter for straight segments that are not the very first or very last
        # This assumes segments_data contains alternating straight and curved segments (or just straights)
        # and the first/last entries correspond to the entrance/exit straights.
        # If the tunnel starts on a boundary, the first segment is the entrance straight.
        # If the tunnel ends on a boundary, the last segment is the exit straight.
        # We want to place obstacles in internal straight sections.
        # A simple heuristic: consider straight segments that are not the first AND not the last,
        # and are at an angle close to 0 (meaning they are aligned with the initial X axis).
        # This might need refinement depending on how complex the tunnel path can become.

        internal_straight_segments = []
        for i, seg in enumerate(segments_data):
             start, end, heading, seg_len = seg
             # Check if it's a straight segment (heading close to 0 or pi)
             # and not the very first or very last segment overall.
             # This logic might need adjustment if the tunnel can end with a curve.
             is_straight = abs(heading) < 1e-3 or abs(heading - math.pi) < 1e-3 or abs(heading + math.pi) < 1e-3
             is_not_first = i > 0
             is_not_last = i < len(segments_data) - 1

             if is_straight and is_not_first and is_not_last and seg_len > MIN_OBSTACLE_DISTANCE * 2: # Ensure segment is long enough
                 internal_straight_segments.append((i, seg)) # Store index and segment


        if not internal_straight_segments:
            print("Not enough suitable internal straight segments for obstacles.")
            return

        used_segment_indices = set()
        placed_positions = []

        tunnel_half_width = self.base_wall_distance
        # Obstacle length should be less than the tunnel width to allow passing
        # It spans from one side of the tunnel to the other, leaving MIN_ROBOT_CLEARANCE
        obstacle_length = 2 * tunnel_half_width - MIN_ROBOT_CLEARANCE - WALL_THICKNESS
        # Ensure obstacle length is positive and reasonable
        if obstacle_length <= 0.1: # Added a small minimum length
             print(f"Obstacle length is too small ({obstacle_length:.2f}). Cannot place obstacles.")
             return


        while added_obstacles_count < num_obstacles and attempts < max_attempts:
            attempts += 1
            # Select a random index from the available internal straight segments
            available_choices = [idx for idx, seg in internal_straight_segments if idx not in used_segment_indices]
            if not available_choices:
                # print("Ran out of available segments for obstacles during placement attempts.") # Reduced print
                break

            chosen_segment_idx_in_segments_data = pyrandom.choice(available_choices)
            # Don't mark segment as used until an obstacle is successfully placed there.
            # used_segment_indices.add(chosen_segment_idx_in_segments_data) # Moved below

            # Find the actual segment data using the index
            chosen_segment = segments_data[chosen_segment_idx_in_segments_data]
            start, end, heading, seg_len = chosen_segment

            # Place obstacle somewhere along the segment, avoiding ends and other obstacles
            min_dist_along_segment = MIN_OBSTACLE_DISTANCE / 2.0 + WALL_THICKNESS / 2.0 # Avoid placing too close to segment start/end
            max_dist_along_segment = seg_len - min_dist_along_segment

            if max_dist_along_segment <= min_dist_along_segment:
                 # print(f"Segment {chosen_segment_idx_in_segments_data} is too short for obstacle placement.") # Reduced print
                 used_segment_indices.add(chosen_segment_idx_in_segments_data) # Mark as unusable for obstacles
                 continue # Choose another segment

            # Generate a random distance along the segment
            d = pyrandom.uniform(min_dist_along_segment, max_dist_along_segment)
            dir_vec = np.array([math.cos(heading), math.sin(heading), 0.0])
            pos = np.array(start) + dir_vec * d

            # Decide which side of the centerline the obstacle is placed on
            side = pyrandom.choice([-1, +1])
            perp = np.array([-dir_vec[1], dir_vec[0], 0.0]) # Perpendicular vector

            # Calculate the shift distance from the centerline
            shift_distance_from_centerline = tunnel_half_width - MIN_ROBOT_CLEARANCE - obstacle_length / 2.0
            # Ensure the shift distance is valid
            if shift_distance_from_centerline < 0:
                 # print(f"[WARNING] Calculated shift distance for obstacle is negative ({shift_distance_from_centerline:.2f}) in segment {chosen_segment_idx_in_segments_data}. Adjusting.") # Reduced print
                 shift_distance_from_centerline = 0

            shift = side * shift_distance_from_centerline

            pos += perp * shift
            pos[2] = WALL_HEIGHT / 2.0 # Set Z height

            # Obstacle rotation is perpendicular to the segment heading
            obstacle_rot_heading = heading + math.pi / 2.0 # Rotate 90 degrees from segment heading
            rot = (0.0, 0.0, 1.0, obstacle_rot_heading)

            # Obstacle size: thickness along segment, length across tunnel, height
            size = (WALL_THICKNESS, obstacle_length, WALL_HEIGHT)

            # Check for proximity to already placed obstacles
            too_close_to_placed = False
            for placed_pos in placed_positions:
                if np.linalg.norm(pos[:2] - placed_pos) < MIN_OBSTACLE_DISTANCE:
                    too_close_to_placed = True
                    # print("Skipping obstacle—too close to another placed obstacle.") # Reduced print
                    break

            if too_close_to_placed:
                continue # Try another random point/segment

            # Check if the obstacle bounding box intersects with any tunnel walls
            # This is a simplified check.
            too_close_to_wall = False
            # Check distance to the expected wall positions for this segment
            expected_left_wall_center = pos[:2] + perp[:2] * (tunnel_half_width - WALL_THICKNESS / 2.0) # Approx center of left wall
            expected_right_wall_center = pos[:2] - perp[:2] * (tunnel_half_width - WALL_THICKNESS / 2.0) # Approx center of right wall

            # Check distance from obstacle center to expected wall centerlines
            dist_to_left_wall_centerline = np.linalg.norm(pos[:2] - expected_left_wall_center)
            dist_to_right_wall_centerline = np.linalg.norm(pos[:2] - expected_right_wall_center)

            # Obstacle half-length + wall half-thickness + buffer
            min_allowed_distance_to_wall_centerline = obstacle_length / 2.0 + WALL_THICKNESS / 2.0 + 0.05

            if dist_to_left_wall_centerline < min_allowed_distance_to_wall_centerline or \
               dist_to_right_wall_centerline < min_allowed_distance_to_wall_centerline:
                 too_close_to_wall = True
                 # print(f"Attempt {attempts}: Proposed obstacle at {pos[:2]} is too close to a tunnel wall centerline. Retrying.") # Reduced print


            if too_close_to_wall:
                 continue # Try another random point/segment


            # If checks pass, create the obstacle wall
            obstacle_node = self.create_wall(pos, rot, size, wall_type='obstacle')

            if obstacle_node:
                added_obstacles_count += 1
                placed_positions.append(pos[:2].copy()) # Store 2D position for proximity checks
                used_segment_indices.add(chosen_segment_idx_in_segments_data) # Mark segment as used
                # print(f"Added obstacle {added_obstacles_count}/{num_obstacles} at {pos[:2]}.") # Reduced print
            else:
                 print(f"[ERROR] Failed to create obstacle node at {pos[:2]}.")


        if added_obstacles_count < num_obstacles:
            print(f"[WARNING] Only added {added_obstacles_count} out of {num_obstacles} requested obstacles after {attempts} attempts.")
        else:
            print(f"Successfully added {added_obstacles_count} obstacles.")


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
        """Checks if the end point of a straight segment is within map bounds."""
        end = T[:3, 3] + T[:3, 0] * length
        return MAP_X_MIN <= end[0] <= MAP_X_MAX and MAP_Y_MIN <= end[1] <= MAP_Y_MAX

    def _within_bounds_after_curve(self, T, angle, seg_len):
        """Checks if the end point after a curved segment is within map bounds."""
        tempT = T.copy()
        step = angle / CURVE_SUBDIVISIONS
        # The length of the centerline arc for one subdivision
        centerline_sub_segment_length = (self.base_wall_distance) * abs(step) # Arc length = radius * angle
        # The length of the straight line segment approximating the arc
        straight_sub_segment_length = 2 * (self.base_wall_distance) * math.sin(abs(step) / 2)

        approx_end_pos = tempT[:3, 3].copy()
        for _ in range(CURVE_SUBDIVISIONS):
            # Rotate first, then translate
            tempT_rotated = tempT @ self._rotation_z(step)
            approx_end_pos += tempT_rotated[:3, 0] * straight_sub_segment_length # Accumulate translation based on new heading
            tempT[:] = tempT_rotated # Update rotation for next step

        end = approx_end_pos
        return MAP_X_MIN <= end[0] <= MAP_X_MAX and MAP_Y_MIN <= end[1] <= MAP_Y_MAX
