import os
import pickle
import sys
from tqdm import tqdm

# Adds the project root directory to sys.path to allow imports from other project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from environment.configuration import get_stage_parameters, MAX_DIFFICULTY_STAGE


def generate_maps(maps_output_dir="evaluation/maps", num_maps_per_difficulty=10,
                  total_difficulty_stages=MAX_DIFFICULTY_STAGE):
    """
    Generates a set of map definitions and saves them as .pkl files.
    Returns a list with the paths to all generated files.
    """
    if not os.path.exists(maps_output_dir):
        os.makedirs(maps_output_dir)

    # Clear old maps to ensure a clean test set
    print(f"Cleaning old maps from '{maps_output_dir}'...")
    for old_map in os.listdir(maps_output_dir):
        os.remove(os.path.join(maps_output_dir, old_map))

    total_maps_to_generate = num_maps_per_difficulty * (total_difficulty_stages + 1)

    # --- FIX: List to store paths of generated files ---
    generated_map_paths = []

    with tqdm(total=total_maps_to_generate, desc="Generating Test Maps") as pbar:
        for difficulty_level in range(total_difficulty_stages + 1):
            for i in range(num_maps_per_difficulty):

                num_curves, curve_angles, clearance, num_obstacles, obstacle_types, passageway_width, straight_length_range = get_stage_parameters(
                    difficulty_level)

                map_params = {
                    'difficulty_level': difficulty_level,
                    'num_curves': num_curves,
                    'curve_angles_list': curve_angles,
                    'clearance_factor': clearance,
                    'num_obstacles': num_obstacles,
                    'obstacle_types': obstacle_types,
                    'passageway_width': passageway_width,
                    'straight_length_range': straight_length_range,
                    'map_index': i
                }

                filename = f"map_diff_{difficulty_level}_index_{i}.pkl"
                filepath = os.path.join(maps_output_dir, filename)

                try:
                    with open(filepath, 'wb') as f:
                        pickle.dump(map_params, f)
                    # Add path to list after successful save
                    generated_map_paths.append(filepath)
                except Exception as e:
                    print(f"\n[ERROR] Failed to save map {filepath}: {e}")

                pbar.update(1)

    print(f"\nMap generation complete. {len(generated_map_paths)} maps were saved to '{maps_output_dir}'.")

    # --- FIX: Return the list of paths ---
    return generated_map_paths


if __name__ == '__main__':
    # This section allows running the script independently if needed
    generate_maps(num_maps_per_difficulty=10)
