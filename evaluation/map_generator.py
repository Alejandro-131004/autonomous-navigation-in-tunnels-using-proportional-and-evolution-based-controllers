import os
import numpy as np
import pickle
from environment.configuration import get_stage_parameters


def generate_maps(
        maps_output_dir: str = "evaluation/maps",
        num_maps_per_difficulty: int = 100,
        total_difficulty_stages: int = 20
) -> list:
    """
    Generates a set of tunnel maps for evaluation and saves their parameters.

    Args:
        maps_output_dir (str): Output directory where map files will be stored.
        num_maps_per_difficulty (int): Number of maps to generate per difficulty level.
        total_difficulty_stages (int): Total number of difficulty stages to cover.

    Returns:
        list: List of file paths to the generated map parameter files.
    """
    print(f"--- Generating {num_maps_per_difficulty * total_difficulty_stages} maps ---")
    os.makedirs(maps_output_dir, exist_ok=True)

    map_files = []

    for difficulty_level in range(0, total_difficulty_stages + 1):
        if os.environ.get('ROBOT_DEBUG_MODE') == '1':
            print(
                f"Generating {num_maps_per_difficulty} maps for Difficulty {difficulty_level}/{total_difficulty_stages}...")

        for i in range(num_maps_per_difficulty):
            # Get parameters for the current difficulty level
            num_curves, angle_range_rad, clearance_factor, num_obstacles, obstacle_types, _ = \
                get_stage_parameters(difficulty_level)

            # Save map parameters in radians as returned by the function
            map_params = {
                "difficulty_level": difficulty_level,
                "num_curves": num_curves,
                "angle_range": angle_range_rad,
                "clearance_factor": clearance_factor,
                "num_obstacles": num_obstacles,
                "obstacle_types": obstacle_types
            }

            map_filename = os.path.join(maps_output_dir, f"map_D{difficulty_level}_N{i + 1}.pkl")
            with open(map_filename, 'wb') as f:
                pickle.dump(map_params, f)
            map_files.append(map_filename)

    print(f"Map generation complete. {len(map_files)} maps saved in '{maps_output_dir}'.")
    return map_files


if __name__ == "__main__":
    # Standalone test example
    MAPS_OUTPUT_DIR = "evaluation/maps_standalone_test"
    NUM_MAPS_PER_DIFFICULTY = 10
    TOTAL_DIFFICULTY_STAGES = 20

    generated_map_files = generate_maps(
        maps_output_dir=MAPS_OUTPUT_DIR,
        num_maps_per_difficulty=NUM_MAPS_PER_DIFFICULTY,
        total_difficulty_stages=TOTAL_DIFFICULTY_STAGES
    )
    print(f"\nStandalone map generation test complete. Maps saved in {MAPS_OUTPUT_DIR}")