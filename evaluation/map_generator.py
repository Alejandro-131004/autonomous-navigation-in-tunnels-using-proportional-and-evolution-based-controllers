import os
import numpy as np
import pickle
# A importação agora é só da função que precisamos
from environment.configuration import get_stage_parameters


def generate_maps(
        maps_output_dir: str = "evaluation/maps",
        num_maps_per_difficulty: int = 100,
        total_difficulty_stages: int = 20  # O valor padrão agora é 20
) -> list:
    """
    Gera um conjunto de mapas de túnel para avaliação, guardando os seus parâmetros.
    """
    print(f"--- A gerar {num_maps_per_difficulty * total_difficulty_stages} mapas ---")
    os.makedirs(maps_output_dir, exist_ok=True)

    map_files = []

    for difficulty_level in range(1, total_difficulty_stages + 1):
        if os.environ.get('ROBOT_DEBUG_MODE') == '1':
            print(
                f"A gerar {num_maps_per_difficulty} mapas para a Dificuldade {difficulty_level}/{total_difficulty_stages}...")

        for i in range(num_maps_per_difficulty):
            # --- CORREÇÃO AQUI ---
            # A chamada à função agora passa apenas o argumento 'difficulty_level',
            # como esperado pela nova versão da função.
            num_curves, angle_range_rad, clearance_factor, num_obstacles, obstacle_types = \
                get_stage_parameters(difficulty_level)

            # Guardamos os parâmetros em radianos, como a função agora retorna
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

    print(f"Geração de mapas concluída. {len(map_files)} mapas guardados em '{maps_output_dir}'.")
    return map_files


if __name__ == "__main__":
    # Exemplo de uso autónomo
    MAPS_OUTPUT_DIR = "evaluation/maps_standalone_test"
    NUM_MAPS_PER_DIFFICULTY = 10
    TOTAL_DIFFICULTY_STAGES = 20

    generated_map_files = generate_maps(
        maps_output_dir=MAPS_OUTPUT_DIR,
        num_maps_per_difficulty=NUM_MAPS_PER_DIFFICULTY,
        total_difficulty_stages=TOTAL_DIFFICULTY_STAGES
    )
    print(f"\nTeste de geração de mapas autónomo concluído. Mapas guardados em {MAPS_OUTPUT_DIR}")
