import os
import pickle
import sys
from tqdm import tqdm

# Adiciona o diretório raiz ao path para permitir imports de outros módulos do projeto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from environment.configuration import get_stage_parameters, MAX_DIFFICULTY_STAGE


def generate_maps(maps_output_dir="evaluation/maps", num_maps_per_difficulty=100,
                  total_difficulty_stages=MAX_DIFFICULTY_STAGE):
    """
    Gera um conjunto de definições de mapas e guarda-os como ficheiros .pkl.
    """
    if not os.path.exists(maps_output_dir):
        os.makedirs(maps_output_dir)
        print(f"Diretório de mapas criado em: {maps_output_dir}")

    total_maps_to_generate = num_maps_per_difficulty * (total_difficulty_stages + 1)
    print(f"A gerar {total_maps_to_generate} ficheiros de mapa...")

    # A barra de progresso (tqdm) ajuda a visualizar o andamento
    with tqdm(total=total_maps_to_generate, desc="A Gerar Mapas") as pbar:
        for difficulty_level in range(total_difficulty_stages + 1):
            for i in range(num_maps_per_difficulty):

                # --- CORREÇÃO PRINCIPAL ---
                # A chamada a get_stage_parameters foi atualizada para receber os 7 valores.
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
                except Exception as e:
                    print(f"\n[ERRO] Falha ao guardar o mapa {filepath}: {e}")

                pbar.update(1)

    print(f"\nGeração de mapas concluída. {total_maps_to_generate} mapas foram guardados em '{maps_output_dir}'.")


if __name__ == '__main__':
    # Esta parte permite executar o script diretamente do terminal para gerar os mapas
    maps_dir = "evaluation/maps"
    if os.path.exists(maps_dir):
        print(f"A limpar diretório de mapas antigo: {maps_dir}")
        for old_map in os.listdir(maps_dir):
            os.remove(os.path.join(maps_dir, old_map))

    generate_maps()

