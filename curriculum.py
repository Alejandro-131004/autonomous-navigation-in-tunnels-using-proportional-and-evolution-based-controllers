# curriculum.py
import os
import pickle
import random
import numpy as np
from environment.configuration import MAX_DIFFICULTY_STAGE
from environment.simulation_manager import SimulationManager
from optimizer.neuralpopulation import NeuralPopulation

# --- Configuração do Checkpoint ---
CHECKPOINT_FILE = "saved_models/checkpoint.pkl"


def _save_checkpoint(data):
    """Guarda o estado do treino num ficheiro de checkpoint."""
    try:
        os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"[ERROR] Não foi possível guardar o checkpoint: {e}")


def _load_checkpoint():
    """Carrega o estado do treino a partir de um ficheiro de checkpoint."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'rb') as f:
                data = pickle.load(f)
            print(f"|--- Checkpoint carregado de {CHECKPOINT_FILE} ---|")
            return data
        except Exception as e:
            print(f"[ERROR] Não foi possível carregar o checkpoint: {e}")
    return None


def _qualify_candidates(candidates, sim_mgr, qualification_stage):
    """Testa uma lista de indivíduos candidatos em todas as fases anteriores."""
    if qualification_stage <= 1:
        return candidates

    qualified_individuals = []
    previous_stages = list(range(1, qualification_stage))
    print(f"--- A qualificar {len(candidates)} melhores candidatos nos níveis anteriores: {previous_stages} ---")

    for ind in candidates:
        is_qualified = all(
            sim_mgr.run_experiment_with_network(ind, stage_to_test, MAX_DIFFICULTY_STAGE)[1]
            for stage_to_test in previous_stages
        )
        if is_qualified:
            qualified_individuals.append(ind)

    print(f"--- Qualificação Concluída: {len(qualified_individuals)}/{len(candidates)} candidatos qualificados. ---")
    return qualified_individuals


def run_curriculum(
        supervisor,
        resume_training: bool = False,
        pop_size: int = 30,
        success_threshold: float = 0.5,
        max_failed_generations: int = 15,
        hidden_size: int = 16,
        mutation_rate: float = 0.15,
        elitism: int = 2,
        top_n_to_qualify: int = 10  # Número de melhores a entrar na qualificação
):
    sim_mgr = SimulationManager(supervisor)
    input_size = len(np.nan_to_num(sim_mgr.lidar.getRangeImage(), nan=0.0))
    output_size = 2

    # --- Inicialização ou Carregamento ---
    population, best_overall_individual, start_stage, success_rate_rolling_avg = None, None, 1, 0.0
    checkpoint_data = _load_checkpoint()
    if resume_training and checkpoint_data:
        population, best_overall_individual, start_stage, success_rate_rolling_avg = (
            checkpoint_data.get('population'), checkpoint_data.get('best_individual'),
            checkpoint_data.get('stage', 1), checkpoint_data.get('rolling_avg', 0.0)
        )
    if population is None:
        population = NeuralPopulation(pop_size, input_size, hidden_size, output_size, mutation_rate, elitism)

    # --- Loop Principal do Treino ---
    try:
        for stage in range(start_stage, MAX_DIFFICULTY_STAGE + 1):
            print(f"\n\n{'=' * 20} INICIANDO ESTÁGIO DE DIFICULDADE {stage} {'=' * 20}")
            if stage != start_stage: success_rate_rolling_avg = 0.0

            for gen_in_stage in range(1, max_failed_generations + 1):
                # 1. Avalia a população APENAS no nível de dificuldade atual.
                print(f"\n--- Geração {gen_in_stage} (Estágio {stage}) | A avaliar no nível: {stage} ---")
                population.evaluate(sim_mgr, [stage], MAX_DIFFICULTY_STAGE)

                # 2. Ordena a população e seleciona os melhores para qualificação.
                population.individuals.sort(key=lambda ind: ind.fitness, reverse=True)
                top_candidates = population.individuals[:top_n_to_qualify]

                # 3. Qualifica os melhores, testando-os nas fases anteriores.
                qualified_pool = _qualify_candidates(top_candidates, sim_mgr, stage)

                # 4. A próxima geração é criada a partir deste grupo de elite qualificado.
                if not qualified_pool:
                    print(
                        "[WARNING] Nenhum candidato qualificado. A usar os melhores não qualificados para evitar estagnação.")
                    qualified_pool = top_candidates  # Fallback

                population.create_next_generation(parent_pool=qualified_pool)

                # ... (cálculo de estatísticas e gravação) ...
                current_success_rate = sum(
                    1 for ind in population.individuals if ind.fitness > 0 and ind.successes > 0) / pop_size
                success_rate_rolling_avg = 0.7 * success_rate_rolling_avg + 0.3 * current_success_rate
                print(f"[ESTATÍSTICAS] Taxa de Sucesso (aprox.): {current_success_rate:.2%}")

                gen_best = population.get_best_individual()
                if best_overall_individual is None or gen_best.fitness > best_overall_individual.fitness:
                    best_overall_individual = gen_best
                    sim_mgr.save_model(best_overall_individual,
                                       filename=f"best_model_stage_{stage}_gen_{gen_in_stage}.pkl")

                _save_checkpoint({'population': population, 'best_individual': best_overall_individual, 'stage': stage,
                                  'rolling_avg': success_rate_rolling_avg})

                if success_rate_rolling_avg > success_threshold and gen_in_stage > 5:
                    print(f"[AVANÇO] Taxa de sucesso suficiente. A avançar.")
                    break
            else:  # Este `else` pertence ao `for`, executa se o loop não for interrompido por `break`
                if stage < MAX_DIFFICULTY_STAGE:
                    print(f"[AVANÇO FORÇADO] A avançar para o estágio {stage + 1}.")
                else:
                    print("[FIM DO TREINO] Currículo completo.")

    except KeyboardInterrupt:
        print("\nTreino interrompido. A guardar checkpoint...")
    finally:
        if population and best_overall_individual:
            _save_checkpoint({'population': population, 'best_individual': best_overall_individual,
                              'stage': stage if 'stage' in locals() else start_stage,
                              'rolling_avg': success_rate_rolling_avg})
        print("Treino concluído ou interrompido.")

    return best_overall_individual
