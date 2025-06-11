# curriculum.py
import os
import pickle
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


def _qualify_candidates(candidates, sim_mgr, qualification_stage, total_stages):
    """
    Testa uma lista de indivíduos candidatos em todos os estágios anteriores.
    Retorna apenas os indivíduos que passam em todos os testes.
    """
    if qualification_stage <= 1:
        return candidates  # No stages prior to stage 1 to qualify in

    qualified_individuals = []
    previous_stages = list(range(1, qualification_stage))
    print(f"--- A qualificar {len(candidates)} melhores candidatos nos níveis anteriores: {previous_stages} ---")

    for ind in candidates:
        # Assume an individual is qualified until proven otherwise
        is_qualified = True
        for stage_to_test in previous_stages:
            # The second element returned by run_experiment_with_network is 'succeeded'
            _, succeeded = sim_mgr.run_experiment_with_network(ind, stage_to_test, total_stages)
            if not succeeded:
                is_qualified = False
                break  # No need to test further stages for this individual

        if is_qualified:
            qualified_individuals.append(ind)

    print(f"--- Qualificação Concluída: {len(qualified_individuals)}/{len(candidates)} candidatos qualificados. ---")
    return qualified_individuals


def run_curriculum(
        supervisor,
        resume_training: bool = False,
        pop_size: int = 30,
        success_threshold: float = 0.5,
        max_generations: int = 1000,
        hidden_size: int = 16,
        mutation_rate: float = 0.15,
        elitism: int = 2,
        top_n_to_qualify: int = 10
):
    """
    Executa o treino com uma abordagem híbrida: avaliação rápida no estágio atual
    e qualificação rigorosa apenas para os melhores candidatos.
    """
    sim_mgr = SimulationManager(supervisor)
    input_size = len(np.nan_to_num(sim_mgr.lidar.getRangeImage(), nan=0.0))
    output_size = 2

    # --- Inicialização ou Carregamento ---
    population, best_overall_individual, start_stage = None, None, 1
    checkpoint_data = _load_checkpoint()
    if resume_training and checkpoint_data:
        population = checkpoint_data.get('population')
        best_overall_individual = checkpoint_data.get('best_individual')
        start_stage = checkpoint_data.get('stage', 1)

    if population is None:
        population = NeuralPopulation(pop_size, input_size, hidden_size, output_size, mutation_rate, elitism)

    # --- Loop Principal do Treino ---
    try:
        current_stage = start_stage
        while current_stage <= MAX_DIFFICULTY_STAGE:
            print(f"\n\n{'=' * 20} INICIANDO ESTÁGIO DE DIFICULDADE {current_stage} {'=' * 20}")

            for gen_in_stage in range(1, max_generations + 1):
                print(f"\n--- Geração {gen_in_stage}/{max_generations} (Estágio {current_stage}) ---")

                # 1. AVALIAÇÃO RÁPIDA: Avalia a população inteira APENAS no estágio atual
                print(f"Avaliando população no nível de dificuldade: {current_stage}")
                # A função 'evaluate' atualiza 'ind.successes' para cada indivíduo
                population.evaluate(sim_mgr, [current_stage], MAX_DIFFICULTY_STAGE)

                # 2. SELEÇÃO E QUALIFICAÇÃO DA ELITE
                # Ordena a população com base na fitness do estágio atual
                population.individuals.sort(key=lambda ind: ind.fitness if ind.fitness is not None else -np.inf,
                                            reverse=True)
                top_candidates = population.individuals[:top_n_to_qualify]

                # **NOVA LÓGICA**: Filtra a elite para incluir apenas quem teve sucesso no estágio ATUAL
                successful_top_candidates = [ind for ind in top_candidates if ind.successes > 0]
                print(
                    f"--- {len(successful_top_candidates)}/{len(top_candidates)} dos melhores candidatos tiveram sucesso no estágio atual. ---")

                # Qualifica apenas os candidatos de topo que TIVERAM SUCESSO no estágio atual
                qualified_pool = _qualify_candidates(successful_top_candidates, sim_mgr, current_stage,
                                                     MAX_DIFFICULTY_STAGE)

                # 3. DEFINIÇÃO DO GRUPO DE PAIS
                parent_pool = qualified_pool
                if not parent_pool:
                    # Fallback: se ninguém se qualificar totalmente, usa os melhores (por fitness) do estágio atual.
                    print(
                        "[WARNING] Nenhum candidato totalmente qualificado (sucesso no atual + anteriores). Usando os melhores do estágio atual como pais.")
                    parent_pool = top_candidates

                # A próxima geração é criada a partir do grupo de pais definido
                population.create_next_generation(parent_pool=parent_pool)

                # 4. VERIFICAÇÃO DE AVANÇO
                # A taxa de avanço é baseada em quantos dos melhores (top_n) se qualificaram totalmente
                qualification_rate = len(qualified_pool) / len(top_candidates) if top_candidates else 0
                print(f"[ESTATÍSTICAS] Taxa de Qualificação Total da Elite: {qualification_rate:.2%}")

                # Guarda o melhor indivíduo e o checkpoint
                gen_best = population.get_best_individual()
                if best_overall_individual is None or (
                        gen_best.fitness is not None and best_overall_individual.fitness is not None and gen_best.fitness > best_overall_individual.fitness):
                    best_overall_individual = gen_best
                    sim_mgr.save_model(best_overall_individual,
                                       filename=f"best_model_stage_{current_stage}_gen_{gen_in_stage}.pkl")

                _save_checkpoint({
                    'population': population,
                    'best_individual': best_overall_individual,
                    'stage': current_stage
                })

                # Condição de avanço baseada na taxa de qualificação total da elite
                if qualification_rate >= success_threshold and gen_in_stage > 5:
                    print(
                        f"[AVANÇO] Taxa de qualificação ({qualification_rate:.2%}) atingiu o limiar ({success_threshold:.2%}). Avançando.")
                    current_stage += 1
                    break

            # Se o loop de gerações terminar sem atingir o limiar
            else:
                if current_stage < MAX_DIFFICULTY_STAGE:
                    print(
                        f"[AVANÇO FORÇADO] Limite de gerações atingido. Avançando para o estágio {current_stage + 1}.")
                    current_stage += 1
                else:
                    print("[FIM DO TREINO] Currículo completo.")
                    break

    except KeyboardInterrupt:
        print("\nTreino interrompido. A guardar checkpoint...")
    finally:
        if population and best_overall_individual:
            _save_checkpoint({
                'population': population,
                'best_individual': best_overall_individual,
                'stage': current_stage
            })
        print("Treino concluído ou interrompido.")

    return best_overall_individual
