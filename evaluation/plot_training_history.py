import matplotlib

# Força o uso de um backend de UI compatível (TkAgg) para evitar erros
matplotlib.use('TkAgg')
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Adiciona o diretório raiz ao path para garantir que os módulos personalizados
# (como as classes de 'Individual') possam ser carregados pelo pickle.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# É necessário importar as classes que foram guardadas no checkpoint para que o pickle as possa reconstruir.
try:
    from optimizer.individualNeural import IndividualNeural
    from optimizer.neuralpopulation import NeuralPopulation
    from optimizer.mlpController import MLPController
    from optimizer.individual import Individual
    from optimizer.population import Population
except ImportError as e:
    print(f"[Aviso] Não foi possível importar algumas classes do projeto: {e}")
    print("Isto pode causar erros se o ficheiro de checkpoint contiver estas classes.")


def plot_history(checkpoint_path):
    """
    Carrega o histórico de treino de um ficheiro de checkpoint e desenha gráficos
    para visualizar a evolução do fitness e das taxas de sucesso.
    """
    if not os.path.exists(checkpoint_path):
        print(f"[ERRO] O ficheiro '{checkpoint_path}' não foi encontrado.")
        return

    try:
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        history = data.get('history', [])
        if not history:
            print("O ficheiro de checkpoint não contém dados de histórico para visualizar.")
            return
    except (pickle.UnpicklingError, EOFError) as e:
        print(f"[ERRO] Falha ao ler o ficheiro de checkpoint: {e}")
        return

    # --- Lógica melhorada para corrigir dados de treino antigos ---
    apply_fix_choice = input(
        "Deseja aplicar a correção para os dados da fase 11? (Remove a fase 11 e ajusta o eixo do tempo) [s/n]: ").lower().strip()
    if apply_fix_choice == 's':
        print("A aplicar correção de dados...")
        history_temp = []
        for entry in history:
            if entry.get('stage') == 11:
                continue  # Ignora todas as entradas da fase 11
            if entry.get('stage') == 12:
                entry['stage'] = 11  # Renomeia a fase 12 para 11
            history_temp.append(entry)

        # --- CORREÇÃO PRINCIPAL: Re-indexa as gerações para remover o espaçamento no gráfico ---
        history_fixed = []
        for i, entry in enumerate(history_temp):
            entry['generation'] = i + 1  # Atribui um número de geração sequencial
            history_fixed.append(entry)

        history = history_fixed
        print("Correção aplicada. O gráfico será contínuo.")

    # Extrai os dados do histórico para os gráficos
    generations = [d['generation'] for d in history]
    fitness_min = [d['fitness_min'] for d in history]
    fitness_avg = [d['fitness_avg'] for d in history]
    fitness_max = [d['fitness_max'] for d in history]
    success_rate_prev = [d.get('success_rate_prev', 0) * 100 for d in history]
    success_rate_curr = [d.get('success_rate_curr', 0) * 100 for d in history]
    stages = [d['stage'] for d in history]

    # --- Criação dos Gráficos (em Inglês) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'Training History - {os.path.basename(checkpoint_path)}', fontsize=16)

    # Gráfico 1: Evolução do Fitness
    ax1.plot(generations, fitness_max, label='Max Fitness', color='green', marker='.', linestyle='-')
    ax1.plot(generations, fitness_avg, label='Average Fitness', color='orange', marker='.', linestyle='-')
    ax1.plot(generations, fitness_min, label='Min Fitness', color='red', marker='.',
             linestyle='--')  # Mantido tracejado como no original
    ax1.fill_between(generations, fitness_min, fitness_max, color='green', alpha=0.1)
    ax1.set_ylabel('Fitness', fontsize=12)
    ax1.set_title('Fitness Evolution per Generation', fontsize=14)
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Adiciona anotações de mudança de fase
    if stages:
        last_stage = -1
        for i, stage in enumerate(stages):
            if stage != last_stage:
                ax1.axvline(x=generations[i], color='grey', linestyle='--', linewidth=1)
                # Verifica se há dados de fitness para evitar erro
                if fitness_max:
                    ax1.text(generations[i], np.max(fitness_max), f' Stage {stage} ', color='blue', rotation=90,
                             verticalalignment='top')
                last_stage = stage

    # Gráfico 2: Taxas de Sucesso
    ax2.plot(generations, success_rate_curr, label='Success Rate (Current Stage)', color='dodgerblue', marker='o',
             markersize=4, linestyle='-')
    ax2.plot(generations, success_rate_prev, label='Success Rate (Previous Stages)', color='lightcoral', marker='x',
             markersize=4, linestyle='--')  # Mantido tracejado
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Success Rate (%)', fontsize=12)
    ax2.set_title('Success Rate per Generation', fontsize=14)
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def main():
    """
    Função principal que pede ao utilizador o caminho para o ficheiro de checkpoint.
    """
    try:
        checkpoint_path = input("Please enter the path to the checkpoint (.pkl) file you want to analyze: > ")
        plot_history(checkpoint_path)
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == '__main__':
    main()
