import os
import sys
from controller import Supervisor
from curriculum import run_unified_curriculum

# Definir os caminhos para os ficheiros de checkpoint
NE_CHECKPOINT_FILE = "saved_models/ne_checkpoint.pkl"
GA_CHECKPOINT_FILE = "saved_models/ga_checkpoint.pkl"


def main():
    """
    Função principal para executar o pipeline de treino.
    """
    # Configurações base e específicas para cada modo
    base_config = {
        'elitism': 2,
    }

    ne_config = {
        'mode': 'NE',
        'pop_size': 30,
        'hidden_size': 16,
        'mutation_rate': 0.15,
        'checkpoint_file': NE_CHECKPOINT_FILE,
    }

    ga_config = {
        'mode': 'GA',
        'pop_size': 30,
        'mutation_rate': 0.15,
        'checkpoint_file': GA_CHECKPOINT_FILE,
    }

    final_config = {}

    # --- LÓGICA DE SELEÇÃO E CHECKPOINT ---

    # 1. O utilizador seleciona o modo de treino
    while True:
        mode_choice = input(
            "Selecione o modo de treino:\n"
            "  1: Neuroevolution (Rede Neuronal)\n"
            "  2: Genetic Algorithm (Parâmetros Reativos)\n"
            "Insira a sua escolha (1 ou 2): "
        ).strip()
        if mode_choice == '1':
            final_config.update(base_config)
            final_config.update(ne_config)
            break
        elif mode_choice == '2':
            final_config.update(base_config)
            final_config.update(ga_config)
            break
        else:
            print("Escolha inválida. Por favor, insira 1 ou 2.")

    # 2. O utilizador seleciona o modo de visualização
    while True:
        debug_choice = input(
            "\nSelecione o modo de visualização:\n"
            "  1: Normal (resumo da geração)\n"
            "  2: Debug (informação detalhada e avisos)\n"
            "Insira a sua escolha (1 ou 2): "
        ).strip()
        if debug_choice == '1':
            # Define uma variável de ambiente para ser acedida por outros módulos
            os.environ['ROBOT_DEBUG_MODE'] = '0'
            break
        elif debug_choice == '2':
            os.environ['ROBOT_DEBUG_MODE'] = '1'
            print("\n--- MODO DEBUG ATIVADO ---")
            break
        else:
            print("Escolha inválida. Por favor, insira 1 ou 2.")

    training_mode = final_config['mode']
    checkpoint_file = final_config['checkpoint_file']

    # 3. Lida com o resumo a partir de um checkpoint
    resume_training = False
    if os.path.exists(checkpoint_file):
        while True:
            resume_choice = input(
                f"\nFoi encontrado um checkpoint de '{training_mode}'.\n"
                f"Deseja continuar o treino (y) ou começar de novo (n)? [y/n]: "
            ).lower().strip()
            if resume_choice == 'y':
                resume_training = True
                print(f"A retomar o treino anterior de {training_mode}...")
                break
            elif resume_choice == 'n':
                try:
                    os.remove(checkpoint_file)
                    print("Checkpoint removido. A iniciar uma nova sessão.")
                except OSError as e:
                    print(f"Erro ao remover o checkpoint: {e}")
                resume_training = False
                break
            else:
                print("Opção inválida. Por favor, insira 'y' ou 'n'.")
    else:
        print(f"Nenhum checkpoint de '{training_mode}' encontrado. A iniciar uma nova sessão.")

    final_config['resume_training'] = resume_training

    # --- FIM DA LÓGICA DE SELEÇÃO E CHECKPOINT ---

    # Inicializa o supervisor e executa o currículo unificado
    sup = Supervisor()
    print(f"\n--- A Iniciar Treino de {training_mode} ---")

    best_model = run_unified_curriculum(
        supervisor=sup,
        config=final_config
    )

    # Mensagem final
    if best_model:
        print(f"\nTreino concluído. O melhor modelo de {training_mode} foi guardado.")
    else:
        print("\nTreino concluído. Não foi identificado um modelo final.")


if __name__ == "__main__":
    main()
