import os
import sys
import pickle
from environment.simulation_manager import SimulationManager
from controller import Supervisor
from curriculum import run_curriculum

# --- Configuração do Checkpoint ---
CHECKPOINT_FILE = "saved_models/checkpoint.pkl"

if __name__ == "__main__":
    sup = Supervisor()
    resume_training = False

    # Verifica se existe um checkpoint e pergunta ao utilizador
    if os.path.exists(CHECKPOINT_FILE):
        while True:
            choice = input(
                f"Foi encontrado um treino guardado em '{CHECKPOINT_FILE}'.\nDeseja continuar (s) ou começar do zero (n)? [s/n]: ").lower().strip()
            if choice == 's':
                resume_training = True
                print("A continuar o treino anterior...")
                break
            elif choice == 'n':
                try:
                    os.remove(CHECKPOINT_FILE)
                    print("Ficheiro de treino anterior removido. A começar do zero...")
                except OSError as e:
                    print(f"Erro ao remover o ficheiro de checkpoint: {e}")
                resume_training = False
                break
            else:
                print("Opção inválida. Por favor, escreva 's' ou 'n'.")
    else:
        print("Nenhum treino guardado encontrado. A começar do zero...")

    # Chama a função principal do currículo
    best_model = run_curriculum(
        supervisor=sup,
        resume_training=resume_training,
        pop_size=30,
        success_threshold=0.5,
        max_failed_generations=15,
        hidden_size=16,
        mutation_rate=0.15,
        elitism=2
    )

    # A lógica de guardar o modelo final já está dentro de run_curriculum.
    # Apenas imprimimos uma mensagem final.
    if best_model:
        print("\nTreino concluído. O melhor modelo foi guardado periodicamente e no final do treino.")
    else:
        print("\nTreino concluído sem um melhor modelo final.")
