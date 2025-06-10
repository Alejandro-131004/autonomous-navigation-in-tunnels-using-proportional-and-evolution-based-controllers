import os
import sys
import numpy as np
from environment.simulation_manager import SimulationManager

# Descomente e ajuste a linha abaixo para o seu caminho de instalação do Webots no Windows
#sys.path.insert(0, 'C:/Program Files/Webots/lib/controller/python') # Ou onde o Webots está instalado

from controller import Supervisor
from curriculum import run_curriculum
if __name__ == "__main__":
    sup = Supervisor()
    best_model = run_curriculum(sup,
                                pop_size=30,
                                generations=5,
                                success_threshold=0.6,
                                max_failed_generations=20)

    # Adicione esta parte para salvar o melhor modelo
    if best_model is not None:
        # Você pode precisar instanciar um SimulationManager aqui
        # ou passar o sim_mgr de run_curriculum para main,
        # ou fazer o save_model dentro de curriculum.py
        # Por simplicidade, vou assumir que você pode instanciar aqui para demonstração
        temp_sim_mgr = SimulationManager(sup) # Pode ser melhor passar o sim_mgr existente
        temp_sim_mgr.save_model(best_model, generation="final_best") # Ou outra identificação
        print(f"Melhor modelo salvo como 'final_best'.pkl'")