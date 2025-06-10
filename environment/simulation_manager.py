import numpy as np
import math
import pickle
import os

from environment.configuration import ROBOT_NAME, ROBOT_RADIUS, TIMEOUT_DURATION, get_stage_parameters, \
    MAX_DIFFICULTY_STAGE
from environment.tunnel import TunnelBuilder
from controllers.utils import cmd_vel


class SimulationManager:
    """
    Versão completa e final do gestor de simulação.
    Mantém todas as funcionalidades originais e é compatível com:
    1. Neuroevolução (via `run_experiment_with_network`).
    2. Algoritmo Genético Clássico (via `run_experiment_with_params`).
    3. Testes genéricos (via `run_experiment`).
    """

    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.robot = self.supervisor.getFromDef(ROBOT_NAME)
        if self.robot is None:
            raise ValueError(f"Robô com nome DEF '{ROBOT_NAME}' não encontrado.")

        self.translation = self.robot.getField("translation")
        self.rotation = self.robot.getField("rotation")

        # Ativar dispositivos
        self.lidar = self.supervisor.getDevice("lidar")
        self.lidar.enable(self.timestep)

        self.touch_sensor = self.supervisor.getDevice("touch sensor")
        self.touch_sensor.enable(self.timestep)

        self.left_motor = self.supervisor.getDevice("left wheel motor")
        self.right_motor = self.supervisor.getDevice("right wheel motor")
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

        # Estatísticas para execuções genéricas
        self.stats = {'total_collisions': 0, 'successful_runs': 0, 'failed_runs': 0}

    def save_model(self, individual, filename="best_model.pkl", save_dir="saved_models"):
        """Salva um indivíduo (modelo) num ficheiro."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, filename)
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(individual, f)
            print(f"Modelo salvo com sucesso em: {filepath}")
        except Exception as e:
            print(f"[ERROR] Falha ao salvar o modelo em {filepath}: {e}")

    def _calculate_fitness(self, success, collided, timeout, initial_dist, final_dist, total_dist, elapsed_time):
        """Função centralizada para calcular a fitness com base nos resultados do episódio."""
        progress = initial_dist - final_dist
        avg_speed = total_dist / (elapsed_time + 1e-6)

        fitness = ((10000.0 if success else 0.0) +
                   (500.0 * (progress / (initial_dist + 1e-6))) +
                   (100.0 * avg_speed) -
                   (5000.0 if collided else 0.0) -
                   (1000.0 if timeout else 0.0) -
                   (2000.0 if total_dist < ROBOT_RADIUS * 3 and not success else 0.0))
        return fitness

    def _run_single_episode(self, controller_callable, stage, total_stages):
        """Função base que executa um episódio de simulação e retorna o resultado completo."""
        # 1. Construir túnel
        num_curves, angle_range, clearance, num_obstacles = get_stage_parameters(stage, total_stages)
        builder = TunnelBuilder(self.supervisor)
        start_pos, end_pos, walls_added, final_heading = builder.build_tunnel(
            num_curves, angle_range, clearance, num_obstacles
        )
        if start_pos is None:
            return {'fitness': -10000.0, 'success': False, 'collided': False, 'timeout': True}

        # 2. Resetar robô
        self.robot.resetPhysics()
        # MODIFICAÇÃO: Coordenada Z definida para 0.0 para garantir que o robô começa no chão.
        self.translation.setSFVec3f([start_pos[0], start_pos[1], 0.0])
        self.rotation.setSFRotation([0, 0, 1, 0])
        self.supervisor.step(5)

        # 3. Iniciar variáveis do episódio
        t0 = self.supervisor.getTime()
        timeout, collided, success = False, False, False
        last_pos = np.array(self.translation.getSFVec3f())
        total_dist = 0.0
        initial_dist_to_goal = np.linalg.norm(last_pos[:2] - end_pos[:2])

        # 4. Loop de simulação
        while self.supervisor.step(self.timestep) != -1:
            elapsed = self.supervisor.getTime() - t0

            if elapsed > TIMEOUT_DURATION:
                timeout = True;
                print("[TIMEOUT]");
                break
            if self.touch_sensor.getValue() > 0:
                collided = True;
                print("[COLLISION]");
                break

            scan = np.nan_to_num(self.lidar.getRangeImage(), nan=np.inf)
            lv, av = controller_callable(scan)
            cmd_vel(self.supervisor, lv, av)

            current_pos = np.array(self.translation.getSFVec3f())
            total_dist += np.linalg.norm(current_pos[:2] - last_pos[:2])
            last_pos = current_pos

            if np.linalg.norm(current_pos[:2] - end_pos[:2]) < ROBOT_RADIUS * 2:
                success = True;
                print(f"[SUCCESS] em {elapsed:.2f}s");
                break

        # 5. Limpar túnel
        builder._clear_walls()

        # 6. Calcular Fitness
        final_dist_to_goal = np.linalg.norm(last_pos[:2] - end_pos[:2])
        fitness = self._calculate_fitness(success, collided, timeout, initial_dist_to_goal, final_dist_to_goal,
                                          total_dist, elapsed)

        return {'fitness': fitness, 'success': success, 'collided': collided, 'timeout': timeout}

    # --- Funções de Interface para Otimizadores e Testes ---

    def run_experiment_with_network(self, individual, stage, total_stages=MAX_DIFFICULTY_STAGE):
        """Interface para a NEUROEVOLUÇÃO (usado por `curriculum.py`)."""
        ind_id = getattr(individual, 'id', 'N/A')
        print(f"[RUN-NETWORK] Ind {ind_id} | Stage {stage}")
        results = self._run_single_episode(individual.act, stage, total_stages)
        print(f"[FITNESS] Ind {ind_id} | Fit: {results['fitness']:.2f} | Success: {results['success']}")
        return results['fitness'], results['success']

    def run_experiment_with_params(self, distP, angleP, stage, total_stages=MAX_DIFFICULTY_STAGE):
        """Interface para o ALGORITMO GENÉTICO CLÁSSICO (usado por `genetic.py`)."""
        print(f"[RUN-PARAMS] distP={distP:.2f}, angleP={angleP:.2f} | Stage {stage}")

        def ga_controller(scan):
            return self._process_lidar_for_ga(scan, distP, angleP)

        results = self._run_single_episode(ga_controller, stage, total_stages)
        print(f"[FITNESS] Params | Fit: {results['fitness']:.2f} | Success: {results['success']}")
        return results['fitness']  # O AG clássico espera apenas a fitness

    def run_experiment(self, num_runs):
        """Função para testes genéricos com um controlador padrão."""
        print("A executar experiência genérica com controlador padrão.")
        self.stats = {'total_collisions': 0, 'successful_runs': 0, 'failed_runs': 0}  # Reset stats

        for run in range(num_runs):
            print(f"\n--- Execução de Teste {run + 1}/{num_runs} ---")
            stage = 1  # Usar o estágio mais fácil para testes

            # Controlador padrão para esta experiência
            def default_controller(scan):
                return self._process_lidar_for_ga(scan, 10.0, 5.0)  # Usar alguns parâmetros padrão

            results = self._run_single_episode(default_controller, stage, MAX_DIFFICULTY_STAGE)

            # Atualizar estatísticas
            if results['success']:
                self.stats['successful_runs'] += 1
            else:
                self.stats['failed_runs'] += 1
            if results['collided']:
                self.stats['total_collisions'] += 1

        self._print_summary()

    def _process_lidar_for_ga(self, dist_values, distP, angleP):
        """Lógica de controlo de parede clássica para o Algoritmo Genético."""
        direction: int = 1
        max_speed: float = 0.12
        wall_dist: float = 0.1

        size: int = len(dist_values)
        if size == 0: return 0.0, 0.0

        min_index = np.argmin(dist_values) if np.any(np.isfinite(dist_values)) else -1
        if min_index == -1: return 0.0, max_speed

        dist_min = dist_values[min_index]
        angle_increment = (2 * math.pi) / size
        angle_min = (size / 2 - min_index) * angle_increment
        dist_front = dist_values[size // 2]

        angular_vel = direction * distP * (dist_min - wall_dist) + angleP * (angle_min - direction * math.pi / 2)

        linear_vel = max_speed
        if dist_front < wall_dist * 1.5:
            linear_vel = 0
        elif dist_front < wall_dist * 2.5:
            linear_vel = max_speed * 0.5

        return np.clip(linear_vel, -max_speed, max_speed), np.clip(angular_vel, -max_speed * 2, max_speed * 2)

    def _print_summary(self):
        """Imprime um resumo das estatísticas das execuções de teste."""
        print("\n=== Resumo Final da Experiência ===")
        print(f"Execuções com sucesso: {self.stats['successful_runs']}")
        print(f"Execuções falhadas: {self.stats['failed_runs']}")
        print(f"Total de colisões: {self.stats['total_collisions']}")
        total_runs = self.stats['successful_runs'] + self.stats['failed_runs']
        if total_runs > 0:
            success_rate = (self.stats['successful_runs'] / total_runs) * 100
            print(f"Taxa de sucesso: {success_rate:.1f}%")
        else:
            print("Nenhuma execução foi completada.")
