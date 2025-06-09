import math
import random as pyrandom  # Import pyrandom for diversity
import numpy as np  # Necessário para np.clip

# --- Configuration ---
ROBOT_NAME = "e-puck"
ROBOT_RADIUS = 0.035  # meters

# Tunnel clearance relative to robot diameter (2 * ROBOT_RADIUS)
# Define ranges for clearance factor
MIN_CLEARANCE_FACTOR_RANGE = (1.8, 2.5)  # Tighter minimum (e.g., 1.8 * 0.07m = 0.126m width)
MAX_CLEARANCE_FACTOR_RANGE = (3.0, 4.0)  # Wider maximum (e.g., 4.0 * 0.07m = 0.28m width)

# Minimum required clearance for the robot to pass an obstacle
MIN_ROBOT_CLEARANCE_FACTOR = 2.1
MIN_ROBOT_CLEARANCE = ROBOT_RADIUS * MIN_ROBOT_CLEARANCE_FACTOR

# Minimum distance between obstacles
MIN_OBSTACLE_DISTANCE_FACTOR = 4.0
MIN_OBSTACLE_DISTANCE = ROBOT_RADIUS * MIN_OBSTACLE_DISTANCE_FACTOR

# Max number of obstacles
MAX_NUM_OBSTACLES = 4

BASE_SPEED = 0.1
# Changed BASE_WALL_LENGTH to ROBOT_RADIUS to ensure compatibility with straight segment length calculations
BASE_WALL_LENGTH = ROBOT_RADIUS  # Base length for segments, now equals robot radius

# New constants for variable straight segment lengths
MIN_STRAIGHT_LENGTH_FACTOR = 6.0  # Minimum straight length in robot radii
MAX_STRAIGHT_LENGTH_FACTOR = 20.0  # Maximum straight length in robot radii
MIN_STRAIGHT_LENGTH = ROBOT_RADIUS * MIN_STRAIGHT_LENGTH_FACTOR
MAX_STRAIGHT_LENGTH = ROBOT_RADIUS * MAX_STRAIGHT_LENGTH_FACTOR
MAX_WALL_PIECES_PER_STRAIGHT = 10  # Max number of BASE_WALL_LENGTH units a straight can be

WALL_THICKNESS = 0.01
WALL_HEIGHT = 0.07
# Define ranges for curve angles (overall possible range, for context)
# Estas constantes definem o intervalo máximo de ângulos que uma curva pode ter.
# A progressão abaixo usará este MAX_CURVE_ANGLE_RANGE[1] como o limite superior final.
MIN_CURVE_ANGLE_RANGE = (math.radians(10), math.radians(30))
MAX_CURVE_ANGLE_RANGE = (math.radians(60), math.radians(90))
CURVE_SUBDIVISIONS = 30  # Increased subdivisions for smoother curves
TIMEOUT_DURATION = 45.0
MAX_NUM_CURVES = 4  # Max number of curves

# Map boundaries (example values, adjust to your world limits)
MAP_X_MIN, MAP_X_MAX = -2.0, 2.0
MAP_Y_MIN, MAP_Y_MAX = -2.0, 2.0

# Small overlap factor to ensure no gaps in curves
OVERLAP_FACTOR = 1.1  # Adjusted overlap factor based on user feedback

# Define a very small gap to prevent visual overlaps between consecutive wall segments
WALL_JOINT_GAP = 0.001  # 1 millimeter gap, adjust if needed

# Define a maximum stage value for normalization (10 levels, so stages 1 to 10)
MAX_DIFFICULTY_STAGE = 10.0  # Use float for calculations, can be converted to int for display


# Function to get difficulty settings based on a progress stage
# This function will be called by the genetic algorithm's training loop
def get_stage_parameters(stage: float, total_stages: float = MAX_DIFFICULTY_STAGE):
    """
    Provides tunnel generation parameters based on a continuous training stage.
    As the stage increases, elements are introduced and their diversity/challenge increases.

    Args:
        stage (float): The current training stage (e.g., from 1.0 to MAX_DIFFICULTY_STAGE).
        total_stages (float): The maximum possible difficulty stage for normalization.

    Returns:
        tuple: (num_curves, angle_range, clearance_factor, num_obstacles)
    """
    # Normalize stage to a 0-1 range (e.g., stage 1 -> 0.1, stage 10 -> 1.0)
    # Ensure stage is at least 1.0 for proper normalization if MAX_DIFFICULTY_STAGE is 10.0
    normalized_stage = min(max(1.0, stage), total_stages) / total_stages

    # --- 1. Progressão do Intervalo de Ângulos das Curvas (Dificuldade 1: Reto, depois 10-10 degs) ---
    effective_max_angle_deg = math.degrees(MAX_CURVE_ANGLE_RANGE[1])  # 90 graus

    if int(stage) == 1:
        chosen_angle_min = 0.0
        chosen_angle_max = 0.0
    else:
        # Calcular o tamanho do "passo" de ângulo para cada estágio (excluindo o estágio 1)
        # Total de 9 estágios para cobrir a gama de 0 a 90 graus
        angle_step_size_deg = effective_max_angle_deg / (total_stages - 1)  # 90 / 9 = 10 graus

        # Calcular os limites inferior e superior do ângulo para o estágio atual
        # Stage 2: (0, 10) graus
        # Stage 3: (10, 20) graus
        # ...
        # Stage 10: (80, 90) graus
        lower_bound_deg = (int(stage) - 2) * angle_step_size_deg
        upper_bound_deg = (int(stage) - 1) * angle_step_size_deg

        # Garantir que os limites não excedem os limites globais definidos
        chosen_angle_min = math.radians(max(0.0, lower_bound_deg))
        chosen_angle_max = math.radians(min(effective_max_angle_deg, upper_bound_deg))

    angle_range = (chosen_angle_min, chosen_angle_max)

    # --- 2. Progressão do Número de Curvas ---
    # Começa com 0 curvas no estágio 1, e aumenta linearmente até MAX_NUM_CURVES (4) no estágio 10.
    # Usamos normalized_stage que vai de 0.1 a 1.0.
    # Para garantir 0 curvas no estágio 1 (0.1), e começar a introduzir a partir daí.
    if normalized_stage <= 0.1:  # Para o estágio 1
        num_curves = 0
    else:
        # Escala de 0 a MAX_NUM_CURVES nos estágios 2-10 (normalized_stage de 0.2 a 1.0)
        scale_factor_curves = (normalized_stage - 0.1) / 0.9  # Mapeia 0.1-1.0 para 0-1
        max_curves_for_stage = scale_factor_curves * MAX_NUM_CURVES
        num_curves = pyrandom.randint(0, math.ceil(max_curves_for_stage))
        num_curves = min(num_curves, MAX_NUM_CURVES)  # Garante que não excede o máximo absoluto

    # --- 3. Progressão da Clareza (largura do túnel) ---
    # Diminui linearmente de MAX_CLEARANCE_FACTOR_RANGE[1] (mais largo) para MIN_CLEARANCE_FACTOR_RANGE[0] (mais apertado).
    # Adiciona aleatoriedade, que aumenta com a dificuldade.
    min_c = MIN_CLEARANCE_FACTOR_RANGE[0]  # 1.8
    max_c = MAX_CLEARANCE_FACTOR_RANGE[1]  # 4.0

    # Calcular o valor alvo da clareza, diminuindo com o aumento da dificuldade
    # Quando normalized_stage é 0.1 (stage 1), target_clearance é quase max_c.
    # Quando normalized_stage é 1.0 (stage 10), target_clearance é min_c.
    target_clearance = max_c - normalized_stage * (max_c - min_c)

    # Introduzir aleatoriedade: A amplitude da aleatoriedade aumenta com a dificuldade.
    # No estágio 1, a aleatoriedade é mínima. No estágio 10, cobre uma gama maior.
    random_spread = (max_c - min_c) * normalized_stage * 0.25  # Ex: até 25% da gama total no estágio 10

    clearance_factor = pyrandom.uniform(
        max(min_c, target_clearance - random_spread),
        min(max_c, target_clearance + random_spread)
    )
    clearance_factor = np.clip(clearance_factor, min_c, max_c)  # Garante que fica dentro dos limites definidos

    # --- 4. Progressão do Número de Obstáculos ---
    # Começa a introduzir obstáculos a partir de um certo estágio (ex: estágio 5 ou 6).
    # Aumenta linearmente até MAX_NUM_OBSTACLES (4) no estágio 10.
    obstacle_introduction_stage_norm = 0.5  # Começa a introduzir obstáculos a partir da metade dos estágios (estágio 5)

    if normalized_stage <= obstacle_introduction_stage_norm:
        num_obstacles = 0
    else:
        # Escala o progresso dos obstáculos do ponto de introdução (0.5) até 1.0
        scale_factor_obstacles = (normalized_stage - obstacle_introduction_stage_norm) / (
                    1.0 - obstacle_introduction_stage_norm)
        max_obstacles_for_stage = scale_factor_obstacles * MAX_NUM_OBSTACLES

        num_obstacles_raw = pyrandom.uniform(0, max_obstacles_for_stage)
        num_obstacles = math.floor(num_obstacles_raw)

        # Regra: se o número aleatório calculado for > 0 mas < 1, define como 1 (garante que um obstáculo aparece)
        if num_obstacles_raw > 0 and num_obstacles == 0:
            num_obstacles = 1
        num_obstacles = min(num_obstacles, MAX_NUM_OBSTACLES)  # Garante que não excede o máximo absoluto

    return num_curves, angle_range, clearance_factor, num_obstacles
