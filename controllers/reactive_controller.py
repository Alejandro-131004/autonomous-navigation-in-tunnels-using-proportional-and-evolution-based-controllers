import math
import numpy as np
from environment.configuration import MAX_VELOCITY  # Importa a velocidade do seu ficheiro de configuração


def reactive_controller_logic(dist_values: list, direction: int = 1, fov_mode: str = 'full') -> tuple[
    float, float]:
    """
    Controlador modificado para replicar o comportamento do main.py, com opções de campo de visão (FOV).

    Args:
        dist_values (list): Lista de distâncias lidas pelo sensor LIDAR.
        direction (int): Direção de rotação (1 para frente, -1 para trás).
        fov_mode (str): Modo do campo de visão ('full', 'left', 'right').

    Returns:
        tuple[float, float]: Velocidade linear e velocidade angular para o robô.
    """
    # --- Parâmetros de Controlo ---
    maxSpeed = MAX_VELOCITY
    distP = 10.0
    angleP = 7.0
    wallDist = 0.3

    # --- Processamento dos dados do LIDAR ---
    original_size = len(dist_values)
    if original_size == 0:
        return 0.0, 0.0

    # Converte NaN para infinito para que não interfiram na busca pelo mínimo
    dist_values_finite = np.nan_to_num(np.array(dist_values), nan=np.inf)

    # Aplica o filtro de FOV
    if fov_mode == 'left':
        # Define a metade direita do FOV como infinito
        dist_values_finite[:original_size // 2] = np.inf
    elif fov_mode == 'right':
        # Define a metade esquerda do FOV como infinito
        dist_values_finite[original_size // 2:] = np.inf
    # 'full' mode não precisa de alterações

    # Encontra o índice do raio que retornou a distância mínima no FOV ativo
    min_index = -1
    if np.any(np.isfinite(dist_values_finite)):
        min_index = np.argmin(dist_values_finite)

    if min_index == -1:
        # Se não houver distâncias finitas no FOV ativo, o robô vagueia
        angular_vel = np.random.normal(loc=0.0, scale=1.0)
        linear_vel = maxSpeed
        return linear_vel, angular_vel

    # Calcula as variáveis de controle com base no FOV ativo
    angle_increment = 2 * math.pi / (original_size - 1)
    angleMin = (original_size // 2 - min_index) * angle_increment
    distMin = dist_values_finite[min_index]

    # Distâncias para pontos de referência no FOV original (mesmo que estejam "desligados" pelo FOV mode)
    # Estes são usados para as condições de bloqueio/desaceleração
    distFront = dist_values_finite[original_size // 2]
    distSide = dist_values_finite[original_size // 4] if (direction == 1) else dist_values_finite[3 * original_size // 4]
    distBack = dist_values_finite[0]

    # Prepara a mensagem para os motores do robô
    linear_vel: float
    angular_vel: float

    # --- Decide o comportamento do robô ---
    if distFront < 1.25 * wallDist and (distSide < 1.25 * wallDist or distBack < 1.25 * wallDist):
        # UNBLOCK: Se estiver muito perto da frente e de um lado/trás, tenta desbloquear
        angular_vel = direction * -1
    else:
        # REGULAR: Comportamento de seguir a parede/evitar obstáculos
        angular_vel = direction * distP * (distMin - wallDist) + angleP * (angleMin - direction * math.pi / 2)

    if distFront < wallDist:
        # TURN: Se a frente estiver muito perto, para e vira
        linear_vel = 0
    elif distFront < 2 * wallDist or distMin < wallDist * 0.75 or distMin > wallDist * 1.25:
        # SLOW: Se a frente ou a distância mínima estiverem em certas faixas, desacelera
        linear_vel = 0.5 * maxSpeed
    else:
        # CRUISE: Velocidade máxima
        linear_vel = maxSpeed

    return np.clip(linear_vel, -maxSpeed, maxSpeed), np.clip(angular_vel, -maxSpeed * 2, maxSpeed * 2)

