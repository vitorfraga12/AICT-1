import numpy as np
import random
from scipy.signal import find_peaks



def lin2db(x):
    return 10*np.log10(x)

def db2lin(x):
    return 10**(x/10)

def gerate_a_ula(m_antennas:int, d_in:float, theta_i:float):
    '''
    Função que gera a matriz A_{ula} 
    m_antennas (int): número de antenas do arranjo
    d_in (float): número de direções de chegada
    theta_i (float): ângulo de chegada em graus

    return: A_{ula} (np.array): matriz A_{ula}
    
    '''



    mu_spatial_frequency = -np.pi*np.sin(np.radians(theta_i))

    A_ula = np.zeros((m_antennas, d_in), dtype=complex)

    for col in range (d_in):
        for row in range (m_antennas):
            A_ula[row, col] = np.exp(1j * row * mu_spatial_frequency[col])

    
    return A_ula



def generate_signal(m_antennas:int, noise_subspace):
    '''
    Função que gera o espectro de potência do sinal recebido
    m_antennas (int): número de antenas do arranjo
    noise_subspace (np.array): subespaço de ruído

    return: angles (np.array): ângulos avaliados, p_spectrum (np.array): espectro de potência do sinal recebido.
    
    '''
    
    # Estimando o espectro de potência
    angles = np.linspace(-90, 90, 181)
    p_spectrum = np.zeros(angles.shape)

    # Calculando o espectro de potência
    for index_angle, angle in enumerate(angles):
        steering_vector = np.exp(1j * np.arange(m_antennas) * (-np.pi * np.sin(np.radians(angle)))) # Vetor de direção, que é o vetor de entrada da matriz A

        numerator = np.abs(np.dot(np.conj(steering_vector.T), steering_vector))
        denominator = np.abs(np.dot(np.conj(steering_vector.T), np.dot(noise_subspace, np.dot(np.conj(noise_subspace.T), steering_vector))))
        
        p_spectrum[index_angle] = numerator / denominator if denominator != 0 else 0

    return angles, p_spectrum



def generate_music(A_ula: np.ndarray, arrival_distance: int, t_snapshot: int, m_antennas: int, snr: float):
    """
    Executa o algoritmo MUSIC para análise espectral.
    
    Parâmetros:
    - A_ula (np.ndarray): Matriz de resposta direcional.
    - arrival_distance (int): Número de sinais incidentes (direções de chegada).
    - t_snapshot (int): Número de snapshots.
    - m_antennas (int): Número de antenas no arranjo.
    - snr (float): Relação sinal-ruído (em dB).
    
    Retorno:
    - angles (np.ndarray): Ângulos (em graus) avaliados.
    - p_spectrum (np.ndarray): Espectro MUSIC correspondente aos ângulos.
    """
    # Gerar o sinal transmitido
    sinal = np.zeros((arrival_distance, t_snapshot), dtype=complex)
    for i in range(arrival_distance):
        sinal[i] =((np.random.normal(size=t_snapshot) + 1j * np.random.normal(size=t_snapshot))/arrival_distance) / np.sqrt(2)

    # Sinal recebido no arranjo de antenas
    sinal_aula = np.dot(A_ula, sinal)

    # Adicionando ruído branco gaussiano ao sinal
    ruido_amplitude = 1/np.sqrt(db2lin(snr))  # Ajusta o ruído com base na SNR
    noise = (np.random.normal(0, ruido_amplitude, (m_antennas, t_snapshot)) + 
             1j * np.random.normal(0, ruido_amplitude, (m_antennas, t_snapshot))) / np.sqrt(2)
    sinal_final = sinal_aula + noise

    # Matriz de autocorrelação
    sinal_final_hermetiano = np.conj(sinal_final.T)
    autocor_matrix_estimada = np.dot(sinal_final, sinal_final_hermetiano) / t_snapshot

    # Decomposição em autovalores e autovetores
    eigenvalues, eigenvec = np.linalg.eig(autocor_matrix_estimada)

    # Ordenando os autovalores em ordem decrescente
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvec = eigenvec[:, idx]

    # Subespaço de ruído
    noise_subspace = eigenvec[:, arrival_distance:]

    # Gerar o espectro MUSIC
    angles, p_spectrum = generate_signal(m_antennas, noise_subspace)

    return angles, p_spectrum



def generate_angles_with_min_diff(d_arrival, min_diff, lower=-50, upper=50):
    """
    Gera ângulos uniformemente distribuídos com uma diferença mínima entre eles.
    
    Parâmetros:
    - d_arrival (int): Número de ângulos a serem gerados.
    - min_diff (float): Diferença mínima permitida entre ângulos consecutivos.
    - lower (float): Limite inferior do intervalo de geração.
    - upper (float): Limite superior do intervalo de geração.
    
    Retorno:
    - np.ndarray: Ângulos gerados com diferença mínima garantida. (em graus)
    """
    while True:
        phi_uniform = np.random.uniform(lower, upper, d_arrival)
        phi_uniform_sorted = np.sort(phi_uniform)
        if np.all(np.diff(phi_uniform_sorted) >= min_diff):
            return phi_uniform_sorted


def find_peaks_d(p_spectrum, d_arrival, prominence=2): # Meu código
    '''
    Encontra os picos no espectro MUSIC, considerando a proeminência dos picos.
    
    Parâmetros:
    - p_spectrum (np.ndarray): Espectro MUSIC.
    - d_arrival (int): Número de picos desejados.
    - prominence (float): Proeminência mínima para considerar um pico.
    
    Retorno:
    - peaks (np.ndarray): Índices dos picos encontrados.
    - values (np.ndarray): Valores dos picos encontrados.
    '''
    # Encontre os picos com base na proeminência
    peaks, properties = find_peaks(p_spectrum, prominence=prominence)

    # Classificar os picos pela proeminência (valores mais altos)
    locs = np.argsort(properties["prominences"])[::-1][:d_arrival]  # Ordena pelos maiores picos de proeminência
    locs = peaks[locs]  # Obtém os índices dos picos mais proeminentes
    locs = locs - 90
    locs = np.sort(locs)
    values = p_spectrum[locs]  # Valores dos picos mais proeminentes

    return locs, values




""" def find_peaks_d(p_spectrum, d_arrival, prominence=2): # Chat GPT
    '''
    Encontra os picos no espectro MUSIC com base na prominência mínima especificada.
    Aplica interpolação para obter picos sub-amostrais.
    
    Parâmetros:
    - p_spectrum (np.ndarray): Espectro MUSIC.
    - d_arrival (int): Número de picos desejados.
    - prominence (float): Prominência mínima do pico.
    
    Retorno:
    - peaks (np.ndarray): Índices dos picos encontrados.
    - _ (dict): Informações adicionais sobre os picos.
    '''
    # Encontra os picos com a função find_peaks, usando a propriedade de prominência
    peaks, properties = find_peaks(p_spectrum, prominence=prominence)
    
    # Verificar se o número de picos encontrados é suficiente
    if len(peaks) < d_arrival:
        print(f"Warning: Apenas {len(peaks)} picos encontrados, esperados {d_arrival} picos.")
        # Se não houver picos suficientes, retornamos todos os picos encontrados ou algum valor padrão
        return peaks, properties
    
    # Ordenando os picos por prominência (para pegar os picos mais destacados)
    sorted_peaks = np.argsort(properties["prominences"])[::-1]  # Ordena pela prominência dos picos
    top_peaks = peaks[sorted_peaks[:d_arrival]]  # Pega os picos mais altos até o número desejado

    # Aplicando a interpolação para obter picos sub-amostrais (refinados)
    refined_peaks = []
    for peak in top_peaks:
        if 1 <= peak < len(p_spectrum) - 1:
            # Pegar os pontos ao redor do pico para ajustar uma parábola
            alpha = p_spectrum[peak - 1]
            beta = p_spectrum[peak]
            gamma = p_spectrum[peak + 1]

            # Calculando o deslocamento do vértice da parábola em relação ao ponto central
            p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma) if (alpha - 2 * beta + gamma) != 0 else 0

            # A posição refinada do pico (sub-amostral)
            refined_peak = peak + p
            refined_peaks.append(refined_peak)
        else:
            refined_peaks.append(peak)  # Caso o pico esteja no limite, não há interpolação

    refined_peaks = np.array(refined_peaks) - 90  # Transladar para que o centro seja zero
    refined_peaks = np.sort(refined_peaks)  # Ordenar os picos refinados
    return refined_peaks, properties
 """

def find_rmse(snr_values: np.ndarray, iterations: int, m_antennas: int, d_arrival: int, t_snapshot: int):
    '''
    Calcula o RMSE (Root Mean Square Error) dos ângulos estimados pelo algoritmo MUSIC em diferentes valores de SNR.

    Parâmetros:
    - snr_values (list or np.ndarray): Lista de valores de SNR (em dB) a serem avaliados.
    - iterations (int): Número de iterações por valor de SNR.
    - m_antennas (int): Número de antenas no arranjo.
    - d_arrival (int): Número de ângulos de chegada (direções de chegada).
    - t_snapshot (int): Número de snapshots para estimativa de autocorrelação.

    Retorno:
    - rmse_maior (list): Lista com os valores de RMSE para os maiores ângulos estimados, correspondentes a cada SNR.
    - rmse_menor (list): Lista com os valores de RMSE para os menores ângulos estimados, correspondentes a cada SNR.
    '''
    
    rmse_maior = []
    rmse_menor = []
    
    for snr_index in snr_values:
        phi_maior = []
        phi_menor = []

        for iteration_index in range(iterations):
            # Gerando os ângulos de chegada
            phit_uniform = generate_angles_with_min_diff(d_arrival, 20)

            A_ula = gerate_a_ula(m_antennas, d_arrival, phit_uniform)
            angles, p_spectrum = generate_music(A_ula, d_arrival, t_snapshot, m_antennas, snr_index)

            # Calculando os ângulos de pico estimados pelo MUSIC
            top_peak_angles, _ = find_peaks_d(p_spectrum, d_arrival)

    

            maior_diferenca = phit_uniform[0] - top_peak_angles[0] 
            menor_diferena = phit_uniform[1] - top_peak_angles[1]

            phi_maior.append(maior_diferenca)
            phi_menor.append(menor_diferena)

        # Calculando o RMSE para os ângulos estimados
        phi_maior = np.array(phi_maior)
        phi_menor = np.array(phi_menor)

        rmse_maior.append(np.sqrt(np.mean(np.abs(phi_maior) ** 2)))
        rmse_menor.append(np.sqrt(np.mean(np.abs(phi_menor) ** 2)))

    return rmse_maior, rmse_menor
