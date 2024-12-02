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
        sinal[i] = (np.random.normal(size=t_snapshot) + 1j * np.random.normal(size=t_snapshot)) / np.sqrt(2)

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



def generate_angles_with_min_diff(d_arrival, min_diff, lower=-90, upper=90):
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


def find_peaks_d(p_spectrum, d_arrival, height=0):
    """
    Encontra os picos no espectro MUSIC.
    
    Parâmetros:
    - p_spectrum (np.ndarray): Espectro MUSIC.
    - height (float): Altura mínima do pico.
    
    Retorno:
    - peaks (np.ndarray): Índices dos picos encontrados.
    - _ (dict): Informações adicionais sobre os picos.
    """
    peaks, _ = find_peaks(p_spectrum, height=0)
    sorted_peaks = np.argsort(p_spectrum[peaks]) # Ordenando picos por altura
    top_peaks = peaks[sorted_peaks[-d_arrival:]] # Obtendo os dois maiores picos
    top_peaks = top_peaks - 90
    top_peaks = np.sort(top_peaks)

    return top_peaks, _







