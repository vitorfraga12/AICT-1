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
    theta_i (float): ângulo de chegada

    return: A_{ula} (np.array): matriz A_{ula}
    
    '''


    mu_spatial_frequency = -np.pi*np.sin(theta_i)

    A_ula = np.zeros((m_antennas, d_in), dtype=complex)

    for col in range (d_in):
        for row in range (m_antennas):
            A_ula[row, col] = np.exp(1j * row * mu_spatial_frequency[col])
    
    return A_ula

def generate_signal(m_antennas:int, noise_subspace):
    
    # Estimando o espectro de potência
    angles = np.linspace(-np.pi/2, np.pi/2, 180)
    p_spectrum = np.zeros(angles.shape)

    # Calculando o espectro de potência
    for index_angle, angle in enumerate(angles):
        steering_vector = np.exp(1j * np.arange(m_antennas) * (-np.pi * np.sin(angle))) # Vetor de direção, que é o vetor de entrada da matriz A

        numerator = np.abs(np.dot(np.conj(steering_vector.T), steering_vector))
        denominator = np.abs(np.dot(np.conj(steering_vector.T), np.dot(noise_subspace, np.dot(np.conj(noise_subspace.T), steering_vector))))
        
        p_spectrum[index_angle] = numerator / denominator if denominator != 0 else 0

    angles = np.degrees(angles)

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
        sinal[i] = np.sin(2 * np.pi * random.random() * np.arange(t_snapshot))

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


def find_rmse(snr_values, iterations, m_antennas, d_arrival, t_snapshot):
    rmse = []
    for snr_index in snr_values:
        all_theta = []

        for iteration_index in range(iterations):
            phi_uniform = np.random.uniform(-60,60,m_antennas) # Gerando os ângulos de chegada
            phit_uniform = np.radians(phi_uniform) # Convertendo para radianos

            A_ula = gerate_a_ula(m_antennas, d_arrival, phit_uniform)
            angles, p_spectrum = generate_music(A_ula, d_arrival, t_snapshot, m_antennas, snr_index)

            # Calculando os ângulos de pico estimados pelo MUSIC
            peaks, _ = find_peaks(p_spectrum, height=0)
            sorted_peaks = np.argsort(p_spectrum[peaks]) # Ordenando picos por altura
            top_peaks = peaks[sorted_peaks[-d_arrival:]] - 90 # Obtendo os dois maiores picos e transladando eles para que o centro seja 0

            all_theta.append(top_peaks)

        all_theta = np.array(all_theta)
        phi_r = all_theta[:,0]
        hat_phi_r = all_theta[:,1]

        # Calculando o RMSE
        rmse.append(np.sqrt(np.mean((phi_r -  hat_phi_r)**2)))

    return rmse

