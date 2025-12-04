import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os

def main():
    print("--- Iniciando Simulação de Canal D2D (Voz + Doppler + Fading) ---")
    
    # =============================================================================
    # 1. Carregamento do Sinal de Voz (Define os Parâmetros da Simulação)
    # =============================================================================
    arquivo_voz_entrada = "voz_entrada.wav"
    try:
        # A taxa de amostragem e duração são lidas diretamente do arquivo
        FS_AUDIO, sinal_voz_original = wavfile.read(arquivo_voz_entrada)
        print(f"Arquivo '{arquivo_voz_entrada}' carregado. Taxa de Amostragem: {FS_AUDIO} Hz.")
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{arquivo_voz_entrada}' não encontrado!")
        print("Por favor, coloque um arquivo .wav com este nome na pasta do script.")
        return

    # Garantir que o sinal seja mono e normalizado
    if sinal_voz_original.ndim > 1:
        sinal_voz_original = np.mean(sinal_voz_original, axis=1)
    sinal_voz_base = sinal_voz_original / np.max(np.abs(sinal_voz_original))

    # Definir os parâmetros de tempo baseados no áudio de entrada
    NUM_AMOSTRAS = len(sinal_voz_base)
    DURACAO_REAL = NUM_AMOSTRAS / FS_AUDIO
    t_sim = np.linspace(0, DURACAO_REAL, NUM_AMOSTRAS)
    print(f"Duração da simulação definida para {DURACAO_REAL:.2f} segundos.")

    # =============================================================================
    # 2. Modelagem do Canal (Física)
    # =============================================================================
    # Os perfis de Doppler e Fading agora se ajustam à duração do áudio
    doppler_max = 6000
    t_centro = DURACAO_REAL / 2
    
    # Perfil Doppler (Curva em S)
    # O divisor no tanh (ex: 10) controla a rapidez da transição. 
    # Para durações curtas, um valor menor pode ser melhor.
    fator_suavizacao = max(1.0, DURACAO_REAL / 90) # Ajusta a curva para a duração
    M_f_perfil = -np.tanh((t_sim - t_centro) / fator_suavizacao) * doppler_max

    # Perfil de Amplitude (Fading)
    # A largura da janela de visada (ex: 60**2) também é ajustada
    largura_fading_seg = DURACAO_REAL / 15 
    B_perfil = np.exp(-((t_sim - t_centro)**2) / (2 * largura_fading_seg**2))
    B_perfil[B_perfil < 0.01] = 0

    # =============================================================================
    # 3. Geração e Modulação do Sinal
    # =============================================================================
    print("Sintetizando áudio e aplicando efeitos...")
    
    freq_portadora = 10000
    freq_instantanea = freq_portadora + M_f_perfil
    fase_portadora_doppler = 2 * np.pi * np.cumsum(freq_instantanea) / FS_AUDIO
    portadora_com_doppler = np.cos(fase_portadora_doppler)

    indice_modulacao = 0.8
    sinal_modulado = (1 + indice_modulacao * sinal_voz_base) * portadora_com_doppler

    sinal_rx = sinal_modulado * B_perfil
    ruido = np.random.normal(0, 0.05, len(sinal_rx))
    sinal_final = sinal_rx + ruido

    # =============================================================================
    # 4. Visualização (Gráficos)
    # =============================================================================
    print("Gerando gráficos...")
    
    plt.figure(figsize=(12, 8))

    # Plot 1: Amplitude no domínio do tempo
    plt.subplot(2, 1, 1)
    plt.plot(t_sim, sinal_final, color='gray', alpha=0.5, label='Sinal Recebido')
    plt.plot(t_sim, B_perfil, 'r', linewidth=2, label='Amplitude (Fading)')
    plt.title('Sinal no Domínio do Tempo')
    plt.ylabel('Amplitude')
    plt.xlabel('Tempo (s)')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Plot 2: Espectrograma
    plt.subplot(2, 1, 2)
    f, t_spec, Sxx = signal.spectrogram(sinal_final, FS_AUDIO, nperseg=4096)
    
    plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud', cmap='inferno')
    plt.ylabel('Frequência [Hz]')
    plt.xlabel('Tempo [s]')
    plt.title('Espectrograma: Visualização do Doppler Shift na Portadora')
    plt.ylim(freq_portadora - doppler_max - 1000, freq_portadora + doppler_max + 1000)
    plt.colorbar(label='dB')

    plt.tight_layout()
    
    # =============================================================================
    # 5. Demodulação e Exportação do Áudio (.wav)
    # =============================================================================
    print("Demodulando e processando o arquivo de áudio...")

    sinal_retificado = np.abs(sinal_final)
    
    b, a = signal.butter(4, 4000 / (FS_AUDIO / 2), btype='low')
    sinal_demodulado_com_dc = signal.filtfilt(b, a, sinal_retificado)

    sinal_demodulado = sinal_demodulado_com_dc - np.mean(sinal_demodulado_com_dc)

    # Normalização final
    max_val = np.max(np.abs(sinal_demodulado))
    if max_val > 0:
        audio_norm = sinal_demodulado / max_val
    else:
        audio_norm = sinal_demodulado
        
    audio_int16 = (audio_norm * 32767).astype(np.int16)

    # Salvar o arquivo com a MESMA taxa de amostragem do original
    arquivo_saida = "voz_recebida.wav"
    wavfile.write(arquivo_saida, FS_AUDIO, audio_int16)
    
    print(f"✅ SUCESSO! Arquivo '{arquivo_saida}' salvo com a mesma duração e taxa de amostragem do original.")
    
    plt.show()

if __name__ == "__main__":
    main()