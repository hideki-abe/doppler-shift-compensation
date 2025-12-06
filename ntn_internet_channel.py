import numpy as np
import matplotlib.pyplot as plt
from skyfield.api import load, EarthSatellite, wgs84

# =============================================================================
# PARÂMETROS
# =============================================================================
ts = load.timescale()
c = 3e8  # Velocidade da luz

# TLE Satélite Starlink
line1_A = '1 58214U 23160A   25341.52083333  .00002177  00000+0  12345-3 0  9997'
line2_A = '2 58214  53.0590 108.4620 0001101  85.5250 274.5850 15.06422233 65000'
sat_A = EarthSatellite(line1_A, line2_A, 'STARLINK-A', ts)

# Posição do usuário (São Paulo)
user = wgs84.latlon(-23.55, -46.63)

# Parâmetros do link
f_downlink = 12e9  # Frequência downlink (12 GHz - Banda Ku)

# =============================================================================
# SIMULAÇÃO
# =============================================================================
# Simular 10 minutos
t0 = ts.now()
t = ts.linspace(t0, t0 + 10/1440, 300)
dt = (10 * 60) / 300  # Intervalo de tempo (segundos)
tempo_minutos = np.linspace(0, 10, 300)

# Posição relativa Satélite → UE
difference_downlink = sat_A.at(t) - user.at(t)

# Métricas do canal
distancia_downlink = difference_downlink.distance().km
elevacao_A = difference_downlink.altaz()[0].degrees
azimute_A = difference_downlink.altaz()[1].degrees

# Velocidade radial e Doppler
velocidade_radial_downlink = np.gradient(distancia_downlink * 1000, dt)
doppler_downlink_sem_comp = -f_downlink * velocidade_radial_downlink / c / 1e3  # kHz

# Compensações de Doppler
doppler_downlink_com_comp = doppler_downlink_sem_comp * 0.05  # Pré-compensação (5% erro)
doppler_pos_comp = np.copy(doppler_downlink_sem_comp)
tempo_aquisicao = 30

for i in range(tempo_aquisicao, len(doppler_pos_comp)):
    doppler_pos_comp[i] = doppler_downlink_sem_comp[i] * 0.02  # Pós-compensação (2% erro)

doppler_pos_comp[:tempo_aquisicao] = doppler_downlink_sem_comp[:tempo_aquisicao] * \
                                     np.linspace(1.0, 0.02, tempo_aquisicao)

# =============================================================================
# MÉTRICAS DE AVALIAÇÃO (do ntn_combined.py e ntn_satellite.py)
# =============================================================================

# 1. DISPONIBILIDADE DO LINK (Tolerâncias Doppler)
tolerancias_doppler = np.array([1.8, 5, 10, 18])  # kHz
cenarios = {
    'Sem Compensação': doppler_downlink_sem_comp,
    'Pré-compensação': doppler_downlink_com_comp,
    'Pós-compensação': doppler_pos_comp
}

disponibilidade_matriz = np.zeros((len(cenarios), len(tolerancias_doppler)))

for i, (nome, doppler) in enumerate(cenarios.items()):
    for j, tolerancia in enumerate(tolerancias_doppler):
        dentro_tolerancia = np.abs(doppler) <= tolerancia
        tempo_disponivel = np.sum(dentro_tolerancia) * dt
        disponibilidade_matriz[i, j] = tempo_disponivel

# 2. TEMPO DE VISIBILIDADE
mascara_visivel = elevacao_A > 0
tempo_visivel = np.sum(mascara_visivel) * dt

# =============================================================================
# VISUALIZAÇÃO
# =============================================================================
plt.rcParams.update({'font.size': 16})
fig = plt.figure(figsize=(20, 16))

# --- GRÁFICO 1: DOPPLER COM ZONAS DE TOLERÂNCIA ---
ax1 = plt.subplot(3, 1, 1)
ax1.plot(tempo_minutos, doppler_downlink_sem_comp, 'r-', linewidth=3, label='Sem Compensação')
ax1.plot(tempo_minutos, doppler_downlink_com_comp, 'b-', linewidth=3, label='Pré-compensação (Satélite)')
ax1.plot(tempo_minutos, doppler_pos_comp, 'g-', linewidth=3, label='Pós-compensação (PLL no Terminal)')

# Zonas de tolerância
ax1.axhspan(-1.8, 1.8, alpha=0.15, color='green', label='Zona ±1.8 kHz')
ax1.axhspan(-5, 5, alpha=0.10, color='yellow', label='Zona ±5 kHz')
ax1.axhspan(-10, 10, alpha=0.08, color='orange', label='Zona ±10 kHz')
ax1.axhspan(-18, 18, alpha=0.05, color='red', label='Zona ±18 kHz')
ax1.axhline(0, color='k', linestyle='--', linewidth=1.5, alpha=0.5)

ax1.set_xlabel('Tempo (minutos)', fontsize=18, fontweight='bold')
ax1.set_ylabel('Desvio Doppler (kHz)', fontsize=18, fontweight='bold')
ax1.set_title('Desvio Doppler com Zonas de Tolerância', fontsize=22, fontweight='bold', pad=20)
ax1.legend(loc='upper right', fontsize=14, framealpha=0.95)
ax1.grid(True, alpha=0.3, linewidth=1.5)
ax1.tick_params(labelsize=16)

# --- GRÁFICO 2: DISPONIBILIDADE DO LINK ---
ax2 = plt.subplot(3, 1, 2)
x_pos = np.arange(len(tolerancias_doppler))
largura_barra = 0.25
cores = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for i, (nome, cor) in enumerate(zip(cenarios.keys(), cores)):
    porcentagem = (disponibilidade_matriz[i, :] / (10 * 60)) * 100
    barras = ax2.bar(x_pos + i * largura_barra, porcentagem, largura_barra, 
                     label=nome, color=cor, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Adicionar valores nas barras
    for barra in barras:
        altura = barra.get_height()
        ax2.text(barra.get_x() + barra.get_width()/2., altura,
                f'{altura:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax2.set_xlabel('Tolerância Doppler (kHz)', fontsize=18, fontweight='bold')
ax2.set_ylabel('Disponibilidade do Link (%)', fontsize=18, fontweight='bold')
ax2.set_title('Disponibilidade do Link por Tolerância Doppler', fontsize=22, fontweight='bold', pad=20)
ax2.set_xticks(x_pos + largura_barra)
ax2.set_xticklabels([f'±{tol}' for tol in tolerancias_doppler], fontsize=16)
ax2.legend(loc='lower right', fontsize=14, framealpha=0.95)
ax2.grid(True, alpha=0.3, axis='y', linewidth=1.5)
ax2.set_ylim([0, 110])
ax2.tick_params(labelsize=16)

# --- GRÁFICO 3: ELEVAÇÃO E AZIMUTE ---
ax3 = plt.subplot(3, 1, 3)
cor_elev = '#E74C3C'
cor_azim = '#3498DB'

ax3.plot(tempo_minutos, elevacao_A, color=cor_elev, linewidth=3, label='Elevação')
ax3.set_xlabel('Tempo (minutos)', fontsize=18, fontweight='bold')
ax3.set_ylabel('Elevação (graus)', fontsize=18, fontweight='bold', color=cor_elev)
ax3.tick_params(axis='y', labelcolor=cor_elev, labelsize=16)
ax3.axhline(0, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
ax3.grid(True, alpha=0.3, linewidth=1.5)

ax3_twin = ax3.twinx()
ax3_twin.plot(tempo_minutos, azimute_A, color=cor_azim, linewidth=3, label='Azimute', linestyle='--')
ax3_twin.set_ylabel('Azimute (graus)', fontsize=18, fontweight='bold', color=cor_azim)
ax3_twin.tick_params(axis='y', labelcolor=cor_azim, labelsize=16)

ax3.set_title('Rastreamento Angular do Satélite', fontsize=22, fontweight='bold', pad=20)

# Legendas combinadas
linhas1, labels1 = ax3.get_legend_handles_labels()
linhas2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(linhas1 + linhas2, labels1 + labels2, loc='upper right', fontsize=14, framealpha=0.95)

plt.tight_layout()
plt.show()

# =============================================================================
# ESTATÍSTICAS DE SAÍDA
# =============================================================================
print("\n" + "="*80)
print("RELATÓRIO DE SIMULAÇÃO - CANAL NTN LEO (Starlink)")
print("="*80)
print(f"Duração da simulação: 10 minutos")
print(f"Frequência downlink: {f_downlink/1e9:.1f} GHz (Banda Ku)")
print(f"Localização do terminal: São Paulo (-23.55°, -46.63°)")
print(f"\nTempo de visibilidade (elevação > 0°): {tempo_visivel:.1f} segundos ({tempo_visivel/60:.2f} min)")

print("\n" + "-"*80)
print("DISPONIBILIDADE DO LINK POR TOLERÂNCIA DOPPLER")
print("-"*80)
for i, nome in enumerate(cenarios.keys()):
    print(f"\n{nome}:")
    for j, tol in enumerate(tolerancias_doppler):
        tempo = disponibilidade_matriz[i, j]
        porcentagem = (tempo / (10 * 60)) * 100
        print(f"  ±{tol:>5.1f} kHz: {tempo:>6.1f}s ({porcentagem:>5.1f}%)")

print("\n" + "-"*80)
print("EFICÁCIA DA COMPENSAÇÃO (Redução do Doppler)")
print("-"*80)
doppler_max_sem = np.max(np.abs(doppler_downlink_sem_comp))
doppler_max_pre = np.max(np.abs(doppler_downlink_com_comp))
doppler_max_pos = np.max(np.abs(doppler_pos_comp))

print(f"Doppler máximo sem compensação: {doppler_max_sem:.2f} kHz")
print(f"Doppler máximo com pré-compensação: {doppler_max_pre:.2f} kHz (redução de {(1-doppler_max_pre/doppler_max_sem)*100:.1f}%)")
print(f"Doppler máximo com pós-compensação: {doppler_max_pos:.2f} kHz (redução de {(1-doppler_max_pos/doppler_max_sem)*100:.1f}%)")
print("="*80 + "\n")