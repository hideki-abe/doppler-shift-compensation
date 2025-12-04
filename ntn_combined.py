import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# === Parâmetros Gerais ===
G = 6.67430e-11  # Constante gravitacional
M = 5.972e24     # Massa da Terra (kg)
Re = 6371e3      # Raio da Terra (m)
c = 3e8          # Velocidade da luz (m/s)
f0 = 3.5e9       # Frequência da portadora (Hz)

# === Parâmetros Temporais ===
T_total = 1200
# Em Python, o último ponto do linspace é inclusivo.
t = np.linspace(-T_total/2, T_total/2, 1200)
# 'dt' é o intervalo de tempo
dt = np.mean(np.diff(t))
t_plot = t + T_total/2

# === Parâmetros Orbitais ===
incl_deg = 53
incl = np.deg2rad(incl_deg)
h = 550e3  # Altitude (m)
R = Re + h # Raio da órbita

# === Posição Inicial da Estação Terrestre (UE) ===
# Assumida no polo norte (0, 0, Re) para simplicidade de cálculo em coordenadas GCRF/ECI
x_usr = 0
y_usr = 0
z_usr = Re
# Posição da UE: array [x, y, z]
pos_usr = np.array([x_usr, y_usr, z_usr])

# === Trajetória do Satélite e Desvio Doppler ===
# Velocidade orbital (circular)
v_orb = np.sqrt(G * M / R)
# Velocidade angular
omega = v_orb / R

# Posição do satélite (vetores ao longo do tempo)
x_sat = R * np.sin(omega * t)
y_sat = R * np.cos(omega * t) * np.cos(incl)
z_sat = R * np.cos(omega * t) * np.sin(incl)

# Vetor da diferença (Satélite - UE)
dx = x_sat - x_usr
dy = y_sat - y_usr
dz = z_sat - z_usr

# Distância (Range)
range_t = np.sqrt(dx**2 + dy**2 + dz**2)

# Velocidade radial (aproximação/afastamento). Usa np.gradient para simular 'gradient' do MATLAB
# A velocidade radial é a taxa de variação da distância (range)
v_radial = np.gradient(range_t, dt)

# Desvio Doppler (fórmula simplificada)
doppler_shift = -f0 * v_radial / c

# === Frequência Recebida (para referência, não usada nos gráficos principais) ===
f_rx_no_comp = f0 + doppler_shift
f_rx_comp_ideal = f0 * np.ones_like(t)

# Estimativas Doppler (simula o deslocamento do vetor, preenchendo o início com zeros)
delay_30 = 30
delay_40 = 40
doppler_est30 = np.concatenate((np.zeros(delay_30), doppler_shift[:-delay_30]))
doppler_est40 = np.concatenate((np.zeros(delay_40), doppler_shift[:-delay_40]))
f_rx_comp30 = f0 + doppler_shift - doppler_est30
f_rx_comp40 = f0 + doppler_shift - doppler_est40

# --- GRÁFICO 1: Desvio Doppler com Áreas Sombreadas ---
limits = np.array([1.8e3, 5e3, 10e3, 18e3])
cores = ['g', 'm', 'c', 'k']
tempo_total = T_total

plt.figure(figsize=(10, 6))
plt.plot(t_plot, doppler_shift/1e3, 'r', linewidth=1.5, label='Doppler')

# Preenchimento das áreas sombreadas (fill)
for i in range(len(limits)):
    # Preenche a região entre os limites de Doppler e seus valores negativos, ao longo do tempo
    # Note: O MATLAB 'fill' aceita [x1, x2, x3, ...], [y1, y2, y3, ...]
    # O Python 'fill_between' é mais idiomático para isso: fill_between(x, y1, y2)
    plt.fill_between(t_plot, limits[i]/1e3, -limits[i]/1e3,
                     color=cores[i], alpha=0.1, label=f'±{limits[i]/1e3:.1f}kHz')

# Linhas horizontais (yline)
for i in range(len(limits)):
    plt.axhline(limits[i]/1e3, color='k', linestyle='--', linewidth=1)
    plt.axhline(-limits[i]/1e3, color='k', linestyle='--', linewidth=1)

plt.xlabel('Time (s)')
plt.ylabel('Doppler Shift (kHz)')

# Ajustando a legenda para que as áreas sombreadas apareçam corretamente
# Lida com o problema de legendas duplicadas das axhline
handles, labels_orig = plt.gca().get_legend_handles_labels()
# Mantém apenas a primeira ocorrência de cada label (o que é gerado pelo plot e fill_between)
legend_labels = ['Doppler'] + [f'±{l/1e3:.1f}kHz' for l in limits]
plt.legend(handles[:len(legend_labels)], legend_labels, loc='upper right')

plt.title('Doppler Shift ao Longo do Tempo')
plt.grid(True)
plt.show()

# --- GRÁFICO 2: Limiares Distintos de Desvio Doppler (Gráfico de Barras) ---
delays = [np.inf, 40, 30, 0] # Inf: sem compensação, 0: ideal
labels = ['No Comp.', 'Partial 40', 'Partial 30', 'Ideal Comp.']
f_limits = np.array([1.8e3, 5e3, 10e3, 18e3])
tempo_util_matriz = np.zeros((len(delays), len(f_limits)))

for i, delay in enumerate(delays):
    if np.isinf(delay) or delay == 0:
        est = np.zeros_like(t) # Sem compensação
        if delay == 0: # Ideal (compensação perfeita, ou seja, a própria shift)
             est = doppler_shift
    else:
        # Cria um vetor de zeros com tamanho 'delay', e concatena com o restante do doppler_shift
        delay_int = int(delay)
        est = np.concatenate((np.zeros(delay_int), doppler_shift[:-delay_int]))

    doppler_corrigido = doppler_shift - est

    for j, limite in enumerate(f_limits):
        # A condição `abs(doppler_corrigido) <= limite` retorna um array booleano
        # 'sum(idx)' conta quantos True existem
        idx = np.abs(doppler_corrigido) <= limite
        tempo_util_matriz[i, j] = np.sum(idx) * dt

# Plota as barras (transpondo a matriz para agrupar por tolerância)
plt.figure(figsize=(10, 6))

# Usa um array de strings para os rótulos do eixo x
x_labels = [f'{l/1e3:.1f}' for l in f_limits]
x = np.arange(len(x_labels))
width = 0.2 # Largura da barra

# Plota uma barra para cada 'delay' (cada linha da matriz transposta)
for i in range(len(delays)):
    offset = i - 1.5 * width
    plt.bar(x + offset * width, tempo_util_matriz[i, :], width, label=labels[i])

# plt.bar(x, tempo_util_matriz.T, width, label=labels) # Tentativa com a transposta do MATLAB 'bar(..., grouped)'

# Configuração do gráfico
plt.xticks(x, x_labels)
plt.xlabel('Doppler Tolerance (kHz)')
plt.ylabel('Link Availability (s)')
plt.legend(loc='upper left')
plt.title('Disponibilidade de Link vs. Tolerância Doppler (Compensação)')
plt.grid(axis='y')
plt.show()

# --- GRÁFICO 3: Doppler Máximo vs. Altitude e Latência ---
altitudes = np.arange(200e3, 800e3 + 200e3, 200e3)
doppler_max = np.zeros_like(altitudes, dtype=float)
latency = np.zeros_like(altitudes, dtype=float)

for i, altitude in enumerate(altitudes):
    R_tmp = Re + altitude
    v_tmp = np.sqrt(G * M / R_tmp)
    omega_tmp = v_tmp / R_tmp

    # Posição do satélite (vetores)
    x_tmp = R_tmp * np.sin(omega_tmp * t)
    y_tmp = R_tmp * np.cos(omega_tmp * t) * np.cos(incl)
    z_tmp = R_tmp * np.cos(omega_tmp * t) * np.sin(incl)

    # Distância (Range)
    d_tmp = np.sqrt((x_tmp - x_usr)**2 + (y_tmp - y_usr)**2 + (z_tmp - z_usr)**2)

    # Velocidade radial e Doppler
    dr_tmp = np.gradient(d_tmp, dt)
    dshift = -f0 * dr_tmp / c

    doppler_max[i] = np.max(np.abs(dshift))
    latency[i] = np.mean(d_tmp) / c

# Cria a figura e os eixos duplos (yyaxis left/right)
fig, ax1 = plt.subplots(figsize=(10, 6))

# Eixo Esquerdo: Doppler Máximo
color = 'tab:blue'
ax1.set_xlabel('Altitude (km)')
ax1.set_ylabel('Max Doppler Shift (kHz)', color=color)
ax1.plot(altitudes/1e3, doppler_max/1e3, '-ob', linewidth=2, color=color, label='Doppler Max')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True)

# Eixo Direito: Latência
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('One-way Propagation Latency (ms)', color=color)
ax2.plot(altitudes/1e3, latency*1e3, '--sr', linewidth=2, color=color, label='Latency')
ax2.tick_params(axis='y', labelcolor=color)

# Linha Vertical (xline) para 550 km
ax1.axvline(550, color='k', linestyle='--', label='550 km (Starlink)')
# Adiciona texto para a linha vertical
ax1.text(550 - 20, ax1.get_ylim()[1] - 5, '550 km (Starlink)',
         color='k', ha='right', verticalalignment='top')


# Adiciona a legenda manualmente, combinando handles
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, ['Doppler Max', 'Latency'], loc='upper right')

plt.title('Max Doppler e Latência vs. Altitude')
plt.show()

# Limpando variáveis (boa prática, embora menos crítica em Python que em MATLAB)
del G, M, Re, c, f0, T_total, t, dt, t_plot, incl_deg, incl, h, R, x_usr, y_usr, z_usr, pos_usr, v_orb, omega, x_sat, y_sat, z_sat, dx, dy, dz, range_t, v_radial, doppler_shift, f_rx_no_comp, f_rx_comp_ideal, doppler_est30, doppler_est40, f_rx_comp30, f_rx_comp40, limits, cores, tempo_total, handles, labels_orig, legend_labels, delays, labels, f_limits, tempo_util_matriz, x_labels, x, width, offset, altitude, R_tmp, v_tmp, omega_tmp, x_tmp, y_tmp, z_tmp, d_tmp, dr_tmp, dshift, doppler_max, latency, fig, ax1, color, ax2, lines, labels2, lines2, delay, delay_int, est, doppler_corrigido, limite, idx, i, j