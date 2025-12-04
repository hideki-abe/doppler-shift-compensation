import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm # Para cores (lines(N))

# === Função Auxiliar (Replica 'plot_inclination_curve' do MATLAB) ===
def plot_inclination_curve(incl_deg, linestyle, linewidth, color,
                           t, omega, R, x_usr, y_usr, z_usr, dt, f0, c, t_plot, figures):
    """
    Calcula a trajetória do satélite, o desvio Doppler, elevação, distância e azimute
    para uma dada inclinação, e plota os resultados nas figuras Matplotlib especificadas.
    """
    incl = np.deg2rad(incl_deg)

    # 1. Trajetória do Satélite
    # Assumindo órbita circular e a UE no Equador para simplificar a transformação
    x_sat = R * np.sin(omega * t)
    y_sat = R * np.cos(omega * t) * np.cos(incl)
    z_sat = R * np.cos(omega * t) * np.sin(incl)

    # 2. Vetor de Distância
    dx = x_sat - x_usr
    dy = y_sat - y_usr
    dz = z_sat - z_usr
    range_t = np.sqrt(dx**2 + dy**2 + dz**2)

    # 3. Elevação (Ângulo de Olhada acima do horizonte, em graus)
    # asind(x) é arcsen(x) em graus. (dz/range_t) é sin(Elevação)
    # np.arcsin retorna em radianos, multiplicamos por 180/pi para converter para graus.
    elevation = np.arcsin(dz / range_t) * 180 / np.pi

    # 4. Azimute (Ângulo horizontal, em graus)
    # atan2d(y, x) em MATLAB é atan2(y, x) em Python e depois * 180/pi.
    # O MATLAB usa (dy, dx) para azimute em coordenadas terrestres, mas neste ECI/GCRF
    # simplificado, a projeção no plano (X, Z) é frequentemente usada para a definição de azimute.
    # Baseado na implementação do MATLAB (atan2d(dx, dz)):
    # Azimute no plano XZ: Eixo Z é 'y' e Eixo X é 'x' para atan2.
    azimuth = np.arctan2(dx, dz) * 180 / np.pi
    azimuth = np.mod(azimuth, 360) # Modulo 360 para garantir 0 a 360

    # 5. Velocidade Radial e Doppler
    v_radial = np.gradient(range_t, dt)
    doppler = -f0 * v_radial / c

    # Rótulo da inclinação para a legenda (usa notação LaTeX)
    label = f'$\\theta = {incl_deg:.1f}^\\circ$'

    # Plotagem nas figuras
    # figure(1): Doppler Shift (kHz)
    figures[0].plot(t_plot, doppler / 1e3, color=color, linestyle=linestyle,
                    linewidth=linewidth, label=label)

    # figure(2): Elevation (degrees)
    figures[1].plot(t_plot, elevation, color=color, linestyle=linestyle,
                    linewidth=linewidth, label=label)

    # figure(3): Distance (km)
    figures[2].plot(t_plot, range_t / 1e3, color=color, linestyle=linestyle,
                    linewidth=linewidth, label=label)

    # figure(4): Azimuth (degrees)
    figures[3].plot(t_plot, azimuth, color=color, linestyle=linestyle,
                    linewidth=linewidth, label=label)


# =====================================================================
# === Execução do Script Principal ===
# =====================================================================

# === Parâmetros ===
G = 6.67430e-11
M = 5.972e24
Re = 6371e3
h = 550e3
R = Re + h
f0 = 3.5e9
c = 3e8

# === Parâmetros Temporais ===
T_total = 1200
t = np.linspace(-T_total/2, T_total/2, 1200)
dt = np.mean(np.diff(t))
t_plot = t + T_total/2

# === Velocidade Orbital ===
v_orb = np.sqrt(G * M / R)
omega = v_orb / R

# === Inclinação ===
inclinations = np.arange(0, 91, 15)
highlight_incl = 53  # Starlink Group 1

# Gera um mapa de cores (lines(N) do MATLAB)
colors = cm.get_cmap('hsv', len(inclinations) + 1)(range(len(inclinations) + 1))
colors[1] = colors[2] # Ajuste visual para separar o 0 do 15

# === Posição Inicial da UE ===
x_usr = 0
y_usr = 0
z_usr = Re

# === Inicialização das Figuras ===
# Cria 4 figuras e seus respectivos eixos
fig1, ax1 = plt.subplots(figsize=(10, 6))
fig2, ax2 = plt.subplots(figsize=(10, 6))
fig3, ax3 = plt.subplots(figsize=(10, 6))
fig4, ax4 = plt.subplots(figsize=(10, 6))
axes = [ax1, ax2, ax3, ax4]

# === Loop de Plotagem ===
# Plota todas as inclinações
for i, incl_deg in enumerate(inclinations):
    plot_inclination_curve(incl_deg, '-', 2, colors[i],
                           t, omega, R, x_usr, y_usr, z_usr, dt, f0, c, t_plot, axes)

# Destaca uma inclinação específica (53 graus)
plot_inclination_curve(highlight_incl, '--', 3, 'k',
                       t, omega, R, x_usr, y_usr, z_usr, dt, f0, c, t_plot, axes)

# === Ajustes Finais para Apresentação ===
figures = [fig1, fig2, fig3, fig4]
ylabels = ['Doppler Shift (kHz)', 'Elevation (degrees)',
           'Distance (km)', 'Azimuth (degrees)']
leglocs = ['lower left', 'lower left', 'upper left', 'upper left'] # Mapeamento de 'southwest' e 'northwest'

for k in range(4):
    ax = axes[k]
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel(ylabels[k], fontsize=14)
    ax.legend(loc=leglocs[k], fontsize=12)
    ax.grid(True)
    ax.set_xlim([0, T_total])
    ax.set_xticks(np.arange(0, T_total + 1, 200))
    # ax.tick_params(labelsize=14) # Substitui 'set(gca, 'FontSize', 14)'

    if k == 3: # Azimuth
        ax.set_ylim([0, 360])

    figures[k].tight_layout() # Ajusta o layout para evitar sobreposição

# === Exportando Figuras (Geração de PDF) ===
fig_names = ['dopplershift_py', 'elevation_py', 'distance_py', 'azimuth_py']

print("\nGerando arquivos PDF dos gráficos...")

for k in range(4):
    # O Matplotlib não usa 'PaperPositionMode' como o MATLAB,
    # usa-se savefig com dpi e bbox_inches='tight' para controlar o tamanho.
    figures[k].savefig(f'{fig_names[k]}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print(f"Salvo: {fig_names[k]}.pdf")

plt.show() # Exibe todos os gráficos gerados