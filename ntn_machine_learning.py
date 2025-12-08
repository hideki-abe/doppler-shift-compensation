import numpy as np
import matplotlib.pyplot as plt
from skyfield.api import load, EarthSatellite, wgs84
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from datetime import datetime, timedelta
import pickle

# =============================================================================
# 1. COLETA DE DADOS REAIS DA STARLINK
# =============================================================================

def baixar_satelites_starlink(num_satelites=50):
    """Baixa TLEs de múltiplos satélites Starlink do Celestrak"""
    print("Baixando dados da constelação Starlink do Celestrak...")
    
    stations_url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle'
    satellites = load.tle_file(stations_url, reload=True)
    
    print(f"Total de satélites Starlink disponíveis: {len(satellites)}")
    
    # Seleciona satélites com órbitas diferentes
    satelites_selecionados = satellites[:num_satelites]
    
    return satelites_selecionados

# =============================================================================
# 2. GERAÇÃO DE DATASET
# =============================================================================

def gerar_dataset_doppler_d2d(satelites, user_A, user_B, duracao_horas=24, amostras_por_minuto=1):
    """
    Gera dataset para comunicação D2D via satélite (relay)
    
    Cenário: Terminal A → Satélite → Terminal B
    
    Features adicionais:
    - Doppler uplink (A→Sat)
    - Doppler downlink (Sat→B)
    - Doppler total (soma dos dois)
    - Distância A→Sat e Sat→B
    - Elevações de A e B
    """
    
    print(f"\nGerando dataset D2D com {len(satelites)} satélites...")
    print(f"Terminal A: {user_A.latitude.degrees:.2f}°, {user_A.longitude.degrees:.2f}°")
    print(f"Terminal B: {user_B.latitude.degrees:.2f}°, {user_B.longitude.degrees:.2f}°")
    
    ts = load.timescale()
    c = 3e8
    f_uplink = 14e9    # 14 GHz uplink (banda Ku)
    f_downlink = 12e9  # 12 GHz downlink (banda Ku)
    
    # Tempo de simulação
    t0 = ts.now()
    num_amostras = int(duracao_horas * 60 * amostras_por_minuto)
    t = ts.linspace(t0, t0 + duracao_horas/24, num_amostras)
    dt = (duracao_horas * 3600) / num_amostras
    
    features_list = []
    labels_list = []
    metadata_list = []
    
    for idx, sat in enumerate(satelites):
        print(f"Processando satélite {idx+1}/{len(satelites)}: {sat.name}", end='\r')
        
        try:
            # Posições
            geocentric = sat.at(t)
            
            # UPLINK: Terminal A → Satélite
            diff_uplink = geocentric - user_A.at(t)
            dist_uplink = diff_uplink.distance().km
            topo_A = diff_uplink.altaz()
            elev_A = topo_A[0].degrees
            azim_A = topo_A[1].degrees
            
            # DOWNLINK: Satélite → Terminal B
            diff_downlink = geocentric - user_B.at(t)
            dist_downlink = diff_downlink.distance().km
            topo_B = diff_downlink.altaz()
            elev_B = topo_B[0].degrees
            azim_B = topo_B[1].degrees
            
            # Posição geográfica do satélite
            subpoint = wgs84.subpoint(geocentric)
            lat_sat = subpoint.latitude.degrees
            lon_sat = subpoint.longitude.degrees
            alt_sat = subpoint.elevation.km
            
            # DOPPLER UPLINK (A→Sat)
            vel_radial_up = np.gradient(dist_uplink * 1000, dt)  # m/s
            doppler_uplink = f_uplink * vel_radial_up / c / 1e3  # kHz (sinal positivo!)
            
            # DOPPLER DOWNLINK (Sat→B)
            vel_radial_down = np.gradient(dist_downlink * 1000, dt)  # m/s
            doppler_downlink = -f_downlink * vel_radial_down / c / 1e3  # kHz
            
            # DOPPLER TOTAL D2D
            doppler_total_d2d = doppler_uplink + doppler_downlink
            
            # Features derivadas
            taxa_dist_up = np.gradient(dist_uplink, dt)
            taxa_dist_down = np.gradient(dist_downlink, dt)
            taxa_elev_A = np.gradient(elev_A, dt)
            taxa_elev_B = np.gradient(elev_B, dt)
            
            # Velocidade do satélite
            pos_xyz = geocentric.position.km
            vel_xyz = np.gradient(pos_xyz, dt, axis=1)
            velocidade_orbital = np.linalg.norm(vel_xyz, axis=0)
            
            # Construir features D2D
            for i in range(len(t)):
                # Apenas se satélite visível para AMBOS os terminais
                if elev_A[i] > 0 and elev_B[i] > 0:
                    features = [
                        # Features uplink (A→Sat)
                        dist_uplink[i],      # 0: Distância A→Sat
                        elev_A[i],           # 1: Elevação vista de A
                        azim_A[i],           # 2: Azimute vista de A
                        taxa_dist_up[i],     # 3: Taxa variação dist A→Sat
                        taxa_elev_A[i],      # 4: Taxa variação elevação A
                        
                        # Features downlink (Sat→B)
                        dist_downlink[i],    # 5: Distância Sat→B
                        elev_B[i],           # 6: Elevação vista de B
                        azim_B[i],           # 7: Azimute vista de B
                        taxa_dist_down[i],   # 8: Taxa variação dist Sat→B
                        taxa_elev_B[i],      # 9: Taxa variação elevação B
                        
                        # Features do satélite
                        lat_sat[i],          # 10: Latitude satélite
                        lon_sat[i],          # 11: Longitude satélite
                        alt_sat[i],          # 12: Altitude satélite
                        velocidade_orbital[i], # 13: Velocidade orbital
                        vel_xyz[0, i],       # 14: Vx
                        vel_xyz[1, i],       # 15: Vy
                        vel_xyz[2, i],       # 16: Vz
                        
                        # Features adicionais D2D
                        doppler_uplink[i],   # 17: Doppler uplink isolado
                        doppler_downlink[i], # 18: Doppler downlink isolado
                    ]
                    
                    # LABEL: Doppler total D2D
                    features_list.append(features)
                    labels_list.append(doppler_total_d2d[i])
                    metadata_list.append({
                        'sat_name': sat.name,
                        'timestamp': t[i].utc_iso(),
                        'elev_A': elev_A[i],
                        'elev_B': elev_B[i],
                        'doppler_up': doppler_uplink[i],
                        'doppler_down': doppler_downlink[i]
                    })
        
        except Exception as e:
            print(f"\nErro ao processar {sat.name}: {e}")
            continue
    
    print(f"\n\nDataset D2D gerado: {len(features_list)} amostras")
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    return X, y, metadata_list

def gerar_dataset_doppler(satelites, user_location, duracao_horas=24, amostras_por_minuto=1):
    """
    Gera dataset de treinamento com features geométricas e Doppler como label
    
    Features:
    - Distância (km)
    - Elevação (graus)
    - Azimute (graus)
    - Taxa de variação da distância (km/s)
    - Taxa de variação da elevação (graus/s)
    - Latitude do satélite
    - Longitude do satélite
    - Altitude do satélite
    - Velocidade orbital (km/s)
    - Componentes de velocidade (Vx, Vy, Vz)
    
    Label:
    - Doppler shift (kHz)
    """
    
    print(f"\nGerando dataset com {len(satelites)} satélites...")
    print(f"Duração: {duracao_horas}h | Amostras: {amostras_por_minuto}/min")
    
    ts = load.timescale()
    c = 3e8
    f_downlink = 12e9
    
    # Tempo de simulação
    t0 = ts.now()
    num_amostras = int(duracao_horas * 60 * amostras_por_minuto)
    t = ts.linspace(t0, t0 + duracao_horas/24, num_amostras)
    dt = (duracao_horas * 3600) / num_amostras
    
    # Listas para armazenar features e labels
    features_list = []
    labels_list = []
    metadata_list = []
    
    for idx, sat in enumerate(satelites):
        print(f"Processando satélite {idx+1}/{len(satelites)}: {sat.name}", end='\r')
        
        try:
            # Posições do satélite
            geocentric = sat.at(t)
            difference = geocentric - user_location.at(t)
            
            # Métricas básicas
            distancia = difference.distance().km
            topocentric = difference.altaz()
            elevacao = topocentric[0].degrees
            azimute = topocentric[1].degrees
            
            # Posição geográfica do satélite
            subpoint = wgs84.subpoint(geocentric)
            lat_sat = subpoint.latitude.degrees
            lon_sat = subpoint.longitude.degrees
            alt_sat = subpoint.elevation.km
            
            # Velocidade radial e Doppler (LABEL)
            velocidade_radial = np.gradient(distancia * 1000, dt)  # m/s
            doppler_khz = -f_downlink * velocidade_radial / c / 1e3
            
            # Features derivadas
            taxa_dist = np.gradient(distancia, dt)  # km/s
            taxa_elev = np.gradient(elevacao, dt)  # graus/s
            
            # Velocidade do satélite (componentes)
            pos_xyz = geocentric.position.km
            vel_xyz = np.gradient(pos_xyz, dt, axis=1)  # km/s
            velocidade_orbital = np.linalg.norm(vel_xyz, axis=0)
            
            # Construir features para cada instante
            for i in range(len(t)):
                features = [
                    distancia[i],           # 0: Distância (km)
                    elevacao[i],            # 1: Elevação (graus)
                    azimute[i],             # 2: Azimute (graus)
                    taxa_dist[i],           # 3: Taxa variação distância (km/s)
                    taxa_elev[i],           # 4: Taxa variação elevação (graus/s)
                    lat_sat[i],             # 5: Latitude satélite
                    lon_sat[i],             # 6: Longitude satélite
                    alt_sat[i],             # 7: Altitude satélite (km)
                    velocidade_orbital[i],  # 8: Velocidade orbital (km/s)
                    vel_xyz[0, i],          # 9: Vx (km/s)
                    vel_xyz[1, i],          # 10: Vy (km/s)
                    vel_xyz[2, i],          # 11: Vz (km/s)
                ]
                
                # Apenas adiciona se satélite está visível (elevação > 0)
                if elevacao[i] > 0:
                    features_list.append(features)
                    labels_list.append(doppler_khz[i])
                    metadata_list.append({
                        'sat_name': sat.name,
                        'timestamp': t[i].utc_iso(),
                        'elevacao': elevacao[i]
                    })
        
        except Exception as e:
            print(f"\nErro ao processar {sat.name}: {e}")
            continue
    
    print(f"\n\nDataset gerado: {len(features_list)} amostras")
    
    # Converter para arrays numpy
    X = np.array(features_list)
    y = np.array(labels_list)
    
    return X, y, metadata_list

# =============================================================================
# 3. TREINAMENTO DE MODELOS
# =============================================================================

def treinar_modelos(X, y):
    """Treina múltiplos modelos de ML para predição de Doppler"""
    
    print("\n" + "="*80)
    print("TREINAMENTO DE MODELOS DE MACHINE LEARNING")
    print("="*80)
    
    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Dicionário de modelos
    modelos = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100, 
            max_depth=20, 
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ),
        'Neural Network (MLP)': MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            max_iter=500,
            random_state=42
        )
    }
    
    # Treinar e avaliar cada modelo
    resultados = {}
    
    for nome, modelo in modelos.items():
        print(f"\nTreinando {nome}...")
        
        # Treinar
        modelo.fit(X_train_scaled, y_train)
        
        # Predições
        y_pred_train = modelo.predict(X_train_scaled)
        y_pred_test = modelo.predict(X_test_scaled)
        
        # Métricas
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2_test = r2_score(y_test, y_pred_test)
        
        resultados[nome] = {
            'modelo': modelo,
            'mae_train': mae_train,
            'mae_test': mae_test,
            'rmse_test': rmse_test,
            'r2_test': r2_test,
            'y_pred_test': y_pred_test
        }
        
        print(f"  MAE Treino: {mae_train:.3f} kHz")
        print(f"  MAE Teste:  {mae_test:.3f} kHz")
        print(f"  RMSE Teste: {rmse_test:.3f} kHz")
        print(f"  R² Teste:   {r2_test:.4f}")
    
    return resultados, scaler, X_test, y_test

# =============================================================================
# 4. VISUALIZAÇÃO E COMPARAÇÃO
# =============================================================================

def visualizar_resultados(resultados, X_test, y_test):
    """Visualiza performance dos modelos"""
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Gráfico 1: Comparação de Métricas
    ax1 = axes[0, 0]
    modelos_nomes = list(resultados.keys())
    mae_values = [resultados[m]['mae_test'] for m in modelos_nomes]
    rmse_values = [resultados[m]['rmse_test'] for m in modelos_nomes]
    
    x_pos = np.arange(len(modelos_nomes))
    width = 0.35
    
    ax1.bar(x_pos - width/2, mae_values, width, label='MAE', color='#3498DB', alpha=0.8)
    ax1.bar(x_pos + width/2, rmse_values, width, label='RMSE', color='#E74C3C', alpha=0.8)
    ax1.set_xlabel('Modelo', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Erro (kHz)', fontsize=12, fontweight='bold')
    ax1.set_title('Comparação de Erros - MAE vs RMSE', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(modelos_nomes, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Gráfico 2: R² Score
    ax2 = axes[0, 1]
    r2_values = [resultados[m]['r2_test'] for m in modelos_nomes]
    bars = ax2.bar(modelos_nomes, r2_values, color='#2ECC71', alpha=0.8)
    ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax2.set_title('Coeficiente de Determinação (R²)', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Gráfico 3: Predição vs Real (Melhor modelo)
    ax3 = axes[1, 0]
    melhor_modelo = min(resultados.keys(), key=lambda k: resultados[k]['mae_test'])
    y_pred_best = resultados[melhor_modelo]['y_pred_test']
    
    ax3.scatter(y_test, y_pred_best, alpha=0.3, s=10, color='#3498DB')
    ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', linewidth=2, label='Predição Perfeita')
    ax3.set_xlabel('Doppler Real (kHz)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Doppler Predito (kHz)', fontsize=12, fontweight='bold')
    ax3.set_title(f'Predição vs Real - {melhor_modelo}', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Distribuição de Erros
    ax4 = axes[1, 1]
    for nome in modelos_nomes:
        y_pred = resultados[nome]['y_pred_test']
        erros = y_test - y_pred
        ax4.hist(erros, bins=50, alpha=0.5, label=nome, density=True)
    
    ax4.set_xlabel('Erro de Predição (kHz)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Densidade', fontsize=12, fontweight='bold')
    ax4.set_title('Distribuição dos Erros de Predição', fontsize=14, fontweight='bold')
    ax4.axvline(0, color='k', linestyle='--', linewidth=2)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 5. COMPENSAÇÃO EM TEMPO REAL
# =============================================================================

def compensacao_ml_tempo_real(satelite, user, modelo, scaler, duracao_min=10):
    """Simula compensação de Doppler em tempo real usando ML"""
    
    print("\n" + "="*80)
    print("SIMULAÇÃO DE COMPENSAÇÃO EM TEMPO REAL COM ML")
    print("="*80)
    
    ts = load.timescale()
    c = 3e8
    f_downlink = 12e9
    
    # Simulação
    t0 = ts.now()
    num_amostras = duracao_min * 60 * 2  # 2 amostras/segundo
    t = ts.linspace(t0, t0 + duracao_min/1440, num_amostras)
    dt = (duracao_min * 60) / num_amostras
    tempo_s = np.linspace(0, duracao_min*60, num_amostras)
    
    # Calcular posições e Doppler real
    geocentric = satelite.at(t)
    difference = geocentric - user.at(t)
    
    distancia = difference.distance().km
    topocentric = difference.altaz()
    elevacao = topocentric[0].degrees
    azimute = topocentric[1].degrees
    
    subpoint = wgs84.subpoint(geocentric)
    lat_sat = subpoint.latitude.degrees
    lon_sat = subpoint.longitude.degrees
    alt_sat = subpoint.elevation.km
    
    velocidade_radial = np.gradient(distancia * 1000, dt)
    doppler_real = -f_downlink * velocidade_radial / c / 1e3
    
    # Features para predição ML
    taxa_dist = np.gradient(distancia, dt)
    taxa_elev = np.gradient(elevacao, dt)
    pos_xyz = geocentric.position.km
    vel_xyz = np.gradient(pos_xyz, dt, axis=1)
    velocidade_orbital = np.linalg.norm(vel_xyz, axis=0)
    
    # Construir features
    X_realtime = np.column_stack([
        distancia, elevacao, azimute, taxa_dist, taxa_elev,
        lat_sat, lon_sat, alt_sat, velocidade_orbital,
        vel_xyz[0], vel_xyz[1], vel_xyz[2]
    ])
    
    # Normalizar e predizer
    X_realtime_scaled = scaler.transform(X_realtime)
    doppler_ml_pred = modelo.predict(X_realtime_scaled)
    
    # ========================================================================
    # CORREÇÃO PRINCIPAL: Simulação correta dos métodos
    # ========================================================================
    
    # 1. SEM COMPENSAÇÃO (baseline)
    doppler_sem_comp = doppler_real  # Erro total = Doppler real
    
    # 2. COMPENSAÇÃO TRADICIONAL (Pré-compensação GPS+ajuste)
    # Satélite estima Doppler com precisão de ~95% e pré-compensa
    # Erro residual = 5% do Doppler original
    erro_tradicional = doppler_real * 0.05  # 5% do Doppler não compensado
    doppler_tradicional = erro_tradicional  # Apenas o erro residual
    
    # 3. COMPENSAÇÃO ML
    # ML prediz Doppler com alta precisão
    # Erro residual = diferença entre real e predito
    doppler_ml_comp = doppler_real - doppler_ml_pred
    
    # ========================================================================
    # VISUALIZAÇÃO
    # ========================================================================
    
    fig, axes = plt.subplots(3, 1, figsize=(18, 14))
    
    # -------------------------------------------------------------------------
    # Gráfico 1: Doppler ao longo do tempo
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(tempo_s, doppler_sem_comp, 'r-', linewidth=2.5, 
             label=f'Sem Compensação (Max: ±{np.max(np.abs(doppler_sem_comp)):.1f} kHz)', alpha=0.7)
    ax1.plot(tempo_s, doppler_tradicional, 'b-', linewidth=2.5, 
             label=f'Tradicional (5% erro residual, Max: ±{np.max(np.abs(doppler_tradicional)):.1f} kHz)')
    ax1.plot(tempo_s, doppler_ml_comp, 'g-', linewidth=2.5, 
             label=f'ML (Max: ±{np.max(np.abs(doppler_ml_comp)):.2f} kHz)')
    ax1.axhline(0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    
    # Zonas de tolerância
    ax1.axhspan(-1.8, 1.8, alpha=0.12, color='green', label='Zona ±1.8 kHz (LTE)')
    ax1.axhspan(-5, 5, alpha=0.08, color='yellow', label='Zona ±5 kHz (5G)')
    ax1.axhspan(-10, 10, alpha=0.05, color='orange', label='Zona ±10 kHz (5G Enhanced)')
    
    ax1.set_xlabel('Tempo (segundos)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Desvio Doppler (kHz)', fontsize=13, fontweight='bold')
    ax1.set_title('Compensação de Doppler: Comparação de Métodos', fontsize=15, fontweight='bold', pad=15)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3, linewidth=1)
    ax1.tick_params(labelsize=11)
    
    # -------------------------------------------------------------------------
    # Gráfico 2: Erro absoluto (escala logarítmica)
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    erro_sem_comp = np.abs(doppler_sem_comp)
    erro_trad = np.abs(doppler_tradicional)
    erro_ml = np.abs(doppler_ml_comp)
    
    ax2.plot(tempo_s, erro_sem_comp, 'r-', linewidth=2, label='Sem Compensação', alpha=0.6)
    ax2.plot(tempo_s, erro_trad, 'b-', linewidth=2.5, label='Tradicional (GPS+Pré-comp)')
    ax2.plot(tempo_s, erro_ml, 'g-', linewidth=2.5, label='Machine Learning')
    
    # Linhas de referência
    ax2.axhline(1.8, color='#27AE60', linestyle='--', label='Limite LTE (±1.8 kHz)', linewidth=2, alpha=0.8)
    ax2.axhline(5, color='#F39C12', linestyle='--', label='Limite 5G (±5 kHz)', linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Tempo (segundos)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Erro Absoluto (kHz)', fontsize=13, fontweight='bold')
    ax2.set_title('Erro de Compensação (Escala Logarítmica)', fontsize=15, fontweight='bold', pad=15)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3, linewidth=1, which='both')
    ax2.set_yscale('log')
    ax2.tick_params(labelsize=11)
    
    # -------------------------------------------------------------------------
    # Gráfico 3: Comparação de disponibilidade
    # -------------------------------------------------------------------------
    ax3 = axes[2]
    
    tolerancias = [1.8, 5, 10, 18]
    disponibilidade_sem = [100*np.sum(erro_sem_comp <= tol)/len(erro_sem_comp) for tol in tolerancias]
    disponibilidade_trad = [100*np.sum(erro_trad <= tol)/len(erro_trad) for tol in tolerancias]
    disponibilidade_ml = [100*np.sum(erro_ml <= tol)/len(erro_ml) for tol in tolerancias]
    
    x = np.arange(len(tolerancias))
    width = 0.25
    
    bars1 = ax3.bar(x - width, disponibilidade_sem, width, label='Sem Compensação', 
                    color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax3.bar(x, disponibilidade_trad, width, label='Tradicional', 
                    color='#3498DB', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars3 = ax3.bar(x + width, disponibilidade_ml, width, label='Machine Learning', 
                    color='#2ECC71', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Adicionar valores nas barras
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3.set_xlabel('Tolerância Doppler (kHz)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Disponibilidade do Link (%)', fontsize=13, fontweight='bold')
    ax3.set_title('Disponibilidade por Método de Compensação', fontsize=15, fontweight='bold', pad=15)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'±{tol}' for tol in tolerancias], fontsize=11)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y', linewidth=1)
    ax3.set_ylim([0, 105])
    ax3.tick_params(labelsize=11)
    
    plt.tight_layout()
    plt.show()
    
    # ========================================================================
    # ESTATÍSTICAS DETALHADAS
    # ========================================================================
    
    print(f"\n{'='*80}")
    print(f"SATÉLITE: {satelite.name}")
    print(f"{'='*80}")
    
    print(f"\n📊 CARACTERÍSTICAS DO DOPPLER:")
    print(f"  Doppler máximo (sem compensação): ±{np.max(np.abs(doppler_real)):.2f} kHz")
    print(f"  Doppler médio (sem compensação): ±{np.mean(np.abs(doppler_real)):.2f} kHz")
    print(f"  Taxa de variação máxima: {np.max(np.abs(np.gradient(doppler_real, dt/60))):.3f} kHz/min")
    
    print(f"\n{'='*80}")
    print(f"COMPARAÇÃO DE MÉTODOS DE COMPENSAÇÃO")
    print(f"{'='*80}")
    
    # Sem compensação
    print(f"\n❌ SEM COMPENSAÇÃO:")
    print(f"  Erro médio: {np.mean(erro_sem_comp):.3f} kHz")
    print(f"  Erro máximo: {np.max(erro_sem_comp):.3f} kHz")
    print(f"  Disponibilidade ±1.8 kHz: {100*np.sum(erro_sem_comp<=1.8)/len(erro_sem_comp):.1f}%")
    print(f"  Disponibilidade ±5 kHz: {100*np.sum(erro_sem_comp<=5)/len(erro_sem_comp):.1f}%")
    
    # Tradicional
    print(f"\n🔵 COMPENSAÇÃO TRADICIONAL (GPS + Pré-compensação):")
    print(f"  Erro médio: {np.mean(erro_trad):.3f} kHz")
    print(f"  Erro máximo: {np.max(erro_trad):.3f} kHz")
    print(f"  Disponibilidade ±1.8 kHz: {100*np.sum(erro_trad<=1.8)/len(erro_trad):.1f}%")
    print(f"  Disponibilidade ±5 kHz: {100*np.sum(erro_trad<=5)/len(erro_trad):.1f}%")
    print(f"  Redução vs sem compensação: {(1-np.mean(erro_trad)/np.mean(erro_sem_comp))*100:.1f}%")
    
    # Machine Learning
    print(f"\n🟢 COMPENSAÇÃO MACHINE LEARNING:")
    print(f"  Erro médio: {np.mean(erro_ml):.3f} kHz")
    print(f"  Erro máximo: {np.max(erro_ml):.3f} kHz")
    print(f"  Disponibilidade ±1.8 kHz: {100*np.sum(erro_ml<=1.8)/len(erro_ml):.1f}%")
    print(f"  Disponibilidade ±5 kHz: {100*np.sum(erro_ml<=5)/len(erro_ml):.1f}%")
    print(f"  Redução vs sem compensação: {(1-np.mean(erro_ml)/np.mean(erro_sem_comp))*100:.1f}%")
    
    print(f"\n{'='*80}")
    print(f"🏆 VANTAGEM DO ML SOBRE TRADICIONAL")
    print(f"{'='*80}")
    melhoria_erro_medio = ((np.mean(erro_trad) - np.mean(erro_ml)) / np.mean(erro_trad)) * 100
    melhoria_erro_max = ((np.max(erro_trad) - np.max(erro_ml)) / np.max(erro_trad)) * 100
    
    print(f"  Redução do erro médio: {melhoria_erro_medio:.1f}%")
    print(f"  Redução do erro máximo: {melhoria_erro_max:.1f}%")
    
    if np.mean(erro_ml) < np.mean(erro_trad):
        print(f"  ✅ ML é {np.mean(erro_trad)/np.mean(erro_ml):.1f}x mais preciso que o método tradicional!")
    else:
        print(f"  ⚠️ ML teve performance inferior (possível overfitting ou dados insuficientes)")
    
    print(f"{'='*80}\n")
    
# =============================================================================
# 6. SCRIPT PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    
    # Configurações D2D
    user_A = wgs84.latlon(-23.55, -46.63)  # Terminal A: São Paulo
    user_B = wgs84.latlon(-22.90, -43.17)  # Terminal B: Rio de Janeiro
    
    print("="*80)
    print("SISTEMA DE COMPENSAÇÃO DOPPLER D2D COM MACHINE LEARNING")
    print("Comunicação: Terminal A → Satélite → Terminal B")
    print("="*80)
    
    # ETAPA 1: Baixar dados
    satelites = baixar_satelites_starlink(num_satelites=20)
    
    # ETAPA 2: Gerar dataset D2D
    X, y, metadata = gerar_dataset_doppler_d2d(
        satelites, 
        user_A, 
        user_B,
        duracao_horas=6,
        amostras_por_minuto=2
    )
    
    print(f"\nShape do dataset D2D: X={X.shape}, y={y.shape}")
    print(f"Range Doppler D2D: [{y.min():.2f}, {y.max():.2f}] kHz")
    print(f"  (Uplink + Downlink combinados)")
    
    # ETAPA 3:
    resultados, scaler, X_test, y_test = treinar_modelos(X, y)
    
    # ETAPA 4: Visualizar resultados
    visualizar_resultados(resultados, X_test, y_test)
    
    # ETAPA 5: Selecionar melhor modelo e testar em tempo real
    melhor_modelo_nome = min(resultados.keys(), key=lambda k: resultados[k]['mae_test'])
    melhor_modelo = resultados[melhor_modelo_nome]['modelo']
    
    print(f"\n🏆 MELHOR MODELO: {melhor_modelo_nome}")
    print(f"   MAE: {resultados[melhor_modelo_nome]['mae_test']:.3f} kHz")
    print(f"   R²: {resultados[melhor_modelo_nome]['r2_test']:.4f}")
    
    # Testar com um satélite específico
    sat_teste = satelites[0]
    compensacao_ml_tempo_real(sat_teste, user, melhor_modelo, scaler, duracao_min=10)
    
    # ETAPA 6: Salvar modelo treinado
    print("\nSalvando modelo treinado...")
    with open('modelo_doppler_ml.pkl', 'wb') as f:
        pickle.dump({'modelo': melhor_modelo, 'scaler': scaler}, f)
    print("✓ Modelo salvo em 'modelo_doppler_ml.pkl'")
    
    print("\n" + "="*80)
    print("PROCESSO CONCLUÍDO COM SUCESSO!")
    print("="*80)