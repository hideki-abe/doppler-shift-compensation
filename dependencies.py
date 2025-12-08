import subprocess
import sys

def verificar_python():
    """Verifica versão do Python"""
    print("="*60)
    print("VERIFICANDO PYTHON")
    print("="*60)
    print(f"Python: {sys.version}")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ é necessário!")
        sys.exit(1)
    print("✓ Versão adequada\n")

def atualizar_pip():
    """Atualiza pip"""
    print("="*60)
    print("ATUALIZANDO PIP")
    print("="*60)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    print("✓ pip atualizado\n")

def instalar_pacotes():
    """Instala todos os pacotes necessários"""
    print("="*60)
    print("INSTALANDO DEPENDÊNCIAS")
    print("="*60)
    
    pacotes = [
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "skyfield>=1.42",
        "requests>=2.28.0",
        "scipy>=1.7.0"
    ]
    
    for pacote in pacotes:
        print(f"\n📦 Instalando: {pacote}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pacote])
            print(f"✓ {pacote} instalado com sucesso")
        except subprocess.CalledProcessError:
            print(f"❌ Erro ao instalar {pacote}")
    
    print("\n")

def verificar_instalacoes():
    """Verifica se todos os pacotes foram instalados"""
    print("="*60)
    print("VERIFICANDO INSTALAÇÕES")
    print("="*60)
    
    modulos = {
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'sklearn': 'Scikit-learn',
        'pandas': 'Pandas',
        'skyfield': 'Skyfield',
        'requests': 'Requests',
        'scipy': 'SciPy'
    }
    
    todos_ok = True
    
    for modulo, nome in modulos.items():
        try:
            mod = __import__(modulo)
            versao = getattr(mod, '__version__', 'versão desconhecida')
            print(f"✓ {nome:15s} {versao}")
        except ImportError:
            print(f"❌ {nome:15s} NÃO INSTALADO")
            todos_ok = False
    
    print("\n" + "="*60)
    if todos_ok:
        print("✅ TODAS AS DEPENDÊNCIAS INSTALADAS COM SUCESSO!")
    else:
        print("⚠️ ALGUMAS DEPENDÊNCIAS NÃO FORAM INSTALADAS")
    print("="*60)

def main():
    print("\n" + "="*60)
    print("INSTALADOR DE DEPENDÊNCIAS")
    print("Doppler Shift Compensation - Machine Learning")
    print("="*60 + "\n")
    
    try:
        verificar_python()
        atualizar_pip()
        instalar_pacotes()
        verificar_instalacoes()
        
        print("\n✨ Instalação concluída! Execute seu script principal agora.\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Instalação cancelada pelo usuário.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro durante instalação: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()