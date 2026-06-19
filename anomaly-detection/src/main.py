import os
from data_loader import DataLoader
from feature_extractor import FeatureExtractor
from anomaly_detector import IsolationForestDetector, LOFDetector, OneClassSVMDetector

def run_experiment():
    print("Iniciando Experimento - O Combate dos 3 Baselines Clássicos...\n")
    
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    data_dir = os.path.join(base_dir, '..', 'data')       
    
    loader = DataLoader(diretorio_dados=data_dir)
    extractor = FeatureExtractor(tamanho_janela=2048)
    
    iforest = IsolationForestDetector()
    lof = LOFDetector()
    ocsvm = OneClassSVMDetector(nu=0.05) 
    
    print("[1/4] Carregando arquivos .mat...")
    sinais_p1 = loader.carregar_sinais('R1F1L1P1.mat')
    sinais_p2 = loader.carregar_sinais('R1F1L1P2.mat')
    sinais_p10 = loader.carregar_sinais('R1F1L1P10.mat')
    
    print("[2/4] Extraindo Características (RMS e Curtose)...")
    X_p1 = extractor.extrair(sinais_p1)
    X_p2 = extractor.extrair(sinais_p2)
    X_p10 = extractor.extrair(sinais_p10)
    
    tamanho_treino = int(len(X_p1) * 0.8)
    X_treino_saudavel = X_p1[:tamanho_treino]
    X_teste_saudavel_restante = X_p1[tamanho_treino:]
    
    print("[3/4] Treinando: Isolation Forest, LOF e One-Class SVM...")
    iforest.treinar(X_treino_saudavel)
    lof.treinar(X_treino_saudavel)
    ocsvm.treinar(X_treino_saudavel)
    
    print("[4/4] Avaliando Modelos...\n")
    
    # ---------------- AVALIAÇÃO I-FOREST ----------------
    print("="*55)
    print(" 1. RESULTADOS DO ISOLATION FOREST (Particionamento)")
    print("="*55)
    iforest.avaliar(X_teste_saudavel_restante, "Engrenagem Saudável")
    iforest.avaliar(X_p2, "Falha Leve (P2)")
    iforest.avaliar(X_p10, "Falha Severa (P10)")

    # ---------------- AVALIAÇÃO LOF ----------------
    print("="*55)
    print(" 2. RESULTADOS DO LOCAL OUTLIER FACTOR (Densidade)")
    print("="*55)
    lof.avaliar(X_teste_saudavel_restante, "Engrenagem Saudável")
    lof.avaliar(X_p2, "Falha Leve (P2)")
    lof.avaliar(X_p10, "Falha Severa (P10)")
    
    # ---------------- AVALIAÇÃO ONE-CLASS SVM ----------------
    print("="*55)
    print(" 3. RESULTADOS DO ONE-CLASS SVM (Fronteira RBF)")
    print("="*55)
    ocsvm.avaliar(X_teste_saudavel_restante, "Engrenagem Saudável")
    ocsvm.avaliar(X_p2, "Falha Leve (P2)")
    ocsvm.avaliar(X_p10, "Falha Severa (P10)")

if __name__ == "__main__":
    run_experiment()