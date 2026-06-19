# Arquivo: src/main_splits.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ShuffleSplit

from data_loader import DataLoader
from feature_extractor import FeatureExtractor
from anomaly_detector import IsolationForestDetector, LOFDetector, OneClassSVMDetector

def run_split_experiment():
    print("Iniciando Análise Estatística de Tamanho de Treino/Teste...\n")
    
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    data_dir = os.path.join(base_dir, '..', 'data')       
    
    loader = DataLoader(diretorio_dados=data_dir)
    extractor = FeatureExtractor(tamanho_janela=2048)
    
    print("[1/3] Carregando dados e extraindo features...")
    X_p1 = extractor.extrair(loader.carregar_sinais('R1F1L1P1.mat'))
    X_p2 = extractor.extrair(loader.carregar_sinais('R1F1L1P2.mat'))
    X_p10 = extractor.extrair(loader.carregar_sinais('R1F1L1P10.mat'))
    
    # Configurações do Experimento Estatístico
    # Testar treinar com 50%, 60%, 70% e 80% do dado Saudável
    tamanhos_treino = [0.5, 0.6, 0.7, 0.8] 
    num_repeticoes = 20
    resultados = [] 
    
    print(f"[2/3] Iniciando o loop de treinamento ({num_repeticoes} repetições por tamanho)...")
    
    for proporcao in tamanhos_treino:
        print(f" -> Avaliando com {int(proporcao*100)}% de Treino e {int((1-proporcao)*100)}% de Teste...")
        # O ShuffleSplit vai embaralhar e fatiar o P1 (Saudável) 20 vezes diferentes
        rs = ShuffleSplit(n_splits=num_repeticoes, train_size=proporcao, random_state=42)
        
        repeticao = 1
        for indice_treino, indice_teste in rs.split(X_p1):
            X_treino_rodada = X_p1[indice_treino]
            X_teste_p1_rodada = X_p1[indice_teste] 
            
            # Instanciando os modelos (zerados para cada repetição)
            modelos = {
                'Isolation Forest': IsolationForestDetector(),
                'LOF': LOFDetector(),
                'One-Class SVM': OneClassSVMDetector(nu=0.05)
            }
            
            for nome_modelo, modelo in modelos.items():
                modelo.treinar(X_treino_rodada)
                # Avalia Falso Positivo no Teste Saudável (P1)
                prev_p1 = modelo.modelo.predict(modelo.scaler.transform(X_teste_p1_rodada))
                taxa_erro_p1 = (np.sum(prev_p1 == -1) / len(prev_p1)) * 100
                
                # Avalia Detecção da Falha Leve (P2)
                prev_p2 = modelo.modelo.predict(modelo.scaler.transform(X_p2))
                taxa_acerto_p2 = (np.sum(prev_p2 == -1) / len(prev_p2)) * 100
                
                # Avalia Detecção da Falha Severa (P10)
                prev_p10 = modelo.modelo.predict(modelo.scaler.transform(X_p10))
                taxa_acerto_p10 = (np.sum(prev_p10 == -1) / len(prev_p10)) * 100
                
                # Salvando os dados 
                resultados.append({
                    'Tamanho_Treino (%)': int(proporcao * 100),
                    'Repeticao': repeticao,
                    'Modelo': nome_modelo,
                    'Falsos_Alarmes_Saudavel (%)': taxa_erro_p1,
                    'Deteccao_P2_Leve (%)': taxa_acerto_p2,
                    'Deteccao_P10_Severa (%)': taxa_acerto_p10
                })
                
            repeticao += 1

    print("[3/3] Consolidando resultados e gerando planilhas/gráficos...")
    df_resultados = pd.DataFrame(resultados)
  
    df_resultados.to_csv('resultados_splits_treino.csv', index=False)
    print(" -> Arquivo 'resultados_splits_treino.csv' foi salvo.")
    
    # Gera os gráficos
    plotar_resultados(df_resultados)

def plotar_resultados(df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico 1: Acurácia no P2 (Falha Leve)
    # O seaborn desenha a linha da Média e a "sombra" do Desvio Padrão automaticamente
    sns.lineplot(data=df, x='Tamanho_Treino (%)', y='Deteccao_P2_Leve (%)',  hue='Modelo', marker='o', errorbar='sd', ax=axes[0])
    axes[0].set_title('Estabilidade: Detecção da Falha Leve (P2)')
    axes[0].set_xlabel('Quantidade de Dados de Treino (%)')
    axes[0].set_ylabel('Taxa de Detecção (%)')
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Gráfico 2: Falsos Alarmes no P1 (Erro)
    sns.lineplot(data=df, x='Tamanho_Treino (%)', y='Falsos_Alarmes_Saudavel (%)', hue='Modelo', marker='s', errorbar='sd', ax=axes[1])
    axes[1].set_title('Robustez: Falsos Alarmes na Máquina Saudável')
    axes[1].set_xlabel('Quantidade de Dados de Treino (%)')
    axes[1].set_ylabel('Falsos Alarmes (%)')
    axes[1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_split_experiment()