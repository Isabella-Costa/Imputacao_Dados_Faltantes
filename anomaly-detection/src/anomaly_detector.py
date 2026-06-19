import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM # <-- Nova biblioteca do SVM

class IsolationForestDetector:
    """Modelo de Baseline utilizando árvores (Isolation Forest)."""
    def __init__(self, contamination='auto', random_state=42):
        self.scaler = StandardScaler()
        self.modelo = IsolationForest(
            n_estimators=100, 
            contamination=contamination, 
            random_state=random_state
        )

    def treinar(self, X_treino):
        X_treino_norm = self.scaler.fit_transform(X_treino)
        self.modelo.fit(X_treino_norm)

    def avaliar(self, X_teste, nome_cenario):
        X_teste_norm = self.scaler.transform(X_teste)
        previsoes = self.modelo.predict(X_teste_norm)
        
        qtd_normal = np.sum(previsoes == 1)
        qtd_anomalia = np.sum(previsoes == -1)
        total = len(previsoes)
        
        print(f"--- [ISOLATION FOREST] Cenário: {nome_cenario} ---")
        print(f"Estado Normal   (Saudável): {qtd_normal} ({qtd_normal/total*100:.2f}%)")
        print(f"Estado Anômalo  (Falha):    {qtd_anomalia} ({qtd_anomalia/total*100:.2f}%)\n")
        
        return previsoes


class LOFDetector:
    """Modelo de Baseline baseado em densidade (Local Outlier Factor)."""
    def __init__(self, n_neighbors=20, contamination='auto'):
        self.scaler = StandardScaler()
        self.modelo = LocalOutlierFactor(
            n_neighbors=n_neighbors, 
            contamination=contamination, 
            novelty=True
        )

    def treinar(self, X_treino):
        X_treino_norm = self.scaler.fit_transform(X_treino)
        self.modelo.fit(X_treino_norm)

    def avaliar(self, X_teste, nome_cenario):
        X_teste_norm = self.scaler.transform(X_teste)
        previsoes = self.modelo.predict(X_teste_norm)
        
        qtd_normal = np.sum(previsoes == 1)
        qtd_anomalia = np.sum(previsoes == -1)
        total = len(previsoes)
        
        print(f"--- [LOF] Cenário: {nome_cenario} ---")
        print(f"Estado Normal   (Saudável): {qtd_normal} ({qtd_normal/total*100:.2f}%)")
        print(f"Estado Anômalo  (Falha):    {qtd_anomalia} ({qtd_anomalia/total*100:.2f}%)\n")
        
        return previsoes


class OneClassSVMDetector:
    """Modelo de Baseline baseado em fronteira de hiperplano (One-Class SVM)."""
    def __init__(self, nu=0.1, kernel='rbf'):
        self.scaler = StandardScaler()
        # 'nu' é o parâmetro do SVM que define a margem de erro permitida (limiar de anomalia)
        # kernel='rbf' permite que a fronteira de decisão seja curva/não-linear
        self.modelo = OneClassSVM(nu=nu, kernel=kernel, gamma='scale')

    def treinar(self, X_treino):
        X_treino_norm = self.scaler.fit_transform(X_treino)
        self.modelo.fit(X_treino_norm)

    def avaliar(self, X_teste, nome_cenario):
        X_teste_norm = self.scaler.transform(X_teste)
        previsoes = self.modelo.predict(X_teste_norm)
        
        qtd_normal = np.sum(previsoes == 1)
        qtd_anomalia = np.sum(previsoes == -1)
        total = len(previsoes)
        
        print(f"--- [ONE-CLASS SVM] Cenário: {nome_cenario} ---")
        print(f"Estado Normal   (Saudável): {qtd_normal} ({qtd_normal/total*100:.2f}%)")
        print(f"Estado Anômalo  (Falha):    {qtd_anomalia} ({qtd_anomalia/total*100:.2f}%)\n")
        
        return previsoes