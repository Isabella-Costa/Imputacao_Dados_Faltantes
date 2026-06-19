# Classe de aplicação do janelamento temporal e extração deindicadores no domínio do tempo (RMS e Curtose).
import numpy as np
from scipy.stats import kurtosis

class FeatureExtractor:
    def __init__(self, tamanho_janela=2048):
        self.tamanho_janela = tamanho_janela

    def extrair(self, matriz_sinais):
        total_amostras, num_sensores = matriz_sinais.shape
        num_janelas = total_amostras // self.tamanho_janela
        
        # matriz vazia (6 sensores * 2 features = 12 colunas)
        features = np.zeros((num_janelas, num_sensores * 2))
        
        for i in range(num_janelas):
            inicio = i * self.tamanho_janela
            fim = inicio + self.tamanho_janela
            
            for s in range(num_sensores):
                sinal_janela = matriz_sinais[inicio:fim, s]
                
                # Extração das características para o monitoramento (CBM)
                rms = np.sqrt(np.mean(sinal_janela**2))
                curt = kurtosis(sinal_janela, fisher=False)
                
                features[i, s*2] = rms
                features[i, s*2 + 1] = curt
                
        return features