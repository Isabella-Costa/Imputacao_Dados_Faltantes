
import os
import scipy.io as sio
import numpy as np

class DataLoader:
    def __init__(self, diretorio_dados):
        self.diretorio_dados = diretorio_dados
        self.campos = [
            'TDMSAnalog50kAcelerometer0', 'TDMSAnalog50kAcelerometer1',
            'TDMSAnalog50kAcelerometer2', 'TDMSAnalog50kAcelerometer3',
            'TDMSAnalog50kAcelerometer4', 'TDMSAnalog50kAcelerometer5'
        ]

    def carregar_sinais(self, nome_arquivo):
        caminho_completo = os.path.join(self.diretorio_dados, nome_arquivo)
        
        try:
            mat = sio.loadmat(caminho_completo)
            dados_struct = mat['data'][0, 0]['Analog50k']
            
            lista_sinais = []
            for campo in self.campos:
                sinal_bruto = dados_struct[campo][0, 0].flatten()
                lista_sinais.append(sinal_bruto)
                
            matriz_dados = np.array(lista_sinais, dtype=float).T
            
            return matriz_dados
            
        except KeyError as e:
            raise KeyError(f"A chave {str(e)} não foi encontrada no arquivo {nome_arquivo}.")
        except Exception as e:
            raise Exception(f"Erro ao ler o arquivo {caminho_completo}: {str(e)}")