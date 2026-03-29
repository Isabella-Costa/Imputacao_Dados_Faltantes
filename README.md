# 🏭 Imputação de Dados Faltantes e Detecção de Anomalias

> Alinhamento da Imputação de Dados Faltantes em Séries Temporais Multivariadas à Tarefa de Detecção de Anomalias no Contexto da Indústria 4.0.

Este projeto explora o impacto e as técnicas de tratamento de dados ausentes (imputação) em séries temporais complexas, com o objetivo final de melhorar a precisão dos modelos de detecção de anomalias aplicados a cenários industriais.

## 📁 Estrutura do Projeto

O repositório está organizado em scripts Python e Jupyter Notebooks (desenvolvidos via Google Colab), divididos pelas etapas de análise e modelagem:

* **`series_temporais.py`**: Script contendo os códigos base para a criação, simulação e manipulação das séries temporais multivariadas.
* **`Análise_de_Séries_Temporais_com_FFT.ipynb`**: Análise exploratória e processamento das séries temporais utilizando a Transformada Rápida de Fourier (FFT) para identificar padrões no domínio da frequência.
* **`transf_de_fourier.ipynb`**: Notebook aprofundando os conceitos e aplicações da Transformada de Fourier nos dados do projeto.
* **`IsolationForest.ipynb`**: Implementação e treinamento do algoritmo de **Isolation Forest**, focado na identificação de *outliers* e anomalias nos dados industriais.
* **`RandomForest.ipynb`**: Implementação de modelos baseados em **Random Forest**, aplicados para suporte na imputação de dados ou classificação de padrões.

## 🛠️ Tecnologias e Ferramentas Utilizadas

* **Linguagem:** Python
* **Ambiente de Desenvolvimento:** Google Colab / Jupyter Notebook/ VsCode
* **Principais Bibliotecas (Estimadas):** * Manipulação de dados: `pandas`, `numpy`
  * Machine Learning: `scikit-learn` (Isolation Forest, Random Forest)
  * Processamento de Sinais: `scipy` (FFT)
  * Visualização: `matplotlib`, `seaborn`

## 🚀 Como Executar

1. Clone este repositório para a sua máquina local:
   ```bash
   git clone [https://github.com/Isabella-Costa/Imputacao_Dados_Faltantes.git)]
   ```

2. Acesse a pasta do projeto:

 ``` bash
  cd Imputacao_Dados_Faltantes
 
 ```

3. Instale as dependências necessárias (recomenda-se o uso de um ambiente virtual):

 ```bash
 
 pip install -r requirements.txt
 
 ```

##  ✒️ Autoria
Orientador:


Contribuidores do projeto:

| [<img src="https://github.com/Danielle-sn.png?size=115" width="115px;" alt="Foto de Danielle"/><br /><sub><b>Danielle-sn</b></sub>](https://github.com/Danielle-sn) | [<img src="https://github.com/Isabella-Costa.png?size=115" width="115px;" alt="Foto de Isabella"/><br /><sub><b>Isabella-Costa</b></sub>](https://github.com/Isabella-Costa) | [<img src="https://github.com/BiaBea.png?size=115" width="115px;" alt="Foto de BiaBea"/><br /><sub><b>BiaBea</b></sub>](https://github.com/BiaBea) |
| :---: | :---: | :---: |
| [✉️ Contato](https://github.com/Danielle-sn) | [✉️ Contato](https://github.com/Isabella-Costa) | [✉️ Contato](https://github.com/BiaBea) |
