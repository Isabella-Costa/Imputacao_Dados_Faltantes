import pandas as pd

print("Lendo e organizando o arquivo CSV...\n")


df = pd.read_csv('resultados_splits_treino.csv')
tabela_organizada = df.groupby(['Modelo', 'Tamanho_Treino (%)']).agg({
    'Falsos_Alarmes_Saudavel (%)': ['mean', 'std'],
    'Deteccao_P2_Leve (%)': ['mean', 'std'],
    'Deteccao_P10_Severa (%)': ['mean', 'std']
})

tabela_organizada = tabela_organizada.round(2)

print("="*80)
print("TABELA DE RESULTADOS CONSOLIDADOS (MÉDIA E DESVIO PADRÃO)")
print("="*80)
print(tabela_organizada.to_string())
print("="*80)