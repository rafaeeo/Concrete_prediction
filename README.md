# Predição da Resistência à Compressão do Concreto

Este projeto utiliza o dataset "Concrete Compressive Strength" do [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength) para analisar os fatores que influenciam a resistência à compressão do concreto e construir modelos preditivos para prever essa resistência.

## Objetivo

O objetivo principal deste projeto é:
- Identificar os fatores mais relevantes que influenciam a resistência à compressão do concreto.
- Construir modelos preditivos para prever a resistência com base nas variáveis disponíveis.
- Avaliar o desempenho dos modelos e interpretar os resultados.

## Dataset

O dataset contém 1.030 amostras e 9 variáveis:
- **Cement (kg/m³)**: Quantidade de cimento.
- **Blast Furnace Slag (kg/m³)**: Quantidade de escória de alto-forno.
- **Fly Ash (kg/m³)**: Quantidade de cinzas volantes.
- **Water (kg/m³)**: Quantidade de água.
- **Superplasticizer (kg/m³)**: Quantidade de superplastificante.
- **Coarse Aggregate (kg/m³)**: Quantidade de agregado graúdo.
- **Fine Aggregate (kg/m³)**: Quantidade de agregado miúdo.
- **Age (dias)**: Idade do concreto.
- **Concrete Compressive Strength (MPa)**: Resistência à compressão do concreto (variável alvo).

## Etapas do Projeto

1. **Exploração e Pré-processamento dos Dados**:
   - Análise descritiva das variáveis.
   - Verificação de valores ausentes e outliers.
   - Normalização e padronização dos dados.

2. **Análise Exploratória de Dados (EDA)**:
   - Visualização das distribuições das variáveis.
   - Análise de correlação entre as variáveis independentes e a variável alvo.

3. **Construção do Modelo**:
   - Divisão dos dados em conjuntos de treino e teste.
   - Treinamento de modelos de regressão linear múltipla, Random Forest e Gradient Boosting.

4. **Avaliação do Modelo**:
   - Métricas utilizadas:
     - **R² (Coeficiente de Determinação)**: Proporção da variância explicada pelo modelo.
     - **MSE (Mean Squared Error)**: Erro médio ao quadrado.
     - **MAE (Mean Absolute Error)**: Erro absoluto médio.
   - Validação cruzada para garantir a robustez dos resultados.

5. **Interpretação dos Resultados**:
   - Identificação das variáveis mais importantes para a resistência à compressão.
   - Comparação do desempenho dos modelos.

## Resultados

- **Regressão Linear**:
  - R² no conjunto de teste: ~0,65.
  - Modelo base com desempenho limitado, indicando relações não lineares entre as variáveis.

- **Random Forest**:
  - R² no conjunto de teste: ~0,85.
  - Capturou melhor as relações não lineares, mostrando robustez.

- **Gradient Boosting**:
  - R² no conjunto de teste: ~0,88.
  - Melhor modelo, explicando ~88% da variância dos dados.

## Tecnologias Utilizadas

- **Linguagem**: Python
- **Bibliotecas**:
  - `pandas` para manipulação de dados.
  - `numpy` para cálculos numéricos.
  - `matplotlib` e `seaborn` para visualização de dados.
  - `scikit-learn` para construção e avaliação dos modelos.

## Como Reproduzir o Projeto

1. Clone o repositório para o seu ambiente local:
   ```bash
   git clone https://github.com/SEU_USUARIO/Concrete_prediction.git
   cd Concrete_prediction

2. Clone o repositório para o seu ambiente local:
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows, use `venv\Scripts\activate`

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt

4. Instale as dependências:
   ```bash
   jupyter notebook Concrete_Project.ipynb


## Resultados

O modelo de Gradient Boosting apresentou o melhor desempenho, explicando ~88% da variância na resistência à compressão do concreto.
As variáveis mais importantes para a resistência foram:
- Quantidade de cimento.
- Quantidade de água.
- Idade do concreto.
O projeto demonstrou que modelos baseados em árvores, como Random Forest e Gradient Boosting, são mais eficazes para prever a resistência à compressão devido à sua capacidade de capturar relações não lineares.

## Recomendações Finais

Engenharia de Features: Criar novas variáveis ou transformar as existentes pode melhorar os resultados.
Ajuste de Hiperparâmetros: Utilizar técnicas como GridSearchCV para otimizar os modelos.
Teste de Generalização: Avaliar o modelo em um conjunto de dados completamente novo para validar sua capacidade de generalização.

## Contribuição
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests para melhorar o projeto.

## Licença
Este projeto está licenciado sob a Licença MIT. Veja o arquivo LICENSE para mais detalhes.

## Referências
UCI Machine Learning Repository - Concrete Compressive Strength Dataset