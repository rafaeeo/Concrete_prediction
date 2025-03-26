# Predição da Resistência à Compressão do Concreto

Este projeto utiliza o dataset "Concrete Compressive Strength" do [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength) para analisar os fatores que influenciam a resistência à compressão do concreto e construir um modelo de regressão para prever essa resistência.

## Objetivo

O objetivo principal deste projeto é prever a resistência à compressão do concreto com base em suas propriedades físicas e químicas, utilizando técnicas de aprendizado de máquina.

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

2. **Análise Exploratória**:
   - Visualização das correlações entre as variáveis.
   - Identificação dos fatores mais relevantes para a resistência à compressão.

3. **Construção do Modelo**:
   - Divisão dos dados em conjuntos de treino e teste.
   - Treinamento de um modelo de regressão linear múltipla.
   - Avaliação do modelo utilizando métricas como R² e RMSE.

4. **Validação Cruzada**:
   - Implementação de K-Fold Cross-Validation para avaliar a robustez do modelo.

5. **Resultados e Conclusões**:
   - Interpretação dos coeficientes do modelo.
   - Identificação dos fatores mais impactantes na resistência do concreto.

## Tecnologias Utilizadas

- **Linguagem**: Python
- **Bibliotecas**:
  - `pandas` para manipulação de dados.
  - `numpy` para cálculos numéricos.
  - `matplotlib` e `seaborn` para visualização de dados.
  - `scikit-learn` para construção e avaliação do modelo.

## Como Executar o Projeto

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

O modelo de regressão linear múltipla apresentou um R² de aproximadamente 0.90 no conjunto de teste, indicando que o modelo explica 90% da variabilidade na resistência à compressão do concreto.

Os fatores mais relevantes para a resistência à compressão foram:
Quantidade de cimento.
Quantidade de água.
Idade do concreto.

## Contribuição
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests para melhorar o projeto.

## Licença
Este projeto está licenciado sob a Licença MIT. Veja o arquivo LICENSE para mais detalhes.

## Referências
UCI Machine Learning Repository - Concrete Compressive Strength Dataset