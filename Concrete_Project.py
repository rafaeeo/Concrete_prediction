import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Configurações iniciais do Streamlit
st.set_page_config(page_title="Dashboard: Resistência à Compressão do Concreto", layout="wide")
st.title("Dashboard Interativo: Resistência à Compressão do Concreto")
st.markdown("""
Este dashboard explora o dataset **Concrete Compressive Strength** e demonstra uma análise exploratória interativa, além de construir um modelo preditivo de regressão para estimar a resistência à compressão do concreto.
""")

# Função para carregar os dados
@st.cache_data
def load_data():
    df = pd.read_excel('Concrete_Data.xls')
    return df

# Carrega os dados
data = load_data()
st.sidebar.header("Configurações e Filtros")
st.sidebar.write("Dataset com", data.shape[0], "registros")

# Exibe o dataset se o usuário desejar
if st.sidebar.checkbox("Mostrar dados brutos"):
    st.subheader("Dados Brutos")
    st.dataframe(data)

# --- Análise Exploratória ---
st.header("1. Análise Exploratória dos Dados")

# Histograma da Resistência à Compressão
st.subheader("Distribuição da Resistência à Compressão do Concreto")
fig1 = px.histogram(data, x="Concrete compressive strength(MPa, megapascals) ", nbins=30, title="Distribuição da Resistência à Compressão do Concreto")
st.plotly_chart(fig1, use_container_width=True)

# Mapa de Correlação
st.subheader("Mapa de Correlação")
numeric_cols = data.select_dtypes(include=[np.number])

# Remover colunas que contêm valores constantes
numeric_cols = numeric_cols.loc[:, (numeric_cols != numeric_cols.iloc[0]).any()]

# Calcular a matriz de correlação
corr_matrix = numeric_cols.corr()

# Plotar o mapa de calor da correlação
fig2, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig2)

# --- Modelagem Preditiva ---
st.header("2. Modelagem Preditiva")

st.markdown("""
Utilizaremos um modelo de **Random Forest** para prever a resistência à compressão do concreto com base em algumas variáveis selecionadas.
""")

# Selecionar features e variável alvo
features = ["Cement (component 1)(kg in a m^3 mixture)", "Blast Furnace Slag (component 2)(kg in a m^3 mixture)", "Fly Ash (component 3)(kg in a m^3 mixture)", "Water  (component 4)(kg in a m^3 mixture)", "Superplasticizer (component 5)(kg in a m^3 mixture)", "Coarse Aggregate  (component 6)(kg in a m^3 mixture)", "Fine Aggregate (component 7)(kg in a m^3 mixture)", "Age (day)"]
X = data[features]
y = data['Concrete compressive strength(MPa, megapascals) ']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo de Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.subheader("Desempenho do Modelo de Random Forest")
st.write(f"**MSE:** {mse:.2f}")
st.write(f"**R²:** {r2:.2f}")

# Plotar previsões vs. valores reais
st.subheader("Previsões vs. Valores Reais")
fig3 = px.scatter(x=y_test, y=y_pred, labels={'x': 'Valores Reais', 'y': 'Previsões'}, title="Previsões vs. Valores Reais")
st.plotly_chart(fig3, use_container_width=True)

# --- Interatividade ---
st.header("3. Previsão Interativa")

st.markdown("""
Insira os valores para prever a resistência à compressão do concreto.
""")

# Campos de entrada para os valores das features
cement = st.number_input('Cimento (kg/m³)', min_value=0.0, max_value=1000.0, value=540.0)
slag = st.number_input('Escória de Alto Forno (kg/m³)', min_value=0.0, max_value=1000.0, value=0.0)
fly_ash = st.number_input('Cinza Volante (kg/m³)', min_value=0.0, max_value=1000.0, value=0.0)
water = st.number_input('Água (kg/m³)', min_value=0.0, max_value=1000.0, value=162.0)
superplasticizer = st.number_input('Superplastificante (kg/m³)', min_value=0.0, max_value=100.0, value=2.5)
coarse_aggregate = st.number_input('Agregado Graúdo (kg/m³)', min_value=0.0, max_value=2000.0, value=1040.0)
fine_aggregate = st.number_input('Agregado Miúdo (kg/m³)', min_value=0.0, max_value=2000.0, value=676.0)
age = st.number_input('Idade (dias)', min_value=1, max_value=365, value=28)

# Prever a resistência à compressão com base nos valores inseridos
input_data = np.array([[cement, slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age]])
prediction = model.predict(input_data)

st.subheader("Previsão da Resistência à Compressão")
st.write(f"A resistência à compressão prevista é: **{prediction[0]:.2f} MPa**")