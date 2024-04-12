import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np


# Configurações iniciais do Streamlit
st.set_page_config(
    page_title="BrogAI",
    page_icon="🏢",
    layout="wide",
)
# #Estlização do Streamlit Utilizando CSS e HTML
# with open("style.css") as f:
#     st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)
# Dados fornecidos
num_corretores = 10
num_imoveis_disponiveis = 350
num_leads = 1300
taxa_venda = 0.0315
vgv_imovel = 800000
num_imoveis_vendidos = 11  # Número de vendas desejado

# Criando o DataFrame
data = {
    'num_corretores': [num_corretores],
    'num_imoveis_disponiveis': [num_imoveis_disponiveis],
    'num_leads': [num_leads],
    'taxa_venda': [taxa_venda],
    'vgv_imovel': [vgv_imovel],
    'num_imoveis_vendidos': [num_imoveis_vendidos]
}
df = pd.DataFrame(data)

# Separando os dados em features (X) e target (y)
X = df[['num_corretores', 'num_imoveis_disponiveis', 'num_leads', 'taxa_venda', 'vgv_imovel']]
y = df['num_imoveis_vendidos']

# Treinando o modelo de regressão linear
model = LinearRegression()
model.fit(X, y)

# Realizando a previsão
num_imoveis_previstos = model.predict(X)[0]

# Logo
st.image("https://www.brognoli.com.br/uploads/logo_0133f73122cb718033a0ef27fe067d10.svg", width=100)

# Título da aplicação
st.title('Calculadora Imobiliária - Lançamentos')

# Dicionário de fatores de sazonalidade
meses = {
    'Janeiro': 0.4,
    'Fevereiro': 0.2,
    'Março': 0.9,
    'Abril': 0.5,
    'Maio': 0.95,
    'Junho': 0.8,
    'Julho': 0.85,
    'Agosto': 1.0,
    'Setembro': 0.9,
    'Outubro': 0.55,
    'Novembro': 0.5,
    'Dezembro': 0.7
}

# Interface da calculadora
st.sidebar.header('Dados do Grupo')
num_corretores_novos = st.sidebar.number_input('Número de Corretores:', min_value=1, step=1, value=num_corretores)
num_leads_novos = st.sidebar.number_input('Número de Leads:', min_value=1, step=1, value=num_leads, help='Número de leads qualificados com score >= 6')

st.sidebar.header('Dados de Imóveis')
num_imoveis_disponiveis_novos = st.sidebar.number_input('Estoque de Imóveis:', min_value=1, step=1, value=num_imoveis_disponiveis)
vgv_imovel_novo = st.sidebar.number_input('VGV Médio por Imóvel (R$):', min_value=0, step=10000, value=vgv_imovel)

X_novo = pd.DataFrame({
    'num_corretores': [num_corretores_novos],
    'num_imoveis_disponiveis': [num_imoveis_disponiveis_novos],
    'num_leads': [num_leads_novos],
    'taxa_venda': [taxa_venda],  
    'vgv_imovel': [vgv_imovel_novo]  
})
num_imoveis_previstos_novos = model.predict(X_novo)[0]

# Seleção do mês
mes_selecionado = st.sidebar.selectbox('Selecione o Mês:', list(meses.keys()))

# Cálculo do fator de sazonalidade
fator_sazonalidade = meses[mes_selecionado]

# Previsão do número de imóveis vendidos e receita
num_imoveis_previstos = int(num_imoveis_previstos_novos * fator_sazonalidade) #int(sum(prever_num_imoveis(valores_imoveis, num_imoveis_disponiveis)) * fator_sazonalidade / 10000)

# Cálculo da receita
vgv_por_imovel = vgv_imovel_novo
receita_min = num_imoveis_previstos * 600000
receita_media = num_imoveis_previstos * vgv_imovel_novo
receita_max = num_imoveis_previstos * 1200000

# Média e desvio padrão das vendas
media_vendas = num_imoveis_previstos
desvio_padrao_vendas = round(np.std([num_imoveis_previstos * meses[mes] for mes in meses.keys()]))

# Gráfico de séries temporais para o número de vendas
df_vendas = pd.DataFrame({
    'Mês': list(meses.keys()),
    'Vendas Mínimas': [max(num_imoveis_previstos * meses[mes] - desvio_padrao_vendas, 0) for mes in meses.keys()],
    'Vendas Médias': [num_imoveis_previstos * meses[mes] for mes in meses.keys()],
    'Vendas Máximas': [num_imoveis_previstos * meses[mes] + desvio_padrao_vendas for mes in meses.keys()]
})
# Mapear nome do mês para o índice
meses_idx = {mes: idx for idx, mes in enumerate(meses.keys())}

# Encontrar o índice do mês atual
mes_atual_idx = meses_idx[mes_selecionado]

# Gráfico de séries temporais para o número de vendas
fig_vendas = go.Figure()
fig_vendas.add_trace(go.Scatter(x=df_vendas['Mês'], y=df_vendas['Vendas Máximas'], mode='lines+markers', name='Máx', line=dict(color='green', dash='dash')))
fig_vendas.add_trace(go.Scatter(x=df_vendas['Mês'], y=df_vendas['Vendas Médias'], mode='lines+markers', name='Vendas', line=dict(color='blue')))
fig_vendas.add_trace(go.Scatter(x=df_vendas['Mês'], y=df_vendas['Vendas Mínimas'], mode='lines+markers', name='Mín', line=dict(color='red', dash='dash')))
fig_vendas.add_vline(x=mes_atual_idx, line_dash="dot", line_color="black", annotation_text="Mês Selecionado", annotation_position="top left")
fig_vendas.update_layout(title='Número de Vendas Previsto por Mês', xaxis_title='Mês', yaxis_title='Número de Vendas')

# Cálculo da receita
receita_min = num_imoveis_previstos * 600000
receita_max = num_imoveis_previstos * 800000
receita_media = (receita_min + receita_max) / 2

# Gráfico de séries temporais para a receita
df_receita = pd.DataFrame({
    'Mês': list(meses.keys()),
    'Receita Mínima': [num_imoveis_previstos * meses[mes] * 600000 for mes in meses.keys()],
    'Receita Média': [((num_imoveis_previstos * meses[mes] * 600000) + (num_imoveis_previstos * meses[mes] * 800000)) / 2 for mes in meses.keys()],
    'Receita Máxima': [num_imoveis_previstos * meses[mes] * 800000 for mes in meses.keys()]
})


fig_receita = go.Figure()
fig_receita.add_trace(go.Scatter(x=df_receita['Mês'], y=df_receita['Receita Máxima'], mode='lines+markers', name='Máx', line=dict(color='green', dash='dash')))
fig_receita.add_trace(go.Scatter(x=df_receita['Mês'], y=df_receita['Receita Média'], mode='lines+markers', name='Receita', line=dict(color='blue')))
fig_receita.add_trace(go.Scatter(x=df_receita['Mês'], y=df_receita['Receita Mínima'], mode='lines+markers', name='Mín', line=dict(color='red', dash='dash')))
fig_receita.add_vline(x=mes_atual_idx, line_dash="dot", line_color="black", annotation_text="Mês Selecionado", annotation_position="top left")
fig_receita.update_layout(title='Receita Prevista por Mês', xaxis_title='Mês', yaxis_title='Receita (R$)')


# Formatação dos números
receita_min_formatada = "{:,.2f}".format(receita_min).replace(",", ";").replace(".", ",").replace(";", ".")
receita_media_formatada = "{:,.2f}".format(receita_media).replace(",", ";").replace(".", ",").replace(";", ".")
receita_max_formatada = "{:,.2f}".format(receita_max).replace(",", ";").replace(".", ",").replace(";", ".")
vgv_imovel_novo_formatada = "{:,.2f}".format(vgv_imovel_novo).replace(",", ";").replace(".", ",").replace(";", ".")

# Exibindo os resultados
st.markdown("---")
# Divisão das colunas em duas linhas
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
col5, col6, col7, col8 = st.columns([1.5, 1.5, 1, 1])

# Preenchendo as colunas com as métricas
with col1:
    st.metric("Mês Selecionado", mes_selecionado)

with col2:
    st.metric("N° de Vendas Previstas", num_imoveis_previstos)

with col3:
    st.metric("Média de Leads por Corretor", round(num_leads_novos/num_corretores_novos))

with col4:
    st.metric(" Média de Imóveis por Corretor", int(num_imoveis_disponiveis_novos/num_corretores_novos))

# Coluna vazia para separar visualmente
with col5:
    st.metric("VGV Previsto", f"R$ {receita_media_formatada}")
    

st.markdown("---")
# Exibindo os gráficos
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_vendas)
with col2:
    st.plotly_chart(fig_receita)





# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# import numpy as np
# import locale

# # Configuração da formatação
# locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

# # Configurações iniciais do Streamlit
# st.set_page_config(
#     page_title="BrogAI",
#     page_icon="🏢",
#     layout="wide",
# )

# # #Estlização do Streamlit Utilizando CSS e HTML
# # with open("style.css") as f:
# #     st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)
# # Logo
# st.image("https://www.brognoli.com.br/uploads/logo_0133f73122cb718033a0ef27fe067d10.svg", width=100)

# # Título da aplicação
# st.title('Calculadora Imobiliária - Lançamentos')

# # Dados fornecidos
# num_corretores = 10
# num_imoveis_disponiveis = 350
# num_leads = 1350
# taxa_venda = 0.0315
# vgv_imovel = 800000


# # Dados fictícios
# np.random.seed(42)
# num_corretores = 10
# clientes_por_corretor = 15
# vgv_min = 600000
# vgv_max = 1200000

# valores_imoveis = np.random.randint(vgv_min, vgv_max, num_imoveis_disponiveis)

# # Inputs para o modelo preditivo
# X = valores_imoveis.reshape(-1, 1)
# y = np.random.randint(1, 10, num_imoveis_disponiveis)  # Número de imóveis vendidos (dados fictícios)

# # Treinamento do modelo preditivo
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Função para prever o número de imóveis vendidos
# def prever_num_imoveis(valores_imoveis, num_imoveis_disponiveis):
#     X_pred = np.array(valores_imoveis).reshape(-1, 1)
#     num_imoveis_previstos = model.predict(X_pred)
#     num_imoveis_previstos[num_imoveis_previstos < num_imoveis_disponiveis] = num_imoveis_disponiveis
#     return num_imoveis_previstos


# # Dicionário de fatores de sazonalidade
# meses = {
#     'Janeiro': 0.4,
#     'Fevereiro': 0.2,
#     'Março': 0.9,
#     'Abril': 0.5,
#     'Maio': 0.95,
#     'Junho': 0.8,
#     'Julho': 0.85,
#     'Agosto': 1.0,
#     'Setembro': 0.9,
#     'Outubro': 0.55,
#     'Novembro': 0.5,
#     'Dezembro': 0.7
# }

# # Interface da calculadora
# st.sidebar.header('Dados do Grupo')
# num_corretores = st.sidebar.number_input('Número de Corretores:', min_value=1, step=1, value=num_corretores)
# # clientes_por_corretor = st.sidebar.number_input('Clientes por Corretor:', min_value=1, step=1, value=clientes_por_corretor)

# st.sidebar.header('Dados de Imóveis')
# num_imoveis_disponiveis = st.sidebar.number_input('Estoque de Imóveis:', min_value=1, step=1, value=num_imoveis_disponiveis)
# vgv = st.sidebar.number_input('VGV (R$):', min_value=0, step=10000, value=vgv_imovel)
# # vgv_min = st.sidebar.number_input('Valor Mínimo do Imóvel (R$):', min_value=0, step=10000, value=vgv_min)
# # vgv_max = st.sidebar.number_input('Valor Máximo do Imóvel (R$):', min_value=vgv_min, step=10000, value=vgv_max)

# valores_imoveis = np.random.randint(vgv_min, vgv_max, num_imoveis_disponiveis)

# # Seleção do mês
# mes_selecionado = st.sidebar.selectbox('Selecione o Mês:', list(meses.keys()))

# # Cálculo do fator de sazonalidade
# fator_sazonalidade = meses[mes_selecionado]

# # Previsão do número de imóveis vendidos
# num_imoveis_previstos = int(sum(prever_num_imoveis(valores_imoveis, num_imoveis_disponiveis)) * fator_sazonalidade / 10000)

# # # Ajuste do número de imóveis disponíveis
# # num_imoveis_disponiveis = min(num_imoveis_disponiveis, num_imoveis_previstos)

# # Cálculo da receita
# vgv_por_imovel = vgv#np.mean(valores_imoveis)
# receita_min = num_imoveis_previstos * vgv_min
# receita_media = num_imoveis_previstos * vgv_por_imovel
# receita_max = num_imoveis_previstos * vgv_max

# # Média e desvio padrão das vendas
# media_vendas = round(num_imoveis_previstos * fator_sazonalidade)
# desvio_padrao_vendas = round(np.std([num_imoveis_previstos * meses[mes] for mes in meses.keys()]))

# # Gráfico de séries temporais para o número de vendas
# df_vendas = pd.DataFrame({
#     'Mês': list(meses.keys()),
#     'Vendas Mínimas': [max(num_imoveis_previstos * meses[mes] - desvio_padrao_vendas, 0) for mes in meses.keys()],
#     'Vendas Médias': [num_imoveis_previstos * meses[mes] for mes in meses.keys()],
#     'Vendas Máximas': [num_imoveis_previstos * meses[mes] + desvio_padrao_vendas for mes in meses.keys()]
# })
# # Mapear nome do mês para o índice
# meses_idx = {mes: idx for idx, mes in enumerate(meses.keys())}

# # Encontrar o índice do mês atual
# mes_atual_idx = meses_idx[mes_selecionado]

# # Gráfico de séries temporais para o número de vendas
# fig_vendas = go.Figure()
# fig_vendas.add_trace(go.Scatter(x=df_vendas['Mês'], y=df_vendas['Vendas Máximas'], mode='lines+markers', name='Máx', line=dict(color='green', dash='dash')))
# fig_vendas.add_trace(go.Scatter(x=df_vendas['Mês'], y=df_vendas['Vendas Médias'], mode='lines+markers', name='Vendas', line=dict(color='blue')))
# fig_vendas.add_trace(go.Scatter(x=df_vendas['Mês'], y=df_vendas['Vendas Mínimas'], mode='lines+markers', name='Mín', line=dict(color='red', dash='dash')))
# fig_vendas.add_vline(x=mes_atual_idx, line_dash="dot", line_color="black", annotation_text="Mês Selecionado", annotation_position="top left")
# fig_vendas.update_layout(title='Número de Vendas Previsto por Mês', xaxis_title='Mês', yaxis_title='Número de Vendas')

# # Cálculo da receita
# receita_min = num_imoveis_previstos * vgv_min
# receita_max = num_imoveis_previstos * vgv_max
# receita_media = (receita_min + receita_max) / 2

# # Gráfico de séries temporais para a receita
# df_receita = pd.DataFrame({
#     'Mês': list(meses.keys()),
#     'Receita Mínima': [num_imoveis_previstos * meses[mes] * vgv_min for mes in meses.keys()],
#     'Receita Média': [((num_imoveis_previstos * meses[mes] * vgv_min) + (num_imoveis_previstos * meses[mes] * vgv_max)) / 2 for mes in meses.keys()],
#     'Receita Máxima': [num_imoveis_previstos * meses[mes] * vgv_max for mes in meses.keys()]
# })


# fig_receita = go.Figure()
# fig_receita.add_trace(go.Scatter(x=df_receita['Mês'], y=df_receita['Receita Máxima'], mode='lines+markers', name='Máx', line=dict(color='green', dash='dash')))
# fig_receita.add_trace(go.Scatter(x=df_receita['Mês'], y=df_receita['Receita Média'], mode='lines+markers', name='Receita', line=dict(color='blue')))
# fig_receita.add_trace(go.Scatter(x=df_receita['Mês'], y=df_receita['Receita Mínima'], mode='lines+markers', name='Mín', line=dict(color='red', dash='dash')))
# fig_receita.add_vline(x=mes_atual_idx, line_dash="dot", line_color="black", annotation_text="Mês Selecionado", annotation_position="top left")
# fig_receita.update_layout(title='Receita Prevista por Mês', xaxis_title='Mês', yaxis_title='Receita (R$)')

# # Formatação dos números
# receita_min_formatada = locale.currency(receita_min, grouping=True)
# receita_media_formatada = locale.currency(receita_media, grouping=True)
# receita_max_formatada = locale.currency(receita_max, grouping=True)

# # Exibindo os resultados
# st.markdown("---")
# col1, col2, col3, col4 = st.columns(4)

# with col1:
#     st.metric("Mês Selecionado", 
#           mes_selecionado)
# with col2:
#     st.metric("N° de Vendas Previstas", 
#           num_imoveis_previstos)
# with col3:
#     st.metric("Receita Prevista", 
#           receita_media_formatada)
# # st.write(f'Receita Mínima: {receita_min_formatada:}')
# # # st.write(f'Receita Aproximada Prevista: {receita_media_formatada:}')
# # st.write(f'Receita Máxima: {receita_max_formatada:}')

# # Exibindo os gráficos
# col1, col2 = st.columns(2)
# with col1:
#     st.plotly_chart(fig_vendas)
# with col2:
#     st.plotly_chart(fig_receita)
