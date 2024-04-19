import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from streamlit_lottie import st_lottie
import requests


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
st.title('Calculadora Imobiliária🏢')

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
st.sidebar.header('Inferências')
opcoes = ["Lançamentos - Estimativa de Negócios", "Vendas - Estimativa de Negócios", "Estimativa de Preços"]
opcao_selecionada  = st.sidebar.radio("", opcoes)
if opcao_selecionada == "Lançamentos - Estimativa de Negócios":
    st.sidebar.header('Dados do Grupo')
    num_corretores_novos = st.sidebar.number_input('Número de Corretores:', min_value=1, step=1, value=10)
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
    num_imoveis_previstos_novos = int(0.02 * num_corretores_novos + 0.03 * num_imoveis_disponiveis_novos + 0.002 * num_leads_novos + ((500 * taxa_venda)/(0.00000365*vgv_imovel_novo)) - 0.000009 * vgv_imovel_novo)
    # num_imoveis_previstos_novos = int(0.9 * num_corretores_novos + 0.001 * num_imoveis_disponiveis_novos + 0.001 * num_leads_novos + 5 * taxa_venda - 0.000001 * vgv_imovel_novo)+1
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
    fig_receita.add_trace(go.Scatter(x=df_receita['Mês'], y=df_receita['Receita Média'], mode='lines+markers', name='VGV', line=dict(color='blue')))
    fig_receita.add_trace(go.Scatter(x=df_receita['Mês'], y=df_receita['Receita Mínima'], mode='lines+markers', name='Mín', line=dict(color='red', dash='dash')))
    fig_receita.add_vline(x=mes_atual_idx, line_dash="dot", line_color="black", annotation_text="Mês Selecionado", annotation_position="top left")
    fig_receita.update_layout(title='VGV Previsto por Mês', xaxis_title='Mês', yaxis_title='Receita (R$)')


    # Formatação dos números
    receita_min_formatada = "{:,.2f}".format(receita_min).replace(",", ";").replace(".", ",").replace(";", ".")
    receita_media_formatada = "{:,.2f}".format(receita_media).replace(",", ";").replace(".", ",").replace(";", ".")
    receita_max_formatada = "{:,.2f}".format(receita_max).replace(",", ";").replace(".", ",").replace(";", ".")
    vgv_imovel_novo_formatada = "{:,.2f}".format(vgv_imovel_novo).replace(",", ";").replace(".", ",").replace(";", ".")

    # Exibindo os resultados
    st.markdown("---")
    # Calcular a média de leads por corretor
    media_leads_por_corretor = int(num_leads_novos / num_corretores_novos)
    media_imoveis_por_corretor = int(num_imoveis_disponiveis_novos/num_corretores_novos)

    # Divisão das colunas em duas linhas
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    col5, col6, col7, col8 = st.columns([1.5, 1.5, 1, 1])

    # Preenchendo as colunas com as métricas
    with col1:
        st.metric("Mês Selecionado", mes_selecionado)

    with col2:
        st.metric("N° de Vendas Previstas", num_imoveis_previstos)

    # Exibir o card com a cor definida
    with col3:
        if 15 <= int(num_leads_novos / num_corretores_novos) <= 20:
            st.write(f'<div style="background-color:yellow; padding: 10px; border-radius: 5px;">'
                    f'Média de Leads por Corretor: {int(num_leads_novos / num_corretores_novos)}'
                    '</div>', unsafe_allow_html=True)
        elif int(num_leads_novos / num_corretores_novos) > 20:
            st.write(f'<div style="background-color:red; padding: 10px; border-radius: 5px;">'
                    f'Média de Leads por Corretor: {int(num_leads_novos / num_corretores_novos)}'
                    '</div>', unsafe_allow_html=True)
        else:
            st.metric("Média de Leads por Corretor", int(num_leads_novos / num_corretores_novos))

    with col4:
        with st.container():
            if 15 <= int(num_imoveis_disponiveis_novos/num_corretores_novos) <= 20:
                st.markdown(
                    f'<div style="background-color:yellow; padding: 10px; border-radius: 5px;">'
                    f'Média de Imóveis por Corretor: {int(num_imoveis_disponiveis_novos/num_corretores_novos)}'
                    '</div>', unsafe_allow_html=True)
            elif int(num_imoveis_disponiveis_novos/num_corretores_novos) > 20:
                st.markdown(
                    f'<div style="background-color:red; padding: 10px; border-radius: 5px;">'
                    f'Média de Imóveis por Corretor: {int(num_imoveis_disponiveis_novos/num_corretores_novos)}'
                    '</div>', unsafe_allow_html=True)
            else:
                st.metric("Média de Imóveis por Corretor", int(num_imoveis_disponiveis_novos/num_corretores_novos))

    # with col4:
    #     st.metric(" Média de Imóveis por Corretor", )

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
elif opcao_selecionada == "Vendas - Estimativa de Negócios":
    st.sidebar.header('Dados do Grupo')
    num_corretores_novos = st.sidebar.number_input('Número de Corretores:', min_value=1, step=1, value=20)
    num_leads_novos = st.sidebar.number_input('Número de Leads:', min_value=1, step=1, value=num_leads, help='Número de leads qualificados com score >= 6')

    st.sidebar.header('Dados de Imóveis')
    num_imoveis_disponiveis_novos = st.sidebar.number_input('Estoque de Imóveis:', min_value=1, step=1, value=num_imoveis_disponiveis)
    vgv_imovel_novo = st.sidebar.number_input('VGV Médio por Imóvel (R$):', min_value=0, step=10000, value=vgv_imovel)

    X_novo = pd.DataFrame({
        'num_corretores': [num_corretores_novos],
        'num_imoveis_disponiveis': [num_imoveis_disponiveis_novos],
        'num_leads': [num_leads_novos],
        'taxa_venda': 0.26,  
        'vgv_imovel': [vgv_imovel_novo]  
    })
    num_imoveis_previstos_novos = model.predict(X_novo)[0]
    num_imoveis_previstos_novos = int(0.02 * num_corretores_novos + 0.1 * num_imoveis_disponiveis_novos + 0.002 * num_leads_novos + ((500 * 0.26)/(0.00000365*vgv_imovel_novo)) - 0.000009 * vgv_imovel_novo)
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
    fig_receita.add_trace(go.Scatter(x=df_receita['Mês'], y=df_receita['Receita Média'], mode='lines+markers', name='VGV', line=dict(color='blue')))
    fig_receita.add_trace(go.Scatter(x=df_receita['Mês'], y=df_receita['Receita Mínima'], mode='lines+markers', name='Mín', line=dict(color='red', dash='dash')))
    fig_receita.add_vline(x=mes_atual_idx, line_dash="dot", line_color="black", annotation_text="Mês Selecionado", annotation_position="top left")
    fig_receita.update_layout(title='VGV Previsto por Mês', xaxis_title='Mês', yaxis_title='Receita (R$)')


    # Formatação dos números
    receita_min_formatada = "{:,.2f}".format(receita_min).replace(",", ";").replace(".", ",").replace(";", ".")
    receita_media_formatada = "{:,.2f}".format(receita_media).replace(",", ";").replace(".", ",").replace(";", ".")
    receita_max_formatada = "{:,.2f}".format(receita_max).replace(",", ";").replace(".", ",").replace(";", ".")
    vgv_imovel_novo_formatada = "{:,.2f}".format(vgv_imovel_novo).replace(",", ";").replace(".", ",").replace(";", ".")

    # Exibindo os resultados
    st.markdown("---")
    # Calcular a média de leads por corretor
    media_leads_por_corretor = int(num_leads_novos / num_corretores_novos)
    media_imoveis_por_corretor = int(num_imoveis_disponiveis_novos/num_corretores_novos)

    # Divisão das colunas em duas linhas
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    col5, col6, col7, col8 = st.columns([1.5, 1.5, 1, 1])

    # Preenchendo as colunas com as métricas
    with col1:
        st.metric("Mês Selecionado", mes_selecionado)

    with col2:
        st.metric("N° de Vendas Previstas", num_imoveis_previstos)

    # Exibir o card com a cor definida
    with col3:
        if 15 <= int(num_leads_novos / num_corretores_novos) <= 20:
            st.write(f'<div style="background-color:yellow; padding: 10px; border-radius: 5px;">'
                    f'Média de Leads por Corretor: {int(num_leads_novos / num_corretores_novos)}'
                    '</div>', unsafe_allow_html=True)
        elif int(num_leads_novos / num_corretores_novos) > 20:
            st.write(f'<div style="background-color:red; padding: 10px; border-radius: 5px;">'
                    f'Média de Leads por Corretor: {int(num_leads_novos / num_corretores_novos)}'
                    '</div>', unsafe_allow_html=True)
        else:
            st.metric("Média de Leads por Corretor", int(num_leads_novos / num_corretores_novos))

    with col4:
        with st.container():
            if 15 <= int(num_imoveis_disponiveis_novos/num_corretores_novos) <= 20:
                st.markdown(
                    f'<div style="background-color:yellow; padding: 10px; border-radius: 5px;">'
                    f'Média de Imóveis por Corretor: {int(num_imoveis_disponiveis_novos/num_corretores_novos)}'
                    '</div>', unsafe_allow_html=True)
            elif int(num_imoveis_disponiveis_novos/num_corretores_novos) > 20:
                st.markdown(
                    f'<div style="background-color:red; padding: 10px; border-radius: 5px;">'
                    f'Média de Imóveis por Corretor: {int(num_imoveis_disponiveis_novos/num_corretores_novos)}'
                    '</div>', unsafe_allow_html=True)
            else:
                st.metric("Média de Imóveis por Corretor", int(num_imoveis_disponiveis_novos/num_corretores_novos))

    # with col4:
    #     st.metric(" Média de Imóveis por Corretor", )

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
elif opcao_selecionada == "Estimativa de Preços": 
    # def render_animation():
    #         animation_response = requests.get('https://lottie.host/cf3926dd-efbd-4508-9cb8-8a319e172be1/UJWBc11teF.json')
    #         animation_json = dict()
            
    #         if animation_response.status_code == 200:
    #             animation_json = animation_response.json()
    #         else:
    #             print("Error in the URL")     
                                
    #         return st_lottie(animation_json, height=200, width=300)
        
    # render_animation()
    
    
    # y = 133245.29266838823 + (-64202.367576423654)*aceita_financiamento + (9168.233870946971)*andar_do_apto + (-8118.868950091412)*andares + (-10565.819973089086)*aptos_andar + (50.79555321114747)*area_privativa + (105282.37547005518)*closet + (0.0)*cobertura + (58148.066619422105)*dormitorios + (-7377.984906381084)*elevadores + (24310.316293653992)*estado_conservacao_imovel + (-46888.68729398165)*midia + (-131566.21440522932)*percentual_comissao + (154144.585405557)*qtd_varandas + (0.1)*valor_condominio 
    
    
  
    aceita_financiamento = st.sidebar.radio('Aceita Financiamento?', ['Sim', 'Não'])
    if aceita_financiamento == 'Não':
        aceita_financiamento = 1
    elif aceita_financiamento == 'Sim':
        aceita_financiamento = 0
    andares = st.sidebar.number_input('N° de Andares do Edifício ', min_value=1, step=1, value=10)
    andar_do_apto = st.sidebar.number_input('N° do Andar ', min_value=1, step=1, value=3)
    aptos_andar = st.sidebar.number_input('N° de Apartamentos por Andar ', min_value=1, step=1, value=5)
    area_privativa = st.sidebar.number_input('M² do Imóvel ', min_value=1, step=1, value=75)
    closet = st.sidebar.radio('Tem Closet?', ['Sim', 'Não'])
    if closet == 'Não':
        closet = 0
    elif closet == 'Sim':
        closet = 1
    dormitorios = st.sidebar.number_input('N° de Dormitórios', min_value=1, step=1, value=4)
    elevadores = st.sidebar.number_input('N° de Elevadores', min_value=1, step=1, value=4)
    estado_conservacao_imovel = st.sidebar.radio('Estado de Conservação', ['Ótimo', 'Bom', 'Regular'])
    if estado_conservacao_imovel == 'Ótimo':
        estado_conservacao_imovel = 7
    elif estado_conservacao_imovel == 'Bom':
        estado_conservacao_imovel = 3
    elif estado_conservacao_imovel == 'Regular':
        estado_conservacao_imovel = 0
    midia= st.sidebar.radio('Possui mídias?', ['Sim', 'Não'], help="Fotos e/ou vídeos")
    if midia == 'Não':
        midia = 0
    elif midia == 'Sim':
        midia = 1
    percentual_comissao = st.sidebar.number_input('% de Comissão', min_value=1, step=1, value=5)
    varanda = st.sidebar.radio('Possui Varanda?', ['Sim', 'Não'])
    if varanda == 'Não':
        qtd_varandas = 0
    elif varanda == 'Sim':
        qtd_varandas = st.sidebar.number_input('N° de Varandas', min_value=1, step=1, value=2)
    valor_condominio = st.sidebar.number_input('Valor de Condomínio', min_value=1, step=1, value=500)
    
    valor = (133245.29266838823 + (-64202.367576423654)*aceita_financiamento + (9168.233870946971)*andar_do_apto + (-8118.868950091412)*andares + (-10565.819973089086)*aptos_andar + (9000.79555321114747)*area_privativa + (105282.37547005518)*closet  + (58148.066619422105)*dormitorios + (7377.984906381084)*elevadores + (24310.316293653992)*estado_conservacao_imovel + (46888.68729398165)*midia + (13156.21440522932)*percentual_comissao + (154144.585405557)*qtd_varandas + (300)*valor_condominio )*0.8

    # Seleção do mês
    # mes_selecionado = st.sidebar.selectbox('Selecione o Mês:', list(meses.keys()))

    # Cálculo do fator de sazonalidade
    # fator_sazonalidade = meses[mes_selecionado]

    # Previsão do número de imóveis vendidos e receita
    # valor_previsto = int(valor * fator_sazonalidade) 
    valor_previsto = valor

    # # Média e desvio padrão dos valores
    # media_valor = valor_previsto
    # desvio_padrao_valor = round(np.std([valor_previsto * meses[mes] for mes in meses.keys()]))

    # # Gráfico de séries temporais para o número de vendas
    # df_valores = pd.DataFrame({
    #     'Mês': list(meses.keys()),
    #     'Valor Mínimo': [max(valor_previsto * meses[mes] - desvio_padrao_valor, 0) for mes in meses.keys()],
    #     'Valor Médio': [valor_previsto * meses[mes] for mes in meses.keys()],
    #     'Valor Máximo': [valor_previsto * meses[mes] + desvio_padrao_valor for mes in meses.keys()]
    # })
    # # Mapear nome do mês para o índice
    # meses_idx = {mes: idx for idx, mes in enumerate(meses.keys())}

    # # Encontrar o índice do mês atual
    # mes_atual_idx = meses_idx[mes_selecionado]

    # # Gráfico de séries temporais para o número de vendas
    # fig_valores = go.Figure()
    # fig_valores.add_trace(go.Scatter(x=df_valores['Mês'], y=df_valores['Valor Máximo'], mode='lines+markers', name='Máx', line=dict(color='green', dash='dash')))
    # fig_valores.add_trace(go.Scatter(x=df_valores['Mês'], y=df_valores['Valor Médio'], mode='lines+markers', name='Vendas', line=dict(color='blue')))
    # fig_valores.add_trace(go.Scatter(x=df_valores['Mês'], y=df_valores['Valor Mínimo'], mode='lines+markers', name='Mín', line=dict(color='red', dash='dash')))
    # fig_valores.add_vline(x=mes_atual_idx, line_dash="dot", line_color="black", annotation_text="Mês Selecionado", annotation_position="top left")
    # fig_valores.update_layout(title='Valor Previsto por Mês', xaxis_title='Mês', yaxis_title='Número de Vendas')
    # valor_previsto = "{:.2f}".format(valor_previsto).replace(",", ";").replace(".", ",").replace(";", ".")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Valor Previsto', "R$ {:,.2f}".format(valor_previsto).replace(',', 'X').replace('.', ',').replace('X', '.'))
    #     st.metric("Mês Selecionado", mes_selecionado)
    # with col2:
        # st.metric('Valor Previsto', "R$ {:,.2f}".format(valor_previsto).replace(',', 'X').replace('.', ',').replace('X', '.'))

    st.markdown("---")
    # st.plotly_chart(fig_valores)
