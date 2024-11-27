import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, classification_report
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests

# Título
st.title("Análise de Modelos para Câncer de Mama")

# Usando HTML para diminuir o tamanho da fonte
st.markdown("""
    <style>
        .small-font {
            font-size: 12px;
        }
    </style>
    <div class="small-font">
        https://github.com/Julio-CSilva/cancer_analysis/blob/master/40X.parquet
    </div>
""", unsafe_allow_html=True)

# Link para a apresentação no Canva
canva_link = "https://www.canva.com/design/DAGXmeQYkh0/RTABBP1BWupt8Hyk94shZA/view?utm_content=DAGXmeQYkh0&utm_campaign=designshare&utm_medium=link&utm_source=editor"

# Criar botão que redireciona
st.write("### Acesse a Apresentação")
button_html = f"""
    <a href="{canva_link}" target="_blank">
        <button style="
            background-color: #4CAF50; 
            border: none; 
            color: white; 
            padding: 10px 20px; 
            text-align: center; 
            text-decoration: none; 
            display: inline-block; 
            font-size: 16px; 
            margin: 4px 2px; 
            cursor: pointer; 
            border-radius: 12px;">
            Abrir Apresentação
        </button>
    </a>
"""
st.markdown(button_html, unsafe_allow_html=True)

st.write("")

# Carregar os dados dinamicamente
@st.cache_data
def load_data(file):
    return pd.read_parquet(file)

uploaded_file = st.file_uploader("Faça upload do arquivo .parquet", type="parquet")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("### Pré-visualização dos Dados")
    st.dataframe(df.head())

    # Análise de classes
    st.write("### Análise de Classes")
    st.bar_chart(df['classe'].value_counts())

    # Seleção de atributos
    features = st.multiselect("Selecione os atributos para o modelo", options=df.columns[:-1], default=df.columns[:-1])
    target = 'classe'

    # Sliders para ajustar os tamanhos dos conjuntos
    st.write("#### Ajuste o tamanho dos conjuntos")
    test_size = st.slider("Porcentagem para Conjunto de Teste:", 10, 60, 20)
    validation_size = st.slider("Porcentagem para Conjunto de Validação:", 10, 40, 20)
    train_size = 100 - (test_size + validation_size)

    # Garantir que as porcentagens somem 100%
    if test_size + validation_size >= 100:
        st.error("A soma dos tamanhos de Teste e Validação deve ser menor que 100%.")
    else:
        st.write(f"**Tamanho dos Conjuntos:** Treino: {train_size}%, Validação: {validation_size}%, Teste: {test_size}%")
        
        # Divisão dos dados
        X = df[features]
        y = LabelEncoder().fit_transform(df[target])

        # Primeira divisão: treino e o resto (validação + teste)
        x_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + validation_size) / 100, random_state=42, stratify=y
        )

        # Segunda divisão: validação e teste
        validation_ratio = validation_size / (test_size + validation_size)  # Proporção relativa
        x_val, x_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1 - validation_ratio, random_state=42, stratify=y_temp
        )

        # Pipeline de pré-processamento
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.7))
        ])

        # Aplicar transformação nos conjuntos
        x_train = pipeline.fit_transform(x_train)
        x_val = pipeline.transform(x_val)
        x_test = pipeline.transform(x_test)

    # Treinamento do Modelo
    if st.button("Treinar Modelo MLP"):
        model = MLPClassifier(random_state=42)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        # Avaliação
        st.write("### Resultados do Modelo")

        # Calculando as métricas
        mcc = matthews_corrcoef(y_test, pred)
        acc = balanced_accuracy_score(y_test, pred)
        report = classification_report(y_test, pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        # Apresentação das principais métricas
        st.metric(label="**MCC (Matthews Correlation Coefficient)**", value=f"{mcc:.2f}")
        st.metric(label="**Acurácia Balanceada**", value=f"{acc:.2f}")

        # Exibindo o Classification Report como tabela
        st.write("#### Relatório de Classificação")
        st.dataframe(report_df.style.background_gradient(cmap="viridis", subset=["precision", "recall", "f1-score"]))

        # Matriz de Confusão
        st.write("#### Matriz de Confusão")
        conf_matrix = pd.crosstab(y_test, pred, rownames=["Real"], colnames=["Predito"])

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        plt.title("Matriz de Confusão")
        plt.ylabel("Classe Real")
        plt.xlabel("Classe Predita")
        st.pyplot(fig)

        # Exibir Gráfico das Métricas do Classification Report
        st.write("#### Métricas de Precisão, Recall e F1-Score por Classe")
        fig, ax = plt.subplots(figsize=(10, 6))
        report_df.drop(["accuracy"], errors="ignore", inplace=True)  # Remove linha 'accuracy' se existir
        report_df[["precision", "recall", "f1-score"]].iloc[:-1].plot(kind="bar", ax=ax, colormap="viridis")
        plt.title("Classificação por Métrica")
        plt.ylabel("Score")
        plt.xlabel("Classes")
        plt.xticks(rotation=0)
        st.pyplot(fig)


    # Visualização PCA
    if st.button("Visualizar PCA"):
        pca = PCA(n_components=2)
        x_pca = pca.fit_transform(pipeline.transform(X))
        pca_df = pd.DataFrame(data=x_pca, columns=['PC1', 'PC2'])
        pca_df['Classe'] = y

        st.write("### Visualização PCA")
        sns.scatterplot(x='PC1', y='PC2', hue='Classe', data=pca_df)
        st.pyplot(plt)

else:
    st.warning("Por favor, faça o upload de um arquivo .parquet para começar.")
