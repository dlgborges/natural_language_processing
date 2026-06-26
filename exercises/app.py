import re
import string

from collections import Counter

import nltk
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from wordcloud import WordCloud

from unidecode import unidecode

@st.cache_resource
def download_nltk():

    nltk.download("punkt")
    nltk.download("stopwords")

download_nltk()

st.set_page_config(

    page_title="Laboratório NLP",

    page_icon="🧠",

    layout="wide"

)

st.title("🧠 Laboratório de Processamento de Linguagem Natural")

st.sidebar.title("Laboratório NLP")

st.sidebar.success("10 atividades")

st.sidebar.write("Tecnologias")

st.sidebar.write("• NLTK")

st.sidebar.write("• Streamlit")

st.sidebar.write("• Pandas")

st.sidebar.write("• Matplotlib")

st.sidebar.write("• WordCloud")

st.sidebar.markdown("---")

st.sidebar.info(
"""
Fluxo do laboratório

1️⃣ Tokenização

2️⃣ Frequência

3️⃣ Palavras negativas

4️⃣ Stopwords

5️⃣ Sentimento

6️⃣ Classificação

7️⃣ Reclamações

8️⃣ Setores

9️⃣ Limpeza

🔟 Pipeline NLP
"""
)

st.markdown("""

Este laboratório implementa 10 atividades utilizando técnicas de NLP.

Cada aba demonstra uma técnica diferente utilizando textos reais.

""")

@st.cache_data
def load_reviews():

    return pd.read_csv(
        "datasets/amazon_reviews.csv"
    )


@st.cache_data
def load_support():

    return pd.read_csv(
        "datasets/customer_support_tickets.csv"
    )

try:

    reviews_df = load_reviews()

except:

    reviews_df = pd.DataFrame()


try:

    support_df = load_support()

except:

    support_df = pd.DataFrame()


STOPWORDS = set(stopwords.words("portuguese"))

POSITIVE_WORDS = {

    "bom",
    "ótimo",
    "excelente",
    "rapido",
    "feliz",
    "maravilhoso",
    "gostei",
    "perfeito",
    "recomendo",
    "funciona"

}

NEGATIVE_WORDS = {

    "ruim",
    "péssimo",
    "erro",
    "defeito",
    "cancelar",
    "lento",
    "atraso",
    "horrível",
    "problema",
    "bug"

}

FINANCE_WORDS = {

    "pagamento",
    "boleto",
    "pix",
    "cartão",
    "fatura",
    "reembolso",
    "cobrança"

}

SUPPORT_WORDS = {

    "erro",
    "instalar",
    "login",
    "senha",
    "acesso",
    "bug",
    "falha"

}

def normalize_text(text):

    text = text.lower()

    text = unidecode(text)

    text = re.sub(r"[^\w\s]", "", text)

    return text

POSITIVE_WORDS = {normalize_text(word) for word in POSITIVE_WORDS}
NEGATIVE_WORDS = {normalize_text(word) for word in NEGATIVE_WORDS}
FINANCE_WORDS = {normalize_text(word) for word in FINANCE_WORDS}
SUPPORT_WORDS = {normalize_text(word) for word in SUPPORT_WORDS}

def tokenize(text):

    return word_tokenize(text)

def preprocess(text):

    text = normalize_text(text)

    tokens = tokenize(text)

    tokens = [

        token

        for token in tokens

        if token not in STOPWORDS

    ]

    return tokens

def word_frequency(tokens):

    return Counter(tokens)

def sentiment(text):

    tokens = preprocess(text)

    positives = [

        t

        for t in tokens

        if t in POSITIVE_WORDS

    ]

    negatives = [

        t

        for t in tokens

        if t in NEGATIVE_WORDS

    ]

    score = len(positives) - len(negatives)

    if score > 0:

        label = "😊 Positivo"

    elif score < 0:

        label = "😞 Negativo"

    else:

        label = "😐 Neutro"

    return {

        "label": label,

        "score": score,

        "positive": positives,

        "negative": negatives,

        "tokens": tokens

    }

def classify(text):

    tokens = preprocess(text)

    finance = [

        t

        for t in tokens

        if t in FINANCE_WORDS

    ]

    support = [

        t

        for t in tokens

        if t in SUPPORT_WORDS

    ]

    if len(finance) > len(support):

        department = "💰 Financeiro"

    elif len(support) > len(finance):

        department = "🛠️ Suporte Técnico"

    else:

        department = "📨 Não Classificado"

    return {

        "department": department,

        "finance": finance,

        "support": support,

        "tokens": tokens

    }

def plot_frequency(counter):

    top = counter.most_common(10)

    words = [i[0] for i in top]

    values = [i[1] for i in top]

    fig, ax = plt.subplots(figsize=(10,5))

    ax.bar(words, values)

    ax.set_title("Top 10 Palavras")

    plt.xticks(rotation=35)

    st.pyplot(fig)

def show_wordcloud(tokens):

    text = " ".join(tokens)

    wc = WordCloud(

        width=900,

        height=400,

        background_color="white"

    ).generate(text)

    fig, ax = plt.subplots(figsize=(12,5))

    ax.imshow(wc)

    ax.axis("off")

    st.pyplot(fig)

tabs = st.tabs([

    "Atividade 1",

    "Atividade 2",

    "Atividade 3",

    "Atividade 4",

    "Atividade 5",

    "Atividade 6",

    "Atividade 7",

    "Atividade 8",

    "Atividade 9",

    "Atividade 10"

])

def get_sample_text(df):

    if df.empty:
        return ""

    column = df.columns[0]

    return df[column].dropna().astype(str).head(100).tolist()

def show_metrics(text, tokens):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Caracteres", len(text))

    with col2:
        st.metric("Palavras", len(text.split()))

    with col3:
        st.metric("Tokens", len(tokens))

with tabs[0]:

    st.header("Atividade 1 - Tokenização")

    st.subheader("Problemática")

    st.write("""
Uma empresa recebe centenas de mensagens diariamente e deseja separar automaticamente
as palavras de cada mensagem para facilitar análises posteriores.
""")

    if not reviews_df.empty:

        sample = st.selectbox(
            "Escolha um comentário do dataset",
            reviews_df.iloc[:100, 0]
        )

    else:

        sample = ""

    text = st.text_area(
        "Ou digite seu próprio texto",
        value=sample,
        height=120,
        key="atividade1"
    )

    if st.button("Executar Tokenização"):

        tokens = tokenize(text)

        st.success("Texto tokenizado com sucesso!")

        st.write(tokens)

        st.metric("Quantidade de Tokens", len(tokens))

        df = pd.DataFrame({

            "Token": tokens

        })

        st.dataframe(df, use_container_width=True)

    with st.expander("Explicação Técnica"):

        st.write("""

A tokenização consiste em dividir um texto em unidades menores
(tokens).

Exemplo:

Produto chegou quebrado hoje

↓

Produto

chegou

quebrado

hoje

Esses tokens serão utilizados nas próximas etapas de NLP.

""")

with tabs[1]:

    st.header("Atividade 2 - Frequência das Palavras")

    st.subheader("Problemática")

    st.write("""
Identificar quais palavras aparecem com maior frequência nas avaliações
dos clientes.
""")

    if not reviews_df.empty:

        sample = st.selectbox(

            "Comentário",

            reviews_df.iloc[:100,0],

            key="atividade2"

        )

    else:

        sample = ""

    text = st.text_area(

        "Texto",

        value=sample,

        height=120

    )

    if st.button("Calcular Frequência"):

        tokens = preprocess(text)

        counter = word_frequency(tokens)

        freq_df = pd.DataFrame(

            counter.items(),

            columns=[

                "Palavra",

                "Frequência"

            ]

        )

        freq_df = freq_df.sort_values(

            "Frequência",

            ascending=False

        )

        st.dataframe(

            freq_df,

            use_container_width=True

        )

        st.subheader("Top Palavras")

        plot_frequency(counter)

        st.metric(

            "Palavras distintas",

            len(counter)

        )

        st.metric(

            "Total de palavras",

            len(tokens)

        )

    with st.expander("Explicação Técnica"):

        st.write("""

Após remover stopwords,
cada palavra é contabilizada utilizando Counter.

Isso permite descobrir rapidamente quais termos aparecem mais.

""")

with tabs[2]:

    st.header("Atividade 3 - Detecção de Palavras Negativas")

    st.subheader("Problemática")

    st.write("""
Detectar automaticamente mensagens contendo palavras negativas
para priorizar o atendimento.
""")

    if not support_df.empty:

        sample = st.selectbox(

            "Mensagem",

            support_df.iloc[:100,0],

            key="atividade3"

        )

    else:

        sample = ""

    text = st.text_area(

        "Mensagem",

        value=sample,

        height=120

    )

    if st.button("Analisar Mensagem"):

        tokens = preprocess(text)

        negatives = []

        for token in tokens:

            if token in NEGATIVE_WORDS:

                negatives.append(token)

        if negatives:

            st.error("Mensagem Prioritária")

            st.write("Palavras negativas encontradas:")

            st.write(negatives)

        else:

            st.success("Nenhuma palavra negativa encontrada.")

        st.subheader("Tokens")

        st.write(tokens)

    with st.expander("Explicação Técnica"):

        st.write("""

Foi utilizada uma abordagem baseada em regras.

Cada token é comparado com um conjunto de palavras negativas.

Caso exista alguma correspondência,
a mensagem recebe prioridade.

""")

with tabs[3]:

    st.header("Atividade 4 - Remoção de Stopwords")

    st.subheader("Problemática")

    st.write("""
Em análises de texto, palavras como "de", "a", "o", "para"
aparecem com muita frequência, mas normalmente não carregam
informação relevante.
""")

    sample = ""

    if not reviews_df.empty:

        sample = st.selectbox(

            "Escolha um comentário",

            get_sample_text(reviews_df),

            key="atividade4_select"

        )

    text = st.text_area(

        "Texto",

        value=sample,

        key="atividade4"

    )

    if st.button("Remover Stopwords"):

        normalized = normalize_text(text)

        original_tokens = tokenize(normalized)

        filtered_tokens = preprocess(text)

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Antes")

            st.write(original_tokens)

            st.metric(

                "Quantidade",

                len(original_tokens)

            )

        with col2:

            st.subheader("Depois")

            st.write(filtered_tokens)

            st.metric(

                "Quantidade",

                len(filtered_tokens)

            )

        removed = [

            t

            for t in original_tokens

            if t not in filtered_tokens

        ]

        st.subheader("Stopwords removidas")

        st.write(sorted(set(removed)))

    with st.expander("Explicação Técnica"):

        st.write("""

As stopwords são palavras extremamente frequentes que,
na maioria das análises estatísticas de texto,
não agregam significado.

Sua remoção reduz o ruído do processamento.

""")

with tabs[4]:

    st.header("Atividade 5 - Classificação de Sentimento")

    st.subheader("Problemática")

    st.write("""
A equipe de marketing deseja identificar rapidamente
se um comentário possui tendência positiva
ou negativa.
""")

    sample = ""

    if not reviews_df.empty:

        sample = st.selectbox(

            "Comentário",

            get_sample_text(reviews_df),

            key="atividade5_select"

        )

    text = st.text_area(

        "Comentário",

        value=sample,

        key="atividade5"

    )

    if st.button("Analisar Sentimento"):

        result = sentiment(text)

        st.subheader("Resultado")

        st.success(result["label"])

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(

                "Score",

                result["score"]

            )

        with col2:

            st.metric(

                "Palavras positivas",

                len(result["positive"])

            )

        with col3:

            st.metric(

                "Palavras negativas",

                len(result["negative"])

            )

        st.subheader("Palavras positivas")

        st.write(result["positive"])

        st.subheader("Palavras negativas")

        st.write(result["negative"])

        st.subheader("Tokens analisados")

        st.write(result["tokens"])

    with st.expander("Explicação Técnica"):

        st.write("""

Foi utilizada uma abordagem baseada em regras.

Cada palavra positiva aumenta o score.

Cada palavra negativa reduz o score.

O resultado final depende da diferença entre ambos.

""")

with tabs[5]:

    st.header("Atividade 6 - Roteamento por Palavras-chave")

    st.subheader("Problemática")

    st.write("""
Um chatbot precisa identificar automaticamente
qual departamento deve receber a solicitação do cliente.
""")

    sample = ""

    if not support_df.empty:

        sample = st.selectbox(

            "Mensagem",

            get_sample_text(support_df),

            key="atividade6_select"

        )

    text = st.text_area(

        "Mensagem",

        value=sample,

        key="atividade6"

    )

    if st.button("Classificar Departamento"):

        result = classify(text)

        st.subheader("Departamento")

        st.success(result["department"])

        col1, col2 = st.columns(2)

        with col1:

            st.write("Palavras Financeiras")

            st.write(result["finance"])

        with col2:

            st.write("Palavras Técnicas")

            st.write(result["support"])

        st.subheader("Tokens")

        st.write(result["tokens"])

    with st.expander("Explicação Técnica"):

        st.write("""

O algoritmo procura palavras associadas
a cada departamento.

O setor com maior número de ocorrências
é escolhido como destino da mensagem.

""")

with tabs[6]:

    st.header("Atividade 7 - Palavras mais Frequentes")

    st.subheader("Problemática")

    st.write("""
Identificar rapidamente quais palavras aparecem com maior frequência
nas reclamações dos clientes.
""")

    sample = ""

    if not support_df.empty:

        sample = st.selectbox(

            "Mensagem",

            get_sample_text(support_df),

            key="atividade7_select"

        )

    text = st.text_area(

        "Texto",

        value=sample,

        key="atividade7"

    )

    if st.button("Analisar Reclamação"):

        tokens = preprocess(text)

        counter = Counter(tokens)

        show_metrics(text, tokens)

        st.subheader("Top 10 Palavras")

        freq = pd.DataFrame(

            counter.most_common(10),

            columns=[

                "Palavra",

                "Frequência"

            ]

        )

        st.dataframe(freq, use_container_width=True)

        plot_frequency(counter)

        st.subheader("WordCloud")

        show_wordcloud(tokens)

    with st.expander("Explicação Técnica"):

        st.write("""

Após a limpeza do texto,
as palavras restantes são contabilizadas.

As mais frequentes representam
os assuntos predominantes das reclamações.

""")

with tabs[7]:

    st.header("Atividade 8 - Classificação de Mensagens")

    st.subheader("Problemática")

    st.write("""
Classificar automaticamente uma mensagem
como Financeiro ou Suporte Técnico.
""")

    sample = ""

    if not support_df.empty:

        sample = st.selectbox(

            "Mensagem",

            get_sample_text(support_df),

            key="atividade8_select"

        )

    text = st.text_area(

        "Mensagem",

        value=sample,

        key="atividade8"

    )

    if st.button("Classificar"):

        result = classify(text)

        show_metrics(text, result["tokens"])

        st.success(result["department"])

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Financeiro")

            st.write(result["finance"])

        with col2:

            st.subheader("Suporte")

            st.write(result["support"])

        st.subheader("Tokens")

        st.write(result["tokens"])

    with st.expander("Explicação Técnica"):

        st.write("""

O algoritmo utiliza um modelo baseado
em regras.

Cada palavra pertence
a um conjunto de palavras-chave.

O conjunto predominante determina
o departamento responsável.

""")

with tabs[8]:

    st.header("Atividade 9 - Limpeza de Texto")

    st.subheader("Problemática")

    st.write("""
Preparar um texto para processamento removendo
pontuação e convertendo tudo para letras minúsculas.
""")

    text = st.text_area(

        "Digite um texto",

        "Olá!! Meu Produto CHEGOU Hoje!!!",

        key="atividade9"

    )

    if st.button("Normalizar Texto"):

        normalized = normalize_text(text)

        tokens = tokenize(normalized)

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Texto Original")

            st.code(text)

        with col2:

            st.subheader("Texto Normalizado")

            st.code(normalized)

        st.subheader("Tokens")

        st.write(tokens)

        show_metrics(normalized, tokens)

    with st.expander("Explicação Técnica"):

        st.write("""

Foram aplicadas três etapas:

• Conversão para minúsculas

• Remoção de acentos

• Remoção de pontuação

Esse processo reduz a quantidade
de palavras equivalentes tratadas como diferentes.

""")

with tabs[9]:

    st.header("Atividade 10 - Pipeline Completo de NLP")

    st.subheader("Problemática")

    st.write("""
Executar todas as etapas de processamento
de linguagem natural em sequência.
""")

    sample = ""

    if not reviews_df.empty:

        sample = st.selectbox(

            "Comentário",

            get_sample_text(reviews_df),

            key="atividade10_select"

        )

    text = st.text_area(

        "Texto",

        value=sample,

        key="atividade10"

    )

    if st.button("Executar Pipeline"):

        status = st.status("Executando Pipeline...", expanded=True)

        status.write("Normalizando texto...")

        normalized = normalize_text(text)

        status.write("Tokenizando...")

        tokens = tokenize(normalized)

        status.write("Removendo Stopwords...")

        filtered = preprocess(text)

        status.write("Calculando Frequências...")

        counter = Counter(filtered)

        status.write("Analisando Sentimento...")

        sentiment_result = sentiment(text)

        status.write("Classificando Departamento...")

        department = classify(text)

        status.update(

            label="Pipeline concluído",

            state="complete"

        )

        show_metrics(text, filtered)

        st.divider()

        st.subheader("Texto Normalizado")

        st.code(normalized)

        st.subheader("Tokens")

        st.write(tokens)

        st.subheader("Tokens após limpeza")

        st.write(filtered)

        st.subheader("Sentimento")

        st.success(sentiment_result["label"])

        st.metric(

            "Score",

            sentiment_result["score"]

        )

        st.subheader("Departamento")

        st.info(department["department"])

        st.subheader("Frequência")

        plot_frequency(counter)

        st.subheader("WordCloud")

        show_wordcloud(filtered)

        st.subheader("Tabela de Frequência")

        st.dataframe(

            pd.DataFrame(

                counter.most_common(),

                columns=[

                    "Palavra",

                    "Ocorrências"

                ]

            ),

            use_container_width=True

        )

    with st.expander("Explicação Técnica"):

        st.write("""

Pipeline executado:

1. Normalização

2. Tokenização

3. Remoção de Stopwords

4. Contagem de Frequências

5. Classificação de Sentimento

6. Classificação do Departamento

7. Visualização dos Resultados

""")
