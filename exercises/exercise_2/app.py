import streamlit as st
import re
import string
from collections import Counter

st.set_page_config(
    page_title="NLP Toolkit",
    page_icon="🧠",
    layout="wide"
)

# ----------------------------
# Funções auxiliares
# ----------------------------

def activity_header(number, title, description):
    st.header(f"Atividade {number}")
    st.subheader(title)
    st.info(description)

    return st.text_area(
        "Digite um texto:",
        height=150,
        key=f"text_{number}"
    )

# ----------------------------
# NLP (Python puro)
# ----------------------------

NEGATIVE_WORDS = {
    "ruim",
    "péssimo",
    "pessimo",
    "erro",
    "problema",
    "defeito",
    "falha"
}

POSITIVE_WORDS = {
    "bom",
    "ótimo",
    "otimo",
    "excelente",
    "gostei",
    "perfeito"
}

STOPWORDS = {
    "a","o","os","as",
    "de","da","do","das","dos",
    "e","é",
    "para","por","em",
    "um","uma","uns","umas",
    "com","sem",
    "na","no","nas","nos"
}


def normalize_text(text):

    text = text.lower()

    text = re.sub(
        f"[{re.escape(string.punctuation)}]",
        " ",
        text
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text):

    text = normalize_text(text)

    return text.split()


def remove_stopwords(tokens):

    return [
        token
        for token in tokens
        if token not in STOPWORDS
    ]


def word_frequency(tokens):

    return Counter(tokens)


def detect_negative(tokens):

    return [
        token
        for token in tokens
        if token in NEGATIVE_WORDS
    ]


def simple_sentiment(tokens):

    positive = sum(
        word in POSITIVE_WORDS
        for word in tokens
    )

    negative = sum(
        word in NEGATIVE_WORDS
        for word in tokens
    )

    if positive > negative:
        return "😊 Positivo"

    if negative > positive:
        return "☹️ Negativo"

    return "😐 Neutro"

# ----------------------------
# Interface
# ----------------------------

st.title("🧠 NLP Toolkit")
st.caption("Processamento de Linguagem Natural com Python")

tabs = st.tabs([
    "🏠 Home",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10"
])

# ----------------------------
# HOME
# ----------------------------

with tabs[0]:

    st.header("Projeto")

    st.write("""
Este projeto implementa 10 atividades de Processamento de Linguagem Natural
utilizando Python, NLTK, spaCy e Streamlit.

Nesta primeira versão estamos apenas validando a estrutura da aplicação.
""")

    st.success("Estrutura carregada com sucesso.")

# ----------------------------
# ATIVIDADE 1
# ----------------------------

with tabs[1]:

    texto = activity_header(
    1,
    "Tokenização",
    "Separar um texto em palavras."
)

    if st.button("Executar", key="b1"):

        tokens = tokenize(texto)

        st.subheader("Tokens")

        st.write(tokens)

# ----------------------------

with tabs[2]:

    texto = activity_header(
    2,
    "Frequência",
    "Contar frequência das palavras."
)

if st.button("Executar", key="b2"):

    tokens = tokenize(texto)

    frequencia = word_frequency(tokens)

    st.subheader("Frequência")

    st.table(frequencia.items())

# ----------------------------

with tabs[3]:

    texto = activity_header(
    3,
    "Palavras negativas",
    "Detectar palavras negativas."
)

if st.button("Executar", key="b3"):

    tokens = tokenize(texto)

    negativas = detect_negative(tokens)

    if negativas:

        st.error("Mensagem prioritária.")

        st.write(negativas)

    else:

        st.success("Nenhuma palavra negativa encontrada.")

# ----------------------------

with tabs[4]:

    texto = activity_header(
    4,
    "Stopwords",
    "Remover palavras irrelevantes."
)

    if st.button("Executar", key="b4"):

        tokens = tokenize(texto)

        resultado = remove_stopwords(tokens)

        st.write(resultado)

# ----------------------------

with tabs[5]:

    texto = activity_header(
    5,
    "Sentimento",
    "Classificação simples."
)

    if st.button("Executar", key="b5"):

        tokens = tokenize(texto)

        resultado = simple_sentiment(tokens)

        st.header(resultado)

# ----------------------------

with tabs[6]:

    texto = activity_header(
        6,
        "Palavras-chave",
        "Detectar palavras-chave."
    )

    if st.button("Executar", key="b6"):
        st.write(texto)

# ----------------------------

with tabs[7]:

    texto = activity_header(
        7,
        "Reclamações",
        "Palavras mais frequentes."
    )

    if st.button("Executar", key="b7"):
        st.write(texto)

# ----------------------------

with tabs[8]:

    texto = activity_header(
        8,
        "Classificação",
        "Suporte ou Financeiro."
    )

    if st.button("Executar", key="b8"):
        st.write(texto)

# ----------------------------

with tabs[9]:

    texto = activity_header(
        9,
        "Normalização",
        "Remover pontuação."
    )

    if st.button("Executar", key="b9"):
        st.write(texto)

# ----------------------------

with tabs[10]:

    texto = activity_header(
        10,
        "Pipeline",
        "Tokenização + Sentimento."
    )

    if st.button("Executar", key="b10"):
        st.write(texto)