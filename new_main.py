texto = "O serviço foi ruim"

if "ruim" in texto:
    print("Sentimento negativo")
else:
    print("Sentimento positivo")
    
    # SIMLULAÇÃO DE ANALISE DE SENTIMENTO BASICA
    
    # COM NLTK
    
from nltk.tokenize import word_tokenize
import nltk

# Baixa os pacotes mais comuns e essenciais (recomendado)
nltk.download(['popular', 'punkt_tab'])

texto = "IA está transformando o mundo"
print(word_tokenize(texto))
