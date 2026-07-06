import re
import string
from collections import Counter
from typing import List, Dict, Iterable, Optional

class TextPreprocessor:
    """
    Contiene cada étapa del proceso de análisis bíblico. 
    Permite hacer modificaciones en las pruebas sin afectar la funcionalidad de las clases.
    """

    def __init__(
        self,
        stopwords: Optional[Iterable[str]] = None,
        min_token_len: int = 2,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = True,
        remove_stopwords: bool = True
    ):
        
        self.stopwords = set(stopwords)
        self.min_token_len = min_token_len
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.vocabulario: Dict[str, int] = {}
        self.frecuencias: Counter = Counter()

    def to_lowercase(self, text: str) -> str:
        """
        Convierte el texto a minúsculas cuando la opción está habilitada.

        @param text: Texto de entrada.
        @return: Texto transformado a minúsculas o sin cambios si la opción está desactivada.
        """

        return text.lower() if self.lowercase else text

    def strip_punctuation(self, text: str) -> str:
        """
        Elimina los signos de puntuación del texto cuando la opción está habilitada.

        @param text: Texto de entrada.
        @return: Texto sin puntuación, sustituyendo cada símbolo por un espacio, o el texto sin cambios si la opción no está activada.
        """

        if not self.remove_punctuation:
            return text

        punct = string.punctuation + "¿¡“”‘’—–"
        return text.translate(str.maketrans(punct, " " * len(punct)))

    def strip_special_and_numbers(self, text: str) -> str:
        """
        Elimina números y caracteres especiales, conservando solo letras y espacios, si la opción está habilitada.

        @param text: Texto de entrada.
        @return: Texto limpio sin números ni caracteres especiales, o sin cambios si la opción no está habilitada.
        """

        if not self.remove_numbers:
            return text

        return re.sub(r"[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ\s]", " ", text)

    def tokenize(self, text: str) -> List[str]:
        """
        Divide el texto en una lista de tokens.

        @param text: Texto de entrada.
        @return: Lista de tokens obtenidos al separar por espacios.
        """

        return text.split()

    def filter_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Elimina las stopwords de la lista de tokens en inglés, definida en el archivo "stopwords.json".

        @param tokens: Lista de tokens de entrada.
        @return: Lista de tokens sin las stopwords, si la opción está habilitada.
        """

        if not self.remove_stopwords:
            return tokens

        return [t for t in tokens if t not in self.stopwords]

    def filter_short_tokens(self, tokens: List[str]) -> List[str]:
        """
        Elimina los tokens que son más cortos que la longitud mínima permitida.

        @param tokens: Lista de tokens de entrada.
        @return: Lista de tokens que cumplen con la longitud mínima.
        """

        return [t for t in tokens if len(t) >= self.min_token_len]

    def process(self, text: str) -> List[str]:
        """
        Aplica todas las etapas del preprocesamiento en orden y devuelve los tokens resultantes.

        @param text: Texto original a procesar.
        @return: Lista de tokens ya limpiados y filtrados.
        """

        text = self.to_lowercase(text)
        text = self.strip_punctuation(text)
        text = self.strip_special_and_numbers(text)
        tokens = self.tokenize(text)
        tokens = self.filter_stopwords(tokens)
        tokens = self.filter_short_tokens(tokens)
        return tokens
    
    def process_ngram(self, text: str) -> List[str]:
        """
        Aplica todas las etapas en orden y devuelve la lista de tokens.
        Para n-grama no pueden eliminarse las palabras "puentes", para ser capaz de hilar palabras.
        """
        text = self.to_lowercase(text)
        text = self.strip_punctuation(text)
        text = self.strip_special_and_numbers(text)
        tokens = self.tokenize(text)
        tokens = self.filter_short_tokens(tokens)
        return tokens

    def process_corpus(self, textos: List[str]) -> List[List[str]]:
        """
        Procesa una lista de textos y actualiza el vocabulario y las frecuencias globales.

        @param textos: Lista de textos a procesar.
        @return: Lista de listas de tokens, una por cada texto procesado.
        """

        resultado = []
        for texto in textos:
            tokens = self.process(texto)
            resultado.append(tokens)
            self.frecuencias.update(tokens)
        self.vocabulario = {palabra: idx for idx, palabra in enumerate(sorted(self.frecuencias))}
        return resultado
    
    def process_corpus_ngram(self, textos: List[str]) -> List[List[str]]:
        """Procesa una lista de textos y actualiza vocabulario/frecuencias globales."""
        resultado = []
        for texto in textos:
            tokens = self.process_ngram(texto)
            resultado.append(tokens)
            self.frecuencias.update(tokens)
        self.vocabulario = {palabra: idx for idx, palabra in enumerate(sorted(self.frecuencias))}
        return resultado

    def palabras_mas_frecuentes(self, n: int = 20):
        """
        Devuelve las palabras más frecuentes según el conteo acumulado del corpus procesado.

        @param n: Número de palabras más frecuentes a devolver.
        @return: Lista de tuplas (palabra, frecuencia) ordenadas por frecuencia.
        """
        
        return self.frecuencias.most_common(n)
