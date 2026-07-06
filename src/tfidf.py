import math
from collections import Counter
from typing import List, Dict
import numpy as np


class TFIDFVectorizer:
    """
    Clase que implementa la técnica de procesamiento de lenguaje natural TF-IDF.
    Mide la importancia de una palabra en un documento (libro o versículo)
    a lo largo de una colección.

    Se aplican las siguientes fórmulas:

    TF  (term frequency o frecuencia de término): tf(t, d) = count(t in d) / len(d)

    IDF (inverse doc frequency o frecuencia inversa de documento):
    idf(t) = log((1 + N) / (1 + df(t))) + 1

    TF-IDF(t, d) = tf(t, d) * idf(t)
    """

    def __init__(self,normalizar=True):
        """
        Args:
        normalizar (bool): si es True, cada fila de la matriz TFIDF resultante
        se normaliza a norma L2 = 1 (vector unitario). Esto ayuda a los clasificadores lineales
        a converger mas rapido y tener mejor desempeño.
        """
        self.vocabulario: Dict[str, int] = {}
        self.idf: np.ndarray = None
        self.n_docs: int = 0
        self.normalizar = normalizar

    def fit(self, documentos_tokenizados: List[List[str]]) -> "TFIDFVectorizer":
        """
        Prepara los documentos ya tokenizados en para convertirlos en vectores TF-IDF.
        Además, calcula los pesos IDF.

        @param documentos_tokenizados: Lista de documentos representados como listas de tokens.
        @return: La instancia del vectorizador.
        """

        self.n_docs = len(documentos_tokenizados)

        vocab_set = set()
        for tokens in documentos_tokenizados:
            vocab_set.update(tokens)

        self.vocabulario = {palabra: i for i, palabra in enumerate(sorted(vocab_set))}

        df = np.zeros(len(self.vocabulario))
        for tokens in documentos_tokenizados:
            presentes = set(tokens)
            for palabra in presentes:
                df[self.vocabulario[palabra]] += 1

        self.idf = np.log((1 + self.n_docs) / (1 + df)) + 1
        return self
    
    def transform(self, documentos_tokenizados: List[List[str]]) -> np.ndarray:
        """
        Convierte los documentos en una matriz TF-IDF.

        @param documentos_tokenizados: Lista de documentos representados como listas de tokens.
        @return: Matriz Numpy de la forma (n_documentos, n_vocabulario) con los pesos TF-IDF.
        """
        
        if self.idf is None:
            raise RuntimeError("Llama a fit() antes de transform().")

        matriz = np.zeros((len(documentos_tokenizados), len(self.vocabulario)))
        for i, tokens in enumerate(documentos_tokenizados):
            if not tokens:
                continue
            conteo = Counter(tokens)
            total = len(tokens)
            for palabra, freq in conteo.items():
                if palabra in self.vocabulario:
                    j = self.vocabulario[palabra]
                    tf = freq / total
                    matriz[i, j] = tf * self.idf[j]

        if self.normalizar:
            normas = np.linalg.norm(matriz, axis=1)
            normas[normas == 0] = 1e-10  # evita dividir entre ceros si hay un documento vacios
            matriz = matriz / normas[:, np.newaxis]

        return matriz

    def fit_transform(self, documentos_tokenizados: List[List[str]]) -> np.ndarray:
        """
        Ajusta el modelo y devuelve la matriz TF-IDF resultante.

        @param documentos_tokenizados: Lista de documentos representados como listas de tokens.
        @return: Matriz Numpy con los pesos TF-IDF calculados.
        """

        self.fit(documentos_tokenizados)
        return self.transform(documentos_tokenizados)

    def vectorizar_texto_nuevo(self, tokens: List[str]) -> np.ndarray:
        """
        Vectoriza un único texto nuevo usando el vocabulario y el IDF ya entrenados.

        @param tokens: Lista de tokens del texto nuevo.
        @return: Vector Numpy con los pesos TF-IDF del texto nuevo.
        """

        return self.transform([tokens])[0]


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calcula la similitud del coseno entre dos vectores.

    @param vec_a: Primer vector.
    @param vec_b: Segundo vector.
    @return: Decimal entre 0 y 1 que indica qué tan similares son los vectores.
    """

    dot = np.dot(vec_a, vec_b)
    norm_a = math.sqrt(np.dot(vec_a, vec_a))
    norm_b = math.sqrt(np.dot(vec_b, vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def cosine_similarity_matrix(matriz: np.ndarray) -> np.ndarray:
    """
    Calcula la similitud de coseno entre todas las filas de una matriz.

    @param matriz: Matriz cuyas filas representan vectores de características.
    @return: Matriz Numpy cuadrada con las similitudes de coseno entre cada par de filas.
    """

    normas = np.linalg.norm(matriz, axis=1)
    normas[normas == 0] = 1e-10  # evita división por cero
    matriz_normalizada = matriz / normas[:, np.newaxis]
    return np.dot(matriz_normalizada, matriz_normalizada.T)
