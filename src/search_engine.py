"""
search_engine.py

Clase que implementa un buscador semantico: dado un texto (una consulta del
usuario o un versiculo del dataset), devuelve los K versiculos mas similares
del corpus usando similitud del coseno sobre los vectores TF-IDF
(implementados en tfidf.py).
"""

import pandas as pd
import numpy as np

from .tfidf import TFIDFVectorizer, cosine_similarity
from .preprocessing import TextPreprocessor


class SemanticSearchEngine:
    """
    Clase que representa un motor de busqueda semantico sobre el corpus.
 
    Atributos:
        preprocessor (TextPreprocessor): preprocesador usado para limpiar y
            tokenizar las consultas del usuario.
        vectorizer (TFIDFVectorizer): vectorizador TF-IDF usado para
            representar numericamente los versiculos y las consultas.
        matriz_tfidf (np.ndarray): matriz TF-IDF de todos los versiculos del
            corpus, calculada durante fit().
        df_corpus (pd.DataFrame): corpus de versiculos con sus metadatos
            (libro, capitulo, versiculo, texto_original).
    """

    def __init__(self, preprocessor, vectorizer):
        """
        Crea el motor de busqueda con un preprocesador y un vectorizador.
        Args:
            preprocessor (TextPreprocessor): preprocesador para las consultas.
            vectorizer (TFIDFVectorizer): vectorizador TF-IDF para el corpus.
        """
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.matriz_tfidf = None
        self.df_corpus = None   

    def fit(self, df_corpus, columna_tokens="texto_procesado"):
        """
        Guarda el corpus y calcula la matriz TF-IDF de todos los versiculos.
        Args:
            df_corpus (pd.DataFrame): corpus con los versiculos y sus metadatos.
            columna_tokens (str): nombre de la columna que contiene los tokens
                ya procesados de cada versiculo.
        Returns:
            SemanticSearchEngine: la misma instancia, ya ajustada.
        """
        self.df_corpus = df_corpus.reset_index(drop=True)
        documentos = df_corpus[columna_tokens].tolist()
        self.matriz_tfidf = self.vectorizer.fit_transform(documentos)
        return self

    def buscar(self, query, k=5):
        """
        Busca los k versiculos mas parecidos a una consulta del usuario.
        Args:
            query (str): la consulta del usuario.
            k (int): el numero de versiculos a devolver.
        Returns:
            pd.DataFrame: los k versiculos mas similares.
        """
        tokens_query = self.preprocessor.process(query)
        vector_query = self.vectorizer.vectorizar_texto_nuevo(tokens_query)
        resultado = self._rankear(vector_query, k)
        return resultado.reset_index(drop=True)

    def buscar_por_indice(self, idx_versiculo, k=5):
        """
        Busca los k versiculos mas similares a un versiculo que ya esta en el
        corpus, excluyendo al propio versiculo de los resultados.
        Args:
            idx_versiculo (int): indice del versiculo de referencia en el corpus.
            k (int): cantidad de versiculos a devolver.
        Returns:
            pd.DataFrame: los k versiculos mas similares con su valor de similitud.
        """
        vector_query = self.matriz_tfidf[idx_versiculo]
        resultado = self._rankear(vector_query, k + 1)
        
        resultado = resultado[resultado.index != idx_versiculo].head(k)
        return resultado.reset_index(drop=True)

    def _rankear(self, vector_query, k):
        """
        Calcula la similitud del vector consulta contra todo el corpus y arma
        el ranking de los k mas similares.
        Args:
            vector_query (np.ndarray): vector TF-IDF de la consulta.
            k (int): cantidad de resultados a devolver.
        Returns:
            pd.DataFrame: los k versiculos mas similares con su similitud,
                manteniendo el indice original del corpus.
        """
        similitudes = np.array([
            cosine_similarity(vector_query, self.matriz_tfidf[i])
            for i in range(self.matriz_tfidf.shape[0])
        ])

        top_idx = np.argsort(similitudes)[::-1][:k]

        resultado = self.df_corpus.iloc[top_idx].copy()
        resultado["similitud"] = similitudes[top_idx]

        columnas = [c for c in ["libro", "capitulo", "versiculo", "texto_original", "similitud"]
                    if c in resultado.columns]
        return resultado[columnas]