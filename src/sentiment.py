"""
sentiment.py
------------
Analizador de sentimiento por versículo y agregacion por capitulo/libro.

El analizador está compuesto por 2 clases,
una funciona de interfaz (SentimentAnalyzer) que actua sobre TextBlobSentimentAnalyzer,
clase que implementa la libreria TextBlob que frece una API sencilla para abordar 
tareas habituales de PLN, como el etiquetado de partes de la palabra, extracción de sintagmas nominales, análisis de sentimiento, 
clasificación y mucho más.
https://textblob.readthedocs.io/en/dev/
"""

import pandas as pd


class SentimentAnalyzer:
    """
    Clase tipo interfaz, cualquier analizador que se implemente a futuro
    (ej: uno en español) debe implementar el metodo score(texto), que
    retorna un valor entre -1 y 1 (negativo o positivo).
    """

    def score(self, texto):
        """
        Calcula el score de sentimiento de un texto.
        Args:
            texto (str): texto a analizar (ej. un versiculo).
        Returns:
            float: puntaje de sentimiento (entre -1 y 1).
        """
        raise NotImplementedError

class TextBlobSentimentAnalyzer(SentimentAnalyzer):
    """
    Clase que analiza sentimiento de textos, usando TextBlob.

    Attributes:
        text_blob_class (type): referencia la clase TextBlob, importada en el init.
    """

    def __init__(self):
        from textblob import TextBlob
        self.text_blob_class = TextBlob

    def score(self, texto):
        """
        Calcula el score de sentimiento de un texto.
        Args:
            texto (str): texto a analizar (ej. un versiculo).
        Returns:
            float: puntaje de sentimiento (entre -1 y 1).
        """
        return self.text_blob_class(texto).sentiment.polarity


def calcular_sentimiento_corpus(df, analyzer, columna_texto="texto_original"):
    """
    Calcula el puntaje de sentimiento para cada texto de un Pandas.DataFrame y agrega los scores
    en una columna 'sentimiento'.
    Args:
        df (pd.DataFrame): DataFrame de versiculos.
        analyzer (SentimentAnalyzer): Analizador.
        zcolumna_texto (str): columna que contiene el texto a analizar. Por default= "texto_original"
    Returns:
        pd.DataFrame: Copia de df con la columna 'sentimiento' agregada.
    """
    df = df.copy()
    df["sentimiento"] = df[columna_texto].apply(analyzer.score)
    return df


def agregar_por_libro(df):
    """
    Agrupa los scores de sentimiento por libro y usa funciones de agregación sobre sus versículos.
    Args:
        df (pd.DataFrame): DataFrame con columnas 'libro' y 'sentimiento'.
    Returns:
        pd.DataFrame: una fila por libro, con mean, std, min, max y count de sentimiento, 
        ordenado de menor a mayor sentimiento promedio.
    """
    return (
        df.groupby("libro")["sentimiento"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .sort_values("mean")
    )


def agregar_por_capitulo(df):
    """
    Agrega los puntajes de sentimiento por libro y capitulo.
    Args:
        df (pd.DataFrame): DataFrame con columnas 'libro', 'capitulo' y 'sentimiento'.
    Returns:
        pd.DataFrame: una fila por (libro, capitulo) con el sentimiento promedio.
    """
    return (
        df.groupby(["libro", "capitulo"])["sentimiento"]
        .mean()
        .reset_index()
    )