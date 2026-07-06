"""
visualization.py

Funciones para graficar informacion del dataset de versiculos biblicos.

Cada funcion construye un grafico usando matplotlib y devuelve la figura
para que pueda mostrarse o guardarse desde el notebook principal.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_longitud_versiculos(df, columna="texto_original"):
    """
    Grafica la distribucion de longitud de los versiculos en cantidad de palabras.

    Args:
        df (pd.DataFrame): DataFrame que contiene los versiculos.
        columna (str): nombre de la columna donde se encuentra el texto original.

    Returns:
        matplotlib.figure.Figure: figura con el histograma de longitudes.
    """
    longitudes = [len(str(texto).split()) for texto in df[columna]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(longitudes, bins=40)
    ax.set_title("Distribucion de longitud de versiculos (en palabras)")
    ax.set_xlabel("Cantidad de palabras")
    ax.set_ylabel("Frecuencia")
    fig.tight_layout()
    return fig


def plot_versiculos_por_libro(df):
    """
    Grafica la cantidad de versiculos que tiene cada libro del corpus.

    Args:
        df (pd.DataFrame): DataFrame con una columna llamada 'libro'.

    Returns:
        matplotlib.figure.Figure: figura con un grafico de barras por libro.
    """
    conteo = df.groupby("libro").size().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(conteo.index, conteo.values)
    ax.set_title("Cantidad de versiculos por libro")
    ax.set_ylabel("N° de versiculos")
    plt.xticks(rotation=90)
    fig.tight_layout()
    return fig


def plot_heatmap_similitud_libros(matriz_similitud, nombres_libros):
    """
    Grafica un mapa de calor con la similitud de coseno entre libros.

    Args:
        matriz_similitud (np.ndarray): matriz cuadrada con las similitudes entre libros.
        nombres_libros (list[str]): lista con los nombres de los libros en el mismo orden
            que la matriz de similitud.

    Returns:
        matplotlib.figure.Figure: figura con el heatmap de similitud.
    """
    fig, ax = plt.subplots(figsize=(18, 16))
    imagen = ax.imshow(matriz_similitud, cmap="Blues")
    fig.colorbar(imagen, ax=ax)

    ax.set_xticks(range(len(nombres_libros)))
    ax.set_yticks(range(len(nombres_libros)))
    ax.set_xticklabels(nombres_libros, rotation=90, fontsize=6)
    ax.set_yticklabels(nombres_libros, fontsize=6)

    ax.set_title("Similitud de coseno entre libros (basado en TF-IDF)")
    fig.tight_layout()
    return fig


def plot_pca_versiculos(matriz_tfidf, etiquetas, titulo="Versiculos proyectados con PCA"):
    """
    Reduce la matriz TF-IDF a dos componentes principales y grafica los versiculos.

    Args:
        matriz_tfidf (np.ndarray): matriz TF-IDF de los versiculos.
        etiquetas (list, np.ndarray o pd.Series): etiquetas usadas para colorear los puntos
            del grafico, por ejemplo libro, testamento o genero.
        titulo (str): titulo del grafico.

    Returns:
        matplotlib.figure.Figure: figura con el grafico de dispersion PCA.
    """
    from sklearn.decomposition import PCA

    if hasattr(matriz_tfidf, "toarray"):
        matriz_tfidf = matriz_tfidf.toarray()

    pca = PCA(n_components=2)
    componentes = pca.fit_transform(matriz_tfidf)

    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100

    if hasattr(etiquetas, "values"):
        etiquetas = etiquetas.values
    df_plot = pd.DataFrame({
        "PC1": componentes[:, 0],
        "PC2": componentes[:, 1],
        "etiqueta": etiquetas,
    })

    fig, ax = plt.subplots(figsize=(10, 8))
    for etiqueta in df_plot["etiqueta"].unique():
        sub = df_plot[df_plot["etiqueta"] == etiqueta]
        ax.scatter(sub["PC1"], sub["PC2"], s=15, alpha=0.5, label=etiqueta)

    ax.set_title(titulo)
    ax.set_xlabel(f"PC1 ({var_pc1:.1f}% var. explicada)")
    ax.set_ylabel(f"PC2 ({var_pc2:.1f}% var. explicada)")

    if df_plot["etiqueta"].nunique() <= 15:
        ax.legend()
    fig.tight_layout()
    return fig


def plot_sentimiento_por_libro(df_sentimiento_agregado):
    """
    Grafica el sentimiento promedio de cada libro.

    Args:
        df_sentimiento_agregado (pd.DataFrame): DataFrame agregado por libro,
            con columnas 'libro' y 'mean'.

    Returns:
        matplotlib.figure.Figure: figura con un grafico de barras horizontales.
    """
    df_ord = df_sentimiento_agregado.sort_values("mean")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(df_ord["libro"], df_ord["mean"])
    ax.set_title("Sentimiento promedio por libro")
    ax.set_xlabel("Sentimiento promedio (negativo <- 0 -> positivo)")
    fig.tight_layout()
    return fig


def plot_wordcloud(frecuencias, titulo="Palabras mas frecuentes"):
    """
    Genera una nube de palabras a partir de un diccionario de frecuencias.

    Args:
        frecuencias (dict): diccionario donde la clave es la palabra y el valor es su frecuencia.
        titulo (str): titulo del grafico.

    Returns:
        matplotlib.figure.Figure: figura con la nube de palabras.
    """
    from wordcloud import WordCloud

    wc = WordCloud(width=900, height=500, background_color="white")
    wc = wc.generate_from_frequencies(frecuencias)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(titulo)
    return fig


def plot_matriz_confusion(cm, clases):
    """
    Grafica la matriz de confusion del clasificador de versiculos.

    Args:
        cm (np.ndarray): matriz de confusion generada durante la evaluacion del modelo.
        clases (list[str] o np.ndarray): nombres de las clases usadas como etiquetas.

    Returns:
        matplotlib.figure.Figure: figura con la matriz de confusion.
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    imagen = ax.imshow(cm, cmap="Blues")
    fig.colorbar(imagen, ax=ax)

    ax.set_xticks(range(len(clases)))
    ax.set_yticks(range(len(clases)))
    ax.set_xticklabels(clases, rotation=90)
    ax.set_yticklabels(clases)

    ax.set_title("Matriz de confusion - Clasificador de versiculos")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    fig.tight_layout()
    return fig
