"""
dashboard.py
---------------------
Endpoints del dashboard 
"""

from collections import Counter
from typing import Optional

from fastapi import APIRouter, Query

from api.singleton import AppState

router = APIRouter()


def _filtrar_df(testamento=None, libro=None, capitulo=None):
    """
    Filtra el DataFrame del corpus segun los parametros ingresados.
    Args:
        testamento (str, optional): Testamento por el que filtrar ("OT" o "NT").
        libro (str, optional): Nombre del libro por el que filtrar.
        capitulo (int, optional): Numero de capitulo por el que filtrar.
    Returns:
        pd.DataFrame: Subconjunto del corpus con los filtros aplicados.
    """
    df = AppState().df.copy()
    if testamento:
        df = df[df["testamento"] == testamento]
    if libro:
        df = df[df["libro"] == libro]
    if capitulo is not None:
        df = df[df["capitulo"] == capitulo]
    return df


@router.get("/filters")
def get_filters():
    """
    Devuelve los valores disponibles para los selectores del dashboard.

    Returns:
        dict: Diccionario con dos listas:
            - "testamentos" (list[str]): Testamentos disponibles en el corpus.
            - "libros" (list[str]): Libros disponibles en el corpus.
    """
    return {
        "testamentos": sorted(AppState().df["testamento"].unique().tolist()),
        "libros": sorted(AppState().df["libro"].unique().tolist()),
    }


@router.get("/chapters")
def get_chapters(book: str = Query(..., description="Nombre del libro")):
    """
    Devuelve los numeros de capitulos de un libro.
    Args:
        book (str): Nombre del libro del que se quieren obtener los capitulos.
            Parametro requerido (Query).
    Returns:
        dict: Diccionario con una lista:
            - "capitulos" (list[int]): Numeros de capitulo disponibles,
              ordenados de menor a mayor.
    """
    capitulos = (
        AppState().df[AppState().df["libro"] == book]["capitulo"]
        .unique()
        .tolist()
    )
    return {"capitulos": sorted(capitulos)}


@router.get("/stats")
def get_stats(
    testament: Optional[str] = Query(None, description="Filtrar por testamento (OT o NT)"),
    book: Optional[str] = Query(None, description="Filtrar por nombre de libro"),
    chapter: Optional[int] = Query(None, description="Filtrar por numero de capitulo"),
):
    """
    Devuelve cantidad de versiculos y longitud promedio agrupados por libro
    Args:
        testament (str, optional): Testamento por el que filtrar.
        book (str, optional): Libro por el que filtrar.
        chapter (int, optional): Capitulo por el que filtrar.
    Returns:
        list[dict]: Lista de diccionarios, uno por libro, con:
            - "libro" (str): Nombre del libro.
            - "n_versiculos" (int): Cantidad de versiculos del libro.
            - "longitud_promedio" (float): Promedio de palabras por versiculo,
              redondeado a 1 decimal.
        La lista viene ordenada de mayor a menor cantidad de versiculos.
    """
    df = _filtrar_df(testament, book, chapter)
    df = df.copy()
    df["n_palabras"] = df["texto_original"].str.split().str.len()

    stats = (
        df.groupby("libro")
        .agg(
            n_versiculos=("texto_original", "count"),
            longitud_promedio=("n_palabras", "mean"),
        )
        .reset_index()
        .sort_values("n_versiculos", ascending=False)
    )
    stats["longitud_promedio"] = stats["longitud_promedio"].round(1)
    return stats.to_dict(orient="records")


@router.get("/top-words")
def get_top_words(
    testament: Optional[str] = Query(None, description="Filtrar por testamento (OT o NT)"),
    book: Optional[str] = Query(None, description="Filtrar por nombre de libro"),
    chapter: Optional[int] = Query(None, description="Filtrar por numero de capitulo"),
    n: int = Query(20, ge=1, le=100, description="Cantidad de palabras a retornar"),
):
    """
    Devuelve las n palabras mas frecuentes del subconjunto filtrado,
    calculadas sobre el texto preprocesado (sin stopwords).
    Args:
        testament (str, optional): Testamento por el que filtrar.
        book (str, optional): Libro por el que filtrar.
        chapter (int, optional): Capitulo por el que filtrar.
        n (int): Cantidad de palabras a retornar. Minimo 1, maximo 100.
            Por defecto 20.
    Returns:
        list[dict]: Lista de n diccionarios ordenados de mayor a menor
        frecuencia, cada uno con:
            - "palabra" (str): La palabra.
            - "frecuencia" (int): Cantidad de veces que aparece en el
              subconjunto filtrado.
    """
    df = _filtrar_df(testament, book, chapter)
    conteo = Counter()
    for tokens in df["texto_procesado"]:
        conteo.update(tokens)
    top = conteo.most_common(n)
    resultado = []
    for p, f in top:
        diccionario = {"palabra": p, "frecuencia": f}
        resultado.append(diccionario)
    return resultado


@router.get("/wordcloud")
def get_wordcloud(
    testament: Optional[str] = Query(None, description="Filtrar por testamento (OT o NT)"),
    book: Optional[str] = Query(None, description="Filtrar por nombre de libro"),
    chapter: Optional[int] = Query(None, description="Filtrar por numero de capitulo"),
    n: int = Query(100, ge=1, le=500, description="Cantidad de palabras a incluir"),
):
    """
    Devuelve las frecuencias de las n palabras mas usadas del conjunto filtrado.
    Args:
        testament (str, optional): Testamento por el que filtrar.
        book (str, optional): Libro por el que filtrar.
        chapter (int, optional): Capitulo por el que filtrar.
        n (int): Cantidad de palabras a incluir. Minimo 1, maximo 500.
            Por defecto 100.
    Returns:
        dict[str, int]: Diccionario con formato {palabra: frecuencia}
            de las n palabras mas frecuentes del conjunto filtrado.
    """
    df = _filtrar_df(testament, book, chapter)
    conteo = Counter()
    for tokens in df["texto_procesado"]:
        conteo.update(tokens)
    return dict(conteo.most_common(n))