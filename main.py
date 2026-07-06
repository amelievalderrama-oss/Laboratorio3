"""
main.py
-------
Orquesta el pipeline completo. Pensado como punto de entrada y como
"guion" para el informe: cada bloque corresponde a una sección del
enunciado (3.1 a 3.7).

Ajusten la ruta y columnas de `cargar_dataset()` según la versión del
dataset de Kaggle que descarguen (las columnas varían: algunos vienen
como b/c/v/t, otros con nombres completos en inglés).
"""

import pandas as pd
from pathlib import Path
import json

from src.models import Biblia
from src.dataloader import cargar_dataset
from src.preprocessing import TextPreprocessor
from src.tfidf import TFIDFVectorizer, cosine_similarity_matrix
from src.search_engine import SemanticSearchEngine
from src.classifier import VerseClassifier
from src.ngram_model import comparar_modelos
from src.sentiment import TextBlobSentimentAnalyzer, calcular_sentimiento_corpus, agregar_por_libro
from src import visualization as viz


def cargar_dataset(path_bible: str,path_key: str,path_genre:str):
    """
    Carga CSV de descargado de Kaggle.
    """
    df = pd.read_csv(path_bible)
    """
    columnas: id,b,c,v,t
    ejemplo: 1001001,1,1,1,In the beginning God created the heavens and the earth.
    """
    df = df.rename(columns={
        "id":"Verse ID",
        "b": "Book",
        "c": "Chapter",
        "v": "Verse",
        "t": "Text"
    })

    df_key = pd.read_csv(path_key)
    """
    columnas: b,n,t,g
    ejemplo: 1,Genesis,OT,1
    """
    df_key = df_key.rename(columns={
        "b":"Book",
        "n": "Book Name",
        "t": "Testament (OT or NT)",
        "g": "Genre ID"
    })

    df_genre = pd.read_csv(path_genre)
    """
    columnas: g,n
    ejemplo: 1,Law
    """
    df_genre = df_genre.rename(columns={
        "g": "Genre ID",
        "n": "Genre name"
    })

    df = pd.merge(df, df_key, how="inner", on="Book")
    df = pd.merge(df, df_genre, how="inner", on="Genre ID")
    return df

def main():
    # Rutas a los archivos del taller basadas en la ubicación de este script
    script_root = Path(__file__).resolve().parent
    dir_dataset = script_root / "data"
    path_bible = dir_dataset / "t_asv.csv"
    path_key = dir_dataset / "key_english.csv"
    path_genre = dir_dataset / "key_genre_english.csv"
    path_stopwords = dir_dataset / "stopwords.json"

    df_raw = cargar_dataset(path_bible, path_key, path_genre)
    
    biblia = Biblia.from_dataframe(
        df_raw,
        col_libro="Book Name",
        col_testamento="Testament (OT or NT)",
        col_capitulo="Chapter",
        col_versiculo="Verse",
        col_texto="Text",
        col_genero="Genre name",
    )
 
    df = biblia.to_dataframe()

    # 2. Preprocesamiento (3.1) 

    with open(path_stopwords) as stopwords_json:
        stopwords = set(json.load(stopwords_json))

    preprocessor = TextPreprocessor(stopwords=stopwords)

    df["texto_procesado"] = preprocessor.process_corpus(df["texto_original"].tolist())

    print("Top 20 palabras más frecuentes:", preprocessor.palabras_mas_frecuentes(20))

    # 3. TF-IDF a nivel de versículo ---------------------------------------------
    vectorizer = TFIDFVectorizer()
    matriz_tfidf_versiculos = vectorizer.fit_transform(df["texto_procesado"].tolist())

    # 4. Visualizaciones (3.2) ----------------------------------------------------
    #viz.plot_longitud_versiculos(df)
    #viz.plot_versiculos_por_libro(df)

    # heatmap de similitud ENTRE LIBROS (obligatorio) -> agregamos texto por libro
    textos_por_libro = df.groupby("libro")["texto_procesado"].sum()  # concatena listas de tokens
    vectorizer_libros = TFIDFVectorizer()
    matriz_tfidf_libros = vectorizer_libros.fit_transform(textos_por_libro.tolist())
    # matriz_similitud_libros = cosine_similarity_matrix(matriz_tfidf_libros)
    # viz.plot_heatmap_similitud_libros(matriz_similitud_libros, textos_por_libro.index.tolist())

    # 5. PCA de versículos (3.3) ---------------------------------------------------
    viz.plot_pca_versiculos(matriz_tfidf_versiculos, df["testamento"], titulo="Versículos por testamento")
    viz.plot_pca_versiculos(matriz_tfidf_versiculos, df["libro"], titulo="Versículos por libro")

    # 6. Motor de búsqueda semántico (3.4) -----------------------------------------
    # motor = SemanticSearchEngine(preprocessor, TFIDFVectorizer())
    # motor.fit(df)
    # print(motor.buscar("amor y fe", k=5))

    # 7. Clasificador de versículos (3.5) -------------------------------------------
    clasificador = VerseClassifier(modelo="logistic")
    clasificador.entrenar(matriz_tfidf_versiculos, df["libro"])
    resultados_clf = clasificador.evaluar()
    print("Accuracy:", resultados_clf["accuracy"])
    viz.plot_matriz_confusion(resultados_clf["matriz_confusion"], resultados_clf["clases"])

    # 8. Generador de texto con n-gramas (3.6) --------------------------------------
    generados = comparar_modelos(df["texto_procesado"].tolist(), ns=(1, 2, 3, 4))
    for n, texto in generados.items():
        print(f"n={n}: {texto}")

    # 9. Análisis de sentimiento (3.7) ------------------------------------------------
    analizador = TextBlobSentimentAnalyzer()
    df = calcular_sentimiento_corpus(df, analizador)
    sentimiento_por_libro = agregar_por_libro(df)
    viz.plot_sentimiento_por_libro(sentimiento_por_libro)

    print("Pipeline completo ejecutado correctamente.")
    return df


if __name__ == "__main__":
    main()