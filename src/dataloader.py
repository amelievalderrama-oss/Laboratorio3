"""
data_loader.py
---------------
Carga y combinacion de los 3 CSV del dataset de Kaggle ('oswinrh/bible').

Se separa de main.py a proposito: asi cada persona puede cargar el
dataset real desde su propio notebook (Colab, Jupyter, etc.) sin tener
que importar ni ejecutar main.py.
"""

import pandas as pd


def cargar_dataset(path_bible, path_key, path_genre):
    """
    Carga los 3 CSV del dataset de Kaggle ('oswinrh/bible') y los combina
    en un unico DataFrame con texto, testamento y genero literario.

    Args:
        path_bible (str): CSV con columnas id,b,c,v,t (ej. t_asv.csv).
        path_key (str): CSV con columnas b,n,t,g (ej. key_english.csv).
        path_genre (str): CSV con columnas g,n (ej. key_genre_english.csv).
    Returns:
        pd.DataFrame: una fila por versiculo, con columnas Book Name,
            Testament (OT or NT), Genre name, Chapter, Verse, Text.
    """
    df = pd.read_csv(path_bible)
    # id,b,c,v,t -> Verse ID, Book, Chapter, Verse, Text
    df = df.rename(columns={
        "id": "Verse ID",
        "b": "Book",
        "c": "Chapter",
        "v": "Verse",
        "t": "Text",
    })

    df_key = pd.read_csv(path_key)
    # b,n,t,g -> Book, Book Name, Testament (OT or NT), Genre ID
    df_key = df_key.rename(columns={
        "b": "Book",
        "n": "Book Name",
        "t": "Testament (OT or NT)",
        "g": "Genre ID",
    })

    df_genre = pd.read_csv(path_genre)
    # g,n -> Genre ID, Genre name
    df_genre = df_genre.rename(columns={
        "g": "Genre ID",
        "n": "Genre name",
    })

    df = pd.merge(df, df_key, how="inner", on="Book")
    df = pd.merge(df, df_genre, how="inner", on="Genre ID")
    return df