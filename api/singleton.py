"""
state.py
--------
Singleton: es la instancia que contiene el estado de la app y sobre la cual se
carga el corpus y entrenan los modelos. Al usarse un patrón singleton, todo se
realiza una única vez y se retorna siempre lo mismo.
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataloader import cargar_dataset
from src.models import Biblia
from src.preprocessing import TextPreprocessor
from src.tfidf import TFIDFVectorizer
from src.search_engine import SemanticSearchEngine
from src.ngram_model import NGramModel



class AppState:
    """
    Singleton de la app que contiene todos los objetos entrenados de la API.

    Atributos:
        df (pd.DataFrame): Corpus completo (una fila por versiculo).
        biblia (Biblia): Jerarquia OOP del corpus.
        preprocessor (TextPreprocessor): Pipeline con stopwords eliminadas
            (usado por TF-IDF y buscador semantico).
        preprocessor_ngram (TextPreprocessor): Pipeline sin eliminacion de
            stopwords (usado exclusivamente por los n-gramas para mantener
            la fluidez gramatical del texto generado).
        vectorizer (TFIDFVectorizer): Vectorizador TF-IDF ya ajustado.
        matriz_tfidf (np.ndarray): Matriz TF-IDF normalizada a norma L2=1.
        motor (SemanticSearchEngine): Motor de busqueda semantico entrenado.
        ngram_models (dict[int, NGramModel]): Modelos de n-gramas entrenados,
            indexados por n (1=unigram, 2=bigram, 3=trigram, 4=n-gram custom).
        word2vec: Modelo Word2Vec entrenado (None si gensim no esta instalado).
        pca_2d_tfidf (np.ndarray): Proyeccion 2D de TF-IDF via PCA.
        pca_3d_tfidf (np.ndarray): Proyeccion 3D de TF-IDF via PCA.
        pca_2d_w2v (np.ndarray): Proyeccion 2D de Word2Vec via PCA.
        pca_3d_w2v (np.ndarray): Proyeccion 3D de Word2Vec via PCA.
    """

    _instance = None
    def __new__(cls):
        """
        Función de control SINGLETON.
        Si instance ya existe, lo retorna. Si no existe, lo crea.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Inicializa los atributos solo la primera vez.
        """
        if hasattr(self, "_inicializado"):
            #Si ya fue inicializado, se retorna
            return
        
        self._inicializado = False
        self.df = None
        self.biblia = None
        self.preprocessor = None
        self.preprocessor_ngram = None
        self.vectorizer = None
        self.matriz_tfidf = None
        self.motor = None
        self.ngram_models = {}
        self.word2vec = None
        self.pca_2d_tfidf = None
        self.pca_3d_tfidf = None
        self.pca_2d_w2v = None
        self.pca_3d_w2v = None

    def esta_cargado(self):
        """
        Retorna un booleano que indica si el corpus y los modelos ya fueron cargados.
        Returns:
            bool: True si cargar() fue ejecutado.
        """
        return self._inicializado

    def cargar(self, path_bible, path_key, path_genre):
        """
        Carga el corpus y entrena los modelos.

        Args:
            path_bible (str | Path): Ruta a t_asv.csv.
            path_key (str | Path): Ruta a key_english.csv.
            path_genre (str | Path): Ruta a key_genre_english.csv.
        """
        if self._inicializado:
            return

        print("Cargando corpus...")
        
        df_raw = cargar_dataset(path_bible, path_key, path_genre)

        self.biblia = Biblia.from_dataframe(
            df_raw,
            col_libro="Book Name",
            col_testamento="Testament (OT or NT)",
            col_capitulo="Chapter",
            col_versiculo="Verse",
            col_texto="Text",
            col_genero="Genre name",
        )
        
        self.df = self.biblia.to_dataframe()
        print(f"Corpus cargado: {len(self.df)} versiculos, "
              f"{self.df['libro'].nunique()} libros")

        print("Preprocesando texto...")
        self.preprocessor = TextPreprocessor()
        self.df["texto_procesado"] = self.preprocessor.process_corpus(
            self.df["texto_original"].tolist()
        )
        self.df["texto_limpio"] = self.df["texto_procesado"].apply(" ".join)
    
        self.preprocessor_ngram = TextPreprocessor(remove_stopwords=False)
        tokens_ngram = self.preprocessor_ngram.process_corpus(
            self.df["texto_original"].tolist()
        )

        print("Calculando TF-IDF...")
        self.vectorizer = TFIDFVectorizer()
        self.matriz_tfidf = self.vectorizer.fit_transform(
            self.df["texto_procesado"].tolist()
        )

        print("Configurando motor de busqueda...")
        self.motor = SemanticSearchEngine(self.preprocessor, self.vectorizer)
        self.motor.df_corpus = self.df.reset_index(drop=True)
        self.motor.matriz_tfidf = self.matriz_tfidf

        print("Entrenando modelos de n-gramas...")
        N_GRAMAS = (1, 2, 3, 4)
        for n in N_GRAMAS:
            self.ngram_models[n] = NGramModel(n).fit(tokens_ngram)
        print(f"N-gramas entrenados: {list(self.ngram_models.keys())}")

        print("Entrenando Word2Vec...")
        self._entrenar_word2vec()

        print("Calculando proyecciones PCA...")
        self._calcular_pca()

        self._inicializado = True
        print("API lista")

    # ----------------------------------------------------------------
    # Metodos privados de apoyo
    # ----------------------------------------------------------------

    def _calcular_pca(self):
        """Calcula proyecciones PCA 2D y 3D sobre TF-IDF y Word2Vec."""
        self.pca_2d_tfidf = self._proyectar(self.matriz_tfidf, 2)
        self.pca_3d_tfidf = self._proyectar(self.matriz_tfidf, 3)

        if self.word2vec is not None:
            vecs_w2v = self._vectores_word2vec()
            self.pca_2d_w2v = self._proyectar(vecs_w2v, 2)
            self.pca_3d_w2v = self._proyectar(vecs_w2v, 3)

    def _proyectar(self, matriz, n_components):
        """
        Aplica PCA a una matriz y devuelve la proyeccion reducida.
        Args:
            matriz (np.ndarray): Matriz a reducir.
            n_components (int): Numero de componentes (2 o 3).
        Returns:
            np.ndarray: Coordenadas proyectadas.
        """
        pca = PCA(n_components=n_components)
        return pca.fit_transform(matriz)

    def _entrenar_word2vec(self):
        """
        Entrena un modelo Word2Vec sobre el corpus tokenizado.
        Si gensim no esta instalado, deja self.word2vec en None
        y muestra un aviso sin interrumpir el arranque de la API.
        """
        try:
            from gensim.models import Word2Vec
            self.word2vec = Word2Vec(
                sentences=self.df["texto_procesado"].tolist(),
                vector_size=100,
                window=5,
                min_count=1,
                workers=4,
                seed=42,
            )
            print("Word2Vec listo")
        except ImportError:
            print("gensim no instalado — Word2Vec no disponible")
            self.word2vec = None

    def _vectores_word2vec(self):
        """
        Representa cada versiculo como el promedio de los vectores Word2Vec
        de sus tokens. Si ningun token tiene vector, devuelve ceros.
        Returns:
            np.ndarray: Matriz (n_versiculos x vector_size).
        """
        size = self.word2vec.vector_size
        vectores = []
        for tokens in self.df["texto_procesado"]:
            vecs = [self.word2vec.wv[t] for t in tokens if t in self.word2vec.wv]
            vectores.append(np.mean(vecs, axis=0) if vecs else np.zeros(size))
        return np.array(vectores)