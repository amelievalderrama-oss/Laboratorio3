"""
ngram_model.py
--------------
Clase que permite crear modelos de lenguaje estadísticos basados en n-gramas
para generar textos de versiculos falsos.

Cada instancia de NGramModel implementa un n-grama de tamaño "n". De esta manera
el unigram, bigram, trigram y "n-gram" son solo instancias distintas
de NGramModel (n=1, 2, 3, n).
"""

import random
from collections import defaultdict, Counter

START = "<START>"
END = "<END>"

class NGramModel:
    """
    Clase que representa un modelo de lenguaje basado en n-gramas.
    Atributos:
        n (int): Tamaño del n-grama.
        contexto_size (int): Cantidad de palabras de contexto usadas para
            predecir la siguiente palabra (n - 1).
        conteos (dict[tuple, Counter]): Para cada contexto (tupla de
            palabras), un contador de cuantas veces aparecio cada palabra
            siguiente en el corpus de entrenamiento.
        vocabulario (set[str]): Conjunto de palabras vistas durante el
            entrenamiento del modelo.
    """

    def __init__(self, n):
        """
        Instancia la clase
        Args:
            n (int): Tamano del n-grama. Debe ser mayor o igual a 1.
        """
        if n < 1:
            raise ValueError("n debe ser >= 1")
        self.n = n
        self.contexto_size = n - 1
        self.conteos = defaultdict(Counter)
        self.vocabulario = set()

    def fit(self, oraciones_tokenizadas):
        """
        Entrena el modelo contando ocurrencias de cada n-grama en el corpus.
        Args:
            oraciones_tokenizadas (list[list[str]]): Lista de versiculos tokenizados.
        Returns:
            NGramModel: la misma instancia.
        """
        for tokens in oraciones_tokenizadas:
            secuencia = [START] * self.contexto_size + tokens + [END]
            self.vocabulario.update(tokens)
            for i in range(len(secuencia) - self.contexto_size):
                contexto = tuple(secuencia[i:i + self.contexto_size])
                siguiente = secuencia[i + self.contexto_size]
                self.conteos[contexto][siguiente] += 1
        return self

    def get_probabilidad(self, contexto, palabra):
        """
        Calcula la probabilidad empirica de que "palabra" siga a "contexto".
        Args:
            contexto (tuple[str]): Tupla de palabras de contexto (tamaño contexto_size).
            palabra (str): Palabra candidata a continuar la secuencia.
        Returns:
            float: probabilidad estimada (frecuencia relativa) entre 0 y 1.
        """
        contador = self.conteos.get(contexto)
        if not contador:
            return 0.0
        total = sum(contador.values())
        return contador[palabra] / total

    def get_siguiente_palabra(self, contexto):
        """
        Elige la siguiente palabra dado el contexto, muestreando segun
        las frecuencias observadas durante el entrenamiento.
        Args:
            contexto (tuple[str]): Tupla de palabras de contexto.
        Returns:
            str: Palabra elegida (puede ser <END>), o una palabra aleatoria
                del vocabulario si el contexto nunca fue visto (backoff simple).
        """
        contador = self.conteos.get(contexto)
        if not contador and self.contexto_size > 0:
            ultima = contexto[-1]
            agregado = Counter()
            for ctx, cnt in self.conteos.items():
                if ctx[-1] == ultima:
                    agregado.update(cnt)
            if agregado:
                contador = agregado
        if not contador:
            return random.choice(list(self.vocabulario)) if self.vocabulario else END
        palabras = list(contador.keys())
        pesos = list(contador.values())
        return random.choices(palabras, weights=pesos, k=1)[0]

    def generar(self, palabra_inicial=None, max_len=30):
        """
        Genera una secuencia de palabras a partir del modelo entrenado.
        Args:
            palabra_inicial (str, optional): Palabra desde la que comenzar
                la generación. Si es None, se comienza desde el contexto inicial <START>.
            max_len (int): Cantidad maxima de palabras a generar (la generación
                tambien puede parar antes si aparece el token <END>).
        Returns:
            str: Texto generado con las palabras separadas por espacios.
        """
        if palabra_inicial:
            if self.contexto_size > 0:
                contexto = tuple([START] * (self.contexto_size - 1) + [palabra_inicial])
            else:
                contexto = tuple()
            generadas = [palabra_inicial]
        else:
            contexto = tuple([START] * self.contexto_size)
            generadas = []

        while len(generadas) < max_len:
            siguiente = self.get_siguiente_palabra(contexto)
            if siguiente == END:
                break
            generadas.append(siguiente)
            if self.contexto_size > 0:
                contexto = tuple((list(contexto) + [siguiente])[-self.contexto_size:])

        return " ".join(generadas)

    def __str__(self):
        return f"NGramModel(n={self.n}, vocabulario={len(self.vocabulario)} palabras)"


def comparar_modelos(oraciones_tokenizadas, ns=(1, 2, 3, 4), palabra_inicial=None, max_len=20):
    """
    Instancia un NGramModel por cada valor de n y genera un texto con cada uno, 
    para la hacer una comparación.
    Args:
        oraciones_tokenizadas (list[list[str]]): versículos tokenizado.
        ns (tuple[int]): tupla con los valores n a comparar (ej. unigram, bigram, trigram, n-gram).
        palabra_inicial (str, optional): palabra inicial para todos los textos que se creen.
        max_len (int): largo maximo de cada texto generado.
    Returns:
        dict[int, str]: diccionario con el texto generado por cada NGramModel.
    """
    resultados = {}
    for n in ns:
        modelo = NGramModel(n).fit(oraciones_tokenizadas)
        resultados[n] = modelo.generar(palabra_inicial=palabra_inicial, max_len=max_len)
    return resultados