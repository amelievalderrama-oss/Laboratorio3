"""
models.py

Jerarquia del corpus biblico usando programacion orientada a objetos.

La estructura se organiza de la siguiente manera:
    Biblia -> Testamento -> Libro -> Capitulo -> Versiculo

Cada libro guarda sus capitulos, versiculos, testamento y genero literario.
El objetivo de este modulo es representar el dataset de manera ordenada para
poder recorrerlo, resumirlo y transformarlo nuevamente a DataFrame.
"""

import pandas as pd


class Versiculo:
    """
    Clase que representa la unidad minima de analisis del corpus biblico.

    Atributos:
        libro (str): nombre del libro al que pertenece el versiculo.
        capitulo (int): numero del capitulo.
        numero (int): numero del versiculo dentro del capitulo.
        texto_original (str): texto original del versiculo.
        texto_procesado (list[str]): lista de tokens obtenidos luego del preprocesamiento.
    """

    def __init__(self, libro, capitulo, numero, texto_original):
        """
        Crea un objeto Versiculo con sus datos principales.

        Args:
            libro (str): nombre del libro.
            capitulo (int): numero del capitulo.
            numero (int): numero del versiculo.
            texto_original (str): texto original del versiculo.
        """
        self.libro = libro
        self.capitulo = capitulo
        self.numero = numero
        self.texto_original = texto_original
        self.texto_procesado = []

    def set_texto_procesado(self, tokens):
        """
        Guarda los tokens procesados del versiculo.

        Args:
            tokens (list[str]): lista de palabras procesadas.
        """
        self.texto_procesado = tokens

    def get_texto_limpio(self):
        """
        Une los tokens procesados en un solo string.

        Returns:
            str: texto procesado unido por espacios.
        """
        return " ".join(self.texto_procesado)

    def __str__(self):
        return f"{self.libro} {self.capitulo}:{self.numero} - {self.texto_original[:40]}..."


class Capitulo:
    """
    Clase que representa un capitulo de un libro biblico.

    Atributos:
        numero (int): numero del capitulo.
        versiculos (list[Versiculo]): lista de versiculos pertenecientes al capitulo.
    """

    def __init__(self, numero):
        """
        Crea un capitulo vacio.

        Args:
            numero (int): numero del capitulo.
        """
        self.numero = numero
        self.versiculos = []

    def agregar_versiculo(self, versiculo):
        """
        Agrega un versiculo al capitulo.

        Args:
            versiculo (Versiculo): versiculo que se desea agregar.
        """
        self.versiculos.append(versiculo)

    def get_cantidad_versiculos(self):
        """
        Cuenta la cantidad de versiculos del capitulo.

        Returns:
            int: numero de versiculos almacenados.
        """
        return len(self.versiculos)

    def get_texto_completo(self):
        """
        Junta el texto original de todos los versiculos del capitulo.

        Returns:
            str: texto completo del capitulo.
        """
        return " ".join(v.texto_original for v in self.versiculos)

    def __str__(self):
        return f"Capitulo {self.numero} ({self.get_cantidad_versiculos()} versiculos)"


class Libro:
    """
    Clase que representa un libro de la Biblia.

    Atributos:
        nombre (str): nombre del libro.
        testamento (str): testamento al que pertenece el libro.
        genero (str): genero literario del libro.
        capitulos (dict[int, Capitulo]): diccionario con los capitulos del libro.
    """

    def __init__(self, nombre, testamento, genero=None):
        """
        Crea un libro con su nombre, testamento y genero.

        Args:
            nombre (str): nombre del libro.
            testamento (str): testamento al que pertenece.
            genero (str, optional): genero literario del libro.
        """
        self.nombre = nombre
        self.testamento = testamento
        self.genero = genero
        self.capitulos = {}

    def set_genero(self, genero):
        """
        Asigna el genero literario del libro.

        Args:
            genero (str): genero literario.
        """
        self.genero = genero

    def get_genero(self):
        """
        Devuelve el genero literario del libro.

        Returns:
            str: genero del libro.
        """
        return self.genero

    def agregar_versiculo(self, versiculo):
        """
        Agrega un versiculo al libro. Si el capitulo no existe, primero lo crea.

        Args:
            versiculo (Versiculo): versiculo que se desea agregar.
        """
        if versiculo.capitulo not in self.capitulos:
            self.capitulos[versiculo.capitulo] = Capitulo(versiculo.capitulo)
        self.capitulos[versiculo.capitulo].agregar_versiculo(versiculo)

    def get_versiculos(self):
        """
        Devuelve todos los versiculos del libro en una sola lista.

        Returns:
            list[Versiculo]: versiculos de todos los capitulos del libro.
        """
        out = []
        for cap in self.capitulos.values():
            out.extend(cap.versiculos)
        return out

    def get_cantidad_versiculos(self):
        """
        Cuenta la cantidad total de versiculos del libro.

        Returns:
            int: numero total de versiculos.
        """
        return len(self.get_versiculos())

    def get_cantidad_capitulos(self):
        """
        Cuenta la cantidad de capitulos del libro.

        Returns:
            int: numero de capitulos.
        """
        return len(self.capitulos)

    def get_texto_completo(self):
        """
        Junta el texto original de todos los versiculos del libro.

        Returns:
            str: texto completo del libro.
        """
        return " ".join(v.texto_original for v in self.get_versiculos())

    def __str__(self):
        return (f"{self.nombre} (Testamento: {self.testamento}, Genero: {self.genero}, "
                f"{self.get_cantidad_versiculos()} versiculos)")


class Testamento:
    """
    Clase que representa un testamento de la Biblia.

    Atributos:
        nombre (str): nombre o codigo del testamento.
        libros (dict[str, Libro]): diccionario con los libros del testamento.
    """

    def __init__(self, nombre):
        """
        Crea un testamento vacio.

        Args:
            nombre (str): nombre o codigo del testamento.
        """
        self.nombre = nombre
        self.libros = {}

    def agregar_libro(self, libro):
        """
        Agrega un libro al testamento.

        Args:
            libro (Libro): libro que se desea agregar.
        """
        self.libros[libro.nombre] = libro

    def get_versiculos(self):
        """
        Devuelve todos los versiculos de todos los libros del testamento.

        Returns:
            list[Versiculo]: lista de versiculos del testamento.
        """
        out = []
        for libro in self.libros.values():
            out.extend(libro.get_versiculos())
        return out

    def get_cantidad_versiculos(self):
        """
        Cuenta la cantidad total de versiculos del testamento.

        Returns:
            int: numero total de versiculos.
        """
        return len(self.get_versiculos())

    def __str__(self):
        return f"Testamento {self.nombre} ({len(self.libros)} libros)"


class Biblia:
    """
    Clase raiz que representa el corpus biblico completo.

    Atributos:
        testamentos (dict[str, Testamento]): diccionario con los testamentos disponibles.
    """

    def __init__(self):
        """
        Crea una Biblia vacia.
        """
        self.testamentos = {}

    def agregar_versiculo(self, versiculo, testamento, nombre_libro, genero=None):
        """
        Agrega un versiculo a la estructura de Biblia.

        Si el testamento o el libro no existen, se crean automaticamente.

        Args:
            versiculo (Versiculo): versiculo que se desea agregar.
            testamento (str): testamento al que pertenece el versiculo.
            nombre_libro (str): nombre del libro.
            genero (str, optional): genero literario del libro.
        """
        if testamento not in self.testamentos:
            self.testamentos[testamento] = Testamento(testamento)
        test_obj = self.testamentos[testamento]

        if nombre_libro not in test_obj.libros:
            test_obj.agregar_libro(Libro(nombre_libro, testamento, genero))
        elif genero is not None and test_obj.libros[nombre_libro].get_genero() is None:
            test_obj.libros[nombre_libro].set_genero(genero)

        test_obj.libros[nombre_libro].agregar_versiculo(versiculo)

    @staticmethod
    def from_dataframe(df, col_libro="libro", col_testamento="testamento",
                       col_capitulo="capitulo", col_versiculo="versiculo",
                       col_texto="texto", col_genero="genero"):
        """
        Construye una Biblia completa a partir de un DataFrame.

        Args:
            df (pd.DataFrame): DataFrame con la informacion de los versiculos.
            col_libro (str): nombre de la columna del libro.
            col_testamento (str): nombre de la columna del testamento.
            col_capitulo (str): nombre de la columna del capitulo.
            col_versiculo (str): nombre de la columna del numero de versiculo.
            col_texto (str): nombre de la columna del texto original.
            col_genero (str): nombre de la columna del genero literario.

        Returns:
            Biblia: objeto Biblia construido desde el DataFrame.
        """
        biblia = Biblia()
        tiene_genero = col_genero in df.columns

        for _, row in df.iterrows():
            versiculo = Versiculo(
                libro=row[col_libro],
                capitulo=int(row[col_capitulo]),
                numero=int(row[col_versiculo]),
                texto_original=str(row[col_texto]),
            )
            genero = row[col_genero] if tiene_genero else None
            biblia.agregar_versiculo(
                versiculo,
                testamento=row[col_testamento],
                nombre_libro=row[col_libro],
                genero=genero,
            )
        return biblia

    def get_libros(self):
        """
        Devuelve todos los libros de la Biblia.

        Returns:
            list[Libro]: lista con todos los libros de todos los testamentos.
        """
        out = []
        for testamento in self.testamentos.values():
            out.extend(testamento.libros.values())
        return out

    def get_versiculos(self):
        """
        Devuelve todos los versiculos de la Biblia.

        Returns:
            list[Versiculo]: lista con todos los versiculos del corpus.
        """
        out = []
        for libro in self.get_libros():
            out.extend(libro.get_versiculos())
        return out

    def get_generos(self):
        """
        Obtiene los generos literarios presentes en la Biblia.

        Returns:
            list[str]: lista de generos sin repetir y ordenados alfabeticamente.
        """
        generos = []
        for libro in self.get_libros():
            g = libro.get_genero()
            if g is not None and g not in generos:
                generos.append(g)
        generos.sort()
        return generos

    def to_dataframe(self):
        """
        Convierte la estructura completa de Biblia en un DataFrame.

        Returns:
            pd.DataFrame: DataFrame con una fila por versiculo y sus metadatos.
        """
        rows = []
        for libro in self.get_libros():
            for versiculo in libro.get_versiculos():
                rows.append({
                    "testamento": libro.testamento,
                    "libro": libro.nombre,
                    "genero": libro.get_genero(),
                    "capitulo": versiculo.capitulo,
                    "versiculo": versiculo.numero,
                    "texto_original": versiculo.texto_original,
                    "texto_procesado": versiculo.texto_procesado,
                    "texto_limpio": versiculo.get_texto_limpio(),
                })
        return pd.DataFrame(rows)

    def get_resumen(self):
        """
        Genera un resumen de cantidad de libros y versiculos por testamento.

        Returns:
            pd.DataFrame: tabla con testamento, numero de libros y numero de versiculos.
        """
        rows = []
        for nombre, testamento in self.testamentos.items():
            rows.append({
                "testamento": nombre,
                "n_libros": len(testamento.libros),
                "n_versiculos": testamento.get_cantidad_versiculos(),
            })
        return pd.DataFrame(rows)

    def get_resumen_generos(self):
        """
        Genera un resumen de cantidad de libros y versiculos por genero literario.

        Returns:
            pd.DataFrame: tabla con genero, numero de libros y numero de versiculos.
        """
        rows = []
        for genero in self.get_generos():
            libros_genero = [l for l in self.get_libros() if l.get_genero() == genero]
            n_versiculos = sum(l.get_cantidad_versiculos() for l in libros_genero)
            rows.append({
                "genero": genero,
                "n_libros": len(libros_genero),
                "n_versiculos": n_versiculos,
            })
        return pd.DataFrame(rows)

    def __str__(self):
        return f"Biblia ({len(self.get_libros())} libros, {len(self.get_versiculos())} versiculos)"
