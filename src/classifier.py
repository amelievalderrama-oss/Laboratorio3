"""
classifier.py
--------------
Clase que implementa un clasificador que predice el libro 
al que pertenece un versiculo, usando elementos como la matriz TF-IDF (implementada
en tfidf.py) y modelos de clasificación estandares de sklearn.
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class VerseClassifier:
    """
    Clase que representa un clasificador de versiculos por libro.
    De momento, se considera el uso de 3 posibles modelos:
    Logistic Regression, Multinomial Naive Bayes y SVL.

    Atributos:
        modelo (object): instancia del modelo de sklearn (LogisticRegression
            o MultinomialNB) usado internamente.
        nombre_modelo (str): string para elegir modelo ("logistic" o "naive_bayes").
        clases (np.ndarray): lista de numpy de libros (etiquetas) vistos durante el entrenamiento.
        X_test (np.ndarray): atributos de prueba guardados tras el entrenamiento para poder evaluar.
        y_test (pd.Series): etiquetas reales de prueba guardadas tras entrenar().
    """

    def __init__(self, modelo="logistic", maximo_iteraciones=1000):
        """
        Crea el clasificador segun el tipo de modelo elegido.
        Args:
            modelo (str): "logistic" para regresion logistica, o "naive_bayes"
                para Naive Bayes Multinomial (baseline simple y rapido).
        """
        if modelo == "logistic":
            self.modelo = LogisticRegression(max_iter=maximo_iteraciones)
        elif modelo == "linear_svm":
            self.modelo = LinearSVC(max_iter=maximo_iteraciones)
        elif modelo == "naive_bayes":
            self.modelo = MultinomialNB()
        else:
            raise ValueError("modelo debe ser 'logistic' o 'naive_bayes'")

        self.nombre_modelo = modelo
        self.clases = None
        self.X_test = None
        self.y_test = None

    def entrenar(self, X, y, test_size=0.2, random_state=42):
        """
        Segmenta el dataset en train y test por medio de la función
        train_test_split() de scikit-learn.
        Luego entrena el modelo.
        Args:
            X (np.ndarray): matriz de atributos (el TF-IDF de los versiculos).
            y (pd.Series): etiquetas (libro al que pertenece cada versiculo, etiqueta objetivo).
            test_size (float): proporcion de datos de prueba (por default dejamos 20%).
            random_state (int): semilla para la separacion (aleatoria) de muestras de train y test .
        Returns:
            tuple: (X_train, X_test, y_train, y_test).
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        self.modelo.fit(X_train, y_train)
        self.clases = self.modelo.classes_
        self.X_test = X_test
        self.y_test = y_test
        return X_train, X_test, y_train, y_test

    def evaluar(self):
        """
        Evalua el modelo sobre el conjunto de prueba guardado en entrenar().
        Returns:
            dict: Contiene accuracy, matriz_confusion, clases, reporte,
                y_test e y_pred.
        """
        if self.X_test is None:
            raise RuntimeError("Llama a entrenar() antes de evaluar().")

        y_pred = self.modelo.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        matriz_confusion = confusion_matrix(self.y_test, y_pred, labels=self.clases)
        reporte = classification_report(self.y_test, y_pred, labels=self.clases, zero_division=0)
        return {
            "accuracy": accuracy,
            "matriz_confusion": matriz_confusion,
            "clases": self.clases,
            "reporte": reporte,
            "y_test": self.y_test,
            "y_pred": y_pred,
        }

    def predecir(self, X_nuevo):
        """
        Predice el libro para nuevas filas de atributos.
        Args:
            X_nuevo (np.ndarray): matriz de atributos de los versiculos a predecir.
        Returns:
            np.ndarray: libros predichos de cada fila de X_nuevo.
        """
        return self.modelo.predict(X_nuevo)

    def __str__(self):
        return f"VerseClassifier(modelo={self.nombre_modelo}, clases={self.clases})"