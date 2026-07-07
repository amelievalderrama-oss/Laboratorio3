"""
    API para obtener las coordenadas PCA de los textos.
    Se puede obtener PCA de TF-IDF o de Word2Vec.
    Se puede obtener PCA en 2D o 3D.
"""

from fastapi import APIRouter, Query, HTTPException
from singleton import AppState
router=APIRouter()

MODELOS = ("tfidf", "word2vec")

def ConstruirRespuesta(coordenadas,dims):
    """
    Construye la respuesta de la API con las coordenadas PCA y metadatos.
    Args:
        coordenadas (np.ndarray): Coordenadas PCA de los textos.
        dims (int): Número de dimensiones (2 o 3).
    Returns:    
        List[Dict]: Lista de diccionarios por versiculo con metadatos y coordenadas PCA.
    """
    df=AppState().df[["testamento","libro","genero","capitulo","versiculo"]].copy()
    df["x"] = coordenadas[:, 0]
    df["y"] = coordenadas[:, 1]
    if dims == 3:
        df["z"] = coordenadas[:, 2]
    return df.to_dict(orient="records")

@router.get("/pca")
def GetPca(dims: int=Query(2, ge=2, le=3)):
    """
    Obtiene las coordenadas PCA de los textos.
    Args:
        dims (int): Número de dimensiones (2 o 3).
    Returns:
        List[Dict]: Lista de diccionarios por versiculo con metadatos y coordenadas PCA.
    """
    coordenadas = AppState().pca_2d_tfidf if dims == 2 else AppState().pca_3d_tfidf
    if coordenadas is None:
        raise HTTPException(status_code=503, detail="PCA no calculado")
    return ConstruirRespuesta(coordenadas, dims)

@router.get("/word2vec")
def GetWord2VecPca(dims: int=Query(2, ge=2, le=3)):
    """
    Obtiene las coordenadas PCA de los textos usando Word2Vec.
    Args:
        dims (int): Número de dimensiones (2 o 3).
    Returns:
        List[Dict]: Lista de diccionarios por versiculo con metadatos y coordenadas PCA.
    """
    if AppState().word2vec is None:
        raise HTTPException(status_code=503, detail="Word2Vec no disponible (instalar gensim y reiniciar la API)")
    coordenadas = AppState().pca_2d_w2v if dims == 2 else AppState().pca_3d_w2v
    if coordenadas is None:
        raise HTTPException(status_code=503, detail="PCA Word2Vec no calculado")
    return ConstruirRespuesta(coordenadas, dims)