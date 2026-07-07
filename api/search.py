"""
API para buscar textos en el corpus.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.singleton import AppState

router = APIRouter()

class SearchRequest(BaseModel):
    """
    Modelo para la solicitud de busqueda.
    Attributes:
        query (str): texto de la consulta del usuario.
        limit (int): cantidad maxima de resultados a devolver (entre 1 y 50).
    """
    query: str = Field(..., example="God is love", description="Consulta de búsqueda")
    limit: int = Field(5, ge=1, le=50, description="Número máximo de resultados a devolver (entre 1 y 50)")

@router.post("")
def buscar(req: SearchRequest):
    """
    Busca textos en el corpus según la consulta proporcionada.
    Args:
        req(SearchRequest): La solicitud de búsqueda y limite de resultados a devolver.
    Returns:
        List[Dict]: Lista de diccionarios con los versiculos más similares a la consulta, incluyendo metadatos y texto.
    """
    df_result = AppState().motor.buscar(req.query, req.limit)
    return df_result.to_dict(orient="records")