
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .singleton import AppState
from src.ngram_model import NGramModel

router = APIRouter()


class GenerateRequest(BaseModel):
    modelo: str = Field(..., description="Modelo a usar: unigram, bigram, trigram o ngram4")
    palabra_inicial: Optional[str] = Field(None, description="Palabra inicial opcional")
    max_len: int = Field(..., ge=1, le=100, description="Largo máximo del texto generado")


def _estado_listo():
    estado = AppState()
    if not estado.esta_cargado():
        raise HTTPException(503, "El corpus aún se está cargando")
    return estado

def _obtener_modelo(estado, n: int) -> NGramModel:
    if n in estado.ngram_models:
        return estado.ngram_models[n]

    tokens = estado.preprocessor_ngram.process_corpus(
        estado.df["texto_original"].tolist()
    ) 

    modelo = NGramModel(n).fit(tokens)
    estado.ngram_models[n] = modelo
    return modelo


NOMBRES_MODELO = {
    1: "unigram",
    2: "bigram",
    3: "trigram",
    4: "4-gram",
}

@router.get("/models")
def listar_modelos():
    estado = _estado_listo()
    return [
        {"nombre": nombre, "n": n, "clave": clave}
        for clave, n in [("unigram",1),("bigram",2),("trigram",3),("ngram4",4)]
        for nombre in [NOMBRES_MODELO[n]]
    ]

def _mapear_modelo(modelo: str) -> int:
    mapa = {
        "unigram": 1,
        "bigram": 2,
        "trigram": 3,
        "ngram4": 4,
    }
    if modelo not in mapa:
        raise HTTPException(400, "Modelo no válido")
    return mapa[modelo]


def _generar_texto(n: int, palabra_inicial: Optional[str], max_len: int):
    estado = _estado_listo()

    if n < 1:
        raise HTTPException(400, "Ingresa un número mayor o igual a 1")

    if n > 4:
        raise HTTPException(400, "Ingresa un número menor o igual a 4")

    if max_len < 1:
        raise HTTPException(400, "Ingresa un largo mayor o igual a 1")

    if max_len > 60:
        raise HTTPException(400, "Ingresa un largo menor o igual a 60")

    modelo = _obtener_modelo(estado, n)

    if palabra_inicial is not None:
        palabra = palabra_inicial.strip().lower()
        if palabra == "":
            raise HTTPException(400, "No hay palabra ingresada")
        if palabra not in modelo.vocabulario:
            raise HTTPException(400, "La palabra no está en el vocabulario")
        texto = modelo.generar(palabra_inicial=palabra, max_len=max_len)
    else:
        texto = modelo.generar(palabra_inicial=None, max_len=max_len)

    return {
        "texto": texto,
        "texto_generado": texto,
        "modelo": NOMBRES_MODELO.get(n, f"{n}-gram"),
        "n": n,
        "palabra_inicial": palabra.lower() if palabra_inicial is not None and palabra_inicial.strip() != "" else None,
        "max_len": max_len,
    }


@router.get("/generate")
def generar(
    n: int,
    palabra_inicial: str = Query("", description="Palabra inicial opcional"),
    max_len: int = Query(...),
):
    return _generar_texto(n, palabra_inicial or None, max_len)


@router.post("")
@router.post("/generate")
def generar_desde_body(req: GenerateRequest):
    n = _mapear_modelo(req.modelo)
    return _generar_texto(n, req.palabra_inicial, req.max_len)
