
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from singleton import AppState
from src.ngram_model import NGramModel

router = APIRouter(prefix="/generator", tags=["generator"])

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
    5: "5-gram",
}

@router.get("/models")
def listar_modelos():
    estado = _estado_listo()
    return {
        "modelos": [
            {
                "n": n,
                "nombre": NOMBRES_MODELO.get(n, f"{n}-gram"),
                "vocabulario_size": len(modelo.vocabulario),
            }
            for n, modelo in sorted(estado.ngram_models.items())
        ]
    }

@router.get("/generate")
def generar(
    n: int,
    palabra_inicial: str,
    max_len: int = Query(...),
):
    estado = _estado_listo()

    if (palabra_inicial.strip() == ""):
        raise HTTPException(400, "No hay palabra ingresada")
    
    if n < 1:
        raise HTTPException(400, "Ingresa un número mayor o igual a 1")
    
    if n > 10:
        raise HTTPException(400, "Ingresa un número menor o igual a 10")
    
    if max_len < 1:
        raise HTTPException(400, "Ingresa un largo mayor o igual a 1")
    
    if max_len > 100:
        raise HTTPException(400, "Ingresa un largo menor o igual a 100")
    
    modelo = _obtener_modelo(estado, n)

    palabra = palabra_inicial.strip().lower()

    if (palabra not in modelo.vocabulario):
        raise HTTPException(400, "La palabra no está en el vocabulario")
    
    texto = modelo.generar(palabra_inicial=palabra, max_len=max_len)
    return {
        "n": n,
        "nombre": NOMBRES_MODELO.get(n, f"{n}-gram"),
        "palabra_inicial": palabra,
        "max_len": max_len,
        "texto_generado": texto,
    }
