from fastapi import FastAPI, Query, HTTPException
from pathlib import Path


from api import dashboard,generator,pca,search,singleton

# Rutas del dataset
DIR_DATA = Path(__file__).resolve().parent.parent / "data"
PATH_BIBLE = DIR_DATA / "t_asv.csv"
PATH_KEY = DIR_DATA / "key_english.csv"
PATH_GENRE = DIR_DATA / "key_genre_english.csv"
PATH_STOPWORDS = DIR_DATA / "stopwords.json"


app = FastAPI(
    title="Biblical Text Mining API",
    description="API para el análisis computacional de un corpus bíblico en inglés de la American Standard Version."
)

app.include_router(dashboard.router, prefix="/dashboard", tags=["Dashboard"])
app.include_router(search.router,    prefix="/search",    tags=["Buscador"])
app.include_router(pca.router,prefix="/visualizer",tags=["PCA y Word2Vec"])
app.include_router(generator.router, prefix="/generator", tags=["Generador"])


@app.on_event("startup")
def startup():
    """
    Función que se ejecuta una sola vez cuando uvicorn levanta la API.
    Instancia el Singleton AppState y llama a cargar().
    Deja todo preparado antes del primer request
    """
    estado = singleton.AppState()
    estado.cargar(PATH_BIBLE, PATH_KEY, PATH_GENRE, PATH_STOPWORDS)

@app.get("/")
def estado_app():
    return {"estado": "API funcionando!"}
