from fastAPI import FastAPI, Query, HTTPException




from api import dashboard,generator,pca,search


app = FastAPI(
    title = "API MAIN"
)

@app.on_event("startup")
def startup():
    """
    Función que se ejecuta una sola vez cuando uvicorn levanta la API.
    Instancia el Singleton AppState y llama a cargar().
    Deja todo preparado antes del primer request
    """
    estado = AppState()
    estado.cargar(PATH_BIBLE, PATH_KEY, PATH_GENRE)

@app.get("/")
def estado_app():
    return {
        "estado": "API funcionando!"
    }