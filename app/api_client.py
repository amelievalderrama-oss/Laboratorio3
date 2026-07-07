import requests

def api_get(base_url: str,path: str,params: dict | None):
    response = requests.get(
        f"{base_url.rstrip("/")}{path}"
        params = params,
        timeout = REQUEST_TIMEOUT
    )
    response.raise_for_status()
    return response.json()

def cargar_metadata(base_url: str):
    return api_get(base_url=base_url,path="/metadata")
