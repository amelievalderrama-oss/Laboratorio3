import os
import request
import streamlit as st
from dotenv import load_dotenv

st.set_page_config(
    page_title = "Taller Tres PC-2026",
    layout = "wide"
)

load_dotenv()

DEFAULT_API_URL = os.getenv("API_URL","http//127.0.0.1:8000")
st.title("Taller 3 de Programación Científica")


from api_client import cargar_metadata
from views import reporte1_view,report2_view


with st.slidebar:
    vista = st.selectbox(
        "Selector de reporte",
        (
            "Reporte 1",
            "Reporte 2",
            "Reporte 3"
        )
    )

if vista == "Reporte 1":
    reporte1_view.render()
    pass
elif vista == "Reporte 2":
    report2_view.render()
    pass
elif vista == "Reporte 3": 
    pass
else:
    pass

try:
    metadata = cargar_metadata(DEFAULT_API_URL)
except request.RequestException as exc:
    st.error("ERROR!")
    st.code(str(exc))