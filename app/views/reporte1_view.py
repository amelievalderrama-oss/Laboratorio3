import streamlit as st
import requests
import numpy as np

from api_client import cargar_metadata


def render(base_url: str):
    print(".....")
    print("Hacer menú-peticiones-etc")

    try:
        cargar_metadata(base_url=base_url)
    except:
        st.error("algo pasó!!")