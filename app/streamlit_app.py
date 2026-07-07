"""
streamlit_app.py
-----------------
Interfaz de usuario del sistema de análisis bíblico.

PARA HACERLO CORRER:
    (la API DEBE CORRER en http://localhost:8000)
    $ streamlit run app/streamlit_app.py
"""

import requests
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

DEFAULT_API_URL = os.getenv("API_URL","http//127.0.0.1:8000")

def get(endpoint, params=None):
    """GET a la API. Devuelve el JSON o muestra un error."""
    try:
        r = requests.get(f"{DEFAULT_API_URL}{endpoint}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Error al conectar con la API: {e}")
        return None


def post(endpoint, body):
    """POST a la API. Devuelve el JSON o muestra un error."""
    try:
        r = requests.post(f"{DEFAULT_API_URL}{endpoint}", json=body, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Error al conectar con la API: {e}")
        return None


#----------- estructura de la página --------
st.set_page_config(
    page_title="Biblical Text Mining",
    layout="wide",
)

st.title("Biblical Text Mining — ASV Corpus")
st.caption("Sistema de análisis computacional del corpus bíblico")

tab_dashboard, tab_search, tab_viz, tab_gen = st.tabs([
    "Dashboard",
    "Buscador semántico",
    "PCA y Word2Vec",
    "Generador de versículos",
])


# -------- dashboard ------------
with tab_dashboard:
    st.header("Dashboard del corpus")

    # Filtros — se obtienen de la API para no hardcodear nada
    filters = get("/dashboard/filters")
    if filters is None:
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        testament = st.selectbox(
            "Testamento",
            ["(todos)"] + filters["testamentos"],
            key="dash_testament",
        )
    with col2:
        book = st.selectbox(
            "Libro",
            ["(todos)"] + filters["libros"],
            key="dash_book",
        )
    with col3:
        chapter = None
        if book != "(todos)":
            caps = get("/dashboard/chapters", params={"book": book})
            if caps:
                cap_opts = ["(todos)"] + [str(c) for c in caps["capitulos"]]
                cap_sel = st.selectbox("Capítulo", cap_opts, key="dash_chapter")
                chapter = int(cap_sel) if cap_sel != "(todos)" else None
        else:
            st.selectbox("Capítulo", ["(selecciona un libro primero)"], disabled=True)

    params = {
        "testament": testament if testament != "(todos)" else None,
        "book": book if book != "(todos)" else None,
        "chapter": chapter,
    }
    params = {k: v for k, v in params.items() if v is not None}

    # Estadísticas
    st.subheader("Versículos y longitud promedio por libro")
    stats = get("/dashboard/stats", params=params)
    if stats:
        df_stats = pd.DataFrame(stats)
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.bar(
                df_stats,
                x="libro", y="n_versiculos",
                labels={"libro": "Libro", "n_versiculos": "N° versículos"},
                title="Versículos por libro",
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig2 = px.bar(
                df_stats,
                x="libro", y="longitud_promedio",
                labels={"libro": "Libro", "longitud_promedio": "Palabras promedio"},
                title="Longitud promedio de versículos",
                color="longitud_promedio",
                color_continuous_scale="Viridis",
            )
            fig2.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)

    # Top palabras
    st.subheader("Palabras más frecuentes")
    n_words = st.slider("Cantidad de palabras", 5, 50, 20, key="dash_nwords")
    top = get("/dashboard/top-words", params={**params, "n": n_words})
    if top:
        df_top = pd.DataFrame(top)
        fig3 = px.bar(
            df_top,
            x="frecuencia", y="palabra",
            orientation="h",
            labels={"frecuencia": "Frecuencia", "palabra": "Palabra"},
            title=f"Top {n_words} palabras más frecuentes",
        )
        fig3.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig3, use_container_width=True)

    # Nube de palabras
    st.subheader("Nube de palabras")
    wc_data = get("/dashboard/wordcloud", params={**params, "n": 150})
    if wc_data:
        wc = WordCloud(
            width=900, height=400, background_color="white", colormap="viridis"
        ).generate_from_frequencies(wc_data)
        fig_wc, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_wc)


# -------- buscador semántico------------
with tab_search:
    st.header("Buscador semántico")
    st.caption(
        "Ingresa una frase y el sistema encuentra los versículos más similares "
        "según similitud de coseno sobre vectores TF-IDF."
    )

    query = st.text_input("Frase de búsqueda", placeholder="e.g. love and faith in the lord")
    k = st.slider("Cantidad de resultados", 1, 20, 5, key="search_k")

    if st.button("Buscar", type="primary") and query:
        resultados = post("/search", {"query": query, "k": k})
        if resultados:
            df_res = pd.DataFrame(resultados)
            df_res["similitud"] = df_res["similitud"].round(4)
            st.dataframe(
                df_res,
                use_container_width=True,
                column_config={
                    "similitud": st.column_config.ProgressColumn(
                        "Similitud", min_value=0, max_value=1, format="%.4f"
                    )
                },
            )


# -------- visualizador PCA y word2vec------------
with tab_viz:
    st.header("Visualizador PCA y Word2Vec")

    col_v1, col_v2, col_v3 = st.columns(3)
    with col_v1:
        representacion = st.radio(
            "Representación vectorial",
            ["TF-IDF + PCA", "Word2Vec + PCA"],
            key="viz_rep",
        )
    with col_v2:
        dims = st.radio("Dimensiones", [2, 3], key="viz_dims")
    with col_v3:
        color_por = st.radio(
            "Colorear por",
            ["testamento", "genero", "libro"],
            key="viz_color",
        )

    endpoint = "/visualizer/pca" if representacion == "TF-IDF + PCA" else "/visualizer/word2vec"

    with st.spinner("Cargando datos de la API..."):
        datos = get(endpoint, params={"dims": dims})

    if datos:
        df_viz = pd.DataFrame(datos)

        if dims == 2:
            fig = px.scatter(
                df_viz,
                x="x", y="y",
                color=color_por,
                hover_data=["libro", "capitulo", "versiculo"],
                title=f"{representacion} — {dims}D (coloreado por {color_por})",
                labels={"x": "PC1", "y": "PC2"},
                opacity=0.6,
            )
        else:
            fig = px.scatter_3d(
                df_viz,
                x="x", y="y", z="z",
                color=color_por,
                hover_data=["libro", "capitulo", "versiculo"],
                title=f"{representacion} — {dims}D (coloreado por {color_por})",
                labels={"x": "PC1", "y": "PC2", "z": "PC3"},
                opacity=0.6,
            )
            fig.update_traces(marker_size=2)

        st.plotly_chart(fig, use_container_width=True)


# -------- generador de versículos ------------
with tab_gen:
    st.header("Generador de versículos falsos")
    st.caption(
        "Genera texto usando modelos estadísticos de lenguaje basados en n-gramas. "
        "Cada modelo usa las n-1 palabras anteriores como contexto."
    )

    modelos_info = get("/generator/modelos")
    if modelos_info:
        opciones = {m["nombre"]: m["clave"] for m in modelos_info}

        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            modelo_nombre = st.selectbox(
                "Modelo",
                list(opciones.keys()),
                key="gen_model",
            )
        with col_g2:
            palabra_inicial = st.text_input(
                "Palabra inicial (opcional)",
                placeholder="e.g. god",
                key="gen_word",
            )
        with col_g3:
            max_len = st.slider("Largo máximo", 10, 60, 20, key="gen_len")

        if st.button("Generar", type="primary"):
            resultado = post("/generator", {
                "modelo": opciones[modelo_nombre],
                "palabra_inicial": palabra_inicial if palabra_inicial else None,
                "max_len": max_len,
            })
            if resultado:
                st.markdown("### Versículo generado")
                st.info(f"*{resultado['texto']}*")
                st.caption(
                    f"Modelo: **{resultado['modelo']}** (n={resultado['n']}) · "
                    f"Largo máximo: {resultado['max_len']} · "
                    f"Palabra inicial: {resultado['palabra_inicial'] or '(ninguna)'}"
                )