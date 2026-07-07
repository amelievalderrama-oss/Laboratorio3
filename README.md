<img src=https://raw.githubusercontent.com/amelievalderrama-oss/Laboratorio3/main/readmepics/images.jpeg align="center" alt="BibliaBanner" >
<h2 align="center"> Taller 03 - Programación Científica </h2>
<h1 align="center"> Aplicación interactiva de Streamlit sobre taller bíblico </h1>
<p align = center>
<a href = "https://www.ucn.cl"><img alt="Static Badge" src="https://img.shields.io/badge/Universidad_Católica_del_Norte-orange"></a>
<a href = "https://eic.ucn.cl"> <img alt="Static Badge" src="https://img.shields.io/badge/Escuela_de_Ingeniería_Coquimbo-blue"></a>
</p>

El presente taller consiste en la implementación interactiva, en Streamlit, del [Taller 02 de Programación Científica](https://github.com/Fifthtaschenmesser4/Taller02-ProgCien).

Consiste en el análisis de un corpus bíblico de 31.103 versículos distribuidos en 66 libros y 2 testamentos.
Se implementa como una arquitectura de cliente-servidor, donde se combina una API REST de FastAPI con una interfaz web interactiva en Streamlit.

La aplicación está dividida en cuatro puntos:
- Dashboard: Estadísticas generales y descriptivas del corpus.
- Buscador semántico: Busca y encuentra los versículos más similares a una frase mediante la similitud del coseno.
- PCA y Word2Vec: Visualización en 2D o 3D de los versículos, aplicando PCA en vectores de TF-IDF o Word2Vec.
- Generador de versículos: Genera texto a partir de una palabra inicial. Está basado en n-gramas.

## Instalación y ejecución 

```bash
git clone https://github.com/amelievalderrama-oss/Laboratorio3.git
cd Laboratorio3

conda create -n taller3 python=3.11 -y
conda activate taller3

pip install -r api/requirements.txt
pip install -r app/requirements.txt
```

Se necesitan **dos terminales**, ambas con el entorno `taller3` activado (`conda activate taller3`).

**Terminal 1 — API**
```bash
uvicorn api.main:app --reload --port 8000
```
Esperar hasta ver `API lista` en la consola.

**Terminal 2 — Streamlit**
```bash
streamlit run app/streamlit_app.py
```

## Acceso

| Servicio | URL |
|---|---|
| App Streamlit | http://localhost:8501 |
| Documentación API (Swagger) | http://localhost:8000/docs |

## Datos

El dataset bíblico fue obtenido de Kaggle: [oswinrh/bible](https://www.kaggle.com/datasets/oswinrh/bible)

Los archivos (ya incluidos en el repositorio) son:
- `t_asv.csv`
- `key_english.csv`
- `key_genre_english.csv`
- `stopwords.json`

## Detener la ejecución

En cada terminal: `Ctrl + C`. Para salir del entorno conda: `conda deactivate`.

## Estructura del proyecto
Laboratorio3/<br>
├── api/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> Backend FastAPI<br> 
├── app/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> Frontend Streamlit<br>
├── src/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> Lógica analítica<br>
└── data/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> Dataset<br>

## Integrantes
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/martindroguett">
        <img src="https://github.com/martindroguett.png" width="100px;" alt="Martín Droguett" style="border-radius:50%"/>
        <br />
        <sub><b>Martín Droguett Robledo</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/amelievalderrama-oss">
        <img src="https://github.com/amelievalderrama-oss.png" width="100px;" alt="Amelie Valderrama" style="border-radius:50%"/>
        <br />
        <sub><b>Amelie Valderrama Campos</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Fifthtaschenmesser4">
        <img src="https://github.com/Fifthtaschenmesser4.png" width="100px;" alt="Francisco Romero" style="border-radius:50%"/>
        <br />
        <sub><b>Francisco Romero Opazo</b></sub>
      </a>
    </td>
  </tr>
</table>