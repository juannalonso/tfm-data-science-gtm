"""
app.py — Cualificador de leads B2B (restaurantes Centro de Madrid)
TFM Data Science e IA — Juan Alonso
"""
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Cualificador de leads · Restaurantes Madrid Centro",
    page_icon="🍽️",
    layout="wide",
)

ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "data" / "gold" / "gold_restaurantes_madrid.parquet"

@st.cache_data
def cargar_gold(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

if not GOLD_PATH.exists():
    st.error(
        f"No encuentro la capa gold en `{GOLD_PATH.relative_to(ROOT)}`.\n\n"
        "Genérala primero con:  `python src/build_gold.py`"
    )
    st.stop()

df = cargar_gold(GOLD_PATH)

st.title("Cualificador de leads B2B — Restaurantes")
st.caption(
    "Universo de trabajo: restaurantes del distrito Centro de Madrid. "
    "Esta pantalla explora la capa gold; en el siguiente paso añadiremos la "
    "cualificación por similitud con tu ICP."
)

st.sidebar.header("Filtros")
barrios = sorted(df["barrio"].dropna().unique())
epigrafes = sorted(df["epigrafe_oficial"].dropna().unique())
cocinas = sorted(df["cocina_osm"].dropna().unique())

sel_barrio = st.sidebar.multiselect("Barrio", barrios)
sel_epigrafe = st.sidebar.multiselect("Epígrafe oficial", epigrafes)
sel_cocina = st.sidebar.multiselect("Cocina (OSM)", cocinas)
solo_web = st.sidebar.checkbox("Solo con web propia")

f = df.copy()
if sel_barrio:
    f = f[f["barrio"].isin(sel_barrio)]
if sel_epigrafe:
    f = f[f["epigrafe_oficial"].isin(sel_epigrafe)]
if sel_cocina:
    f = f[f["cocina_osm"].isin(sel_cocina)]
if solo_web:
    f = f[f["tiene_web"]]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Restaurantes", f"{len(f):,}".replace(",", "."))
c2.metric("Barrios", f["barrio"].nunique())
c3.metric("Con web", f"{f['tiene_web'].mean() * 100:.0f}%")
c4.metric("Con teléfono", f"{f['tiene_telefono'].mean() * 100:.0f}%")

st.subheader("Mapa")
mapa = f[["latitud", "longitud"]].dropna().rename(
    columns={"latitud": "lat", "longitud": "lon"}
)
st.map(mapa, size=8)

st.subheader(f"Listado ({len(f):,} restaurantes)".replace(",", "."))
cols = [
    "nombre", "barrio", "epigrafe_oficial", "cocina_osm",
    "competencia_500m", "tiene_web", "tiene_terraza", "codigo_postal",
]
st.dataframe(f[cols], use_container_width=True, hide_index=True, height=420)

st.download_button(
    "Descargar selección (CSV)",
    f[cols].to_csv(index=False).encode("utf-8"),
    file_name="restaurantes_seleccion.csv",
    mime="text/csv",
)
