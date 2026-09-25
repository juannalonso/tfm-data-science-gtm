"""
app.py — Cualificador de leads B2B (restaurantes Centro de Madrid)
TFM Data Science e IA — Juan Alonso · Tutor: Julio Valero · 2025-2026

Pestañas:
- 🔎 Explorar: universo de restaurantes con filtros, KPIs y mapa.
- 🎯 Cualificar: seleccionas 3 restaurantes como ICP y devuelve top-K similares.
- ℹ️ Sobre el proyecto: cifras del TFM, fuentes y features del modelo.

Ejecutar desde la raíz del repo:
    streamlit run app/app.py
"""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.similitud import construir_vectores, calcular_scores  # noqa: E402

# =========================================================================== #
# Configuración
# =========================================================================== #
st.set_page_config(
    page_title="Cualificador de leads · Restaurantes Madrid Centro",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

GOLD_PATH = ROOT / "data" / "gold" / "gold_restaurantes_madrid.parquet"

# --------------------------------------------------------------------------- #
# Presets de demo con narrativa (esto es lo que cuenta la historia al tribunal)
# --------------------------------------------------------------------------- #
DEMOS: dict[str, dict] = {
    "demo_premium": {
        "titulo": "Demo 1 · Software de reservas premium",
        "cliente": "Consultora GTM que vende un CRM/software de reservas a restaurantes de gama alta.",
        "icp_narrativa": (
            "El comercial ha elegido 3 restaurantes que representan su cliente ideal: "
            "**Sagardi**, **Sandó** y **Dray Martina**. Los tres comparten posicionamiento premium, "
            "presencia digital activa, distribuidos en tres barrios distintos del Centro y con ambientes complementarios (tradicional, formal, moderno)."
        ),
        "seeds": [
            "b4c626a9f914",  # Sagardi (Cortes)
            "5ee59f48bde1",  # Sandó (Palacio)
            "1d4eb6981e34",  # Dray Martina (Justicia)
        ],
    },
    "demo_asiatico": {
        "titulo": "Demo 2 · Distribuidor de producto asiático",
        "cliente": "Distribuidor mayorista de producto asiático (fideos, salsas, especias) buscando restaurantes objetivo.",
        "icp_narrativa": (
            "El ICP son 3 restaurantes asiáticos de perfil operativo similar: "
            "**Okashi Sanda**, **Hunan Restaurant** y **Tuk Tuk**. Cocina asiática, "
            "ambiente casual/moderno, concentrados en la zona norte del Centro."
        ),
        "seeds": [
            "268924f9373e",  # Okashi Sanda (Universidad)
            "9b4c6f30c242",  # Hunan Restaurant (Universidad)
            "a7da6460e258",  # Tuk Tuk (Justicia)
        ],
    },
}


# =========================================================================== #
# Carga
# =========================================================================== #
@st.cache_data
def cargar_gold(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_resource
def preparar_modelo(df: pd.DataFrame):
    return construir_vectores(df)


def col_cocina(df: pd.DataFrame) -> str:
    return "cocina" if "cocina" in df.columns else "cocina_osm"


def bonito(v) -> str:
    """Nulos y placeholders 'desconocido' -> guion largo, para tablas."""
    if pd.isna(v):
        return "—"
    s = str(v).strip()
    return "—" if s.lower() in {"none", "nan", "desconocido", ""} else s


if not GOLD_PATH.exists():
    st.error(
        f"No encuentro la capa gold en `{GOLD_PATH.relative_to(ROOT)}`.\n\n"
        "Genérala primero con:  `python src/build_gold.py`"
    )
    st.stop()

df = cargar_gold(GOLD_PATH)
modelo = preparar_modelo(df)
COCINA = col_cocina(df)

# =========================================================================== #
# Cabecera del proyecto
# =========================================================================== #
st.markdown(
    """
    <div style="background: linear-gradient(90deg,#0f172a 0%,#1e293b 100%);
                padding: 18px 24px; border-radius: 10px; margin-bottom: 10px;">
      <div style="color:#f8fafc; font-size:24px; font-weight:700;">
        🍽️ Cualificador de leads B2B — Restaurantes
      </div>
      <div style="color:#cbd5e1; font-size:14px; margin-top:4px;">
        TFM Data Science e IA · Juan Alonso · Tutor: Julio Valero · Curso 2025-2026
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# KPIs del proyecto (visibles siempre)
n_llm = int(df["resumen_llm"].notna().sum()) if "resumen_llm" in df.columns else 0
n_ine = int(df["renta_media"].notna().sum()) if "renta_media" in df.columns else 0
n_censo = int((df["metodo_enriquecimiento_censo"] != "no_matcheado").sum()) \
    if "metodo_enriquecimiento_censo" in df.columns else 0

kc1, kc2, kc3, kc4 = st.columns(4)
kc1.metric("Universo", f"{len(df):,}".replace(",", "."), help="Restaurantes del distrito Centro de Madrid")
kc2.metric("Enriquecido con censo", f"{n_censo/len(df)*100:.0f}%",
           help="Cruce con censo del Ayuntamiento por dirección o proximidad")
kc3.metric("Enriquecido con INE",
           f"{n_ine/len(df)*100:.0f}%" if "renta_media" in df.columns else "—",
           help="Datos socioeconómicos por sección censal")
kc4.metric("Enriquecido con LLM",
           f"{n_llm/len(df)*100:.0f}%" if "resumen_llm" in df.columns else "—",
           help="Variables cualitativas generadas con Gemini")

st.markdown("")  # espacio

tab_explorar, tab_cualificar, tab_proyecto = st.tabs(
    ["🔎 Explorar", "🎯 Cualificar (ICP)", "ℹ️ Sobre el proyecto"]
)

# =========================================================================== #
# TAB 1 · EXPLORAR
# =========================================================================== #
with tab_explorar:
    with st.sidebar:
        st.header("Filtros")
        st.caption("Solo afectan a la pestaña *Explorar*.")
        barrios = sorted(df["barrio"].dropna().unique())
        epigrafes = sorted(df["epigrafe_oficial"].dropna().unique())
        cocinas = sorted(df[COCINA].dropna().unique())

        sel_barrio = st.multiselect("Barrio", barrios)
        sel_epigrafe = st.multiselect("Epígrafe oficial", epigrafes)
        sel_cocina = st.multiselect("Cocina", cocinas)
        solo_web = st.checkbox("Solo con web propia")

        if "ambiente" in df.columns:
            ambientes = sorted(df["ambiente"].dropna().unique())
            sel_ambiente = st.multiselect("Ambiente (LLM)", ambientes)
        else:
            sel_ambiente = []

    f = df.copy()
    if sel_barrio:   f = f[f["barrio"].isin(sel_barrio)]
    if sel_epigrafe: f = f[f["epigrafe_oficial"].isin(sel_epigrafe)]
    if sel_cocina:   f = f[f[COCINA].isin(sel_cocina)]
    if solo_web:     f = f[f["tiene_web"]]
    if sel_ambiente: f = f[f["ambiente"].isin(sel_ambiente)]

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
    cols_base = ["nombre", "barrio", "epigrafe_oficial", COCINA, "competencia_500m",
                 "tiene_web", "codigo_postal"]
    cols_extra = [c for c in ["ambiente", "presencia_digital", "posicionamiento_precio"]
                  if c in f.columns]
    cols_mostrar = cols_base + cols_extra

    # Vista bonita: reemplazar Nones/desconocido por guion en columnas de string
    f_view = f[cols_mostrar].copy()
    for c in f_view.select_dtypes(include=["object", "string"]).columns:
        f_view[c] = f_view[c].map(bonito)

    st.dataframe(f_view, use_container_width=True, hide_index=True, height=380)
    st.download_button(
        "📥 Descargar selección (CSV)",
        f[cols_mostrar].to_csv(index=False).encode("utf-8"),
        file_name="restaurantes_seleccion.csv",
        mime="text/csv",
    )

# =========================================================================== #
# TAB 2 · CUALIFICAR (ICP)
# =========================================================================== #
with tab_cualificar:
    # Ocultamos el sidebar de filtros aquí para no confundir (los filtros
    # eran de Explorar y no afectan al ranking del ICP).

    st.markdown(
        "#### Cómo funciona\n"
        "1. Elige **3 restaurantes** que representen tu cliente ideal (ICP), o carga un preset.\n"
        "2. El sistema aprende qué caracteriza a esos 3 y **rankea el resto** por similitud coseno.\n"
        "3. Cada lead trae su **perfil enriquecido** (LLM + INE) para arrancar la conversación."
    )

    st.markdown("##### 🎬 Presets de demo (casos reales de agencia GTM)")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        d = DEMOS["demo_premium"]
        st.markdown(f"**{d['titulo']}**")
        st.caption(d["cliente"])
        if st.button("Cargar Demo 1", use_container_width=True, key="btn_d1"):
            st.session_state["seeds"] = d["seeds"]
            st.session_state["preset_activo"] = "demo_premium"

    with col_d2:
        d = DEMOS["demo_asiatico"]
        st.markdown(f"**{d['titulo']}**")
        st.caption(d["cliente"])
        if st.button("Cargar Demo 2", use_container_width=True, key="btn_d2"):
            st.session_state["seeds"] = d["seeds"]
            st.session_state["preset_activo"] = "demo_asiatico"

    # Contexto del preset cargado
    preset = st.session_state.get("preset_activo")
    if preset and preset in DEMOS:
        st.info(f"**Caso activo — {DEMOS[preset]['titulo']}**\n\n{DEMOS[preset]['icp_narrativa']}")

    st.markdown("---")
    st.markdown("##### Tu selección de ICP")

    df_sel = df.assign(
        etiqueta=df["nombre"].fillna("(sin nombre)") + " — " + df["barrio"].fillna("¿?")
    ).sort_values("etiqueta")
    etiqueta_por_id = dict(zip(df_sel["restaurante_id"], df_sel["etiqueta"]))
    id_por_etiqueta = {v: k for k, v in etiqueta_por_id.items()}

    default_ids = st.session_state.get("seeds", [])
    default_labels = [etiqueta_por_id[i] for i in default_ids if i in etiqueta_por_id]

    sel_labels = st.multiselect(
        "Restaurantes seed (elige exactamente 3)",
        options=list(id_por_etiqueta.keys()),
        default=default_labels,
        max_selections=3,
        placeholder="Escribe para buscar por nombre…",
    )
    seeds = [id_por_etiqueta[l] for l in sel_labels]

    col_k, col_btn = st.columns([2, 1])
    with col_k:
        top_k = st.slider("Cuántos leads mostrar", 10, 100, 20, step=5)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        buscar = st.button("🎯 Buscar leads similares", type="primary",
                           disabled=len(seeds) != 3, use_container_width=True)

    if len(seeds) != 3:
        st.info(f"Selecciona 3 restaurantes (llevas {len(seeds)}).")

    if buscar:
        st.session_state["seeds"] = seeds
        scores = calcular_scores(modelo, seeds)
        ranking = (
            df.merge(scores.rename("score"), left_on="restaurante_id", right_index=True)
            .dropna(subset=["score"])
            .sort_values("score", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )
        st.session_state["ranking"] = ranking

    if "ranking" in st.session_state:
        ranking = st.session_state["ranking"]
        seeds_df = df[df["restaurante_id"].isin(st.session_state["seeds"])]

        st.divider()
        st.subheader(f"🏆 Top {len(ranking)} leads más similares al ICP")

        k1, k2, k3 = st.columns(3)
        k1.metric("Score medio top-K", f"{ranking['score'].mean():.3f}")
        k2.metric("Score máximo", f"{ranking['score'].max():.3f}")
        k3.metric("% con web propia", f"{ranking['tiene_web'].mean() * 100:.0f}%")

        cols_rank = ["score", "nombre", "barrio", COCINA, "epigrafe_oficial",
                     "tiene_web", "competencia_500m", "codigo_postal"]
        cols_rank += [c for c in ["ambiente", "presencia_digital", "posicionamiento_precio"]
                      if c in ranking.columns]

        r_view = ranking[cols_rank].copy()
        for c in r_view.select_dtypes(include=["object", "string"]).columns:
            r_view[c] = r_view[c].map(bonito)

        st.dataframe(
            r_view.style.format({"score": "{:.3f}"}),
            use_container_width=True, hide_index=True, height=380,
        )
        st.download_button(
            "📥 Descargar ranking (CSV)",
            ranking[cols_rank].to_csv(index=False).encode("utf-8"),
            file_name="leads_ranking.csv",
            mime="text/csv",
        )

        # Mapa · seeds vs leads
        st.subheader("🗺️ Mapa · seeds vs leads")
        puntos = pd.concat([
            seeds_df.assign(tipo="seed"),
            ranking.assign(tipo="lead"),
        ])[["latitud", "longitud", "tipo"]].dropna()
        puntos["color"] = puntos["tipo"].map(
            {"seed": [34, 197, 94, 220], "lead": [59, 130, 246, 180]}
        )
        puntos["radio"] = puntos["tipo"].map({"seed": 60, "lead": 30})
        st.map(
            puntos.rename(columns={"latitud": "lat", "longitud": "lon"}),
            color="color", size="radio",
        )
        st.caption("🟢 Seeds (tu ICP)   ·   🔵 Leads recomendados")

        # Ficha detalle del lead
        st.subheader("📋 Ficha detalle del lead")
        etiquetas_rank = (ranking["nombre"].fillna("(sin nombre)") + " ("
                          + ranking["barrio"].fillna("¿?") + ")").tolist()
        opcion = st.selectbox("Elige un lead para ver su ficha",
                              options=etiquetas_rank, index=0)
        idx = etiquetas_rank.index(opcion)
        r = ranking.iloc[idx]

        # Enlace a Google Maps con las coordenadas del lead
        gmaps = (f"https://www.google.com/maps/search/?api=1&query="
                 f"{r['latitud']},{r['longitud']}")

        cA, cB = st.columns(2)
        with cA:
            st.markdown(f"### {r['nombre']}")
            st.markdown(f"**Barrio:** {bonito(r['barrio'])}  ·  **CP:** {bonito(r['codigo_postal'])}")
            st.markdown(f"**Cocina:** {bonito(r[COCINA])}  ·  **Epígrafe:** {bonito(r['epigrafe_oficial'])}")
            st.markdown(f"**Score de similitud:** `{r['score']:.3f}`")
            if "renta_media" in ranking.columns and pd.notna(r.get("renta_media")):
                renta = f"{int(r['renta_media']):,}".replace(",", ".")
                st.markdown(f"**Renta media del entorno:** {renta} €")
            st.markdown(f"[🗺️ Ver en Google Maps]({gmaps})")

        with cB:
            st.markdown("**Señales operativas**")
            web_txt = f"  [{r['url_web']}]({r['url_web']})" if pd.notna(r.get("url_web")) else ""
            st.markdown(f"- Web: {'✅' if r['tiene_web'] else '—'}{web_txt}")
            st.markdown(f"- Teléfono: {'✅' if r['tiene_telefono'] else '—'}")
            st.markdown(f"- Horario publicado: {'✅' if r['tiene_horario'] else '—'}")
            st.markdown(f"- Terraza: {'✅' if r['tiene_terraza'] else '—'}")
            st.markdown(f"- Delivery: {'✅' if r['ofrece_delivery'] else '—'}")
            st.markdown(f"- Takeaway: {'✅' if r['ofrece_takeaway'] else '—'}")
            st.markdown(f"- Competencia en 500 m: **{int(r['competencia_500m'])}**")

        # Perfil cualitativo LLM
        campos_llm = [c for c in ["ambiente", "presencia_digital",
                                  "posicionamiento_precio", "tipo_clientela"]
                      if c in ranking.columns]
        if campos_llm or "resumen_llm" in ranking.columns:
            st.markdown("---")
            st.markdown("#### 🤖 Perfil cualitativo (enriquecimiento LLM)")
            if campos_llm:
                cols_llm = st.columns(len(campos_llm))
                for col, campo in zip(cols_llm, campos_llm):
                    valor = r.get(campo)
                    col.metric(campo.replace("_", " ").capitalize(), bonito(valor))
            if "resumen_llm" in ranking.columns and pd.notna(r.get("resumen_llm")):
                st.markdown("**Resumen generado por el LLM:**")
                st.info(r["resumen_llm"])

        # Dossier para el comercial (markdown copy-paste)
        st.markdown("---")
        st.markdown("#### 📝 Dossier del lead (para pegar en HubSpot / Notion)")
        dossier = f"""**{r['nombre']}** — {bonito(r['barrio'])}
- Cocina: {bonito(r[COCINA])} · Epígrafe: {bonito(r['epigrafe_oficial'])} · CP: {bonito(r['codigo_postal'])}
- Score similitud con ICP: {r['score']:.3f}
- Web: {r.get('url_web') if pd.notna(r.get('url_web')) else '—'}
- Teléfono publicado: {'sí' if r['tiene_telefono'] else 'no'}
- Competencia en 500m: {int(r['competencia_500m'])}
"""
        if campos_llm:
            for c in campos_llm:
                dossier += f"- {c.replace('_',' ').capitalize()}: {bonito(r.get(c))}\n"
        if "resumen_llm" in ranking.columns and pd.notna(r.get("resumen_llm")):
            dossier += f"\n> {r['resumen_llm']}\n"
        dossier += f"\n[Ver en Google Maps]({gmaps})\n"

        st.code(dossier, language="markdown")

# =========================================================================== #
# TAB 3 · SOBRE EL PROYECTO
# =========================================================================== #
with tab_proyecto:
    st.markdown("### El problema")
    st.markdown(
        "En estrategias de outbound B2B, la construcción y priorización del listado "
        "de empresas-objetivo es un cuello de botella recurrente. Las herramientas del "
        "mercado (Apollo, ZoomInfo, scrapers) resuelven la recolección, pero devuelven "
        "listas planas y sin criterio de priorización."
    )

    st.markdown("### La solución")
    st.markdown(
        "Un sistema que, a partir de **3 restaurantes de referencia** (ICP) aportados "
        "por el usuario, aprende automáticamente el perfil ideal mediante técnicas de "
        "*few-shot learning* y prioriza el universo restante por similitud coseno."
    )

    st.markdown("### Fuentes de datos")
    fuentes = pd.DataFrame([
        {"Fuente": "OpenStreetMap (API Overpass)",
         "Aporta": "Universo, ubicación, cocina, servicios operativos",
         "Volumen": "1.688 restaurantes"},
        {"Fuente": "Censo Ayuntamiento Madrid",
         "Aporta": "Clasificación oficial, barrio, sección censal",
         "Volumen": "1.626 locales cruzados"},
        {"Fuente": "INE (sección censal)",
         "Aporta": "Renta media del entorno del establecimiento",
         "Volumen": f"{n_ine}/{len(df)} restaurantes" if "renta_media" in df.columns else "—"},
        {"Fuente": "Gemini LLM",
         "Aporta": "Ambiente, presencia digital, posicionamiento, resumen",
         "Volumen": f"{n_llm}/{len(df)} restaurantes" if "resumen_llm" in df.columns else "—"},
    ])
    st.dataframe(fuentes, hide_index=True, use_container_width=True)

    st.markdown("### Modelo de cualificación")
    st.markdown(
        "**Modelo 1 · Similitud coseno sobre vector combinado.** "
        "Se construye un vector por restaurante con features numéricas escaladas, "
        "banderas booleanas y one-hot de categóricas. El score de cada lead es el "
        "coseno entre su vector y el centroide del ICP."
    )
    cols_u = modelo.columnas_usadas
    with st.expander("Features usadas por el modelo", expanded=False):
        st.markdown(f"- **Numéricas:** {', '.join(cols_u['numericas']) or '—'}")
        st.markdown(f"- **Booleanas:** {', '.join(cols_u['booleanas']) or '—'}")
        st.markdown(f"- **Categóricas (one-hot):** {', '.join(cols_u['categoricas']) or '—'}")
        if cols_u["excluidas_por_sesgo"]:
            st.markdown(
                f"- **Excluidas por baja varianza (análisis de sesgo):** "
                f"{', '.join(cols_u['excluidas_por_sesgo'])}"
            )
        st.caption(f"Total de features en el vector: {len(modelo.feature_names)}")

# =========================================================================== #
# Footer
# =========================================================================== #
st.markdown(
    """
    <div style="margin-top:32px; padding-top:12px; border-top:1px solid #e2e8f0;
                text-align:center; color:#64748b; font-size:12px;">
      TFM Data Science e IA · Juan Alonso · Tutor: Julio Valero · Curso 2025-2026<br>
      Datos abiertos: OpenStreetMap · Ayuntamiento de Madrid · INE · Enriquecimiento cualitativo: Gemini
    </div>
    """,
    unsafe_allow_html=True,
)
