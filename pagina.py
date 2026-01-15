import os
import html
import base64
import json
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import plotly.express as px
from PIL import Image

from build_ecuador_map import build_ecuador_map

# --------------------------------------------------
# CONFIG STREAMLIT (DEBE SER LO PRIMERO)
# --------------------------------------------------
st.set_page_config(page_title="Inmovision", layout="wide")

# --------------------------------------------------
# VIDEO DE FONDO (ESTABLE)
# --------------------------------------------------
def set_video_background(video_path: str):
    if not os.path.exists(video_path):
        st.error(f"⚠️ No se encontró el video: {video_path}")
        return

    with open(video_path, "rb") as f:
        video_base64 = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        #bgVideo {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%;
            min-height: 100%;
            object-fit: cover;
            z-index: -2;
        }}

        .video-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.50);
            z-index: -1;
        }}

        .stApp {{
            background: transparent;
        }}
        </style>

        <video autoplay muted loop playsinline id="bgVideo">
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        </video>
        <div class="video-overlay"></div>
        """,
        unsafe_allow_html=True
    )

VIDEO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fondo_inmobiliaria.mp4")
set_video_background(VIDEO_PATH)

# --------------------------------------------------
PRIMARY_COLOR = "#0056b3"
SECONDARY_COLOR = "#FFD700"
TEXT_COLOR = "#222"
# --------------------------------------------------
# --------------------------------------------------
# CSS (tu mismo estilo
# --------------------------------------------------
st.markdown(
    """
<style>
iframe { min-height: 520px !important; }
header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stHeader"] { display: none; }
.block-container { padding-top: 0.5rem !important; padding-bottom: 2rem; }

h1, h2, h3 {
  color: white !important;
  font-weight: 900 !important;
  text-shadow: 0px 6px 18px rgba(0,0,0,0.90) !important;
}

.title-band {
  background: rgba(0,0,0,0.35);
  padding: 15px 25px;
  border-radius: 18px;
  display: inline-block;
  backdrop-filter: blur(10px);
  box-shadow: 0px 8px 20px rgba(0,0,0,0.25);
}

.login-card {
  max-width: 900px;
  margin: auto;
  margin-top: 5vh;
  padding: 60px;
  border-radius: 26px;
  background: rgba(255,255,255,0.18);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(255,255,255,0.35);
  box-shadow: 0 16px 45px rgba(0,0,0,0.35);
  text-align: center;
}

.login-sub {
  font-size: 1.4rem;
  font-weight: 700;
  color: rgba(255,255,255,0.95);
  margin-top: -10px;
  margin-bottom: 25px;
  text-shadow: 0px 3px 10px rgba(0,0,0,0.75);
}

.login-btn .stButton>button {
  width: 70%;
  margin: auto;
  border-radius: 18px;
  padding: 15px;
  font-size: 1.15rem;
  font-weight: 900;
  color: white;
  background: linear-gradient(135deg, #00b894, #0984e3);
  transition: 0.25s;
  border: none;
}

.login-btn .stButton>button:hover {
  background: linear-gradient(135deg, #FFD700, #ffb400);
  color: #111;
  transform: scale(1.05);
  box-shadow: 0px 10px 25px rgba(0,0,0,0.45);
}

.section-card {
  background: rgba(255,255,255,0.95);
  padding: 25px;
  border-radius: 18px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.25);
  margin-top: 20px;
}

.section-card h1, .section-card h2, .section-card h3 {
  color: """ + PRIMARY_COLOR + """ !important;
  text-shadow: none !important;
}

.comentario-box-wide {
  background:#fff;
  padding:18px 20px;
  border-radius:12px;
  border-left:6px solid #4A90E2;
  box-shadow:0 2px 10px rgba(0,0,0,0.10);
  margin-top:18px;
  width:100%;
  white-space:pre-line;
}

.footer {
  position: fixed;
  left: 0;
  bottom: 0;
  width: 100%;
  background-color: rgba(0,0,0,0.55);
  color: white;
  text-align: center;
  padding: 10px;
  font-size: 0.9em;
  z-index: 100;
}
</style>
""",
    unsafe_allow_html=True,
)
# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def norm_title_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.title()

import html

def build_commentary_analitico(
    region_sel, prov_sel, canton_sel, tipo_sel, años_sel,
    precio_actual_m2, precio_futuro_m2,
    precio_actual_total=None, precio_futuro_total=None,
    area_med=None,
    growth=None
):
    # Sanitizar / seguridad
    region_sel = region_sel or ""
    prov_sel = prov_sel or ""
    canton_sel = canton_sel or ""
    tipo_sel = tipo_sel or "(sin tipo)"

    # Cálculos base
    delta_m2 = float(precio_futuro_m2 - precio_actual_m2)
    pct_m2 = (delta_m2 / precio_actual_m2) * 100 if precio_actual_m2 else 0.0

    # Total
    delta_total = None
    pct_total = None
    if precio_actual_total is not None and precio_futuro_total is not None and precio_actual_total != 0:
        delta_total = float(precio_futuro_total - precio_actual_total)
        pct_total = (delta_total / precio_actual_total) * 100

    # Frases “inteligentes” (heurísticas)
    impacto_unitario = "moderado"
    if abs(pct_m2) >= 35:
        impacto_unitario = "alto"
    elif abs(pct_m2) >= 20:
        impacto_unitario = "relevante"
    elif abs(pct_m2) >= 10:
        impacto_unitario = "moderado"
    else:
        impacto_unitario = "leve"

    # Explicación principal
    explicacion = []
    if area_med is not None and area_med > 0 and delta_total is not None:
        # amplificación por área
        explicacion.append(
            "Este resultado muestra que el efecto económico no proviene solo del precio unitario, "
            "sino de cómo ese aumento se amplifica al aplicarse sobre superficies extensas."
        )

        # detectar “terreno grande” (ajusta umbral si quieres)
        if area_med >= 10000:
            explicacion.append(
                "En propiedades de gran extensión, variaciones pequeñas en el valor por m² "
                "pueden traducirse en incrementos monetarios elevados en el valor total."
            )
        else:
            explicacion.append(
                "Incluso con áreas medianas, el incremento por m² se traduce en un aumento total "
                "significativo al multiplicarse por la superficie."
            )
    else:
        explicacion.append(
            "El precio por m² describe el valor unitario, mientras que el precio total depende además del tamaño. "
            "Si no hay área suficiente, la estimación total no puede calcularse con precisión."
        )

    # Construcción del texto estilo “análisis”
    intro = (
        f"En un horizonte de {años_sel} años, para {tipo_sel} en {canton_sel} ({prov_sel}, {region_sel}), "
        f"el precio promedio por metro cuadrado pasa de ${precio_actual_m2:,.0f} a ${precio_futuro_m2:,.0f}, "
        f"lo que representa un incremento absoluto de ${delta_m2:,.0f} por m² "
        f"({pct_m2:.1f}%)."
    )

    if delta_total is not None and precio_actual_total is not None and precio_futuro_total is not None and area_med is not None:
        segundo = (
            f" Aunque a nivel unitario el crecimiento puede parecer {impacto_unitario}, su efecto se vuelve más visible "
            f"cuando se proyecta sobre el área típica observada en los datos (≈ {area_med:,.0f} m²). "
            f"Con esa referencia, el valor total estimado pasa de aproximadamente ${precio_actual_total:,.0f} "
            f"a ${precio_futuro_total:,.0f}, lo que implica un incremento absoluto cercano a ${delta_total:,.0f} "
            f"en {años_sel} años."
        )
    else:
        segundo = " No fue posible estimar el valor total porque faltan datos de área suficientes para esta selección."

    # opcional: añadir crecimiento anual usado
    cola = ""
    if growth is not None:
        cola = f" (Supuesto de crecimiento anual aplicado: {growth*100:.2f}%.)"

    texto = intro + segundo + "\n\n" + " ".join(explicacion) + cola
    return texto

# --------------------------------------------------
# PREPROCESAMIENTO (tu función original)
# --------------------------------------------------
@st.cache_data
def load_and_process_data(file_path="datos.csv"):
    df = pd.read_csv(file_path, encoding="latin1")
    df_original = df.copy()
    df.columns = df.columns.str.strip()

    df["Precio_USD"] = (
        df["Precio"].astype(str)
        .str.replace("USD", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(" ", "", regex=False)
        .astype(float)
    )

    df["Total_m2"] = (
        df["Total construido"].astype(str)
        .str.replace("m²", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    df["Total_m2"] = pd.to_numeric(df["Total_m2"], errors="coerce")

    df["Superficie_m2"] = (
        df["Superficie"].astype(str)
        .str.replace("m²", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    df["Superficie_m2"] = pd.to_numeric(df["Superficie_m2"], errors="coerce")

    df["Antiguedad"] = df["Antiguedad"].replace("A Estrenar", 0)
    df["Antiguedad"] = pd.to_numeric(df["Antiguedad"], errors="coerce")

    nulos_antes = df.isnull().sum()

    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    nulos_despues = df.isnull().sum()
    return df_original, df, nulos_antes, nulos_despues


# --------------------------------------------------
# MODELOS
# --------------------------------------------------
@st.cache_resource
def load_models(base_dir: str):
    model_path = os.path.join(base_dir, "models_export", "inmovision_rf_price_model.pkl")
    cfg_path = os.path.join(base_dir, "models_export", "forecast_cfg.json")
    rf_model = joblib.load(model_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        forecast_cfg = json.load(f)
    return rf_model, forecast_cfg
#--------------------------------------------------
#modelos demás (lightgbm, xgboost) con carga segura
@st.cache_resource
def load_all_models_safe(models_dir: str):
    """
    Carga modelos de models_export de forma segura.
    - Si falta lightgbm o xgboost, no revienta la app: solo omite ese modelo.
    - Siempre intenta cargar el preprocessor.
    """
    models = {}
    errors = []

    # ---- Preprocessor ----
    preproc_path = os.path.join(models_dir, "ml_preprocessor.pkl")
    preprocessor = None
    if os.path.exists(preproc_path):
        try:
            preprocessor = joblib.load(preproc_path)
        except Exception as e:
            errors.append(f"Preprocessor: {e}")
            preprocessor = None
    else:
        errors.append("Preprocessor: ml_preprocessor.pkl no existe.")

    # ---- RF (sklearn) ----
    rf_path = os.path.join(models_dir, "inmovision_rf_price_model.pkl")
    if os.path.exists(rf_path):
        try:
            models["ML - RandomForest Price"] = joblib.load(rf_path)
        except Exception as e:
            errors.append(f"RandomForest: {e}")

    # ---- XGBoost (requiere xgboost) ----
    xgb_path = os.path.join(models_dir, "inmovision_xgb_model.pkl")
    if os.path.exists(xgb_path):
        try:
            __import__("xgboost")  # prueba si existe
            models["ML - XGBoost"] = joblib.load(xgb_path)
        except Exception as e:
            errors.append(f"XGBoost: {e}")

    # ---- LightGBM (requiere lightgbm) ----
    lgbm_path = os.path.join(models_dir, "inmovision_lightgbm_price_model.pkl")
    if os.path.exists(lgbm_path):
        try:
            __import__("lightgbm")  # prueba si existe
            models["ML - LightGBM"] = joblib.load(lgbm_path)
        except Exception as e:
            errors.append(f"LightGBM: {e}")

    return models, preprocessor, errors
#--------------------------------------------------
# --------------------------------------------------
# ADMIN DATA (Camino A + B)
# --------------------------------------------------
@st.cache_data
def load_admin_data(base_dir: str):
    data_admin = os.path.join(base_dir, "data_admin")

    def rp(name):
        return os.path.join(data_admin, name)

    out = {}
    out["admin_points"] = pd.read_parquet(rp("admin_points.parquet"))

    market_path = rp("admin_points_market.parquet")
    out["admin_points_market"] = pd.read_parquet(market_path) if os.path.exists(market_path) else pd.DataFrame()
    return out


# --------------------------------------------------
# MAPA (cache para performance)
# --------------------------------------------------
@st.cache_resource
def get_map(data_dir_abs: str):
    return build_ecuador_map(data_dir=data_dir_abs)


# --------------------------------------------------
# INIT PATHS + LOADS
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

try:
    modelo_rf, forecast_cfg = load_models(BASE_DIR)
except Exception as e:
    st.error(f"❌ Error cargando modelos: {e}")
    st.stop()

try:
    admin = load_admin_data(BASE_DIR)
except Exception as e:
    st.error(f"❌ Error cargando data_admin: {e}")
    st.stop()


# --------------------------------------------------
# LOGIN
# --------------------------------------------------
def show_welcome_screen():
    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.markdown('<div class="title-band"><h1>Bienvenido a la Plataforma</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="login-sub">Plataforma de Analítica Inmobiliaria</div>', unsafe_allow_html=True)
    st.info("📌 Herramienta integral para gestión, análisis predictivo y clustering.")

    st.markdown('<div class="login-btn">', unsafe_allow_html=True)
    if st.button("✅ ACCEDER A LA PLATAFORMA"):
        st.session_state["logged_in"] = True
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="footer">© 2023 Inmovisión. Todos los derechos reservados.</div>', unsafe_allow_html=True)


# --------------------------------------------------
# MAIN UI
# --------------------------------------------------
def show_main_interface():
    with st.sidebar:
        st.header("⚙️ Menú del Proyecto")
        choice = st.radio(
            "Navegación",
            [
                "📊 Preprocesamiento de Datos",
                "📈 Visualización Gráfica",
                "🏙️ Predicción Inmobiliaria AI",  
            ],
        )
        st.markdown("---")
        if st.button("Cerrar Sesión"):
            st.session_state["logged_in"] = False
            st.rerun()

    st.markdown(f"<div class='title-band'><h2>{choice}</h2></div>", unsafe_allow_html=True)

    # -------------------------
    # PREPROCESAMIENTO
    # -------------------------
    if choice == "📊 Preprocesamiento de Datos":
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📊 Proceso Completo de Preprocesamiento")

        df_original, df_clean, nulos_antes, nulos_despues = load_and_process_data()

        st.markdown("### ✅ Datos Originales")
        st.dataframe(df_original.head(20), use_container_width=True)

        st.markdown("### 🔎 Valores nulos ANTES")
        st.dataframe(nulos_antes[nulos_antes > 0], use_container_width=True)

        st.markdown("### ✅ Transformaciones realizadas")
        st.success("✔ Precio → Precio_USD")
        st.success("✔ Total construido → Total_m2")
        st.success("✔ Superficie → Superficie_m2")
        st.success("✔ Antigüedad: 'A Estrenar' → 0")
        st.success("✔ Relleno de nulos (mediana y moda)")

        st.markdown("### ✅ Valores nulos DESPUÉS")
        st.dataframe(nulos_despues[nulos_despues > 0], use_container_width=True)

        st.markdown("### ✅ Datos Procesados")
        st.dataframe(df_clean.head(20), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    # -------------------------
    # VISUALIZACIÓN
    # -------------------------
    elif choice == "📈 Visualización Gráfica":
        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        _, df, _, _ = load_and_process_data()

        # Helper: mostrar imagen grande con Plotly (mismo estilo que ya usas)
        def show_image_plotly_big(img_path: str, titulo: str, height: int = 650):
            st.markdown(f"### {titulo}")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                fig = px.imshow(img)
                fig.update_layout(
                    height=height,
                    margin=dict(l=0, r=0, t=30, b=0),
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
                st.markdown("---")
            else:
                st.warning(f"⚠️ No encuentro: {img_path}")

        # =========================
        # TÍTULO 1: Histograma
        # =========================
        st.markdown("### ✅ TÍTULO 1: 📊 Histograma de Precios")
        st.plotly_chart(px.histogram(df, x="Precio_USD", nbins=30), use_container_width=True)

        # --- IMAGEN debajo del histograma ---
        st.markdown("## DISTRIBUCIONES:")
        img1_path = os.path.join(BASE_DIR, "assets", "assetsimagen_histograma.png")
        if os.path.exists(img1_path):
            img = Image.open(img1_path)
            fig_img1 = px.imshow(img)
            fig_img1.update_layout(
                height=750,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
            )
            st.plotly_chart(fig_img1, use_container_width=True, config={"displaylogo": False})
        else:
            st.warning(f"⚠️ No encuentro la imagen: {img1_path}")

        # =========================
        # TÍTULO 2: Boxplot
        # =========================
        st.markdown("### ✅ TÍTULO 2: 📌 Boxplot Precio por Tipo de Propiedad")
        st.plotly_chart(px.box(df, x="Tipo de Propiedad", y="Precio_USD"), use_container_width=True)

        # =========================
        # TÍTULO 3: 3 imágenes
        # =========================
        st.markdown("## COMPARACIÓN DE RELACIONES:")

        imgA = os.path.join(BASE_DIR, "assets", "assetsgrafico_A.png")
        imgB = os.path.join(BASE_DIR, "assets", "assetsgrafico_B.png")
        imgC = os.path.join(BASE_DIR, "assets", "assetsgrafico_C.png")

        show_image_plotly_big(imgA, "✅ TOTAL DE VENTAS POR PROVINCIA  SRI", height=650)
        show_image_plotly_big(imgB, "✅ TOTAL DE COMPRAS POR PROVINCIA  SRI", height=650)
        show_image_plotly_big(imgC, "✅ DISTRIBUCIÓN DE PROPIEDADES", height=650)

        # =========================
        # TÍTULO 4: 1 imagen
        # =========================
        st.markdown("### ✅ TÍTULO 4: 🖼️ Distribución por Tipo de Propiedad")

        img4_path = os.path.join(BASE_DIR, "assets", "assetsgrafico_tipo.png")
        if os.path.exists(img4_path):
            img = Image.open(img4_path)
            fig_img4 = px.imshow(img)
            fig_img4.update_layout(
                height=750,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
            )
            st.plotly_chart(fig_img4, use_container_width=True, config={"displaylogo": False})
        else:
            st.warning(f"⚠️ No encuentro la imagen: {img4_path}")

        # =====================================================
        # ✅ DASHBOARD NACIONAL (IMÁGENES) - MISMO ESTILO (Plotly)
        # =====================================================
        st.markdown("---")
        st.subheader("📊 Dashboard Nacional (Resumen en Imágenes)")

        imgR1 = os.path.join(BASE_DIR, "assets", "region1.png")
        imgR2 = os.path.join(BASE_DIR, "assets", "region2.png")

        colA, colB = st.columns(2)

        with colA:
            st.markdown("### ✅ Dashboard 1: Estadística por región")
            if os.path.exists(imgR1):
                img = Image.open(imgR1)
                fig = px.imshow(img)
                fig.update_layout(
                    height=520,
                    margin=dict(l=0, r=0, t=30, b=0),
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            else:
                st.warning(f"⚠️ No encuentro: {imgR1}")

        with colB:
            st.markdown("### ✅ Dashboard 2: Estadística por región")
            if os.path.exists(imgR2):
                img = Image.open(imgR2)
                fig = px.imshow(img)
                fig.update_layout(
                    height=520,
                    margin=dict(l=0, r=0, t=30, b=0),
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            else:
                st.warning(f"⚠️ No encuentro: {imgR2}")

        st.markdown("</div>", unsafe_allow_html=True)
   
  # ---------------------------
#---------------------------
    elif choice == "🏙️ Predicción Inmobiliaria AI":
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.title("🏙️ Sistema de Predicción Inmobiliaria AI")
        # ---------- fuentes ----------
        admin_points = admin["admin_points"].copy()
        for c in ["region", "provincia", "canton"]:
            if c in admin_points.columns:
                admin_points[c] = norm_title_series(admin_points[c])

        admin_points_market = admin.get("admin_points_market", pd.DataFrame()).copy()
        has_market = (not admin_points_market.empty) and ("tipo_inmueble" in admin_points_market.columns)
        if has_market:
            for c in ["region", "provincia", "canton", "tipo_inmueble"]:
                if c in admin_points_market.columns:
                    admin_points_market[c] = norm_title_series(admin_points_market[c])

        all_cantons = sorted(admin_points["canton"].dropna().unique().tolist())
        if not all_cantons:
            st.error("❌ admin_points no tiene cantones para construir filtros.")
            st.stop()

        # ---------- session defaults ----------
        if "f_canton" not in st.session_state:
            st.session_state["f_canton"] = all_cantons[0]
        if st.session_state["f_canton"] not in all_cantons:
            st.session_state["f_canton"] = all_cantons[0]

        if "f_tipo" not in st.session_state:
            st.session_state["f_tipo"] = ""
        if "f_years" not in st.session_state:
            st.session_state["f_years"] = 5

        # ==========================================================
        # ✅ FORM ARRIBA (entre título y mapa)
        # ==========================================================
        st.markdown("### 🔎 Selección de variables")

        with st.form("search_form", clear_on_submit=False):
            c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1.2, 1.6, 1.2])

            # Cantón master
            with c3:
                canton_sel = st.selectbox(
                    "📍 Cantón",
                    all_cantons,
                    index=all_cantons.index(st.session_state["f_canton"]),
                )

            # Región/Provincia autocompletadas
            rows_c = admin_points[admin_points["canton"] == canton_sel]
            prov_sel = rows_c["provincia"].mode().iloc[0] if ("provincia" in rows_c.columns and not rows_c["provincia"].dropna().empty) else ""
            region_sel = rows_c["region"].mode().iloc[0] if ("region" in rows_c.columns and not rows_c["region"].dropna().empty) else ""

            with c1:
                st.selectbox("🧭 Región", [region_sel] if region_sel else ["(sin datos)"], disabled=True)
            with c2:
                st.selectbox("🌍 Provincia", [prov_sel] if prov_sel else ["(sin datos)"], disabled=True)

            # Tipo market filtrado por cantón
            with c4:
                if has_market:
                    mp_local = admin_points_market[admin_points_market["canton"] == canton_sel]
                    tipos = sorted(mp_local["tipo_inmueble"].dropna().unique().tolist())
                    if not tipos:
                        tipos = sorted(admin_points_market["tipo_inmueble"].dropna().unique().tolist())
                else:
                    tipos = ["Desconocido"]

                tipo_default = st.session_state["f_tipo"] if st.session_state["f_tipo"] in tipos else tipos[0]
                tipo_sel = st.selectbox("🏠 Tipo de Propiedad (Market)", tipos, index=tipos.index(tipo_default))

            # Años (defendible)
            with c5:
                años_sel = st.slider("⏳ Proyección (Años)", 1, 10, int(st.session_state["f_years"]))

            submitted = st.form_submit_button("🔎 Buscar")

        # Guardar selección
        st.session_state["f_canton"] = canton_sel
        st.session_state["f_tipo"] = tipo_sel
        st.session_state["f_years"] = años_sel

        # ==========================================================
        # ✅ MAPA (debajo del form)
        # ==========================================================
        st.subheader("🗺️ Mapa del Ecuador (Inmovisión)")
        try:
            mapa = get_map(DATA_DIR)
            st.components.v1.html(mapa._repr_html_(), height=560, scrolling=True)
        except Exception as e:
            st.error(f"❌ Error renderizando mapa. Detalle: {e}")
            st.stop()

        # No buscar hasta apretar botón
        if not submitted:
            st.info("Selecciona variables y luego presiona **🔎 Buscar**.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        # ==========================================================
        # ✅ CÁLCULO (ADMIN) + RESULTADOS
        # ==========================================================
        ap_local = admin_points[admin_points["canton"] == canton_sel].copy()
        ap_local["precio_promedio"] = pd.to_numeric(ap_local.get("precio_promedio"), errors="coerce")
        ap_local["area_promedio"] = pd.to_numeric(ap_local.get("area_promedio"), errors="coerce")
        ap_local = ap_local.dropna(subset=["precio_promedio"])

        if ap_local.empty:
            st.warning("No hay datos suficientes para calcular precio en ese cantón.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        precio_actual_m2 = float(ap_local["precio_promedio"].median())

        # Total aprox con área mediana
        precio_actual_total = None
        area_med = None
        if "area_promedio" in ap_local.columns and ap_local["area_promedio"].notna().any():
            area_med = float(ap_local["area_promedio"].median())
            if area_med and area_med > 0:
                precio_actual_total = float(precio_actual_m2 * area_med)

        if area_med is not None and area_med > 200000:
            st.warning("⚠️ El área mediana es muy grande. Verifica unidades (m² vs hectáreas) o outliers.")

        # Growth por provincia (forecast_cfg)
        growth = forecast_cfg.get("growth_by_prov", {}).get(prov_sel, forecast_cfg.get("DEFAULT_GROWTH", 0.05))

        precio_futuro_m2 = float(precio_actual_m2 * ((1 + growth) ** años_sel))
        pct_m2 = float(((precio_futuro_m2 - precio_actual_m2) / precio_actual_m2) * 100)

        precio_futuro_total = None
        pct_total = None
        if precio_actual_total is not None:
            precio_futuro_total = float(precio_actual_total * ((1 + growth) ** años_sel))
            pct_total = float(((precio_futuro_total - precio_actual_total) / precio_actual_total) * 100)

        # Resultados
        colA, colB = st.columns([1.0, 1.0], gap="large")
        with colA:
            st.subheader("📊 Resultados (m²)")
            st.metric("Precio Actual (m²)", f"${precio_actual_m2:,.0f}")
            st.metric(f"Precio (m²) en {años_sel} años", f"${precio_futuro_m2:,.0f}", delta=f"{pct_m2:.1f}%")

        with colB:
            st.subheader("💰 Resultados (Total aprox.)")
            if precio_actual_total is not None and precio_futuro_total is not None and pct_total is not None:
                st.metric("Precio Actual (Total aprox.)", f"${precio_actual_total:,.0f}")
                st.metric(f"Precio Total en {años_sel} años", f"${precio_futuro_total:,.0f}", delta=f"{pct_total:.1f}%")
            else:
                st.info("No hay área suficiente para estimar total (solo m²).")
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📝 Análisis de Resultados")
        # -----------------------------
        # COMENTARIO ANALÍTICO (ancho)
        # -----------------------------
        comentario = build_commentary_analitico(
            region_sel=region_sel,
            prov_sel=prov_sel,
            canton_sel=canton_sel,
            tipo_sel=tipo_sel,
            años_sel=años_sel,
            precio_actual_m2=precio_actual_m2,
            precio_futuro_m2=precio_futuro_m2,
            precio_actual_total=precio_actual_total,
            precio_futuro_total=precio_futuro_total,
            area_med=area_med,
            growth=growth,
        )

        st.markdown(
            f"""
            <div style="
                background:#fff;
                color:#111;
                padding:18px 20px;
                border-radius:12px;
                border-left:6px solid #4A90E2;
                box-shadow:0 2px 10px rgba(0,0,0,0.10);
                margin-top:18px;
                width:100%;
                white-space:pre-line;
                line-height:1.55;
                font-size:16px;
            ">
                {html.escape(comentario)}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
# ==========================================================   
# --------------------------------------------------
# --------------------------------------------------
# SESSION + ARRANQUE DE LA APP (OBLIGATORIO)
# --------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    show_welcome_screen()
else:
    show_main_interface()



   

