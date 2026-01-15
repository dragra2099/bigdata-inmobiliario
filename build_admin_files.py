import os
import pandas as pd
import geopandas as gpd

# -----------------------------
# CONFIG
# -----------------------------
SPATIAL_DIR = "data"          # tu mapa actual
ADMIN_DIR = "data_admin"      # nuevos outputs
CANTON_GEOJSON = os.path.join(ADMIN_DIR, "gadm_ecuador_cantones.geojson")

os.makedirs(ADMIN_DIR, exist_ok=True)

# -----------------------------
# 1) Mapeo Provincia -> Región
# (ajústalo si quieres)
# -----------------------------
PROV_TO_REGION = {
    # Costa
    "Esmeraldas": "Costa", "Manabí": "Costa", "Guayas": "Costa", "Santa Elena": "Costa",
    "Los Ríos": "Costa", "El Oro": "Costa", "Santo Domingo De Los Tsáchilas": "Costa",
    # Sierra
    "Carchi": "Sierra", "Imbabura": "Sierra", "Pichincha": "Sierra", "Cotopaxi": "Sierra",
    "Tungurahua": "Sierra", "Chimborazo": "Sierra", "Bolívar": "Sierra", "Cañar": "Sierra",
    "Azuay": "Sierra", "Loja": "Sierra",
    # Oriente
    "Sucumbíos": "Oriente", "Napo": "Oriente", "Orellana": "Oriente", "Pastaza": "Oriente",
    "Morona Santiago": "Oriente", "Zamora Chinchipe": "Oriente",
    # Insular
    "Galápagos": "Insular",
}

def _norm_title(s: str) -> str:
    if pd.isna(s):
        return s
    s = str(s).strip()
    if not s:
        return s
    return s.title()

def _find_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

# -----------------------------
# 2) Cargar puntos (tu base real)
# -----------------------------
def load_points():
    path = os.path.join(SPATIAL_DIR, "clustering_features.parquet")
    df = pd.read_parquet(path)

    lat_col = _find_col(df, ["latitude", "lat"])
    lon_col = _find_col(df, ["longitude", "lon", "lng"])
    if not lat_col or not lon_col:
        raise ValueError("No encuentro columnas de lat/lon en clustering_features.")

    # columnas esperadas (tu caso)
    # latitude, longitude, precio_promedio, area_promedio, cluster, precio_total_usd, provincia, tipo_inmueble
    df = df.copy()
    df["provincia"] = df.get("provincia", None).apply(_norm_title)

    # GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"
    )
    return gdf

# -----------------------------
# 3) Cargar cantones (GADM level 2) y hacer spatial join
# -----------------------------
def add_canton_by_spatial_join(gdf_points):
    if not os.path.exists(CANTON_GEOJSON):
        raise FileNotFoundError(
            f"No existe {CANTON_GEOJSON}. Descarga GADM ECU nivel 2 (cantones) y guárdalo ahí."
        )

    gadm2 = gpd.read_file(CANTON_GEOJSON)
    if gadm2.crs is None:
        gadm2 = gadm2.set_crs("EPSG:4326")
    else:
        gadm2 = gadm2.to_crs("EPSG:4326")

    # nombres típicos GADM:
    # NAME_1 = provincia, NAME_2 = canton
    prov_col = _find_col(gadm2, ["NAME_1", "provincia"])
    cant_col = _find_col(gadm2, ["NAME_2", "canton", "cantón"])

    if not prov_col or not cant_col:
        raise ValueError("El geojson de cantones no tiene NAME_1/NAME_2 (o equivalentes).")

    gadm2 = gadm2[[prov_col, cant_col, "geometry"]].copy()
    gadm2 = gadm2.rename(columns={prov_col: "provincia_gadm", cant_col: "canton"})

    joined = gpd.sjoin(gdf_points, gadm2, how="left", predicate="within")

    joined["provincia_gadm"] = joined["provincia_gadm"].apply(_norm_title)
    joined["canton"] = joined["canton"].apply(_norm_title)

    # Si ya traías provincia por datos, la respetas; si no, usas la GADM
    if "provincia" not in joined.columns or joined["provincia"].isna().all():
        joined["provincia"] = joined["provincia_gadm"]
    else:
        joined["provincia"] = joined["provincia"].fillna(joined["provincia_gadm"])

    joined = joined.drop(columns=["provincia_gadm", "index_right"], errors="ignore")
    return joined

# -----------------------------
# 4) Región + columnas numéricas finales
# -----------------------------
def finalize_admin_columns(gdf):
    gdf = gdf.copy()
    gdf["provincia"] = gdf["provincia"].apply(_norm_title)
    gdf["region"] = gdf["provincia"].map(PROV_TO_REGION).fillna("Unknown")

    # asegurar numéricos
    for col in ["precio_promedio", "area_promedio", "precio_total_usd"]:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

    # “precio_total_usd” si no existe, lo aproximas con precio_promedio * area_promedio
    if "precio_total_usd" not in gdf.columns:
        if "precio_promedio" in gdf.columns and "area_promedio" in gdf.columns:
            gdf["precio_total_usd"] = gdf["precio_promedio"] * gdf["area_promedio"]

    return gdf

# -----------------------------
# 5) Agregaciones (promedios + conteos)
# -----------------------------
def build_aggregations(df_admin):
    base_metrics = {
        "n_registros": ("cluster", "size") if "cluster" in df_admin.columns else ("region", "size"),
        "precio_m2_prom": ("precio_promedio", "mean"),
        "area_prom_m2": ("area_promedio", "mean"),
        "precio_total_prom": ("precio_total_usd", "mean"),
    }

    # Región
    region_prices = (
        df_admin.groupby("region")
        .agg(**base_metrics)
        .reset_index()
        .sort_values("n_registros", ascending=False)
    )

    # Provincia
    province_prices = (
        df_admin.groupby(["region", "provincia"])
        .agg(**base_metrics)
        .reset_index()
        .sort_values("n_registros", ascending=False)
    )

    # Cantón
    canton_prices = (
        df_admin.groupby(["region", "provincia", "canton"])
        .agg(**base_metrics)
        .reset_index()
        .sort_values("n_registros", ascending=False)
    )

    # Conteos por tipo
    if "tipo_inmueble" in df_admin.columns:
        counts_tipo_region = (
            df_admin.groupby(["region", "tipo_inmueble"])
            .size().reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        counts_tipo_province = (
            df_admin.groupby(["region", "provincia", "tipo_inmueble"])
            .size().reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        counts_tipo_canton = (
            df_admin.groupby(["region", "provincia", "canton", "tipo_inmueble"])
            .size().reset_index(name="count")
            .sort_values("count", ascending=False)
        )
    else:
        counts_tipo_region = counts_tipo_province = counts_tipo_canton = None

    return (region_prices, province_prices, canton_prices,
            counts_tipo_region, counts_tipo_province, counts_tipo_canton)

# -----------------------------
# 6) Guardar outputs
# -----------------------------
def main():
    print("1) Cargando puntos...")
    gdf_points = load_points()

    print("2) Spatial join con cantones...")
    df_admin = add_canton_by_spatial_join(gdf_points)

    print("3) Finalizando columnas admin...")
    df_admin = finalize_admin_columns(df_admin)

    # Guardar base admin (con canton/region)
    admin_points_path = os.path.join(ADMIN_DIR, "admin_points.parquet")
    df_admin.drop(columns="geometry", errors="ignore").to_parquet(admin_points_path, index=False)
    print("✅ Guardado:", admin_points_path)

    print("4) Agregaciones...")
    (region_prices, province_prices, canton_prices,
     counts_tipo_region, counts_tipo_province, counts_tipo_canton) = build_aggregations(df_admin)

    region_prices.to_parquet(os.path.join(ADMIN_DIR, "region_prices.parquet"), index=False)
    province_prices.to_parquet(os.path.join(ADMIN_DIR, "province_prices.parquet"), index=False)
    canton_prices.to_parquet(os.path.join(ADMIN_DIR, "canton_prices.parquet"), index=False)

    print("✅ Guardados: region_prices / province_prices / canton_prices")

    if counts_tipo_region is not None:
        counts_tipo_region.to_parquet(os.path.join(ADMIN_DIR, "counts_tipo_region.parquet"), index=False)
        counts_tipo_province.to_parquet(os.path.join(ADMIN_DIR, "counts_tipo_province.parquet"), index=False)
        counts_tipo_canton.to_parquet(os.path.join(ADMIN_DIR, "counts_tipo_canton.parquet"), index=False)
        print("✅ Guardados: counts_tipo_region / counts_tipo_province / counts_tipo_canton")

    # Conteos “totales” (sin tipo) por nivel
    df_admin.assign(_one=1).groupby("region")["_one"].sum().reset_index(name="count") \
        .to_parquet(os.path.join(ADMIN_DIR, "counts_region.parquet"), index=False)

    df_admin.assign(_one=1).groupby(["region", "provincia"])["_one"].sum().reset_index(name="count") \
        .to_parquet(os.path.join(ADMIN_DIR, "counts_province.parquet"), index=False)

    df_admin.assign(_one=1).groupby(["region", "provincia", "canton"])["_one"].sum().reset_index(name="count") \
        .to_parquet(os.path.join(ADMIN_DIR, "counts_canton.parquet"), index=False)

    print("✅ Guardados: counts_region / counts_province / counts_canton")

    print("\nLISTO. Ya tienes capa administrativa sin tocar tu mapa.")

if __name__ == "__main__":
    main()
