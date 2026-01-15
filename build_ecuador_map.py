import os
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from branca.colormap import linear


def _safe_breaks_in_range(vmin, vmax, candidate_ticks):
    """Devuelve breaks que no se salgan del rango real (evita leyendas raras)."""
    ticks = [t for t in candidate_ticks if vmin < t < vmax]
    return sorted([float(vmin)] + ticks + [float(vmax)])


def build_ecuador_map(data_dir="data"):
    """
    data_dir puede ser:
      - "data" relativo
      - o una ruta absoluta (recomendado desde Streamlit)
    """

    # ---- rutas robustas (Windows/Streamlit) ----
    if not os.path.isabs(data_dir):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, data_dir)

    def p(fname):
        return os.path.join(data_dir, fname)

    # -----------------------------
    # Cargar artefactos persistentes
    # -----------------------------
    clustering_features = pd.read_parquet(p("clustering_features.parquet"))
    cluster_centroids = pd.read_parquet(p("cluster_centroids.parquet"))
    cluster_centroids_total = pd.read_parquet(p("cluster_centroids_total.parquet"))
    cluster_centroids_kmeans = pd.read_parquet(p("cluster_centroids_kmeans.parquet"))
    predominant = pd.read_parquet(p("predominant_provinces.parquet"))
    geodf_provinces = gpd.read_file(p("provinces_gadm.geojson"))

    # -----------------------------
    # Centro del mapa
    # -----------------------------
    mean_lat_all = cluster_centroids["latitude"].dropna().mean()
    mean_lon_all = cluster_centroids["longitude"].dropna().mean()
    m = folium.Map(location=[mean_lat_all, mean_lon_all], zoom_start=7)

    # -----------------------------
    # Escala cromática: Precio/m²
    # -----------------------------
    vmin_m2 = float(cluster_centroids["precio_promedio"].min())
    vmax_m2 = float(cluster_centroids["precio_promedio"].max())
    candidate_ticks_m2 = [500, 1000, 1500, 2000, 2500, 3000]
    price_m2_breaks = _safe_breaks_in_range(vmin_m2, vmax_m2, candidate_ticks_m2)

    colormap_m2 = linear.RdYlGn_08.scale(vmin=price_m2_breaks[0], vmax=price_m2_breaks[-1])
    colormap_m2.index = price_m2_breaks
    colormap_m2.caption = "Precio promedio por m² (USD)"

    # -----------------------------
    # Escala cromática: Precio total
    # -----------------------------
    min_total_price = float(cluster_centroids_total["precio_total_promedio"].min())
    max_total_price = float(cluster_centroids_total["precio_total_promedio"].max())

    total_price_breaks = np.linspace(min_total_price, max_total_price, 8).tolist()
    colormap_total = linear.PuBu_08.scale(vmin=total_price_breaks[0], vmax=total_price_breaks[-1])
    colormap_total.index = total_price_breaks
    colormap_total.caption = "Precio total promedio por cluster (USD)"

    # -----------------------------
    # Límites provinciales (GADM)
    # -----------------------------
    folium.GeoJson(
        geodf_provinces,
        name="Límites Provinciales (GADM)",
        style_function=lambda x: {"color": "#333333", "weight": 0.8, "fillOpacity": 0},
        overlay=True,
        control=True,
    ).add_to(m)

    # -----------------------------
    # Feature Groups
    # -----------------------------
    layer_centroids_m2 = folium.FeatureGroup(name="📍 Centroides (Precio/m²)", show=True)
    layer_centroids_total = folium.FeatureGroup(name="💰 Centroides (Precio Total)", show=False)
    layer_properties_new = folium.FeatureGroup(name="🏠 Propiedades Individuales", show=False)
    layer_density_new = folium.FeatureGroup(name="🔥 Densidad de Oferta", show=False)
    layer_price_hotspots_new = folium.FeatureGroup(name="💰 Hotspots de Precio (m²)", show=False)
    layer_clusters_new = folium.FeatureGroup(name="🧩 Clusters Espaciales", show=False)
    layer_zonas_kmeans = folium.FeatureGroup(name="🟣 Zonas Territoriales (KMeans)", show=False)

    # -----------------------------
    # Centroides (Precio/m²)
    # -----------------------------
    for _, row in cluster_centroids.iterrows():
        cluster_id = row["cluster"]
        mean_lat = row["latitude"]
        mean_lon = row["longitude"]
        avg_price_m2 = row["precio_promedio"]

        provs = predominant[predominant["cluster"] == cluster_id]["provincia"].tolist()
        provinces_str = ", ".join(provs)

        popup_text = (
            f"<b>Cluster ID:</b> {cluster_id}<br>"
            f"<b>Precio promedio por m²:</b> ${avg_price_m2:,.2f}<br>"
            f"<b>Provincias predominantes:</b> {provinces_str}"
        )

        folium.CircleMarker(
            location=[mean_lat, mean_lon],
            radius=10,
            color=colormap_m2(avg_price_m2),
            fill=True,
            fill_color=colormap_m2(avg_price_m2),
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=300),
        ).add_to(layer_centroids_m2)

    # -----------------------------
    # Centroides (Precio Total)
    # -----------------------------
    for _, row in cluster_centroids_total.iterrows():
        cluster_id = row["cluster"]
        mean_lat = row["latitude"]
        mean_lon = row["longitude"]
        avg_total_price = row["precio_total_promedio"]

        provs = predominant[predominant["cluster"] == cluster_id]["provincia"].tolist()
        provinces_str = ", ".join(provs)

        popup_text = (
            f"<b>Cluster ID:</b> {cluster_id}<br>"
            f"<b>Precio total promedio:</b> ${avg_total_price:,.2f}<br>"
            f"<b>Provincias predominantes:</b> {provinces_str}"
        )

        folium.CircleMarker(
            location=[mean_lat, mean_lon],
            radius=10,
            color=colormap_total(avg_total_price),
            fill=True,
            fill_color=colormap_total(avg_total_price),
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=300),
        ).add_to(layer_centroids_total)

    # -----------------------------
    # Propiedades individuales
    # -----------------------------
    for _, row in clustering_features.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3,
            color="gray",
            fill=True,
            fill_opacity=0.4,
            popup=(f"Precio por m²: ${row['precio_promedio']:,.0f}<br>" f"Cluster: {row['cluster']}"),
        ).add_to(layer_properties_new)

    # -----------------------------
    # Heatmap densidad
    # -----------------------------
    heat_data = clustering_features[["latitude", "longitude"]].values.tolist()
    HeatMap(heat_data, radius=12, blur=15, max_zoom=10).add_to(layer_density_new)

    # -----------------------------
    # Heatmap precios (m²)
    # -----------------------------
    heat_price_data = clustering_features[["latitude", "longitude", "precio_promedio"]].values.tolist()
    HeatMap(heat_price_data, radius=18, blur=20, max_zoom=10).add_to(layer_price_hotspots_new)

    # -----------------------------
    # Zonas territoriales (KMeans)
    # -----------------------------
    zone_colors = {"Bajo costo": "#2c7bb6", "Emergente": "#fdae61", "Premium": "#d7191c"}

    for _, row in cluster_centroids_kmeans.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        zona = row["zona"]
        avg = row["precio_promedio_cluster"]

        popup = (
            f"<b>Zona territorial:</b> {zona}<br>"
            f"<b>Cluster KMeans:</b> {row['cluster']}<br>"
            f"<b>Precio promedio:</b> ${avg:,.0f}"
        )

        folium.CircleMarker(
            location=[lat, lon],
            radius=12,
            color=zone_colors.get(zona, "gray"),
            fill=True,
            fill_color=zone_colors.get(zona, "gray"),
            fill_opacity=0.75,
            popup=folium.Popup(popup, max_width=350),
        ).add_to(layer_zonas_kmeans)

    # -----------------------------
    # Visualización clusters (colores)
    # -----------------------------
    colors = ["red", "blue", "green", "purple", "orange", "darkred", "cadetblue"]
    for _, row in clustering_features.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            color=colors[int(row["cluster"]) % len(colors)],
            fill=True,
            fill_opacity=0.6,
        ).add_to(layer_clusters_new)

    # -----------------------------
    # Agregar capas al mapa
    # -----------------------------
    layer_centroids_m2.add_to(m)
    layer_centroids_total.add_to(m)
    layer_properties_new.add_to(m)
    layer_density_new.add_to(m)
    layer_price_hotspots_new.add_to(m)
    layer_clusters_new.add_to(m)
    layer_zonas_kmeans.add_to(m)

    # -----------------------------
    # CSS (leyenda)
    # -----------------------------
    m.get_root().html.add_child(
        folium.Element(
            """
            <style>
            .leaflet-control .legend {
                position: absolute;
                bottom: 40px;
                left: 50px;
                z-index: 9999;
            }
            </style>
            """
        )
    )

    # Controles finales
    m.add_child(colormap_m2)
    m.add_child(colormap_total)
    folium.LayerControl(collapsed=False).add_to(m)

    return m
