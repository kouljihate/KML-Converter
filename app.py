import io
import tempfile
import zipfile
from pathlib import Path

import folium
import geopandas as gpd
import rasterio
import streamlit as st
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from streamlit_folium import st_folium


# ============================================================
# Streamlit application configuration
# ============================================================

st.set_page_config(
    page_title="KML / KMZ Converter",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Load stylesheets
# Theme first → application CSS second → local font is declared
# inside static/css/app.css
# ============================================================

st.markdown(
    """
    <link rel="stylesheet" href="/app/static/themes/bs-lumen.css">
    <link rel="stylesheet" href="/app/static/css/app.css">
    """,
    unsafe_allow_html=True,
)


# ============================================================
# File loading
# ============================================================

@st.cache_data(show_spinner=False)
def read_kml_or_kmz(file_bytes: bytes, filename: str) -> gpd.GeoDataFrame:
    """
    Load geographic features from a KML or KMZ file.

    KMZ is a ZIP archive; Google Earth commonly stores its main
    document as doc.kml. The function extracts it temporarily and
    reads it through GeoPandas/Pyogrio.
    """
    file_extension = Path(filename).suffix.lower()

    if file_extension not in {".kml", ".kmz"}:
        raise ValueError("Unsupported input. Upload a .kml or .kmz file.")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        if file_extension == ".kml":
            kml_path = temp_path / "uploaded_file.kml"
            kml_path.write_bytes(file_bytes)

        else:
            kmz_path = temp_path / "uploaded_file.kmz"
            kmz_path.write_bytes(file_bytes)

            with zipfile.ZipFile(kmz_path, "r") as kmz_archive:
                kml_files = [
                    archive_file
                    for archive_file in kmz_archive.namelist()
                    if archive_file.lower().endswith(".kml")
                    and not archive_file.endswith("/")
                ]

                if not kml_files:
                    raise ValueError(
                        "The KMZ archive does not contain a readable .kml file."
                    )

                selected_kml = next(
                    (
                        archive_file
                        for archive_file in kml_files
                        if Path(archive_file).name.lower() == "doc.kml"
                    ),
                    kml_files[0],
                )

                kml_path = temp_path / "document.kml"

                with kmz_archive.open(selected_kml) as source_file:
                    kml_path.write_bytes(source_file.read())

        try:
            gdf = gpd.read_file(kml_path, engine="pyogrio")
        except Exception as error:
            raise ValueError(
                "Unable to read this KML/KMZ file. "
                "Ensure Pyogrio/GDAL includes KML support and that the file "
                "contains valid geographic features."
            ) from error

    if gdf.empty:
        raise ValueError("No geographic features were found in the uploaded file.")

    # KML normally uses longitude/latitude WGS84 coordinates.
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    # Remove invalid/no-geometry records before map and conversion operations.
    gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    ].copy()

    if gdf.empty:
        raise ValueError("The file has no valid geometries to process.")

    return gdf


# ============================================================
# GeoTIFF conversion
# ============================================================

def create_geotiff(
    gdf: gpd.GeoDataFrame,
    output_crs: str,
    resolution_meters: float,
    inside_value: int,
    no_data_value: int,
) -> bytes:
    """
    Convert all vector geometries to a single-band raster mask.

    Raster cells touched by input geometry receive inside_value.
    Empty/background cells receive no_data_value.
    """
    if gdf.crs is None:
        raise ValueError("The input dataset has no coordinate reference system.")

    if resolution_meters <= 0:
        raise ValueError("Raster cell size must be greater than zero.")

    if not 0 <= inside_value <= 255:
        raise ValueError("Raster value inside features must be from 0 to 255.")

    if not 0 <= no_data_value <= 255:
        raise ValueError("No-data value must be from 0 to 255.")

    try:
        projected_gdf = gdf.to_crs(output_crs)
    except Exception as error:
        raise ValueError(
            f"Unable to reproject the data to '{output_crs}'. "
            "Enter a valid projected CRS, such as EPSG:32629."
        ) from error

    geometries = [
        geometry
        for geometry in projected_gdf.geometry
        if geometry is not None and not geometry.is_empty
    ]

    if not geometries:
        raise ValueError("No valid geometries are available to rasterize.")

    min_x, min_y, max_x, max_y = projected_gdf.total_bounds

    # One-pixel external margin avoids clipping features at the output border.
    min_x -= resolution_meters
    min_y -= resolution_meters
    max_x += resolution_meters
    max_y += resolution_meters

    raster_width = max(
        1,
        int((max_x - min_x) / resolution_meters) + 1,
    )
    raster_height = max(
        1,
        int((max_y - min_y) / resolution_meters) + 1,
    )

    raster_transform = from_bounds(
        min_x,
        min_y,
        max_x,
        max_y,
        raster_width,
        raster_height,
    )

    raster_data = rasterize(
        shapes=[
            (geometry, inside_value)
            for geometry in geometries
        ],
        out_shape=(raster_height, raster_width),
        transform=raster_transform,
        fill=no_data_value,
        all_touched=True,
        dtype="uint8",
    )

    profile = {
        "driver": "GTiff",
        "height": raster_height,
        "width": raster_width,
        "count": 1,
        "dtype": "uint8",
        "crs": output_crs,
        "transform": raster_transform,
        "nodata": no_data_value,
        "compress": "deflate",
    }

    output_buffer = io.BytesIO()

    with rasterio.open(output_buffer, "w", **profile) as output_file:
        output_file.write(raster_data, 1)

    return output_buffer.getvalue()


# ============================================================
# Interactive map helpers
# ============================================================

def get_popup_fields(gdf: gpd.GeoDataFrame) -> list[str]:
    """Choose existing KML attributes that are useful in map popups."""
    desired_order = [
        "Name",
        "name",
        "description",
        "Description",
        "id",
        "timestamp",
        "begin",
        "end",
        "altitudeMode",
        "tessellate",
        "extrude",
        "visibility",
        "drawOrder",
        "icon",
    ]

    available_fields = [
        field
        for field in desired_order
        if field in gdf.columns and field != "geometry"
    ]

    if available_fields:
        return available_fields

    return [
        column
        for column in gdf.columns
        if column != "geometry"
    ][:10]


def create_map(gdf: gpd.GeoDataFrame) -> folium.Map:
    """
    Build a Folium map displaying every geographic element:
    points/wells, lines/pipes, polygons/boundaries and multiparts.
    """
    map_gdf = gdf.to_crs("EPSG:4326").copy()
    min_x, min_y, max_x, max_y = map_gdf.total_bounds

    center = [
        (min_y + max_y) / 2,
        (min_x + max_x) / 2,
    ]

    map_view = folium.Map(
        location=center,
        tiles="OpenStreetMap",
        zoom_start=14,
        control_scale=True,
    )

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri World Imagery",
        name="Satellite imagery",
        overlay=False,
        control=True,
        show=False,
    ).add_to(map_view)

    popup_fields = get_popup_fields(map_gdf)
    popup_aliases = [
        field.replace("_", " ").title()
        for field in popup_fields
    ]

    tooltip_fields = [
        field
        for field in ["Name", "name", "id"]
        if field in map_gdf.columns
    ][:1]

    if not tooltip_fields and popup_fields:
        tooltip_fields = popup_fields[:1]

    folium.GeoJson(
        data=map_gdf.to_json(),
        name="All KML / KMZ features",
        style_function=lambda feature: {
            "color": "#158cba",
            "weight": 3,
            "opacity": 0.90,
            "fillColor": "#28a745",
            "fillOpacity": 0.30,
        },
        highlight_function=lambda feature: {
            "color": "#fd7e14",
            "weight": 5,
            "opacity": 1.0,
            "fillColor": "#ffc107",
            "fillOpacity": 0.55,
        },
        marker=folium.CircleMarker(
            radius=7,
            color="#dc3545",
            weight=2,
            fill=True,
            fill_color="#dc3545",
            fill_opacity=0.95,
        ),
        popup=folium.GeoJsonPopup(
            fields=popup_fields,
            aliases=popup_aliases,
            labels=True,
            localize=True,
            max_width=420,
        ) if popup_fields else None,
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=["Feature"],
            sticky=True,
            labels=True,
        ) if tooltip_fields else None,
        smooth_factor=0,
    ).add_to(map_view)

    map_view.fit_bounds([
        [min_y, min_x],
        [max_y, max_x],
    ])

    folium.LayerControl(collapsed=False).add_to(map_view)

    return map_view


# ============================================================
# Interface header and sidebar
# ============================================================

st.title("KML / KMZ → GeoJSON & GeoTIFF")

st.sidebar.header("GeoTIFF options")

generate_geotiff = st.sidebar.checkbox(
    "Create GeoTIFF",
    value=True,
)

cell_size = st.sidebar.number_input(
    "Raster cell size (meters)",
    min_value=0.5,
    max_value=1000.0,
    value=10.0,
    step=0.5,
)

target_crs = st.sidebar.text_input(
    "GeoTIFF CRS",
    value="EPSG:32629",
    help=(
        "Use a projected CRS measured in meters. "
        "EPSG:32629 is UTM Zone 29N, useful for much of western Morocco."
    ),
)

feature_value = st.sidebar.number_input(
    "Raster value inside features",
    min_value=0,
    max_value=255,
    value=1,
    step=1,
)

background_value = st.sidebar.number_input(
    "No-data value",
    min_value=0,
    max_value=255,
    value=0,
    step=1,
)


# ============================================================
# Upload interface and conversion workflow
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a KML or KMZ file",
    type=["kml", "kmz"],
    help="Supported input formats: Google Earth KML (.kml) and KMZ (.kmz).",
)

if uploaded_file is None:
    st.info("Upload a `.kml` or `.kmz` file to begin.")

else:
    try:
        with st.spinner("Reading geographic data..."):
            geo_data = read_kml_or_kmz(
                file_bytes=uploaded_file.getvalue(),
                filename=uploaded_file.name,
            )

        st.success(
            f"Loaded {len(geo_data)} geographic feature(s) "
            f"from `{uploaded_file.name}`."
        )

        # ----------------------------------------------------
        # Data preview
        # ----------------------------------------------------

        st.subheader("Data preview")

        preview_data = geo_data.drop(
            columns="geometry",
            errors="ignore",
        ).head(50)

        st.dataframe(
            preview_data,
            width="stretch",
            hide_index=True,
        )

        geometry_types = sorted(
            geo_data.geom_type.dropna().unique()
        )

        metric_1, metric_2, metric_3 = st.columns(3)

        metric_1.metric(
            "Features",
            len(geo_data),
        )

        metric_2.metric(
            "Geometry types",
            len(geometry_types),
            help=", ".join(geometry_types),
        )

        metric_3.metric(
            "Coordinate system",
            str(geo_data.crs),
        )

        st.caption(
            f"Geometry types detected: {', '.join(geometry_types)}"
        )

        st.divider()

        # ----------------------------------------------------
        # Full-width map preview
        # ----------------------------------------------------

        st.subheader("Map preview — all KML elements")

        with st.spinner("Preparing the interactive map..."):
            interactive_map = create_map(geo_data)

        st_folium(
            interactive_map,
            width=None,
            height=650,
            returned_objects=[],
            key=f"map-{uploaded_file.name}",
        )

        st.divider()

        # ----------------------------------------------------
        # Downloads
        # ----------------------------------------------------

        st.subheader("Download converted files")

        output_base_name = Path(uploaded_file.name).stem

        geojson_output = geo_data.to_json(
            drop_id=True,
        ).encode("utf-8")

        geojson_column, geotiff_column = st.columns(2)

        with geojson_column:
            st.download_button(
                label="Download GeoJSON",
                data=geojson_output,
                file_name=f"{output_base_name}.geojson",
                mime="application/geo+json",
                width="stretch",
            )

        with geotiff_column:
            if generate_geotiff:
                try:
                    with st.spinner("Creating GeoTIFF raster mask..."):
                        geotiff_output = create_geotiff(
                            gdf=geo_data,
                            output_crs=target_crs,
                            resolution_meters=float(cell_size),
                            inside_value=int(feature_value),
                            no_data_value=int(background_value),
                        )

                    st.download_button(
                        label="Download GeoTIFF",
                        data=geotiff_output,
                        file_name=f"{output_base_name}.tif",
                        mime="image/tiff",
                        width="stretch",
                    )

                    st.caption(
                        f"Mask values: `{feature_value}` for mapped features, "
                        f"`{background_value}` for background. "
                        f"Cell size: `{cell_size} m`."
                    )

                except Exception as geotiff_error:
                    st.error(
                        "GeoJSON conversion succeeded, but GeoTIFF generation "
                        f"failed: {geotiff_error}"
                    )

            else:
                st.info("GeoTIFF generation is disabled in the sidebar.")

    except Exception as upload_error:
        st.error(f"Could not read the file: {upload_error}")


# ============================================================
# Fixed footer
# All visual rules are defined in static/css/app.css.
# ============================================================

st.html(
    """
    <div class="footer-page-spacer"></div>

    <footer class="fixed-app-footer">
        <div class="fixed-footer-col fixed-footer-left">
            koul ©2026
        </div>

        <div class="fixed-footer-col fixed-footer-center">
            KML · KMZ · GeoJSON · GeoTIFF
        </div>

        <div class="fixed-footer-col fixed-footer-right">
            Version 1.0.0
        </div>
    </footer>
    """
)