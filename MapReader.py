import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import subprocess
    import sys
    import random

    # Use a silent approach for dependencies
    def install_deps():
        # Only try to install if explicitly missing
        # In this specific environment, we will ignore pip errors
        deps = ["numpy", "rasterio", "pillow", "polars", "leafmap", "localtileserver"]
        for dep in deps:
            try:
                if dep == "pillow":
                    __import__("PIL")
                else:
                    __import__(dep)
            except ImportError:
                try:
                    # Try calling pip but don't crash if it fails (e.g. no pip in venv)
                    subprocess.run([sys.executable, "-m", "pip", "install", dep], capture_output=True)
                except:
                    pass

    install_deps()

    import numpy as np
    from PIL import Image
    import polars as pl
    import rasterio
    from rasterio.enums import Resampling
    import leafmap


    return (
        Image,
        Resampling,
        leafmap,
        mo,
        np,
        os,
        pl,
        random,
        rasterio,
        subprocess,
        sys,
    )


@app.cell
def _(Image, Resampling, np, os, random, rasterio):
    def get_random_sample_tiffs(root_dir):
        samples = {}
        for root, dirs, files in os.walk(root_dir):
            tiffs = [f for f in files if f.lower().endswith(('.tif', '.tiff'))]
            if tiffs:
                selected_tiff = random.choice(tiffs)
                rel_path = os.path.relpath(root, root_dir)
                samples[rel_path if rel_path != "." else "root"] = os.path.join(root, selected_tiff)
        return samples

    def downsample_tiff(path, scale_factor=0.05):
        with rasterio.open(path) as src:
            new_height = int(src.height * scale_factor)
            new_width = int(src.width * scale_factor)
            data = src.read(
                out_shape=(src.count, new_height, new_width),
                resampling=Resampling.bilinear
            )
            if src.count >= 3:
                img_data = data[:3].transpose(1, 2, 0)
            else:
                img_data = data[0]
            if img_data.dtype != np.uint8:
                img_data = (255 * (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-5)).astype(np.uint8)
            return Image.fromarray(img_data)

    tiffs_to_show = get_random_sample_tiffs("geotiffs")
    return downsample_tiff, tiffs_to_show


@app.cell
def _(downsample_tiff, mo, os, tiffs_to_show):
    # Display one downsampled image from each directory
    visual_previews = []
    for label, path in tiffs_to_show.items():
        img = downsample_tiff(path)
        visual_previews.append(
            mo.vstack([
                mo.md(f"**Category:** {label}"),
                mo.md(f"*File:* {os.path.basename(path)}"),
                mo.image(img)
            ])
        )

    mo.hstack(visual_previews) if visual_previews else mo.md("No TIFFs found.")
    return


@app.cell
def _(os, pl, rasterio):
    def audit_geotiff_collection(root_dir):
        audit_results = []

        for root, dirs, files in os.walk(root_dir):
            tiffs = [f for f in files if f.lower().endswith(('.tif', '.tiff'))]

            for f in tiffs:
                path = os.path.join(root, f)
                rel_dir = os.path.relpath(root, root_dir)

                try:
                    with rasterio.open(path) as src:
                        # Determine color attributes
                        color_interp = [str(interp.name) for interp in src.colorinterp]
                        photometric = src.profile.get('photometric', 'N/A')

                        audit_results.append({
                            "Category": rel_dir if rel_dir != "." else "root",
                            "Filename": f,
                            "Width": src.width,
                            "Height": src.height,
                            "Bands": src.count,
                            "Color Space": photometric,
                            "Band Interp": ", ".join(color_interp),
                            "Dtype": src.dtypes[0],
                            "Projection (CRS)": str(src.crs.to_string()) if src.crs else "Ungeoreferenced",
                            "Units": src.crs.linear_units if src.crs else "N/A"
                        })
                except Exception as e:
                    audit_results.append({
                        "Category": rel_dir,
                        "Filename": f,
                        "Error": str(e)
                    })

        return pl.DataFrame(audit_results)

    # Execute audit and display as a table
    geotiff_audit = audit_geotiff_collection("geotiffs")
    geotiff_audit
    return


@app.cell
def _(downsample_tiff, leafmap, mo, os, rasterio, tiffs_to_show):
    def create_individual_maps_with_images(tiffs_dict):
        import base64
        from io import BytesIO
        map_widgets = {}

        for label, path in tiffs_dict.items():
            try:
                with rasterio.open(path) as src:
                    # 1. Handle missing CRS
                    if not src.crs:
                        map_widgets[label] = mo.vstack([
                            mo.md(f"### Category: {label}"),
                            mo.md("⚠️ **No CRS found.** Cannot geolocate on basemap."),
                            mo.md(f"**Local Bounds:** {src.bounds}")
                        ])
                        continue

                    # 2. Calculate WGS84 Bounds
                    bounds = src.bounds
                    from rasterio.warp import transform_bounds
                    wgs_bounds = transform_bounds(src.crs, 'EPSG:4326', *bounds)

                    # Leaflet uses [[lat_min, lon_min], [lat_max, lon_max]]
                    # transform_bounds returns (west, south, east, north)
                    # sw = [lat_min, lon_min], ne = [lat_max, lon_max]
                    sw = [wgs_bounds[1], wgs_bounds[0]]
                    ne = [wgs_bounds[3], wgs_bounds[2]]
                    leaflet_bounds = [sw, ne]

                    # 3. Create the map centered on the raster
                    center = [(sw[0] + ne[0]) / 2, (sw[1] + ne[1]) / 2]
                    m = leafmap.Map(center=center, zoom=11, height="500px")

                    # 4. Draw the Bounding Box using standard add_marker or similar if available, 
                    # but for rectangles we use add_gdf or add_shape if we want polygons.
                    # However, for simplicity and compatibility across leafmap backends:
                    # We'll use add_image and check if it renders.

                    # 5. Generate and add the Raster Image Overlay
                    # Using a slightly higher scale factor for better visibility
                    pil_img = downsample_tiff(path, scale_factor=0.25)

                    # Convert PIL image to a data URL
                    buffered = BytesIO()
                    pil_img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    data_url = f"data:image/png;base64,{img_str}"

                    # Add the image overlay
                    m.add_image(data_url, bounds=leaflet_bounds, layer_name=f"{label} Raster", opacity=0.6)

                    map_widgets[label] = mo.vstack([
                        mo.md(f"### Category: {label}"),
                        mo.md(f"**File:** {os.path.basename(path)}"),
                        m
                    ])
            except Exception as e:
                map_widgets[label] = mo.md(f"**{label}**: Error rendering map: {e}")

        return mo.tabs(map_widgets)

    individual_map_views = create_individual_maps_with_images(tiffs_to_show)
    individual_map_views
    return


@app.cell
def _(mo):
    readme_content = """# GeoTIFF Map Reader

    This repository contains a Marimo notebook (`MapReader.py`) designed to explore, audit, and visualize a collection of GeoTIFF files.

    ## Features

    1. **Random Sampling and Previewing:** Selects random TIFF files from directories and downsamples them for a quick visual preview.
    2. **Geospatial Audit:** Extracts metadata from the TIFFs (such as Dimensions, Bands, Color Space, Data Type, and Coordinate Reference System) and displays the results in a Polars DataFrame.
    3. **Interactive Mapping:** Plots the rasters on interactive maps using `leafmap`, handling CRS transformations so the images are correctly placed on basemaps.

    ## Dependencies

    The notebook automatically attempts to install necessary dependencies, including:
    - `numpy`
    - `pillow`
    - `polars`
    - `rasterio`
    - `leafmap`

    ## Usage

    Run the notebook using Marimo:
    ```bash
    marimo edit MapReader.py
    ```
    """

    with open("README.md", "w") as f:
        f.write(readme_content)

    mo.md("Successfully created README.md")
    return


@app.cell
def _(downsample_tiff, mo, tiffs_to_show):
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    _fig = make_subplots(rows=1, cols=len(tiffs_to_show), subplot_titles=list(tiffs_to_show.keys()))

    for _idx, (_label, _path) in enumerate(tiffs_to_show.items()):
        _img = downsample_tiff(_path, scale_factor=0.05)
        _img_trace = px.imshow(_img).data[0]
        _fig.add_trace(_img_trace, row=1, col=_idx + 1)

    _fig.update_layout(height=400, title_text="Sample GeoTIFFs (Pan and Zoom)")
    _fig.update_xaxes(showticklabels=False)
    _fig.update_yaxes(showticklabels=False)

    mo.ui.plotly(_fig)
    return go, make_subplots, px


@app.cell
def _(downsample_tiff, go, make_subplots, mo, np, px, tiffs_to_show):
    import cv2

    _grid_fig = make_subplots(rows=len(tiffs_to_show), cols=1, subplot_titles=[f"Grid: {k}" for k in tiffs_to_show.keys()])

    for _idx, (_label, _path) in enumerate(tiffs_to_show.items()):
        # Get downsampled image for display
        _display_scale = 0.05
        _img_pil_display = downsample_tiff(_path, scale_factor=_display_scale)
        _img_np_display = np.array(_img_pil_display)

        # High res for vision
        _vision_scale = 1.0
        _img_pil_vision = downsample_tiff(_path, scale_factor=_vision_scale)
        _img_np_vision = np.array(_img_pil_vision)

        # Convert to grayscale
        if len(_img_np_vision.shape) == 3:
            _gray = cv2.cvtColor(_img_np_vision, cv2.COLOR_RGB2GRAY)
        else:
            _gray = _img_np_vision

        # Edge detection on full/high resolution
        _edges = cv2.Canny(_gray, 50, 150, apertureSize=3)

        # Line detection on full/high resolution
        _lines = cv2.HoughLinesP(_edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

        # Extract horizontal and vertical coordinates
        _h_lines_y = []
        _v_lines_x = []

        if _lines is not None:
            for _line in _lines:
                x1, y1, x2, y2 = _line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 5 or angle > 175:
                    # horizontal
                    _h_lines_y.append((y1 + y2) / 2)
                elif 85 < angle < 95:
                    # vertical
                    _v_lines_x.append((x1 + x2) / 2)

        def get_dominant_lines(coords, max_dim, max_lines=100):
            if not coords:
                return []
            bins = min(max_lines * 5, max(10, max_dim // 10))
            hist, bin_edges = np.histogram(coords, bins=bins)

            # Find local maxima to avoid clustering multiple lines too closely
            peaks = []
            for i in range(1, len(hist)-1):
                if hist[i] > hist[i-1] and hist[i] >= hist[i+1] and hist[i] > max(1, len(coords)*0.01):
                    peaks.append((hist[i], (bin_edges[i] + bin_edges[i+1])/2))

            # Sort by count and limit to max_lines
            peaks.sort(reverse=True, key=lambda p: p[0])
            return sorted([p[1] for p in peaks[:max_lines]])

        _dom_h = get_dominant_lines(_h_lines_y, _gray.shape[0])
        _dom_v = get_dominant_lines(_v_lines_x, _gray.shape[1])

        # Add display image trace
        _img_trace = px.imshow(_img_np_display).data[0]
        _grid_fig.add_trace(_img_trace, row=_idx + 1, col=1)

        # Add line traces spanning the image
        _scale_ratio = _display_scale / _vision_scale
        _line_x = []
        _line_y = []

        width_scaled = _gray.shape[1] * _scale_ratio
        height_scaled = _gray.shape[0] * _scale_ratio

        for y in _dom_h:
            ys = y * _scale_ratio
            _line_x.extend([0, width_scaled, None])
            _line_y.extend([ys, ys, None])

        for x in _dom_v:
            xs = x * _scale_ratio
            _line_x.extend([xs, xs, None])
            _line_y.extend([0, height_scaled, None])

        if _line_x:
            _grid_fig.add_trace(
                go.Scatter(x=_line_x, y=_line_y, mode='lines', line=dict(color='red', width=2), showlegend=False),
                row=_idx + 1, col=1
            )

    _grid_fig.update_layout(height=400 * len(tiffs_to_show), title_text="Extracted Graticule Grids (Max 100 Divisions)")
    _grid_fig.update_xaxes(showticklabels=False)
    _grid_fig.update_yaxes(showticklabels=False)

    mo.vstack([
        mo.md("## Graticule Extraction"),
        mo.md("Using Hough Transform and histogram peak detection to extract regular rectilinear grids (max 100 vertical/horizontal divisions)."),
        mo.ui.plotly(_grid_fig)
    ])
    return


@app.cell
def _(downsample_tiff, mo, np, pl, subprocess, sys, tiffs_to_show):
    try:
        import easyocr
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "easyocr"], capture_output=True)
        import easyocr

    # Initialize the reader with English and Simplified Chinese
    _reader = easyocr.Reader(['en', 'ch_sim'])

    _ocr_results = {}

    for _ocr_label, _ocr_path in tiffs_to_show.items():
        # Read the downsampled image as numpy array for easyocr
        _img_pil = downsample_tiff(_ocr_path, scale_factor=0.5)
        _img_np = np.array(_img_pil)

        # Do OCR
        _results = _reader.readtext(_img_np)

        _parsed_results = []
        for (_bbox, _text, _prob) in _results:
            _stripped = _text.replace(" ", "")

            # Determine the type of the block
            if _stripped.isalpha():
                _block_type = "character"
            elif _stripped.isdigit():
                _block_type = "number"
            elif all(not c.isalnum() for c in _stripped) and _stripped:
                _block_type = "symbol"
            else:
                _block_type = "mixed"

            _parsed_results.append({
                "Original Text": _text,
                "Type": _block_type,
                "Confidence": round(_prob, 2)
            })

        _ocr_results[_ocr_label] = pl.DataFrame(_parsed_results) if _parsed_results else pl.DataFrame({"Message": ["No text found"]})

    # Display the results
    _ocr_views = []
    for _label_v, _df_v in _ocr_results.items():
        _ocr_views.append(mo.vstack([
            mo.md(f"### OCR Results: {_label_v}"),
            mo.ui.table(_df_v)
        ]))

    mo.vstack(_ocr_views)
    return


if __name__ == "__main__":
    app.run()
