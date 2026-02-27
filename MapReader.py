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

    # All third-party dependencies in one place
    _deps = {
        "numpy": "numpy",
        "PIL": "pillow",
        "polars": "polars",
        "rasterio": "rasterio",
        "leafmap": "leafmap",
        "localtileserver": "localtileserver",
        "plotly": "plotly",
        "cv2": "opencv-python",
        "easyocr": "easyocr",
        "scipy": "scipy",
    }

    for _import_name, _pip_name in _deps.items():
        try:
            __import__(_import_name)
        except ImportError:
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", _pip_name],
                    capture_output=True,
                )
            except Exception:
                pass

    import numpy as np
    from PIL import Image
    import polars as pl
    import rasterio
    from rasterio.enums import Resampling
    import leafmap
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import cv2
    import easyocr

    return (
        Image,
        Resampling,
        cv2,
        easyocr,
        go,
        leafmap,
        make_subplots,
        mo,
        np,
        os,
        pl,
        px,
        random,
        rasterio,
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
def _(downsample_tiff, make_subplots, mo, px, tiffs_to_show):
    # preview current maps
    _fig = make_subplots(rows=1, cols=len(tiffs_to_show), subplot_titles=list(tiffs_to_show.keys()))

    for _idx, (_label, _path) in enumerate(tiffs_to_show.items()):
        _img = downsample_tiff(_path, scale_factor=0.05)
        _img_trace = px.imshow(_img).data[0]
        _fig.add_trace(_img_trace, row=1, col=_idx + 1)

    _fig.update_layout(height=400, title_text="Sample GeoTIFFs (Pan and Zoom)")
    _fig.update_xaxes(showticklabels=False)
    _fig.update_yaxes(showticklabels=False)

    mo.ui.plotly(_fig)
    return


@app.cell
def _(cv2, downsample_tiff, go, make_subplots, mo, np, px, rasterio, tiffs_to_show):
    from scipy.ndimage import map_coordinates
    from scipy.signal import find_peaks
    from rasterio.warp import transform_bounds, transform
    from rasterio.transform import rowcol

    # --- Helper functions ---

    def _detect_neatline(gray_img):
        """Find the map's inner boundary to exclude collar/margins."""
        h, w = gray_img.shape
        row_means = gray_img.mean(axis=1)
        col_means = gray_img.mean(axis=0)

        def _find_edge(profile, length):
            """Find strongest gradient in each half of a 1-D intensity profile."""
            grad = np.abs(np.diff(profile.astype(float)))
            mid = length // 2
            lo = int(np.argmax(grad[:mid])) if grad[:mid].max() > 10 else int(length * 0.05)
            hi_region = grad[mid:]
            hi = mid + int(np.argmax(hi_region)) if hi_region.max() > 10 else int(length * 0.95)
            return lo, hi

        row_min, row_max = _find_edge(row_means, h)
        col_min, col_max = _find_edge(col_means, w)
        return row_min, row_max, col_min, col_max

    def _generate_crs_candidates(path, analysis_scale):
        """Generate candidate graticule lines at standard cartographic intervals."""
        try:
            with rasterio.open(path) as src:
                if not src.crs:
                    return None
                wgs_bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                lon_min, lat_min, lon_max, lat_max = wgs_bounds
                lon_span = lon_max - lon_min
                lat_span = lat_max - lat_min

                full_transform = src.transform
                map_crs = src.crs
                full_h, full_w = src.height, src.width
        except Exception:
            return None

        intervals = [10, 5, 2, 1, 0.5, 1/4, 1/6, 1/12]
        candidates = []

        for interval in intervals:
            n_meridians = int(lon_span / interval) - 1
            n_parallels = int(lat_span / interval) - 1
            total = n_meridians + n_parallels
            if total < 2 or total > 50:
                continue

            meridian_curves = []
            first_lon = np.ceil(lon_min / interval) * interval
            for i in range(n_meridians):
                lon = first_lon + i * interval
                if lon <= lon_min or lon >= lon_max:
                    continue
                lats = np.linspace(lat_min, lat_max, 100)
                lons = np.full_like(lats, lon)
                try:
                    xs, ys = transform("EPSG:4326", map_crs, lons, lats)
                    rows, cols = rowcol(full_transform, xs, ys)
                    rows = np.array(rows, dtype=float) * analysis_scale
                    cols = np.array(cols, dtype=float) * analysis_scale
                    meridian_curves.append((rows, cols))
                except Exception:
                    continue

            parallel_curves = []
            first_lat = np.ceil(lat_min / interval) * interval
            for i in range(n_parallels):
                lat = first_lat + i * interval
                if lat <= lat_min or lat >= lat_max:
                    continue
                lons = np.linspace(lon_min, lon_max, 100)
                lats = np.full_like(lons, lat)
                try:
                    xs, ys = transform("EPSG:4326", map_crs, lons, lats)
                    rows, cols = rowcol(full_transform, xs, ys)
                    rows = np.array(rows, dtype=float) * analysis_scale
                    cols = np.array(cols, dtype=float) * analysis_scale
                    parallel_curves.append((rows, cols))
                except Exception:
                    continue

            if meridian_curves or parallel_curves:
                candidates.append((interval, meridian_curves, parallel_curves))

        return candidates if candidates else None

    def _score_candidate(edge_img, neatline, meridians, parallels):
        """Score how well a candidate interval matches actual image edges."""
        row_min, row_max, col_min, col_max = neatline
        h, w = edge_img.shape
        total_hits = 0
        total_samples = 0

        all_curves = list(meridians) + list(parallels)
        if not all_curves:
            return 0.0

        for rows, cols in all_curves:
            for offset in [-2, -1, 0, 1, 2]:
                # For meridians (vertical), offset applies to cols; for parallels, to rows
                sample_rows = np.clip(rows + offset * 0.5, 0, h - 1)
                sample_cols = np.clip(cols + offset * 0.5, 0, w - 1)
                # Only sample within neatline
                mask = (sample_rows >= row_min) & (sample_rows <= row_max) & \
                       (sample_cols >= col_min) & (sample_cols <= col_max)
                if mask.sum() == 0:
                    continue
                vals = map_coordinates(edge_img, [sample_rows[mask], sample_cols[mask]], order=0)
                total_hits += (vals > 0).sum()
                total_samples += mask.sum()

        if total_samples == 0:
            return 0.0

        edge_density = total_hits / total_samples
        n_lines = len(all_curves)
        coverage = min(1.0, n_lines / 4.0)
        return edge_density * coverage

    def _detect_graticule_pixel_space(gray_img, edge_img, neatline):
        """Fallback for ungeoreferenced TIFFs — find regular spacing via FFT."""
        row_min, row_max, col_min, col_max = neatline
        cropped = edge_img[row_min:row_max, col_min:col_max]
        if cropped.size == 0:
            return [], []

        # Project edges onto row and column axes
        row_profile = cropped.mean(axis=1).astype(float)
        col_profile = cropped.mean(axis=0).astype(float)

        def _find_spacing(profile):
            if len(profile) < 8:
                return []
            # FFT to find dominant periodic spacing
            fft_vals = np.abs(np.fft.rfft(profile - profile.mean()))
            # Ignore DC and very low frequencies (spacing > half the profile)
            fft_vals[:2] = 0
            dominant_freq_idx = np.argmax(fft_vals)
            if dominant_freq_idx == 0:
                return []
            spacing = len(profile) / dominant_freq_idx

            if spacing < 10 or spacing > len(profile) / 2:
                return []

            # Find actual peaks at roughly this spacing
            min_dist = int(spacing * 0.5)
            peaks, _ = find_peaks(profile, distance=max(1, min_dist), height=profile.mean())

            if len(peaks) < 2:
                return []

            # Validate regularity: reject peaks deviating >30% from median spacing
            diffs = np.diff(peaks)
            if len(diffs) == 0:
                return []
            median_sp = np.median(diffs)
            if median_sp == 0:
                return []
            regular = np.abs(diffs - median_sp) / median_sp < 0.3
            # Keep only peaks connected by regular intervals
            valid = [peaks[0]]
            for i, is_reg in enumerate(regular):
                if is_reg:
                    valid.append(peaks[i + 1])
            return valid

        h_positions = _find_spacing(row_profile)
        v_positions = _find_spacing(col_profile)

        # Convert from cropped coords back to full image coords
        h_positions = [p + row_min for p in h_positions]
        v_positions = [p + col_min for p in v_positions]
        return h_positions, v_positions

    def _format_interval(deg):
        """Format degree value as human-readable string."""
        minutes = deg * 60
        if abs(deg - round(deg)) < 1e-6 and deg >= 1:
            return f"{int(round(deg))}\u00b0"
        if abs(minutes - round(minutes)) < 0.1:
            return f"{int(round(minutes))}'"
        return f"{deg:.4f}\u00b0"

    # --- Main loop ---
    _analysis_scale = 0.15
    _display_scale = 0.05
    _grid_fig = make_subplots(
        rows=len(tiffs_to_show), cols=1,
        subplot_titles=[f"Grid: {k}" for k in tiffs_to_show.keys()]
    )
    _summaries = []

    for _idx, (_label, _path) in enumerate(tiffs_to_show.items()):
        # Load at analysis and display scales
        _img_analysis = np.array(downsample_tiff(_path, scale_factor=_analysis_scale))
        _img_display = np.array(downsample_tiff(_path, scale_factor=_display_scale))

        # Grayscale + Canny
        if len(_img_analysis.shape) == 3:
            _gray = cv2.cvtColor(_img_analysis, cv2.COLOR_RGB2GRAY)
        else:
            _gray = _img_analysis
        _edges = cv2.Canny(_gray, 50, 150, apertureSize=3)

        # Detect neatline
        _neatline = _detect_neatline(_gray)

        # Try CRS-aware path
        _method = None
        _interval_str = "N/A"
        _draw_curves = []
        _line_color = "cyan"
        _h_count = 0
        _v_count = 0

        _candidates = _generate_crs_candidates(_path, _analysis_scale)
        if _candidates:
            _best_score = 0.0
            _best = None
            for _interval, _meridians, _parallels in _candidates:
                _s = _score_candidate(_edges, _neatline, _meridians, _parallels)
                if _s > _best_score:
                    _best_score = _s
                    _best = (_interval, _meridians, _parallels)

            if _best is not None and _best_score > 0.02:
                _method = "CRS-aware"
                _interval_str = _format_interval(_best[0])
                _line_color = "cyan"
                # Collect curves for drawing (meridians + parallels)
                for rows, cols in _best[1]:
                    _draw_curves.append((rows, cols))
                    _v_count += 1
                for rows, cols in _best[2]:
                    _draw_curves.append((rows, cols))
                    _h_count += 1

        # Pixel-space fallback
        if _method is None:
            _h_positions, _v_positions = _detect_graticule_pixel_space(_gray, _edges, _neatline)
            _method = "Pixel-space (FFT)"
            _line_color = "red"
            _h_count = len(_h_positions)
            _v_count = len(_v_positions)
            # Convert to curves for uniform drawing
            a_h, a_w = _gray.shape
            for y in _h_positions:
                _draw_curves.append(
                    (np.array([y, y], dtype=float), np.array([0, a_w - 1], dtype=float))
                )
            for x in _v_positions:
                _draw_curves.append(
                    (np.array([0, a_h - 1], dtype=float), np.array([x, x], dtype=float))
                )

        _summaries.append(
            f"**{_label}**: {_method} | interval: {_interval_str} | "
            f"{_h_count} parallels, {_v_count} meridians"
        )

        # --- Draw on plotly figure ---
        _img_trace = px.imshow(_img_display).data[0]
        _grid_fig.add_trace(_img_trace, row=_idx + 1, col=1)

        _scale_ratio = _display_scale / _analysis_scale
        _line_x = []
        _line_y = []
        for rows, cols in _draw_curves:
            _line_x.extend((cols * _scale_ratio).tolist() + [None])
            _line_y.extend((rows * _scale_ratio).tolist() + [None])

        if _line_x:
            _grid_fig.add_trace(
                go.Scatter(
                    x=_line_x, y=_line_y, mode='lines',
                    line=dict(color=_line_color, width=2), showlegend=False
                ),
                row=_idx + 1, col=1
            )

    _grid_fig.update_layout(
        height=400 * len(tiffs_to_show),
        title_text="CRS-Aware Graticule Detection"
    )
    _grid_fig.update_xaxes(showticklabels=False)
    _grid_fig.update_yaxes(showticklabels=False)

    mo.vstack([
        mo.md("## Graticule Extraction"),
        mo.md("CRS-aware detection scores projected candidate grids against edge evidence. "
               "Cyan lines = CRS-aware, Red lines = pixel-space FFT fallback."),
        mo.md("\n\n".join(_summaries)),
        mo.ui.plotly(_grid_fig)
    ])
    return


@app.cell
def _(downsample_tiff, easyocr, mo, np, pl, tiffs_to_show):
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
