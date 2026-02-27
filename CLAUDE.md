# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Geospatial AI Map Reader for UCSB DreamLab. Transforms scans of paper maps into structured, queryable data using machine vision and computational cartography. Built as interactive Marimo notebooks for exploring, auditing, and visualizing GeoTIFF collections.

Serves as new implementations of [MapReader](https://mapreader.readthedocs.io/en/latest/) and [MapKurator](https://knowledge-computing.github.io/mapkurator-doc/#/docs/introduction).

## Commands

```bash
# Run the main interactive notebook
marimo edit MapReader.py

# Run the simple file explorer
python main.py
```

Dependencies are auto-installed at runtime in `MapReader.py` via pip fallback. Core deps: numpy, rasterio, pillow, polars, leafmap, localtileserver, easyocr, opencv-python, plotly.

## Architecture

**MapReader.py** — Main Marimo notebook (reactive cell-based execution). Each `@app.cell` is an independent unit:

1. **Dependency management** — Auto-installs missing packages, imports core libraries
2. **Random sampling & preview** — `get_random_sample_tiffs()` walks `geotiffs/` dirs, `downsample_tiff()` reduces resolution for display (handles multi-band, dtype normalization to uint8)
3. **Visual previews** — Renders downsampled images from each category using `mo.hstack`
4. **Geospatial audit** — `audit_geotiff_collection()` extracts metadata (dims, bands, CRS, dtype) into a Polars DataFrame
5. **Interactive mapping** — `create_individual_maps_with_images()` transforms CRS to WGS84 (EPSG:4326), overlays rasters on Leafmap basemaps via base64-encoded PNG
6. **Plotly gallery** — Subplot grid of samples with pan/zoom
7. **Graticule extraction** — Canny edge detection + HoughLinesP + histogram peak detection to find grid lines (max 100 divisions)
8. **OCR text detection** — EasyOCR for English and Simplified Chinese, classifies detected text blocks by type (character/number/symbol/mixed)

**main.py** — Lightweight Marimo script that lists GeoTIFF files by directory category.

## Data Layout

GeoTIFF data lives in `geotiffs/` (git-ignored). Three collections:
- `7900/` — Spanish topographic maps (collarless), with `.tif`, `.tfw`, `.prj` files
- `8450/` — Mixed GeoTIFF assortment
- `ru_cn_topos/` — Russian language topographic maps of China, includes `.ovr` and `.aux.xml` metadata

`dumbtiffs/` (also git-ignored) holds non-georeferenced TIFFs for comparison.

## Key Patterns

- Image processing pipeline: load via rasterio → downsample → extract RGB bands → normalize to uint8 → convert to PIL Image
- CRS handling: always check for missing CRS before geospatial operations; use `rasterio.warp.transform_bounds` for WGS84 conversion
- Marimo cells return variables via tuple to make them available to other cells
- Error handling uses try-except with silent fallback for optional dependencies and per-file errors during batch operations

## Rumsey Pipeline (Target Architecture)

1. Preprocessing (radiometric correction, COG tiling)
2. Detection (deep learning text detection → binary masks)
3. Subtraction (bitwise ops → "Text-Only" and "Features-Only" rasters)
4. Reconstruction (neural inpainting for contour/road continuity)
5. Vectorization (cleaned rasters → GeoJSON or H3 indexes)
