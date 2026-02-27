# GeoTIFF Map Reader

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
