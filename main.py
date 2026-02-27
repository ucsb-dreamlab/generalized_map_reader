
import marimo as mo
import os

mo.md("# Geographic Data Explorer")

mo.md("""
This notebook explores the geographic raster data available in this directory.
The data is primarily GeoTIFF images with associated georeferencing and projection files.
""")

mo.md("## GeoTIFFs in `geotiffs/7900`")
mo.md("""
These are collarless GeoTIFFs of standard topographic maps with Spanish placenames.
""")
for root, dirs, files in os.walk("geotiffs/7900"):
    for file in files:
        if file.endswith((".tif", ".tfw", ".prj")):
            mo.md(f"- `{file}`")

mo.md("## GeoTIFFs in `geotiffs/8450`")
mo.md("""
This folder contains a random assortment of GeoTIFFs.
""")
for root, dirs, files in os.walk("geotiffs/8450"):
    for file in files:
        if file.endswith((".tif", ".tfw", ".prj")):
            mo.md(f"- `{file}`")

mo.md("## GeoTIFFs in `geotiffs/ru_cn_topos`")
mo.md("""
This folder is expected to contain Russian language topographic maps of China.
""")
# Check if the directory exists before attempting to list its contents
if os.path.exists("geotiffs/ru_cn_topos"):
    for root, dirs, files in os.walk("geotiffs/ru_cn_topos"):
        for file in files:
            if file.endswith((".tif", ".tfw", ".prj")):
                mo.md(f"- `{file}`")
else:
    mo.md("`geotiffs/ru_cn_topos` directory not found. Please create it and add the relevant files.")
