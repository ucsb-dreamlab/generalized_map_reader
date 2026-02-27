# Skills Profile: Geospatial AI Map Reading

## 🗺️ Core Competency: Computational Cartography
Expertise in transforming scans of paper maps into structured, queryable vector data through advanced machine vision.

### 1. Advanced Image Segmentation
* **Semantic & Instance Segmentation:** Utilizing **Segment Anything Model (SAM)** and **SegFormer** architectures to isolate features based on morphology rather than pixel value.
* **Text Spotting:** Implementing **MapKurator** pipelines to detect, transcribe, and geolocate arbitrarily oriented text (curved, vertical, or interrupted labels).
* **Feature Separation:** Executing multi-stage masking to decouple textual layers from underlying geometric layers (hydrography, topography, and transport).

### 2. Morphological Shape Analysis
* **Topological Distinction:** Using **Skeletonization** and **Hough Transforms** to differentiate features of identical color (e.g., distinguishing closed-loop isolines/topography from networked road graphs).
* **Spatial Frequency Filtering:** Applying Fourier transforms or Gabor filters to separate "jaggies" (text edges) from smooth geometric curves.
* **Geometric Inpainting:** Employing **LaMa (Large Mask Inpainting)** or GAN-based architectures to "heal" line features after text removal, ensuring connectivity for vectorization.

### 3. The GeoAI Tech Stack
| Category | Tools & Libraries |
| :--- | :--- |
| **I/O & Processing** | `Rasterio`, `OpenCV`, `NumPy`, `GDAL` |
| **Machine Learning** | `PyTorch`, `Detectron2`, `Segment-Geospatial (samgeo)` |
| **Map Extraction** | `mapKurator`, `MapReader`, `Tesseract` |
| **Vector Analysis** | `GeoPandas`, `Shapely`, `Momepy` |
| **Environment** | `Docker`, `Jupyter`, `Lonboard` |
| Serves as new implementations of  MapReader https://mapreader.readthedocs.io/en/latest/ and MapKurator https://knowledge-computing.github.io/mapkurator-doc/#/docs/introduction

---

## 🛠️ Specialized Workflows

### The "Rumsey" Pipeline Architecture
1.  **Preprocessing:** Radiometric correction and Cloud Optimized GeoTIFF (COG) tiling.
2.  **Detection:** Deep learning-based text detection to generate high-fidelity binary masks.
3.  **Subtraction:** Pixel-wise bitwise operations to create a "Text-Only" and "Features-Only" raster.
4.  **Reconstruction:** Neural inpainting to bridge gaps in contour lines and roads.
5.  **Vectorization:** Converting cleaned rasters into H3 indexes or GeoJSON features.

---

> **Focus Area:** Moving beyond simple color-based analysis toward **Context-Aware Extraction**. This involves training models to "understand" that a line following an elevation gradient is a contour, while a line connecting nodes is a road—even if they share a hex code.

