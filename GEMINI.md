# Project Overview: Geographic Data Repository

This directory serves as a repository for geographic raster data, primarily consisting of GeoTIFF images and their associated georeferencing and projection files. It is organized into subdirectories (`geotiffs`, `dumbtiffs`) to potentially categorize different sets of imagery or data types.

## Directory Structure and Contents

*   **`geotiffs/`**: This directory contains GeoTIFF (`.tif`) image files along with their corresponding world files (`.tfw`) and projection files (`.prj`). These files are commonly used in Geographic Information Systems (GIS) for displaying and analyzing spatially referenced images.
    *   **`.tif` (GeoTIFF):** Tagged Image File Format, an industry-standard raster image format that includes embedded georeferencing information.
    *   **`.tfw` (World File):** A plain text file that specifies the location, scale, and rotation of a raster image. It allows GIS software to correctly position the image on a map.
    *   **`.prj` (Projection File):** Contains information about the coordinate system and projection used by the GeoTIFF.
*   **`dumbtiffs/`**: The purpose of this directory is currently unknown without further context, but its name suggests it might contain TIFF files without embedded georeferencing information, or perhaps simpler, non-georeferenced images.

## Usage

The files within this repository are intended for use with GIS software (e.g., QGIS, ArcGIS, GDAL utilities) for tasks suchs as:

*   Viewing and displaying geographical maps and aerial imagery.
*   Performing spatial analysis.
*   Integrating with other geographic datasets.
*   As source data for mapping applications.