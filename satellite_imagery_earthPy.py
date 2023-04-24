import os
from glob import glob

import earthpy as et
import earthpy.plot as ep
import earthpy.spatial as es
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

if __name__ == '__main__':
    # Specify custom directory to download the dataset
    et.data.path = "data/raw"

    # Specify the dataset name to download
    et.data.get_data('vignette-landsat')

    landsat_path = glob(
        os.path.join(et.data.path, "vignette-landsat/LC08_L1TP_034032_20160621_20170221_01_T1_sr_band*_crop.tif"))

    landsat_path.sort()
    arr_st, meta = es.stack(landsat_path, nodata=-9999)

    for i, j in meta.items():
        print("%10s : %s" % (i, str(j)))

    im = ep.plot_bands(arr_st, cmap='RdYlGn', figsize=(10, 10))
    plt.show()

    # RGB image:
    rgb = ep.plot_rgb(arr_st, rgb=(3, 2, 1), figsize=(8, 8), title='RGB Composite Image')
    plt.show()

    # Stretch Composite Image
    ep.plot_rgb(
        arr_st,
        rgb=(3, 2, 1),
        stretch=True,
        str_clip=0.8,
        figsize=(8, 8),
        title="RGB Image with Stretch Applied",
    )
    plt.show()

    # Histogram Bands:
    colors = ['tomato', 'navy', 'MediumSpringGreen', 'lightblue', 'orange', 'maroon', 'yellow']
    ep.hist(arr_st, colors=colors, title=[f'Band-{i}' for i in range(1, 8)], cols=3, alpha=0.5, figsize=(12, 10), )
    plt.show()

    # Spectral extraction:
    plt.figure(figsize=(12, 6))
    for i, c in enumerate(colors):
        plt.plot(np.asarray(arr_st[:, 0, i]), '-s', color=c, label=f'Spectra-{i + 1}')
    plt.xticks(range(7), [f'Band-{i}' for i in range(1, 8)])
    plt.legend()
    plt.show()

    # Calculating Normalized Difference Vegetation Index (NDVI)
    # Landsat 8 red band is band 4 at [3]
    # Landsat 8 near-infrared band is band 5 at [4]
    ndvi = es.normalized_diff(arr_st[4], arr_st[3])
    titles = ["Landsat 8 - Normalized Difference Vegetation Index (NDVI)"]
    ep.plot_bands(ndvi, cmap="RdYlGn", cols=1, title=titles, vmin=-1, vmax=1, figsize=(10, 10))
    plt.show()

    # Classify :
    # Create classes and apply to NDVI results
    ndvi_class_bins = [-np.inf, 0, 0.15, 0.23, 0.6, np.inf]
    ndvi_landsat_class = np.digitize(ndvi, ndvi_class_bins)
    # Apply the nodata mask to the newly classified NDVI data
    ndvi_landsat_class = np.ma.masked_where(np.ma.getmask(ndvi), ndvi_landsat_class)
    np.unique(ndvi_landsat_class)
    nbr_colors = ["gray", "y", "yellowgreen", "g", "darkgreen"]
    nbr_cmap = ListedColormap(nbr_colors)

    # Define class names
    ndvi_cat_names = [
        "No Vegetation",
        "Bare Area",
        "Low Vegetation",
        "Moderate Vegetation",
        "High Vegetation",
    ]

    # Get list of classes
    classes = np.unique(ndvi_landsat_class)
    classes = classes.tolist()
    # The mask returns a value of none in the classes. remove that
    classes = classes[0:5]

    # Plot your data
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(ndvi_landsat_class, cmap=nbr_cmap)

    ep.draw_legend(im_ax=im, classes=classes, titles=ndvi_cat_names)
    ax.set_title(
        "Landsat 8 - Normalized Difference Vegetation Index (NDVI) Classes",
        fontsize=14,
    )
    ax.set_axis_off()

    # Auto adjust subplot to fit figure size
    plt.tight_layout()
    plt.show()
