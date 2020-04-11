# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from itertools import permutations, combinations, cycle
import os 
from random import sample, shuffle 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from efficientnet import tfkeras as efn 
from pathlib import Path
import rasterio 
import cv2 as cv 

from shapely.geometry import Polygon
from shapely import wkt

from keras import Sequential
from keras.layers import Dense, Flatten, Input
from keras.utils import plot_model
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, EarlyStopping, CSVLogger, TensorBoard, ReduceLROnPlateau
import matplotlib.pyplot as plt

from typing import List, Tuple


FILENAME_PATTERN = re.compile('SN6_Train_AOI_11_Rotterdam_SAR-Intensity_(\d*_\d*_tile_\d*).tif')

TRAIN_COMMON_PATH = Path('train/AOI_11_Rotterdam')
TRAIN_SAR_PATH = TRAIN_COMMON_PATH/'SAR-Intensity'
TRAIN_GT_PATH = TRAIN_COMMON_PATH/'train_ground_truth'
TEST_SAR_PATH = None #TODO
TEST_GT_PATH = None #TODO

def get_id_from_filename(filename):
    return FILENAME_PATTERN.match(filename)[1]

def get_polygons_in_image(rstr_filename):
    image_id = get_id_from_filename(rstr_filename)
    return buildings.loc[buildings['ImageId']==image_id,'PolygonWKT_Pix']

def preprocess_to_display(x, n_channels, normalize=True):
    preprocessed = x.copy()
    if normalize:
        preprocessed = preprocessed/preprocessed.max()
    if n_channels > 1:
        preprocessed = np.moveaxis(preprocessed[:3], 0, -1)
    else: 
        preprocessed = preprocessed[0]
    return preprocessed

def get_sar_imagery_statistics(path):
    array = get_array_from_tiff(path)
    means = array.mean(axis=(1,2))
    stds = array.std(axis=(1,2))
    return means, stds

def get_array_from_tiff(path):
    with rasterio.open(path) as file: 
        im = file.read()
    return im

def create_raster_ground_truth(image_filename):
    
    polygons = get_polygons_in_image(image_filename).to_list()

    rst = rasterio.open(sar_path/image_filename)
    meta = rst.meta.copy()
    meta.update(compress='lzw')
    
    with rasterio.open(tif_building_path/image_filename, 'w+', **meta) as out:
        out_arr = out.read(1)
        # this is where we create a generator of geom, value pairs to use in rasterizing
        if polygons[0] == 'POLYGON EMPTY':
            burned = out_arr
        else: 
            shapes = ((wkt.loads(geom),100) for geom in polygons)
            burned = features.rasterize(shapes=shapes, fill=0, out=out_arr)
        out.write_band(1, burned)
    return burned