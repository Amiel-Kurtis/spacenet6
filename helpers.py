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