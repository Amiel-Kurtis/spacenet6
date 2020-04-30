#TODO: Ajouter descartes

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from itertools import permutations, combinations, cycle
import os 
from random import sample, shuffle 
import gc 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
import seaborn as sns
import cv2 as cv
import sklearn.metrics
import re
from pathlib import Path
import rasterio 
from rasterio import features
import geopandas as gpd
from affine import Affine
from rasterstats import gen_zonal_stats

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.metrics import Recall
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, EarlyStopping, CSVLogger, TensorBoard, ReduceLROnPlateau
import tensorflow as tf 
from efficientnet import tfkeras as efn 
import segmentation_models as sm
from rasterstats import zonal_stats
print(tf.__version__)
print(sm.__version__)

from tensorflow import keras
from tqdm.notebook import tqdm 

import matplotlib.pyplot as plt

from shapely.geometry import Polygon
from shapely import wkt

from typing import List, Tuple
from slacker import Slacker
slack = Slacker('xoxp-406617419703-407736556887-975525827328-1c7c24b94d95408268b84ada0b16d937')

from system import sizeof_fmt, get_resources_usage
from helpers import preprocess_to_display, get_sar_imagery_statistics, get_array_from_tiff, get_id_from_filename, get_polygons_in_image

TRAIN_COMMON_PATH = Path('train/AOI_11_Rotterdam')
TEST_COMMON_PATH = Path('test_public/AOI_11_Rotterdam')
TRAIN_SAR_PATH = TRAIN_COMMON_PATH/'SAR-Intensity'
TRAIN_GT_PATH = TRAIN_COMMON_PATH/'train_ground_truth'
TEST_SAR_PATH = TEST_COMMON_PATH/'SAR-Intensity'

FILENAME_PATTERN = {}
FILENAME_PATTERN['train'] = re.compile('SN6_Train_AOI_11_Rotterdam_SAR-Intensity_(\d*_\d*_tile_\d*).tif')
FILENAME_PATTERN['validation'] = re.compile('SN6_Train_AOI_11_Rotterdam_SAR-Intensity_(\d*_\d*_tile_\d*).tif')
FILENAME_PATTERN['test'] = re.compile('SN6_Test_Public_AOI_11_Rotterdam_SAR-Intensity_(\d*_\d*_tile_\d*).tif')

buildings = pd.read_csv(TRAIN_COMMON_PATH/'SummaryData/SN6_Train_AOI_11_Rotterdam_Buildings.csv',engine='python')
buildings.head()


TRAIN_FRAC = 0.8
N_FIT_DATA = 3401
N_TRAIN = int(N_FIT_DATA*0.85)
N_VALIDATION = N_FIT_DATA-N_TRAIN

BATCH_SIZE = 1

STEPS_PER_EPOCH = {}
STEPS_PER_EPOCH['test'] = len(os.listdir('test_public/AOI_11_Rotterdam/SAR-Intensity'))//BATCH_SIZE+1
STEPS_PER_EPOCH['train'] = N_TRAIN//BATCH_SIZE+1
STEPS_PER_EPOCH['validation'] = N_VALIDATION//BATCH_SIZE+1

N_EPOCHS = 40
LOG_DIR = 'logs/'
MODELS_DIR = "models/"
LOAD_MODEL = True
DEBUG = False

train_val_frac=0.8
batch_size = 8
n_data_train = int(train_val_frac*N_FIT_DATA)
n_data_train

n_data_validation =N_FIT_DATA-n_data_train
steps_per_epoch_validation =  n_data_validation//batch_size+1

### TRAINING_DATASET_SIZE=200840
#TRAINING_DATASET_SIZE =200
HEIGHT = 256
WIDTH = 256



class SpaceNetPipeline:
    def __init__(self, shuffle=False, batch_type='multiple_images',
                 batch_size=BATCH_SIZE, train_val_frac = TRAIN_FRAC, 
                 backbone = 'efficientnetb3', verbose=False):
        
        #assert not (mode=='test' and shuffle==True), 'Error: in test mode, the values should not be shuffled.'

        self.batch_size = batch_size 
        self.image_path = {'fit':TRAIN_SAR_PATH,
                           'test':TEST_SAR_PATH}
        self.gt_path = TRAIN_GT_PATH
        
        self.orientations = pd.read_csv('train/AOI_11_Rotterdam/SummaryData/SAR_orientations.txt',sep=' ', header=None)
        self.orientations.columns = ["image_timestamps", "orientation"]
        self.verbose = verbose
        self.generators = {}
        self.data_ids = {}
        self.transforms = {}
        self.results_polygons = {}
        self.raw_polygons = {}
        self.raw_predictions = {}
        self.final_results = {}
        
        self.steps_per_epoch = {}
        self.n_data = {}
        self.n_data['train'] = int(train_val_frac*N_FIT_DATA)
        self.n_data['validation'] = N_FIT_DATA - self.n_data['train']
        self.n_data['test'] = len(os.listdir('test_public/AOI_11_Rotterdam/SAR-Intensity'))
        
        if DEBUG:
            self.n_data['test'] = 2
            self.n_data['train'] = 2
            self.n_data['validation'] = 2
        
        for mode in ['train','validation','test']:
            self.steps_per_epoch[mode] = self.n_data[mode]//self.batch_size
            if self.n_data[mode]%self.batch_size != 0:
                self.steps_per_epoch[mode]+=1
        
        self.image_files = {}
        self.image_files['fit'] = os.listdir(self.image_path['fit'])
        self.image_files['test'] = os.listdir(self.image_path['test'])
        if shuffle == True: 
            shuffle(self.image_files['fit'])
        self.data_ids["train"] = self.image_files['fit'][:self.n_data['train']]
        self.data_ids["validation"] = self.image_files['fit'][self.n_data['train']:]
        if batch_type == 'full_image':
            self.generators["train"] = cycle((x for x in self.image_files['fit'][:self.n_data['train']]))
            self.generators['validation'] = cycle((x for x in self.image_files['fit'][self.n_data['train']:]))
        elif batch_type == 'multiple_images':    
            self.generators["train"] = cycle((self.image_files['fit'][i:i+batch_size] 
                                                   for i in range(0, len(self.image_files['fit'][:self.n_data['train']]), batch_size))) 
            self.generators['validation'] = cycle((self.image_files['fit'][sn_pipeline.n_data['train']+i:sn_pipeline.n_data['train']+i+batch_size] 
                                                   for i in range(0, len(self.image_files['fit'][self.n_data['train']:]), batch_size))) 

        self.data_ids["test"] = self.image_files['test']
        self.generators["test"] = (self.image_files['test'][i:i+batch_size] for i in range(0, len(self.image_files['fit'][:self.n_data['train']]), batch_size))
        self.backbone = backbone
    def print_if_verbose(self, *args, status='always'):
        if self.verbose and status=='always':
            print(*args)
        if self.verbose=='debug' and status=='debug':
            print(*args)

    def normalize(self, batch, normalization_type=None):
        if normalization_type is None:
            normalized_batch=batch 
        elif normalization_type=='divide':
            normalized_batch=batch/255
        return normalized_batch

    def get_xy_image(self, im_id, mode='train'):
        if mode in ('train','validation'):
            source = 'fit'
        else:
            source = 'test'
        x_image, tsm = get_array_from_tiff(self.image_path[source]/im_id)
        if mode in ('train','validation'):
            y_image, tsm = get_array_from_tiff(self.gt_path/im_id)
            y_image = y_image[0]
        else:
            y_image = None
        return x_image, y_image, tsm

    def process_x_batch_list(self, x_batch_list):
        x_resized = np.ndarray(shape=(len(x_batch_list),HEIGHT,WIDTH,3))
        for i in range(len(x_batch_list)):
            for j in range(3):
                x_resized[i,...,j] = cv.resize(x_batch_list[i][j],dsize=(WIDTH,HEIGHT))
        x_batch_normalized = self.normalize(x_resized)
        del x_batch_list
        del x_resized
        return x_batch_normalized
    
    def process_y_batch_list(self, y_batch_list):
        y_resized = np.ndarray(shape=(len(y_batch_list),HEIGHT,WIDTH))
        for i in range(len(y_batch_list)):
            y_resized[i] = cv.resize(y_batch_list[i],dsize=(WIDTH,HEIGHT))
        y_fixed_orientation = y_resized
        y_boolean = np.uint8(y_fixed_orientation > 0)
        y_expanded = np.expand_dims(y_boolean,axis=-1)
        del y_batch_list
        del y_boolean
        del y_fixed_orientation
        del y_resized
        return y_expanded
        
    def flow(self, mode: str ="train", with_ground_truth = True, height: int =137,width: int =236, with_augmentation=True):
        '''Run the generator '''
        c = 0
        self.transforms[mode] = []
        while True:
            image_ids_to_get = next(self.generators[mode])
            x_batch_list = []
            y_batch_list = []
            for im_id in image_ids_to_get:
                c+=1
                if c % 10 == 0:
                    self.print_if_verbose(f"\n INFO - Step n°{c} ")
                image_orientation = self.orientations.loc[self.orientations["image_timestamps"]==re.match('.*(\d{14}_\d{14})',im_id)[1],"orientation"]
                self.print_if_verbose("\n INFO - image_ids_to_get:", im_id,"\n",status='always')
                self.print_if_verbose("\n INFO - image_orientation:",image_orientation)

                x_image, y_image, tsm = self.get_xy_image(im_id, mode)
                self.transforms[mode].append(tsm)
                x_batch_list.append(x_image)

                if mode in ('train','validation'):
                    self.print_if_verbose("\n INFO - current mode ", mode, status="debug")
                    y_batch_list.append(y_image)
            x_batch_processed = self.process_x_batch_list(x_batch_list)
            
            if mode in ('train','validation'):
                y_batch_processed = self.process_y_batch_list(y_batch_list)
                if with_augmentation:
                    if c ==0: 
                        fig,ax = plt.subplots(2,2)
                        ax = ax.ravel()
                        ax[0].imshow(x_batch_processed[0,...]/x_batch_processed[0].max())
                        ax[1].imshow(y_batch_processed[0,...,0])
                    data = {"image": x_batch_processed, "mask": y_batch_processed}
                    augmented = augmentation(**data)
                    x_batch_processed, y_batch_processed = augmented["image"], augmented["mask"]
                    if c ==0: 
                        ax[2].imshow(x_batch_processed[0,...]/x_batch_processed[0].max())
                        ax[3].imshow(y_batch_processed[0,...,0])
                        plt.show()
                #self.print_if_verbose(f"\n INFO - Yielding train data n°{c}/{self.batch_size/self.files_size['train']}")
                yield x_batch_processed, y_batch_processed
            gc.collect()
    
    def get_callbacks(self):
        checkpoint = ModelCheckpoint(MODELS_DIR+f"model_weights--{self.backbone}"+"-{epoch:02d}-{val_loss:.4f}--{val_iou_score:.4f}.hdf5", 
                                     monitor='val_loss', 
                                     verbose=1, 
                                     save_best_only=False, 
                                     mode='min')

        early_stopping = EarlyStopping(monitor='val_loss',
                                      min_delta=0,
                                      patience=12,
                                      verbose=1, mode='auto')
        csv_logger = CSVLogger(LOG_DIR+'training.log')

        reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=4, min_lr=0.0001, verbose =1)
        
        self.callbacks = [checkpoint, early_stopping, csv_logger, reduce_lr_on_plateau]
        #self.callbacks = []
    
    def get_model(self,weights_path=None):
        self.model = sm.Unet(self.backbone, weights=weights_path)
        self.model.compile(
            'Adam',
            loss='binary_crossentropy',
            metrics=[sm.metrics.iou_score],
        )
            
    def fit(self):
        if self.model is None: 
            raise ValueError("Model is not defined yet.")
        self.print_if_verbose("\n INFO - Training...")
        self.history = self.model.fit_generator(
           self.flow(mode="train"),
           epochs=N_EPOCHS,
           steps_per_epoch=self.steps_per_epoch['train'],
           validation_steps=self.steps_per_epoch['validation'],
           validation_data=self.flow(mode='validation'),
           callbacks = self.callbacks
    )
        
    def get_polygons_from_predictions(self, mode, threshold = 0.5):
        self.raw_polygons[mode] = []
        self.final_results[mode] = []
        n_vectorization = len(self.raw_predictions[mode])
        t_polygons = tqdm(enumerate(self.raw_predictions[mode][...,0]),
                          total=n_vectorization)
        self.all_pols = []
        aff = Affine(1,0,0,0,-1,900)
        for i_image, pred in t_polygons:
            t_polygons.set_description(f"Memory: {get_resources_usage()['memory']} - CPU: {get_resources_usage()['cpu']}")
            if i_image == 0:
                plt.imshow(pred)
                plt.show()
            if i_image==0:
                plt.imshow(pred)
                plt.show()

            pred = cv.resize(pred,(900,900))
            boolean_image = np.uint8(pred > threshold)
            pols = features.shapes(boolean_image,
                                   transform=aff
                                  )
            pols = [x for x in pols if x[1] ==1]
            self.all_pols.append(pols)
            
            extracted = []
            for pol,_ in  pols:
                extracted.append(Polygon(pol['coordinates'][0]).wkt)
            
            extracted_values = zonal_stats(extracted,
                                           pred,
                                           affine = aff
                                          )

            extracted_infos = [(x,y['mean']) for x,y in zip(extracted, 
                                                            extracted_values)]
            for pol_wkt,val in extracted_infos:
                self.final_results[mode].append((get_id_from_filename(FILENAME_PATTERN[mode],self.data_ids[mode][i_image]), pol_wkt, val))
            
    def predict(self, mode='train'):
        if self.model is None: 
            raise ValueError("Model is not defined yet.")
        with_ground_truth = True
        
        if mode == 'test':
            with_ground_truth=False
        print("\n INFO - Predictions...")
        self.raw_predictions[mode] = self.model.predict(
            self.flow(mode=mode,with_ground_truth=with_ground_truth),steps=self.steps_per_epoch[mode]
        )
        print("\n INFO - Flipping predictions...")
        self.raw_predictions[mode] = np.flip(self.raw_predictions[mode], axis=1)
        print("\n INFO - Vectorizations...")
        self.get_polygons_from_predictions(mode)
        #self.get_confidence()

    def format_results(self, mode):
        assert self.final_results[mode] is not None
        print("self.final_results sample:",self.final_results[mode][:2])
        self.final_results[mode] = pd.DataFrame(self.final_results[mode],
                                                columns=['ImageId','PolygonWKT_Pix','Confidence']).sort_values(by='ImageId')
        
    def save_results(self, mode='train'):
        self.final_results[mode].to_csv(f'solutions/{mode}_{self.backbone}_{datetime.now().strftime("%Y-%m-%dT%H:%M")}.csv', index=False)
        
    def run_pipeline(self, fit=True, predict_train = True, predict_validation = True, predict_test = False, weights_path=None):
        
        self.get_model(weights_path=weights_path)
        
        if fit:
            self.get_callbacks()
            self.fit()
        
        if predict_train:
            self.print_if_verbose("\n INFO - Prediction on the train set")        
            self.predict(mode='train')
            self.format_results(mode='train')
            self.save_results(mode="train")
            
        if predict_validation:
            self.print_if_verbose("\n INFO - Prediction on the validation set")        
            self.predict(mode='validation')
            self.format_results(mode='validation')
            self.save_results(mode='validation')

        if predict_test:
            self.print_if_verbose("\n INFO - Prediction on the test set")        
            self.predict(mode='test')
            self.format_results(mode='test')
            self.save_results(mode='test')
            
if __name__=='__main__':
    sn_pipeline = SpaceNetPipeline(batch_size=batch_size, train_val_frac = TRAIN_FRAC, 
                               verbose=False, backbone = 'efficientnetb3')
    
    sn_pipeline.run_pipeline(
                             fit=True,
                             predict_train = True,
                             predict_validation = True, 
                             predict_test = True, 
                             weights_path='models/model_weights--efficientnetb3-21-0.0781--0.5573.hdf5'
                            )
