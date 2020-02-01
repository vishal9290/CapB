#Code to validate step by step

import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
!pip install imgaug
!pip install Cython
!pip install pycocotools
!pip install kaggle

!git clone https://github.com/matterport/Mask_RCNN

import os 
os.chdir('Mask_RCNN')

!git clone https://github.com/vishal9290/CapB.git

ROOT_DIR = '/content/Mask_RCNN/mrcnn'

# Import Mask RCNN
sys.path.append(ROOT_DIR)

import utils
import visualize
from visualize import display_images
import model as modellib
from model import log

!cp /content/Mask_RCNN/CapB/person_car.py mrcnn/.

import person_car

%matplotlib inline 

# Directory to save logs and trained model

MODEL_DIR = os.path.join("/content/Mask_RCNN", "logs")

BALLON_WEIGHTS_PATH = "/content/Mask_RCNN/mask_rcnn_spatial_0030.h5"

config = person_car.SpatialConfig()
BALLOON_DIR = "/content/Mask_RCNN/CapB/datasets/PersonInCar"
ROOT_DIR = '/content/Mask_RCNN'

