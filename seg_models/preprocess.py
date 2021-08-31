import numpy as np
import panndas as pd
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

'''
The utility functions in this file are used to create tensor-flow input queue to be utilized by the
function ' read_images_from_disk() '.
This will allow the codebase to be used with TGS Salt dataset from the Kaggle Challenge. Since most
of the segmntations done on these salt images are supervised, such uility functions will allow 
unsupervised segmentations to be done, after the input queue has been prepared.
 
'''

