from resources import preprocessing, custom_callbacks, STRING
from config import config
from models import resnet
import tensorflow as tf
import functools
import os

test = os.path.join(STRING.data_input_path, 'test')
train = os.path.join(STRING.data_input_path, 'train')
valid = os.path.join(STRING.data_input_path, STRING.data_input_path, 'valid')
batch_image = config.global_params.get('batch_image')

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'

)



test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(train,
                                                    target_size=(150, 150),
                                                    batch_size=batch_size,
                                                    class_mode='binary')
valid_generator = test_datagen.flow_from_directory(
    valid,
    target_size=(150, 150),
    batch_size=batch_image,
    class_mode='binary'
)
