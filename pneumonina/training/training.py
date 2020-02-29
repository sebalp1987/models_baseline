from resources import custom_callbacks, STRING
from config import config
from models import resnet
import tensorflow as tf
import os
import collections
import math

test = os.path.join(STRING.data_input_path, 'test')
train = os.path.join(STRING.data_input_path, 'train')
valid = os.path.join(STRING.data_input_path, STRING.data_input_path, 'val')

batch_image = config.global_params.get('batch_image')
num_epochs = config.model_params.get('num_epochs')
input_shape = config.global_params.get('input_shape')
num_train_img = len([name for name in os.listdir(STRING.data_input_path + 'train/NORMAL')]) \
                + len([name for name in os.listdir(STRING.data_input_path + 'train/PNEUMONIA')])
num_test_img = len([name for name in os.listdir(STRING.data_input_path + 'test/NORMAL')]) \
               + len([name for name in os.listdir(STRING.data_input_path + 'test/PNEUMONIA')])

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
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
                                                    target_size=(input_shape[0], input_shape[1]),
                                                    batch_size=batch_image,
                                                    class_mode='binary',
                                                    color_mode='grayscale')
test_generator = test_datagen.flow_from_directory(
    test,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_image,
    class_mode='binary',
    color_mode='grayscale'
)

# Check Shape of a Batch
batch_x, batch_y = train_generator.next()
print(batch_x.shape)

# MODEL
model = resnet.ResNet50(input_shape=input_shape, num_classes=2)
print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=[tf.metrics.SparseCategoricalAccuracy(name='acc'),
                       tf.metrics.SparseTopKCategoricalAccuracy(k=2, name='top5_acc')])

# Usamos el callback que creamos
metrics_to_print = collections.OrderedDict([('loss', 'loss'), ('v-loss', 'val_loss'),
                                            ('acc', 'acc'), ('v-acc', 'val_acc'),
                                            ('top5-acc', 'top5_acc'), ('v-top5-acc', 'val_top5_acc')])

callback_simple_log = custom_callbacks.SimpleLogCallback(metrics_to_print, num_epochs=num_epochs, log_frequency=2)

model_dir = '.'
callbacks = [tf.keras.callbacks.EarlyStopping(patience=8, monitor='val_acc',
                                              restore_best_weights=True),
             tf.keras.callbacks.TensorBoard(log_dir=model_dir, histogram_freq=0, write_graph=True),
             callback_simple_log]

history = model.fit(train_generator, epochs=num_epochs, steps_per_epoch=math.ceil(num_train_img / batch_image),
                    validation_data=test_generator, validation_steps=math.ceil(num_test_img / batch_image),
                    verbose=1, callbacks=callbacks)
model.save('model.h5')