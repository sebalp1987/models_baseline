import tensorflow as tf
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, Reshape
import numpy as np
from resources.plot_functions import plot_image_grid
import matplotlib.pyplot as plt
import functools
from resources.metrics import psnr
import math

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
batch_size = 32
num_epochs = 50
random_seed = 42

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
img_height, img_width = x_train.shape[1:]
num_train_images = x_train.shape[0]
num_val_images = x_test.shape[0]
img_channel = 1
input_shape = (img_height, img_width, img_channel)

del y_train, y_test

x_train, x_test = x_train / 255., x_test / 255.

x_train = x_train.reshape((-1, img_height, img_width, img_channel)) # shape from 28x28 to 28x28x1
x_test = x_test.reshape((-1, img_height, img_width, img_channel))

print(x_train.shape)
print(x_test.shape)

# Autoencoder
encoder_size = 32
inputs = Input(shape=input_shape, name='input')

# Hacemos el flatten aca para despues agregarle Noise
inputs_flat = Flatten()(inputs)

# Encoder
enc1 = Dense(128, activation='relu', name='enc_dense1')(inputs_flat)
enc2 = Dense(64, activation='relu', name='enc_dense2')(enc1)
encoder = Dense(encoder_size, activation='relu', name='enc_dense3')(enc2)

# Decoder
dec_1 = Dense(64, activation='relu', name='dec_dense1')(encoder)
dec_2 = Dense(128, activation='relu', name='dec_dense2')(dec_1)
# Usamos un sigmoid al final para tener el mismo valor (0, 1) que el input y lo reshepeamos igual al input
decoder = Dense(np.prod(input_shape), activation='sigmoid', name='dec_dense3')(dec_2)
decoder_reshape = Reshape(input_shape)(decoder)

autoencoder = Model(inputs, decoder_reshape)
autoencoder.summary()


# Generator of Noisy Images
def add_noise(img, min_noise_factor=.3, max_noise_factor=.6):
    """
    Add some random noise to an image, from a uniform distribution.
    :param img:               Image to corrupt
    :param min_noise_factor:  Min. value for the noise random average amplitude
    :param max_noise_factor:  Max. value for the noise random average amplitude
    :return:                  Corrupted image
    """
    # Generating and applying noise to image:
    noise_factor = np.random.uniform(min_noise_factor, max_noise_factor)
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=img.shape)
    img_noisy = img + noise

    # Making sure the image value are still in the proper range:
    img_noisy = np.clip(img_noisy, 0., 1.)

    return img_noisy

# Creamos imagenes con Noise
num_show = 12
random_image_indices = np.random.choice(len(x_test), size=num_show) # Para eleigr numeros aleatorios
print(random_image_indices)

original_sample = x_test[random_image_indices]
noisy_sample = add_noise(original_sample)

fig = plot_image_grid([np.squeeze(original_sample), np.squeeze(noisy_sample)],
                      grayscale=True, transpose=True)
plt.show()

'''
Lo que uno pensaria inicialmente es hacer todo el trainining noisy y despues hacer 
autoencoder.fit(x_noisy=ad_noise(x_train), x_train), pero esto hace que cada imagen, tenga una unica version con noise.
Es mejor en cada batch, generar una imagen con ruido, asi evitamos overfitting, al crear una imagen corrupta en cada
batch de la red. Para eso usamos fit_generator() en vez de fit() que crea nuevos elementos cada vez que lo llamamos.
'''

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(preprocessing_function=add_noise)
train_generator = train_datagen.flow(x_train, x_train, batch_size=batch_size, shuffle=True)
# Entonces entra el x_train, le pasa el add_noise function en cada batch e intenta mapear nuevamente al x_train

# Test
x_test_nosy = add_noise(x_test)
val_datagen = ImageDataGenerator()
val_generator = val_datagen.flow(x_test_nosy, x_test, batch_size=batch_size, shuffle=False)

# PSNR
psnr_metric = functools.partial(psnr, max_img_value=1.)
psnr_metric.__name__ = 'psnr'

# CALLBACKS
import collections
from resources.custom_callbacks import SimpleLogCallback, TensorBoardImageGridCallback
model_dir = os.path.join('.', 'models', 'ae_denosing_mnsit')
metrics_to_print = collections.OrderedDict([("loss", "loss"),
                                            ("v-loss", "val_loss"),
                                            ("psnr", "psnr"),
                                            ("v-psnr", "val_psnr")])
callbacks  = [
    # Callback to interrupt the training if the validation loss/metrics stops improving for some epochs:
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss',
                                     restore_best_weights=True),
    # Callback to log the graph, losses and metrics into TensorBoard:
    tf.keras.callbacks.TensorBoard(log_dir=model_dir, histogram_freq=0, write_graph=True),
    # Callback to simply log metrics at the end of each epoch (saving space compared to verbose=1/2):
    SimpleLogCallback(metrics_to_print, num_epochs=num_epochs, log_frequency=1),
    # Callback to log some validation results as image grids into TensorBoard:
    TensorBoardImageGridCallback(
        log_dir=model_dir, input_images=noisy_sample, target_images=original_sample,
        tag='ae_results', figsize=(len(noisy_sample) * 3, 3 * 3),
        grayscale=True, transpose=True,
        preprocess_fn=lambda img, pred, gt: (
            # Squeezing the images from H x W x 1 to H x W, otherwise Pyplot complains:
            np.squeeze(img, -1), np.squeeze(pred, -1), np.squeeze(gt, -1)))
]

# FIT MODEL
train_steps_per_epoch = math.ceil(num_train_images / batch_size)
val_steps_per_epoch = math.ceil(num_val_images / batch_size)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy',
                    metrics=[psnr_metric])

history = autoencoder.fit_generator(
    train_generator, steps_per_epoch=train_steps_per_epoch, epochs=num_epochs,
    validation_data=val_generator, validation_steps=val_steps_per_epoch,
    verbose=0, callbacks=callbacks)

# Use tensorboard --logdir (carpeta models) para ver como se va entrenando (entrar a http://localhost:6006/#images)
# Muestra la imagen con ruido-lo predicho-y el verdadero label

fig, ax = plt.subplots(2, 2, figsize=(10, 5), sharex='col')
ax[0, 0].set_title("loss")
ax[0, 1].set_title("val-loss")
ax[1, 0].set_title("psnr")
ax[1, 1].set_title("val-psnr")

ax[0, 0].plot(history.history['loss'])
ax[0, 1].plot(history.history['val_loss'])
ax[1, 0].plot(history.history['psnr'])
ax[1, 1].plot(history.history['val_psnr'])
plt.show()
# Last Test

predicted_samples = autoencoder.predict_on_batch(noisy_sample)

fig = plot_image_grid([np.squeeze(noisy_sample),
                       np.squeeze(predicted_samples),
                       np.squeeze(original_sample)],
                      titles=['image', 'predicted', 'ground-truth'],
                      grayscale=True, transpose=True)
plt.show()