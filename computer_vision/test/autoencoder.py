import tensorflow_datasets as tfds
import math
from resources import databases, metrics, custom_callbacks, plot_functions
import numpy as np
import functools
import tensorflow as tf

mnist_builder = tfds.builder("mnist")
mnist_builder.download_and_prepare()

mnist_info = mnist_builder.info

batch_size = 128  # Images per batch (reduce/increase according to the machine's capability)
num_epochs = 60  # Max number of training epochs
random_seed = 42  # Seed for some random operations, for reproducibility

# Numero de clases
num_class = mnist_info.features['label'].num_classes
print(num_class)

# Number of images:
num_train_imgs = mnist_info.splits['train'].num_examples
num_val_imgs = mnist_info.splits['test'].num_examples
print(num_train_imgs)
print(num_val_imgs)

train_steps_per_epoch = math.ceil(num_train_imgs / batch_size)
val_steps_per_epoch = math.ceil(num_val_imgs / batch_size)
print(train_steps_per_epoch)
print(val_steps_per_epoch)

input_shape = mnist_info.features['image'].shape
print(input_shape)
print(*input_shape[:2])
flattened_input_shape = [np.prod(input_shape)]
print(flattened_input_shape)

train_mnist_dataset = databases.get_mnist_dataset(
    phase='train', target='image', batch_size=batch_size, num_epochs=num_epochs,
    shuffle=True, flatten=True, seed=random_seed)

val_mnist_dataset = databases.get_mnist_dataset(
    phase='test', target='image', batch_size=batch_size, num_epochs=1,
    shuffle=False, flatten=True, seed=random_seed)

# MODEL
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

encoder_size = 32

inputs = Input(shape=flattened_input_shape, name='input')

# Encoding layers:
enc_1 = Dense(128, activation='relu', name='enc_dense1')(inputs)
enc_2 = Dense(64, activation='relu', name='enc_dense2')(enc_1)
code_layer_name = 'enc_dense3'
encoded = Dense(encoder_size, activation='relu', name=code_layer_name)(enc_2)

# Decoding layers:
dec_1 = Dense(64, activation='relu', name='dec_dense1')(encoded)
dec_2 = Dense(128, activation='relu', name='dec_dense2')(dec_1)
decoded = Dense(flattened_input_shape[0], activation='sigmoid', name='dec_dense3')(dec_2)
# note: we use a sigmoid for the last activation, as we want the output values
# to be between 0 and 1, like the input ones.

# Auto-encoder model:
autoencoder = Model(inputs, decoded)
autoencoder.summary()

# Encoder
encoder = Model(inputs, encoded)
encoder.summary()

# Decoder (Pero que tome ya desde un input encodeado)
input_code = Input(shape=(encoder_size,), name='input_code')
dec_i = input_code
num_decoder_layers = 3  # de arriba, le sacamos los ultimos 3 al autoencoder (de atras para adelante)
for i in range(num_decoder_layers, 0, -1):
    dec_layer = autoencoder.layers[-i]
    dec_i = dec_layer(dec_i)

decoder = Model(input_code, dec_i)
decoder.summary()

# Training
psnr_metrics = functools.partial(metrics.psnr, max_img_value=1.)  # Esto wrapea la funcion que creamos a Keras
psnr_metrics.__name__ = 'psnr'

autoencoder.compile(optimizer='adam', loss='binary_crossentropy',
                    metrics=[psnr_metrics])

history = autoencoder.fit(
    train_mnist_dataset, epochs=num_epochs, steps_per_epoch=train_steps_per_epoch,
    validation_data=val_mnist_dataset, validation_steps=val_steps_per_epoch,
    verbose=0, callbacks=[custom_callbacks.DynamicPlotCallback()])

# Test sample
num_show = 12

x_test_sample = next(val_mnist_dataset.__iter__())[0][:num_show].numpy()

x_decoded = autoencoder.predict_on_batch(x_test_sample).numpy()
print(type(x_decoded))

plot_functions.show_pairs(x_test_sample.reshape(num_show, *input_shape[:2]),
                          x_decoded.reshape(num_show, *input_shape[:2]))

# Test sample del codificador

x_encoded = encoder.predict_on_batch(x_test_sample)
print(type(x_encoded))
# We scale up the code, to better visualize it:
x_encoded_show = np.tile(x_encoded.reshape(num_show, 1, encoder_size), (1, 15, 1))

plot_functions.show_pairs(x_test_sample.reshape(num_show, *input_shape[:2]),
                          x_encoded_show, plot_fn_b="matshow")

# t-SNE
'''
While multiple solutions exist to find interesting projections given a multi-dimensional dataset (PCA, LDA, etc.), 
we will use here t-SNE. The t-Distributed Stochastic Neighbor Embedding$^2$ is an unsupervised technique developed by 
Laurens van der Maatens and Geoffrey Hinton in 2008. Given a dataset, this method returns a non-linear projection which 
tries to maximize the distance between all the elements after projection. 
'''
from sklearn.manifold import TSNE

# Creating a t-SNE instance to project our codes into 2D elements:
tsne = TSNE(n_components=2, verbose=1, random_state=0)

_, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test / 255.0
x_test = x_test.reshape(x_test.shape[0], *flattened_input_shape)

x_encoded = encoder.predict_on_batch(x_test)
x_2d = tsne.fit_transform(x_encoded)

import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'normal',
        'weight': 'bold',
        'size': 22}
matplotlib.rc('font', **font)
figure = plt.figure(figsize=(15, 10))
plt.scatter(x_2d[:, 0], x_2d[:, 1],  # 2D image projections
            c=y_test  # per-class colors
            , cmap=plt.cm.get_cmap("jet", num_class)
            # replace "jet" with "plasma" for colors more consistent when printed grayscale
            )
plt.colorbar(ticks=range(num_class))
plt.clim(-0.5, num_class - 0.5)
plt.show()

# Aca ploteamos los digitos encodeados (coloreamos con el verdadero label) para ver dos
# features que compactan la imagen
