import tensorflow as tf
import os
from matplotlib import pyplot as plt
import math
import tensorflow_datasets as tfds
from resources.plot_functions import plot_image_grid
from resources.image_preprocessing import _prepare_data_fn_resize
import math
import functools
'''
Sirve para aumentar la calidad de imágenes de categorías específicas
'''

# Some hyper-parameters:
batch_size = 32  # Images per batch (reduce/increase according to the machine's capability)
num_epochs = 300  # Max number of training epochs
random_seed = 42  # Seed for some random operations, for reproducibility
scale_factor = 4

# Usamos la base de datos de Rock-Paper-Scissors (esta en beta asi que usamos nighty-tfds
hands_builder = tfds.builder('rock_paper_scissors')
hands_builder.download_and_prepare()
print(hands_builder.info)

# Ploteamos
num_show = 5
hands_val_dataset = hands_builder.as_dataset(split=tfds.Split.TEST).batch(num_show)
hands_val_dataset_iter = hands_val_dataset.skip(1).__iter__() # Saltea el primer batch porque no tiene tanta diversidad de manos
batch = next(hands_val_dataset_iter)
fig = plot_image_grid([batch['image'].numpy()], titles=['image'], transpose=True)
plt.show()

# Training
num_train_imgs = hands_builder.info.splits['train'].num_examples
num_val_imgs = hands_builder.info.splits['test'].num_examples

train_steps_per_epoch = math.ceil(num_train_imgs / batch_size)
val_steps_per_epoch = math.ceil(num_val_imgs / batch_size)

input_shape = hands_builder.info.features['image'].shape

# Vamos a ver cómo funciona el upsampling. Entonces primero reducimos la calidad de las imagenes
prepare_data_fn_downscaled = functools.partial(_prepare_data_fn_resize, scale_factor=scale_factor, augment=True,
                                               return_batch_as_tuple=True, seed=random_seed)

train_hands = hands_builder.as_dataset(split=tfds.Split.TRAIN)
train_hands = train_hands.repeat(num_epochs)  # Repite el dataset el numero de epochs
train_hands = train_hands.shuffle(hands_builder.info.splits['train'].num_examples, seed=random_seed)  # shuffle
train_hands = train_hands.batch(batch_size)  # Divide el dataset en batches
train_hands = train_hands.map(prepare_data_fn_downscaled,
                              num_parallel_calls=4)  # Le aplica la funcion paralelamente de a 4 elementos
train_hands = train_hands.prefetch(1)  # Esto toma n (=1) batches de (batch_size=32) ejemplos cada uno.

# Validation
prepare_data_fn_downscaled_valid = functools.partial(_prepare_data_fn_resize, scale_factor=scale_factor, augment=False,
                                                     return_batch_as_tuple=True, seed=random_seed)
val_hands = hands_builder.as_dataset(split=tfds.Split.TEST)
val_hands = val_hands.repeat(num_epochs)  # No le aplicamos shuffle
val_hands = val_hands.batch(batch_size)
val_hands = val_hands.map(prepare_data_fn_downscaled_valid, num_parallel_calls=4)
val_hands = val_hands.prefetch(1)

# =====================================================================================================================
# Vemos si lo hicimos bien, y visualizamos un batch
val_hands_dataset_show = val_hands.take(1)  # tomamos 1
val_img_input, val_img_target = next(val_hands_dataset_show.__iter__())
val_img_input = val_img_input[num_show:(num_show * 2)]  # saltamos el primer batch que no es tan diverso
val_img_target = val_img_target[num_show: (num_show * 2)]

# La UPSCALEAMOS de vuelta (el target es el original), para ver el efecto de estas transformaciones
val_img_input_upscale = tf.image.resize(val_img_input, tf.shape(val_img_target)[1:3])
val_psnr_result = tf.image.psnr(val_img_target, val_img_input_upscale, max_val=1.)

fig = plot_image_grid([val_img_input_upscale.numpy(), val_img_target.numpy()], titles=['scaled', 'original'],
                      transpose=True)
plt.show()
print("PSNR for each pair: {}".format(val_psnr_result.numpy()))

"""

We can clearly see the upscaling artifacts / missing details in the tampered images, as confirmed by their low PSNR.
"""
# =====================================================================================================================

# AUTOENCODER
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda, Conv2D, Conv2DTranspose


'''
Lo que buscamos es devolver a la imagen original, rescatando lo que se perdió primero con el downscaling
(las transformaciones que le aplicamos antes), y si le metemos algun upscaling.
'''

# 2) Creamos el Deep AE
def simple_dae(inputs, kernel_size=3, filters_orig=16, layer_depth=4):
    """
       Build a simple fully-convolutional DAE network.
       :param inputs:        Input tensor/placeholder
       :param kernel_size:   Kernel size for the convolutions
       :param filters_orig:  Number of filters for the 1st CNN layer (then multiplied by 2 each layer)
       :param layer_depth:   Number of layers composing the encoder/decoder
       :return:              DAE network (Keras Functional API)
    """

    # Encoding Layers
    filters = filters_orig
    x = inputs
    for i in range(layer_depth):
        x = Conv2D(filters, kernel_size, strides=2, padding='same', activation='relu',
                   name='enc_conv{}'.format(i))(x)
        filters = min(filters*2, 512)

    # Decoding Layers
    for i in range(layer_depth):
        filters = max(filters // 2, filters_orig)
        x = Conv2DTranspose(filters, kernel_size, strides=2, padding='same', activation='relu',
                   name='dec_conv{}'.format(i))(x)
    decoded = Conv2D(inputs.shape[-1], kernel_size=1, padding='same', activation='sigmoid',
                   name='dec_output')(x)

    return decoded

input_example = Input(shape=(91, 91, 3), name='input')
output_example = simple_dae(inputs=input_example, kernel_size=4, filters_orig=32, layer_depth=4)
ae_example = Model(input_example, output_example)
ae_example.summary()

'''
Si vemos este ejemplo, donde las dimensiones de las imagenes no estan normalizadas o no son parejas, puede que no 
obtengamos el tamaño original exacto. (91 vs 96 pixeles). Le agregamos entonces un layer que resizee a lo mismo
'''
# 1) Primero le aplicamos un upscale a la imagen que fue donwsized, para que vuelva a la escala original
Upscale = lambda name: Lambda(lambda images: tf.image.resize(images, tf.shape(images)[-3:-1]*scale_factor), name=name)

# Este corrige los pequeños errores de pixeles
ResizeToSame = lambda name: Lambda(
    lambda images: tf.image.resize(images[0], tf.shape(images[1])[-3:-1]),
    # `images` is a tuple of 2 tensors.
    # We resize the first image tensor to the shape of the 2nd
    name=name)

def dae_super_resolution(inputs, kernel_size=3, filters_orig=16, layer_depth=4):
    # Para tener un autoencoder simetrico, le hacemos un upscaling a la imagen que sea igual al del target.
    resized_inputs = Upscale(name='upscale_input')(inputs)

    # El autoencoder aplicado, lo que hace es remover el ruido generado por esta operacion.
    decoded = simple_dae(inputs=resized_inputs, kernel_size=kernel_size, filters_orig=filters_orig, layer_depth=layer_depth)

    # Para que no haya ese pequeño error de pixeles, le hacemos el ResizeToSame
    decoded = ResizeToSame(name='dec_output_scale')([decoded, resized_inputs])

    return decoded

inputs = Input(shape=(None, None, input_shape[-1]), name='input')
decoded = dae_super_resolution(inputs, kernel_size=4, filters_orig=32, layer_depth=4)
autoencoder = Model(inputs, decoded)
autoencoder.summary(positions=[.35, .65, .75, 1.])

# TRAINING
# Callbacks and Metrics
import collections
from resources.custom_callbacks import SimpleLogCallback, TensorBoardImageGridCallback

psnr_metric = functools.partial(tf.image.psnr, max_val=1.)
psnr_metric.__name__ = 'psnr'

model_dir = os.path.join('.', 'models', 'superres_dae')

metrics_to_print = collections.OrderedDict([("loss", "loss"), ("v-loss", "val_loss"),
                                            ("psnr", "psnr"), ("v-psnr", "val_psnr")])
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
        log_dir=model_dir, input_images=val_img_input.numpy(), target_images=val_img_target.numpy(),
        tag='ae_super_res_results', figsize=(num_show * 3, 3 * 3),
        grayscale=True, transpose=True)
]

optimizer = tf.optimizers.Adam(learning_rate=1e-4)
autoencoder.compile(optimizer=optimizer, loss='mae', metrics=[psnr_metric])

history = autoencoder.fit(train_hands, epochs=num_epochs, steps_per_epoch=train_steps_per_epoch,
                          validation_steps=val_steps_per_epoch, validation_data=val_hands, verbose=0,
                          callbacks=callbacks)

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

'''
Los resultados muestran que apenas estamos por debajo de 27 PSNR, cuando vimos que con un simple upscaling estabamos en
31 (lo llama bilinear interpolation que es como un Average Unpooling. 
NOTA: PSNR EVALUA QUE TAN MALO EL RUIDO ES. DOS IMAGENES CON EL MISMO PSNR PUEDEN TENER DIFERENCIAS DE CALIDAD, POR 
EJ UNA IMAGEN QUE TENGA MUCHO RUIDO EN APENAS UNA REGION, Y QUE PARA EL OJO HUMANO NO PAREZCA.
Si vemos el Tensorboard, notamos que hay algunas deficiencias de calidad.
'''

