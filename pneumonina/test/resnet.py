from resources import custom_callbacks, STRING
from config import config
from models import resnet
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import collections
import math

test = os.path.join(STRING.data_input_path, 'test')
train = os.path.join(STRING.data_input_path, 'train')
valid = os.path.join(STRING.data_input_path, STRING.data_input_path, 'val')

batch_image = config.global_params.get('batch_image')
num_epochs = config.model_params.get('num_epochs')
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
                                                    target_size=(224, 224),
                                                    batch_size=batch_image,
                                                    class_mode='binary',
                                                    color_mode='grayscale')
test_generator = test_datagen.flow_from_directory(
    test,
    target_size=(224, 224),
    batch_size=batch_image,
    class_mode='binary',
    color_mode='grayscale'
)

# Check Shape of a Batch
batch_x, batch_y = train_generator.next()
print(batch_x.shape)

# MODEL
'''
conv = tf.keras.applications.ResNet50(
    include_top=False,
    input_shape=(224, 224, 1),
    weights=None
)
print(conv.summary())

model = tf.keras.models.Sequential()
model.add(conv)
model.add(tf.keras.layers.GlobalAveragePooling2D(name='avg_poll'))
model.add(tf.keras.layers.Dense(units=2, activation='softmax', kernel_initializer='he_normal'))
print(model.summary())
'''
model = resnet.ResNet50(input_shape=config.global_params.get('input_shape'), num_classes=2)
print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=[tf.metrics.SparseCategoricalAccuracy(name='acc'),
                       tf.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_acc')])

# Usamos el callback que creamos
metrics_to_print = collections.OrderedDict([('loss', 'loss'), ('v-loss', 'val_loss'),
                                            ('acc', 'acc'), ('v-acc', 'val_acc'),
                                            ('top5-acc', 'top5_acc'), ('v-top5-acc', 'val_top5_acc')])

callback_simple_log = custom_callbacks.SimpleLogCallback(metrics_to_print, num_epochs=num_epochs, log_frequency=2)

model_dir = '.'
callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_acc',
                                              restore_best_weights=True),
             tf.keras.callbacks.TensorBoard(log_dir=model_dir, histogram_freq=0, write_graph=True),
             callback_simple_log]

history = model.fit(train_generator, epochs=num_epochs, steps_per_epoch=math.ceil(num_train_img / batch_image),
                    validation_data=test_generator, validation_steps=math.ceil(num_test_img / batch_image),
                    verbose=1, callbacks=callbacks)

# PLOT
fig, ax = plt.subplots(3, 2, figsize=(15, 10),
                       sharex='col')  # add parameter `sharey='row'` for a more direct comparison
ax[0, 0].set_title("loss")
ax[0, 1].set_title("val-loss")
ax[1, 0].set_title("acc")
ax[1, 1].set_title("val-acc")
ax[2, 0].set_title("top5-acc")
ax[2, 1].set_title("val-top5-acc")

ax[0, 0].plot(history.history['loss'])
ax[0, 1].plot(history.history['val_loss'])
ax[1, 0].plot(history.history['acc'])
ax[1, 1].plot(history.history['val_acc'])
ax[2, 0].plot(history.history['top5_acc'])
ax[2, 1].plot(history.history['val_top5_acc'])

plt.show()

best_val_acc = max(history.history['val_acc']) * 100
best_val_top5 = max(history.history['val_top5_acc']) * 100

print('Best val acc:  {:2.2f}%'.format(best_val_acc))
print('Best val top5: {:2.2f}%'.format(best_val_top5))
