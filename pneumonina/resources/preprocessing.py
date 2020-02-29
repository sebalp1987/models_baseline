import tensorflow as tf


def _prepare_data_fn(features, input_shape, augment=False):
    input_shape = tf.convert_to_tensor(input_shape)
    image = features['image']
    label = features['label']

    image = tf.image.convert_image_dtype(image, tf.float32)  # convertimos la imagen a float

    if augment:  # Si data augmentation le mandamos transformaciones random
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.clip_by_value(image, 0., 1.)

        random_scale_factor = tf.random.uniform([1], minval=1., maxval=1.4, dtype=tf.float32)
        scaled_height = tf.cast(tf.cast(input_shape[0], tf.float32) * random_scale_factor, tf.int32)
        scaled_width = tf.cast(tf.cast(input_shape[1], tf.float32) * random_scale_factor, tf.int32)
        scaled_shape = tf.squeeze(tf.stack([scaled_height, scaled_width]))
        image = tf.image.resize(image, scaled_shape)
        image = tf.image.random_crop(image, input_shape)
    else:
        image = tf.image.resize(image, input_shape[:2])

    return image, label