import tensorflow as tf

def _prepare_data_fn(features, target='label', flatten=True,
                     return_batch_as_tuple=True, seed=None):
    """
    Resize image to expected dimensions, and opt. apply some random transformations.
    :param features:              Data
    :param target                 Target/ground-truth data to be returned along the images
                                  ('label' for categorical labels, 'image' for images, or None)
    :param flatten:               Flag to flatten the images, from (28, 28, 1) to (784,)
    :param return_batch_as_tuple: Flag to return the batch data as tuple instead of dict
    :param seed:                  Seed for random operations
    :return:                      Processed data
    """

    # Tensorflow-Dataset returns batches as feature dictionaries, expected by Estimators.
    # To train Keras models, it is more straightforward to return the batch content as tuples.

    image = features['image']
    # Convert the images to float type, also scaling their values from [0, 255] to [0., 1.]:
    image = tf.image.convert_image_dtype(image, tf.float32)

    if flatten:
        is_batched = len(image.shape) > 3
        flattened_shape = (-1, 784) if is_batched else (784,)
        image = tf.reshape(image, flattened_shape)

    if target is None:
        return image if return_batch_as_tuple else {'image': image}
    else:
        features['image'] = image
        return (image, features[target]) if return_batch_as_tuple else features


def _prepare_data_fn_resize(features, scale_factor=4, augment=True,
                     return_batch_as_tuple=True, seed=None):
    """
    Resize image to expected dimensions, and opt. apply some random transformations.
    :param features:              Data
    :param scale_factor:          Scale factor for the task
    :param augment:               Flag to augment the images with random operations
    :param return_batch_as_tuple: Flag to return the batch data as tuple instead of dict
    :param seed:                  Seed for random operations
    :return:                      Processed data
    """

    # Tensorflow-Dataset returns batches as feature dictionaries, expected by Estimators.
    # To train Keras models, it is more straightforward to return the batch content as tuples.

    image = features['image']
    # Convert the images to float type, also scaling their values from [0, 255] to [0., 1.]:
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Computing the scaled-down shape:
    original_shape = tf.shape(image)
    original_size = original_shape[-3:-1]
    scaled_size = original_size // scale_factor
    # Just in case the original dimensions were not a multiple of `scale_factor`,
    # we slightly resize the original image so its dimensions now are
    # (to make the loss/metrics computations easier during training):
    original_size_mult = scaled_size * scale_factor

    # Opt. augmenting the image:
    if augment:
        original_shape_mult = (original_size_mult, [tf.shape(image)[-1]])
        if len(image.shape) > 3:  # batched data:
            original_shape_mult = ([tf.shape(image)[0]], *original_shape_mult)
        original_shape_mult = tf.concat(original_shape_mult, axis=0)

        # Randomly applied horizontal flip:
        image = tf.image.random_flip_left_right(image)

        # Random B/S changes:
        image = tf.image.random_brightness(image, max_delta=0.1, seed=seed)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5, seed=seed)
        image = tf.clip_by_value(image, 0.0, 1.0)  # keeping pixel values in check

        # Random resize and random crop back to expected size:
        random_scale_factor = tf.random.uniform([1], minval=1.0, maxval=1.2,
                                                dtype=tf.float32, seed=seed)
        scaled_height = tf.cast(tf.multiply(tf.cast(original_size[0], tf.float32),
                                            random_scale_factor),
                                tf.int32)
        scaled_width = tf.cast(tf.multiply(tf.cast(original_size[1], tf.float32),
                                           random_scale_factor),
                               tf.int32)
        scaled_shape = tf.squeeze(tf.stack([scaled_height, scaled_width]))
        image = tf.image.resize(image, scaled_shape)
        image = tf.image.random_crop(image, original_shape, seed=seed)

    # Generating the data pair for super-resolution task,
    # i.e. the downscaled image + its original version
    image_downscaled = tf.image.resize(image, scaled_size)

    # Just in case the original dimensions were not a multiple of `scale_factor`,
    # we slightly resize the original image so its dimensions now are
    # (to make the loss/metrics computations easier during training):
    original_size_mult = scaled_size * scale_factor
    image = tf.image.resize(image, original_size_mult)

    features = (image_downscaled, image) if return_batch_as_tuple else {'image': image_downscaled,
                                                                        'label': image}
    return features
