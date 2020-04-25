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