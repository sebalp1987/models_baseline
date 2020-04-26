import functools
from .image_preprocessing import _prepare_data_fn
import tensorflow_datasets as tfds

def get_mnist_dataset(phase='train', target='label', batch_size=32, num_epochs=None,
                      shuffle=True, flatten=True, return_batch_as_tuple=True, seed=None):
    """
    Instantiate a CIFAR-100 dataset.
    :param phase:                 Phase ('train' or 'val')
    :param target                 Target/ground-truth data to be returned along the images
                                  ('label' for categorical labels, 'image' for images, or None)
    :param batch_size:            Batch size
    :param num_epochs:            Number of epochs (to repeat the iteration - infinite if None)
    :param shuffle:               Flag to shuffle the dataset (if True)
    :param flatten:               Flag to flatten the images, from (28, 28, 1) to (784,)
    :param return_batch_as_tuple: Flag to return the batch data as tuple instead of dict
    :param seed:                  Seed for random operations
    :return:                      Iterable Dataset
    """
    mnist_builder = tfds.builder("mnist")
    mnist_builder.download_and_prepare()

    mnist_info = mnist_builder.info
    print(mnist_info)
    assert (phase == 'train' or phase == 'test')

    prepare_data_fn = functools.partial(_prepare_data_fn, return_batch_as_tuple=return_batch_as_tuple,
                                        target=target, flatten=flatten, seed=seed)

    mnist_dataset = mnist_builder.as_dataset(split=tfds.Split.TRAIN if phase == 'train' else tfds.Split.TEST)
    mnist_dataset = mnist_dataset.repeat(num_epochs)
    if shuffle:
        mnist_dataset = mnist_dataset.shuffle(10000, seed=seed)
    mnist_dataset = mnist_dataset.batch(batch_size)
    mnist_dataset = mnist_dataset.map(prepare_data_fn, num_parallel_calls=4)
    mnist_dataset = mnist_dataset.prefetch(1)

    return mnist_dataset

