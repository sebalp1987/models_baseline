import tensorflow as tf
import functools


class ConvBatchNorm(tf.keras.layers.Conv2D):
    """ Convultional Layer with Batch Normalization"""

    def __init__(self, activation='relu', name='convbn', **kwargs):
        """
        Initialize the layer.
        :param activation:   Activation function (name or callable)
        :param name:         Name suffix for the sub-layers.
        :param kwargs:       Mandatory and optional parameters of tf.keras.layers.Conv2D
        """
        self.activation = tf.keras.layers.Activation(activation, name=name + '_act')
        super().__init__(activation=None, name=name + '_c', **kwargs)
        self.batch_norm = tf.keras.layers.BatchNormalization(axis=-1, name=name + '_bn')

    def call(self, inputs, training=None):
        """
        Call the layer.
        :param inputs:         Input tensor to process
        :param training:       Flag to let TF knows if it is a training iteration or not
                                       (this will affect the behavior of BatchNorm)
        :return:               Convolved tensor
        """
        x = super().call(inputs)
        x = self.batch_norm(x, training=training)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ResMerge(tf.keras.layers.Layer):
    def __init__(self, name='block', **kwargs):
        """
        Initialize the layer.
        :param activation:   Activation function (name or callable)
        :param name:         Name suffix for the sub-layers.
        :param kwargs:       Optional parameters of tf.keras.layers.Conv2D
        """
        super().__init__(name=name)
        self.shortcut = None
        self.kwargs = kwargs

    def build(self, input_shape):
        x_shape = input_shape[0]
        x_residual_shape = input_shape[1]
        if x_shape[1] == x_residual_shape[1] and \
                x_shape[2] == x_residual_shape[2] and \
                x_shape[3] == x_residual_shape[3]:
            self.shortcut = functools.partial(tf.identity, name=self.name + '_shortcut')
        else:
            strides = (
                int(round(x_shape[1] / x_residual_shape[1])),  # vertical stride
                int(round(x_shape[2] / x_residual_shape[2]))  # horizontal stride
            )
            x_residual_channels = x_residual_shape[3]
            self.shortcut = ConvBatchNorm(filters=x_residual_channels, kernel_size=(1, 1),
                                          strides=strides, activation=None, name=self.name + 'shortcut_c',
                                          **self.kwargs)

    def call(self, inputs):
        """
        Call the layer.
        :param inputs:         Tuple of two input tensors to merge
        :return:               Merged tensor
        """
        x, x_residual = inputs
        x_shorcut = self.shortcut(x)
        x_merge = tf.keras.layers.add([x_shorcut, x_residual])
        return x_merge


class ResBlock(tf.keras.Model):
    def __init__(self, filters=64, kernel_size=3, strides=1, activation='relu',
                 kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                 name='resblock', **kwargs):
        """
        Initialize the layer.
        :param filters:                 Number of filters
        :param kernel_size:             Kernel size
        :param strides:                 Convolution strides
        :param activation:              Activation function (name or callable)
        :param kernel_initializer:      Kernel initialisation method name
        :param kernel_regularizer:      Kernel regularizer
        :param name:                    Name suffix for the sub-layers.
        :param kwargs:                  Optional parameters of tf.keras.layers.Conv2D
        """
        super().__init__()
        self.conv1 = ConvBatchNorm(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                                   activation=activation, kernel_initializer=kernel_initializer,
                                   kernel_regularizer=kernel_regularizer, name=name + '_c1',
                                   **kwargs)
        self.conv2 = ConvBatchNorm(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                                   activation=None, kernel_initializer=kernel_initializer,
                                   kernel_regularizer=kernel_regularizer, name=name + '_c2',
                                   **kwargs)
        self.merge = ResMerge(kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
                              name=name)

        self.activation = tf.keras.layers.Activation(activation, name=name + '_act')

    def call(self, inputs, training=None):
        """
        Call the block.
        :param inputs:         Input tensor to process
        :param training:       Flag to let TF knows if it is a training iteration or not
                               (this will affect the behavior of BatchNorm)
        :return:               Block output tensor
        """
        x = inputs
        x_res = self.conv1(x, training=training)
        x_res = self.conv2(x_res, training=training)

        x_merge = self.merge([x, x_res])
        x_merge = self.activation(x_merge)
        return x_merge


class ResBlockBottleneck(tf.keras.Model):
    def __init__(self, filters=16, kernel_size=1, strides=1, activation='relu', name='resblock',
                 kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                 **kwargs):

        """
        Initialize the block.
        :param filters:                 Number of filters
        :param kernel_size:             Kernel size
        :param strides:                 Convolution strides
        :param activation:              Activation function (name or callable)
        :param kernel_initializer:      Kernel initialisation method name
        :param kernel_regularizer:      Kernel regularizer
        :param name:                    Name suffix for the sub-layers.
        :param kwargs:                  Optional parameters of tf.keras.layers.Conv2D
        """
        super().__init__(name=name)

        self.conv0 = ConvBatchNorm(filters=filters, kernel_size=1, strides=strides,
                                   padding='valid', kernel_initializer=kernel_initializer,
                                   kernel_regularizer=kernel_regularizer, activation=activation, name=name + '_cb0',
                                   **kwargs)
        self.conv1 = ConvBatchNorm(filters=filters, kernel_size=kernel_size, strides=strides,
                                   padding='same', kernel_initializer=kernel_initializer,
                                   kernel_regularizer=kernel_regularizer, activation=activation, name=name + '_cb1',
                                   **kwargs)
        self.conv2 = ConvBatchNorm(filters=filters, kernel_size=1, strides=strides,
                                   padding='valid', kernel_initializer=kernel_initializer,
                                   kernel_regularizer=kernel_regularizer, activation=None, name=name + '_cb2',
                                   **kwargs)
        self.merge = ResMerge(kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
                              name=name)
        self.activation = tf.keras.layers.Activation(activation=activation, name=name + '_act')

    def call(self, inputs, training=None):
        """
        Call the layer.
        :param inputs:         Input tensor to process
        :param training:       Flag to let TF knows if it is a training iteration or not
                               (this will affect the behavior of BatchNorm)
        :return:               Block output tensor
        """
        x = inputs
        x_res = self.conv0(x, training=training)
        x_res = self.conv1(x_res, training=training)
        x_res = self.conv2(x_res, training=training)
        merge = self.merge([x, x_res])
        merge = self.activation(merge)
        return merge


class ResMacroBlock(tf.keras.models.Sequential):
    """ Macro-block, chaining multiple residual blocks (as a Sequential model)"""

    def __init__(self, block_class=ResBlockBottleneck, repetitions=3,
                 filters=16, kernel_size=1, strides=1, activation='relu',
                 kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                 name='res_macroblock', **kwargs):
        """
        Initialize the block.
        :param block_class:             Block class to be used.
        :param repetitions:             Number of times the block should be repeated inside.
        :param filters:                 Number of filters
        :param kernel_size:             Kernel size
        :param strides:                 Convolution strides
        :param activation:              Activation function (name or callable)
        :param kernel_initializer:      Kernel initialisation method name
        :param kernel_regularizer:      Kernel regularizer
        :param name:                    Name suffix for the sub-layers.
        :param kwargs:                  Optional parameters of tf.keras.layers.Conv2D
        """
        super().__init__(
            [block_class(
                filters=filters, kernel_size=kernel_size, activation=activation,
                strides=strides if i == 0 else 1, name="{}_{}".format(name, i),
                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
                for i in range(repetitions)],
            name=name)

class ResNet(tf.keras.Sequential):
    def __init__(self, input_shape, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-04),
                 block_class=ResBlockBottleneck, repetitions=(2, 2, 2, 2),
                 num_classes=1000, name='resnet'):
        filters = 64
        strides = 2
        super().__init__(
            # Input y First Layers
            [tf.keras.Input(shape=input_shape, name='input'),
             ConvBatchNorm(filters=filters, kernel_size=7, activation='relu', padding='same',
                           strides=strides, kernel_initializer=kernel_initializer,
                           kernel_regularizer=kernel_regularizer, name='conv'),
                tf.keras.layers.MaxPooling2D(pool_size=3, strides=strides, padding='same', name='max_pool')

            ] + \
            # Residual
            [
                ResMacroBlock(
                    block_class=block_class, repetitions=repet, filters=min(filters * (2 ** i), 1024),
                    kernel_size=3, activation='relu', strides=strides if i!=0 else 1, name='block_{}'.format(i),
                    kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer) for i, repet in
                    enumerate(repetitions)

            ] + \

            # Final Layer
            [
                tf.keras.layers.GlobalAveragePooling2D(name='avg_poll'),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(units=num_classes, activation='softmax', kernel_initializer=kernel_initializer)
            ], name=name

        )


# Standard ResNet versions:
class ResNet18(ResNet):
    def __init__(self, input_shape, num_classes=1000, name='resnet18',
                 kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4)):
        super().__init__(input_shape=input_shape, num_classes=num_classes,
                         block_class=ResBlock, repetitions=(2, 2, 2, 2),
                         kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)


class ResNet34(ResNet):
    def __init__(self, input_shape, num_classes=1000, name='resnet34',
                 kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4)):
        super().__init__(input_shape=input_shape, num_classes=num_classes,
                         block_class=ResBlock, repetitions=(3, 4, 6, 3),
                         kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)


class ResNet50(ResNet):
    def __init__(self, input_shape, num_classes=1000, name='resnet50',
                 kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4)):
        super().__init__(input_shape=input_shape, num_classes=num_classes,
                         block_class=ResBlockBottleneck, repetitions=(3, 4, 6, 3),
                         kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)


class ResNet101(ResNet):
    def __init__(self, input_shape, num_classes=1000, name='resnet101',
                 kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4)):
        super().__init__(input_shape=input_shape, num_classes=num_classes,
                         block_class=ResBlockBottleneck, repetitions=(3, 4, 23, 3),
                         kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)


class ResNet152(ResNet):
    def __init__(self, input_shape, num_classes=1000, name='resnet152',
                 kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4)):
        super().__init__(input_shape=input_shape, num_classes=num_classes,
                         block_class=ResBlockBottleneck, repetitions=(3, 8, 36, 3),
                         kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)


if __name__ == "__main__":
    input_shape = [224, 224, 3]
    num_classes = 1000
    model = ResNet50(input_shape=input_shape, num_classes=num_classes)
    print(model.summary())