import tensorflow as tf

def log_n(x, n=10):
    """
    Compute log_n(x), i.e. the log base `n` value of `x`.
    :param x:   Input tensor
    :param n:   Value of the log base
    :return:    Log result
    """
    log_e = tf.math.log(x)
    div_log_n = tf.math.log(tf.constant(n, dtype=log_e.dtype))
    return log_e / div_log_n


def psnr(img_a, img_b, max_img_value=255):
    """
    Compute the PSNR (Peak Signal-to-Noise Ratio) between two images.
    it measures the quality of a corrupted or recovered signal/image compared to its original version.
    The higher the value, the closer to the original image (the value is in decibels, i.e. following a
    logarithmic scale)
    :param img_a:           Image A
    :param img_b:           Image B
    :param max_img_value:   Maximum possible pixel value of the images
    :return:                PSNR value
    """
    mse = tf.reduce_mean((img_a - img_b) ** 2)
    return 20 * log_n(max_img_value, 10) - 10 * log_n(mse, 10)