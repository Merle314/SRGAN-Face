import tensorflow as tf
import numpy as np

distance_matrix = None

def get_dist_matrix():

    global distance_matrix

    # distance matrix for blur process
    if type(distance_matrix)!=np.ndarray:
        kernel_radius = 3
        kernel_size = int(np.ceil(6*kernel_radius))
        kernel_size = kernel_size + 1 if kernel_size%2==0 else kernel_size
        distance_matrix = np.zeros([kernel_size, kernel_size])
        center = kernel_size/2
        for i in range(kernel_size):
            for j in range(kernel_size):
                distance_matrix[i,j]= ((center-i)**2 + (center-j)**2)**.5
        distance_matrix = np.expand_dims(distance_matrix,2)
        distance_matrix = np.expand_dims(distance_matrix,3)

    return distance_matrix

def subtracttract_mean(image, mean_value):

    return image - mean_value

def reverse_grayscale(image):
    """ Reverse pixel values
    """

    return tf.reduce_max(image)-(image-tf.reduce_min(image))

def random_reverse_grayscale(image):

    rand_value = tf.random_uniform([])
    rand_cond = tf.greater_equal(rand_value,0.5)

    return tf.cond(rand_cond, lambda: reverse_grayscale(image),
                              lambda: image)

def random_brightness(image, max_delta, always=False):

    if always:
        return tf.image.random_brightness(image, max_delta=max_delta)
    else:
        rand_value = tf.random_uniform([])
        rand_cond = tf.greater_equal(rand_value, 0.5)

        return tf.cond(rand_cond, lambda: tf.image.random_brightness(image, max_delta=max_delta),
                                  lambda: image)

def random_contrast(image, lower, upper, always=False):

    if always:
        return tf.image.random_contrast(image, lower=lower, upper=upper)
    else:
        rand_value = tf.random_uniform([])
        rand_cond = tf.greater_equal(rand_value, 0.5)

        return tf.cond(rand_cond, lambda: tf.image.random_contrast(image, lower=lower, upper=upper),
                                  lambda: image)

def adjust_gamma(image, gamma=1.0, gain=1.0):

    #NOTE: pixel values should lie within [0,1]
    return image ** gamma * gain

def random_gamma(image, max_delta, gain=1, always=False):

    def _random_adjust(image, max_delta, gain=1):
        rand_gamma = tf.random_uniform([], minval=1.0-max_delta, maxval=1.0+max_delta)
        return adjust_gamma(image, gamma=rand_gamma, gain=gain)

    if always:
        return _random_adjust(image, max_delta=max_delta, gain=gain)
    else:
        rand_value = tf.random_uniform([])
        rand_cond = tf.greater_equal(rand_value, 0.5)
        return tf.cond(rand_cond, lambda: _random_adjust(image, max_delta=max_delta, gain=gain),
                                  lambda: image)

def blur(image, sigma, device_id):

    dist = get_dist_matrix()
    kernel = tf.exp(-dist/(2*sigma**2))
    kernel = kernel/tf.reduce_sum(kernel)
    image = tf.expand_dims(image,0)
    with tf.device('/gpu:%d'%device_id):
        processed = tf.nn.conv2d(image, kernel, [1,1,1,1], 'SAME')
    processed = tf.squeeze(processed, axis=0)

    return processed

def random_blur(image, min_sigma, max_sigma, device_id, always=True):
    
    def _random_adjust(image):
        sigma = tf.random_uniform([], minval=min_sigma, maxval=max_sigma)
        return blur(image, sigma, device_id)

    if always:
        return _random_adjust(image)
    else:
        rand_value = tf.random_uniform([])
        rand_cond = tf.greater_equal(rand_value, 0.5)
        return tf.cond(rand_cond, lambda: _random_adjust(image),
                                  lambda: image)
        
def sharpen(image, sigma, amount, device_id):
    blurred = blur(image, sigma, device_id)
    processed = image + (image - blurred) * amount

    return processed

def random_sharpen(image, min_sigma, max_sigma, min_amount, max_amount, device_id, always=True):
    
    def _random_adjust(image):
        sigma = tf.random_uniform([], minval=min_sigma, maxval=max_sigma)
        amount = tf.random_uniform([], minval=min_amount, maxval=max_amount)
        return sharpen(image, sigma, amount, device_id)

    if always:
        return _random_adjust(image)
    else:
        rand_value = tf.random_uniform([])
        rand_cond = tf.greater_equal(rand_value, 0.5)
        return tf.cond(rand_cond, lambda: _random_adjust(image),
                                  lambda: image)

def random_blur_or_sharpen(image, params, device_id):
    sigma = tf.random_uniform([], minval=params[0], maxval=params[1])
    amount = tf.random_uniform([], minval=params[2], maxval=params[3])

    rand_value = tf.random_uniform([])
    rand_cond = tf.greater_equal(rand_value, 0.5)

    return tf.cond(rand_cond, lambda: blur(image, sigma, device_id), lambda: sharpen(image, sigma, amount, device_id))
