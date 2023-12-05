import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def get_weights(layer_name, field_name, weights_path="models/pre-training/open_nsfw-weights.npy"):
    weights_o = np.load(weights_path, allow_pickle=True,
                        encoding='latin1').item()
    if layer_name not in weights_o:
        raise ValueError("No weights for layer named '{}'".format(layer_name))
    w = weights_o[layer_name]
    if field_name not in w:
        raise (ValueError("No entry for field '{}' in layer named '{}'").format(
            field_name, layer_name))
    return w[field_name]


def batch_norm(name, trainable=True):
    bn_epsilon = 1e-5
    return layers.BatchNormalization(trainable=trainable, epsilon=bn_epsilon,
                                     gamma_initializer=tf.constant_initializer(
                                         get_weights(name, "scale")),
                                     beta_initializer=tf.constant_initializer(
                                         get_weights(name, "offset")),
                                     moving_mean_initializer=tf.compat.v1.constant_initializer(
                                         get_weights(name, "mean")),
                                     moving_variance_initializer=tf.constant_initializer(get_weights(name, "variance")), name=name)


def fully_connected(name, num_outputs):
    return layers.Dense(
        units=num_outputs, name=name,
        kernel_initializer=tf.constant_initializer(
            get_weights(name, "weights")),
        bias_initializer=tf.constant_initializer(
            get_weights(name, "biases")))


def standardize(img):
    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean) / std
    return img
