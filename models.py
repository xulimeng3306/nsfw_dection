import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras import models, layers
from basic_layers import Conv2d, get_block_arrs
from utils import batch_norm, fully_connected


class ResModel(models.Model):
    def __init__(self, num_blocks, block_strides, input_type=1, output_classer=2, filter_depths=[32, 32, 128], kernel_size=3):
        super(ResModel, self).__init__()
        self.input_type = input_type

        self.conv2 = Conv2d("conv_1", filter_depth=64,
                            kernel_size=7, stride=2, padding="valid")
        self.bn = batch_norm(name="bn_1")
        self.pool = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')

        self.all_blocks = get_block_arrs(
            filter_depths=filter_depths, num_blocks=num_blocks, kernel_size=kernel_size, block_strides=block_strides)
        self.average_pooling2d = layers.AveragePooling2D(
            pool_size=7, strides=1, padding="valid", name="avergepool")

        self.logits = fully_connected(
            name="fc_nsfw", num_outputs=output_classer)

    def call(self, inputs, is_training=None):

        x = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], 'CONSTANT')
        # print("-----------------input=", inputs, "x=", x.shape)
        # print("pad", x.shape)
        x = self.conv2(x)
        x = self.bn(x)
        x = tf.nn.relu(x)

        # print("__conv2d : ", x.shape)
        x = self.pool(x)
        # print("MaxPool2D  : ", x.shape)

        for stage_arr in self.all_blocks:
            for block in stage_arr:
                x = block(x)

        x = self.average_pooling2d(x)
        # print("before reshape", x.shape)
        x = tf.reshape(tensor=x, shape=(-1, 1024))
        x = self.logits(x)
        # print("logits: ", x.shape)
        if not is_training:
            x = tf.nn.softmax(logits=x, name="predictions")
        return x
