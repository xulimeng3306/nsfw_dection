import math
import tensorflow as tf
from tensorflow.keras import layers
from utils import get_weights, batch_norm


class Conv2d(layers.Layer):
    def __init__(self, name, filter_depth, kernel_size, stride=1, padding='same', trainable=True):
        super(Conv2d, self).__init__()
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv2 = layers.Conv2D(filter_depth, kernel_size=(kernel_size, kernel_size), strides=(stride, stride), padding="valid", activation=None, trainable=trainable, name=name, kernel_initializer=tf.constant_initializer(
            get_weights(name, "weights")), bias_initializer=tf.constant_initializer(
            get_weights(name, "biases"))
        )

    def call(self, inputs):
        if self.padding.lower() == 'same' and self.kernel_size > 1:
            if self.kernel_size > 1:
                oh = inputs.shape.as_list()[1]
                h = inputs.shape.as_list()[1]
                p = int(math.floor(
                    ((oh - 1) * self.stride + self.kernel_size - h) // 2))
                inputs = tf.pad(inputs, [[0, 0], [p, p], [p, p], [0, 0]])
            else:
                raise Exception(
                    "unsuported kernel size for padding: '{}'".format(self.kernel_size))
        return self.conv2(inputs)


class BasicBlock(layers.Layer):
    def __init__(self, stage, block, filterdepths, kernel_size=3, stride=2):
        super(BasicBlock, self).__init__()

        self.filter_depth1, self.filter_depth2, self.filter_depth3 = filterdepths

        self.conv_name_base = "conv_stage{}_block{}_branch".format(
            stage, block)
        self.bn_name_base = "bn_stage{}_block{}_branch".format(stage, block)
        self.shortcut_name_post = "_stage{}_block{}_proj_shortcut".format(
            stage, block)

        # unit_1
        self.conv0 = Conv2d("conv{}".format(self.shortcut_name_post), self.filter_depth3, kernel_size=1, stride=stride,
                            padding='same')

        self.bn0 = batch_norm("bn{}".format(self.shortcut_name_post))

        # self.relu0 = layers.Activation('relu')

        # 2a
        self.conv1 = Conv2d("{}2a".format(self.conv_name_base), self.filter_depth1, kernel_size=1, stride=stride,
                            padding='same')

        self.bn1 = batch_norm("{}2a".format(self.bn_name_base))

        self.relu1 = layers.Activation('relu')

        # 2b
        self.conv2 = Conv2d("{}2b".format(self.conv_name_base), self.filter_depth2, kernel_size=kernel_size, stride=1,
                            padding='same')

        self.bn2 = batch_norm("{}2b".format(self.bn_name_base))

        self.relu2 = layers.Activation('relu')

        # 2c
        self.conv3 = Conv2d("{}2c".format(self.conv_name_base), self.filter_depth3, kernel_size=1, stride=1,
                            padding='same')

        self.bn3 = batch_norm("{}2c".format(self.bn_name_base))

    def call(self, inputs, training=None):
        # print("input shape", inputs.shape)
        # print(self.conv0(inputs).shape)
        shortcut = self.bn0(self.conv0(inputs))

        x = self.conv1(inputs)
        x = self.relu1(self.bn1(x))

        x = self.conv2(x)
        x = self.relu2(self.bn2(x))

        x = self.conv3(x)
        x = self.bn3(x)

        x = tf.add(x, shortcut)

        return tf.nn.relu(x)


class Identity_block_layer(layers.Layer):
    def __init__(self, stage, block, filter_depths, kernel_size):
        super(Identity_block_layer, self).__init__()
        self.filter_depth1, self.filter_depth2, self.filter_depth3 = filter_depths
        self.conv_name_base = "conv_stage{}_block{}_branch".format(
            stage, block)
        self.bn_name_base = "bn_stage{}_block{}_branch".format(stage, block)

        # 2a
        self.conva = Conv2d("{}2a".format(self.conv_name_base), filter_depth=self.filter_depth1, kernel_size=1,
                            stride=1, padding="same")
        self.bna = batch_norm("{}2a".format(self.bn_name_base))

        # 2b
        self.convb = Conv2d("{}2b".format(self.conv_name_base), filter_depth=self.filter_depth2,
                            kernel_size=kernel_size, stride=1, padding="same")
        self.bnb = batch_norm("{}2b".format(self.bn_name_base))

        # 2c
        self.convc = Conv2d("{}2c".format(self.conv_name_base), filter_depth=self.filter_depth3, kernel_size=1,
                            stride=1, padding="same")
        self.bnc = batch_norm("{}2c".format(self.bn_name_base))

    def call(self, inputs):
        x = self.bna(self.conva(inputs))
        x = tf.nn.relu(x)

        x = self.bnb(self.convb(x))
        x = tf.nn.relu(x)

        x = self.bnc(self.convc(x))
        x = tf.add(x, inputs)
        return tf.nn.relu(x)
