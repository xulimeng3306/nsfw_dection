import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from basic_layers import Conv2d, BasicBlock, Identity_block_layer
from utils import batch_norm, fully_connected


class ResModel(models.Model):
    def __init__(self, input_type=1):
        super(ResModel, self).__init__()
        self.input_type = input_type

        self.conv2 = Conv2d("conv_1", filter_depth=64,
                            kernel_size=7, stride=2, padding="valid")
        self.bn = batch_norm("bn_1")
        self.pool = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')

        # stage0
        self.conv_bolck_0_0 = BasicBlock(
            0, 0, filterdepths=[32, 32, 128], kernel_size=3, stride=1)
        self.identity_0_1 = Identity_block_layer(
            stage=0, block=1, filter_depths=[32, 32, 128], kernel_size=3)
        self.identity_0_2 = Identity_block_layer(
            stage=0, block=2, filter_depths=[32, 32, 128], kernel_size=3)

        # stage1
        self.conv_bolck_1_0 = BasicBlock(
            1, 0, filterdepths=[64, 64, 256], kernel_size=3, stride=2)
        self.identity_1_1 = Identity_block_layer(
            stage=1, block=1, filter_depths=[64, 64, 256], kernel_size=3)
        self.identity_1_2 = Identity_block_layer(
            stage=1, block=2, filter_depths=[64, 64, 256], kernel_size=3)
        self.identity_1_3 = Identity_block_layer(
            stage=1, block=3, filter_depths=[64, 64, 256], kernel_size=3)

        # stage2
        self.conv_bolck_2_0 = BasicBlock(
            2, 0, filterdepths=[128, 128, 512], kernel_size=3, stride=2)
        self.identity_2_1 = Identity_block_layer(stage=2, block=1, filter_depths=[
                                                 128, 128, 512], kernel_size=3)
        self.identity_2_2 = Identity_block_layer(stage=2, block=2, filter_depths=[
                                                 128, 128, 512], kernel_size=3)
        self.identity_2_3 = Identity_block_layer(stage=2, block=3, filter_depths=[
                                                 128, 128, 512], kernel_size=3)
        self.identity_2_4 = Identity_block_layer(stage=2, block=4, filter_depths=[
                                                 128, 128, 512], kernel_size=3)
        self.identity_2_5 = Identity_block_layer(stage=2, block=5, filter_depths=[
                                                 128, 128, 512], kernel_size=3)

        # stage3
        self.conv_bolck_3_0 = BasicBlock(
            3, 0, filterdepths=[256, 256, 1024], kernel_size=3, stride=2)
        self.identity_3_1 = Identity_block_layer(stage=3, block=1, filter_depths=[
                                                 256, 256, 1024], kernel_size=3)
        self.identity_3_2 = Identity_block_layer(stage=3, block=2, filter_depths=[
                                                 256, 256, 1024], kernel_size=3)

        self.average_pooling2d = layers.AveragePooling2D(
            pool_size=7, strides=1, padding="valid", name="avergepool")

        self.logits = fully_connected(name="fc_nsfw", num_outputs=2)

    def call(self, inputs):

        # print("before pad : ", self.input_tensor.shape)
        x = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], 'CONSTANT')
        # print("pad", x.shape)
        x = self.conv2(x)
        x = self.bn(x)
        x = tf.nn.relu(x)

        # print("__conv2d : ", x.shape)
        x = self.pool(x)
        # print("MaxPool2D  : ", x.shape)
        # stage0
        x = self.conv_bolck_0_0(x)
        x = self.identity_0_1(x)
        x = self.identity_0_2(x)

        # stage1
        x = self.conv_bolck_1_0(x)
        x = self.identity_1_1(x)
        x = self.identity_1_2(x)
        x = self.identity_1_3(x)
        # stage2
        x = self.conv_bolck_2_0(x)
        x = self.identity_2_1(x)
        x = self.identity_2_2(x)
        x = self.identity_2_3(x)
        x = self.identity_2_4(x)
        x = self.identity_2_5(x)

        # stage3
        x = self.conv_bolck_3_0(x)
        x = self.identity_3_1(x)
        x = self.identity_3_2(x)

        x = self.average_pooling2d(x)
        # print("before reshape", x.shape)
        x = tf.reshape(x, shape=(-1, 1024))

        x = self.logits(x)
        # print("logits: ", x.shape)
        x = tf.nn.softmax(x, name="predictions")
        return x


def getModel():
    inputs = keras.Input((224, 224, 3))
    x = layers.Lambda(lambda x: tf.pad(
        x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'CONSTANT'))(inputs)

    x = Conv2d("conv_1", filter_depth=64, kernel_size=7,
               stride=2, padding="valid")(x)
    x = batch_norm("bn_1")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # stage0
    x = BasicBlock(0, 0, filterdepths=[
                   32, 32, 128], kernel_size=3, stride=1)(x)
    x = Identity_block_layer(stage=0, block=1, filter_depths=[
                             32, 32, 128], kernel_size=3)(x)
    x = Identity_block_layer(stage=0, block=2, filter_depths=[
                             32, 32, 128], kernel_size=3)(x)

    # stage1
    x = BasicBlock(1, 0, filterdepths=[
                   64, 64, 256], kernel_size=3, stride=2)(x)
    x = Identity_block_layer(stage=1, block=1, filter_depths=[
                             64, 64, 256], kernel_size=3)(x)
    x = Identity_block_layer(stage=1, block=2, filter_depths=[
                             64, 64, 256], kernel_size=3)(x)
    x = Identity_block_layer(stage=1, block=3, filter_depths=[
                             64, 64, 256], kernel_size=3)(x)

    # stage2
    x = BasicBlock(2, 0, filterdepths=[
                   128, 128, 512], kernel_size=3, stride=2)(x)
    x = Identity_block_layer(stage=2, block=1, filter_depths=[
                             128, 128, 512], kernel_size=3)(x)
    x = Identity_block_layer(stage=2, block=2, filter_depths=[
                             128, 128, 512], kernel_size=3)(x)
    x = Identity_block_layer(stage=2, block=3, filter_depths=[
                             128, 128, 512], kernel_size=3)(x)
    x = Identity_block_layer(stage=2, block=4, filter_depths=[
                             128, 128, 512], kernel_size=3)(x)
    x = Identity_block_layer(stage=2, block=5, filter_depths=[
                             128, 128, 512], kernel_size=3)(x)

    # stage3
    x = BasicBlock(3, 0, filterdepths=[
                   256, 256, 1024], kernel_size=3, stride=2)(x)
    x = Identity_block_layer(stage=3, block=1, filter_depths=[
                             256, 256, 1024], kernel_size=3)(x)
    x = Identity_block_layer(stage=3, block=2, filter_depths=[
                             256, 256, 1024], kernel_size=3)(x)

    x = layers.AveragePooling2D(
        pool_size=7, strides=1, padding="valid", name="avergepool")(x)
    print("averagePooling: ", x)
    x = layers.Flatten()(x)
    logits = fully_connected(name="fc_nsfw", num_outputs=2)(x)
    output = layers.Activation("softmax")(logits)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model
