import tensorflow as tf
import os
from tfrecord_utils import read_TFRecond_file
from models import ResModel

data_path = "./data/tfrecords/ht_all_imgs.tfrecord"
weight_decay = 2e-4

if __name__ == '__main__':
    dataset = read_TFRecond_file(data_path).shuffle(10000).batch(32)

    model = ResModel(num_blocks=[3, 4, 6, 3], block_strides=[
                     1, 2, 2, 2], input_type=1, output_classer=2, filter_depths=[32, 32, 128], kernel_size=3)

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )

    model.build(input_shape=(None, 224, 224, 3))

    model.summary()

    model.fit(dataset, steps_per_epoch=20)

    name = "ht_model-1"

    save_model_path = "./models/%s/" % name
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    model.save(save_model_path)
