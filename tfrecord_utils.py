from PIL import Image
from utils import standardize
import tensorflow as tf
import os
from matplotlib import pyplot as plt

labels = {0: 'drawings', 1: 'hentai', 2: 'neutral', 3: 'porn', 4: 'sexy'}
two_labels = {0: 'sfw', 1: 'nsfw'}


def build_images_and_labels(label_to_dir: dict):
    """
    :param label_to_dir: dict of labels to directories
    :return: images, labels
    """
    images = []
    labels = []
    for label, dir in label_to_dir.items():
        for file in os.listdir(dir):
            if file.endswith(".jpg"):
                images.append(os.path.join(dir, file))
                labels.append(label)
    return images, labels


def write_to_tfrecord(images: list, labels: list, tfrecord_file: str, image_size: tuple = (64, 64)):
    """
    :param images: list of image paths
    :param labels: list of labels
    :param tfrecord_file: path to tfrecord file
    :return: None
    """
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for image, label in zip(images, labels):
            img = Image.open(image)
            exterma = img.convert("L").getextrema()
            if exterma[0] == exterma[1]:
                print("Continue image: {}".format(image))
                continue
            img = img.resize(image_size)
            img = standardize(img)
            print("----------------------------", img.shape)
            img_bytes = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))
            writer.write(example.SerializeToString())
            break


# 定义Feature结构，告诉解码器每个Feature的类型是什么，要与生成的TFrecord的类型一致
feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64)
}

# 将TFRecord 文件中的每一个序列化的 tf.train.Example 解码


def parse_example(example_string):
    temp = tf.io.parse_single_example(example_string, feature_description)
    print(temp['image'])
    img = tf.io.decode_raw(temp['image'], tf.uint8)
    print(img.shape)
    img = tf.reshape(img, [224, 224, 3])
    img = img / 255
    label = temp['label']
    return (img, label)


def read_TFRecond_file(tfrecord_file):
    # 读取TFRecord 文件
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
    # 解码
    dataset = raw_dataset.map(parse_example)

    return dataset


if __name__ == '__main__':
    label_to_dir = {
        0: "./data/sfw",
        1: "./data/nsfw"
    }

    images, labels = build_images_and_labels(label_to_dir)

    write_to_tfrecord(
        images, labels, "./data/tfrecords/nsfw.tfrecord", (224, 224))

    dataset_train = read_TFRecond_file("./data/tfrecords/nsfw.tfrecord")

    dataset_batch_1 = dataset_train.take(1)

    for images, labels in dataset_batch_1:
        print("images.shape: ", images.shape, "labels.shape: ", labels.shape)
