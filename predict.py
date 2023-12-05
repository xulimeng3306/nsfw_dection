import tensorflow as tf
from models import ResModel
from image_utils import create_tensorflow_image_loader
from image_utils import create_yahoo_image_loader
from PIL import Image
import numpy as np
import os

IMAGE_LOADER_TENSORFLOW = "tensorflow"
IMAGE_LOADER_YAHOO = "yahoo"
model_path = './nsfwmodel'


def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.jpg') or f.endswith('.png'):
                # if re.match(r'.*\d.*', f):
                fullname = os.path.join(root, f)
                yield fullname


def load_image(input_type=1, image_loader="yahoo"):
    if input_type == 1:
        print('TENSOR...')
        if image_loader == IMAGE_LOADER_TENSORFLOW:
            print('IMAGE_LOADER_TENSORFLOW...')
            fn_load_image = create_tensorflow_image_loader(
                tf.Session(graph=tf.Graph()))
        else:
            print('create_yahoo_image_loader')
            fn_load_image = create_yahoo_image_loader()
    elif input_type == 2:
        print('BASE64_JPEG...')
        import base64
        def fn_load_image(filename): return np.array(
            [base64.urlsafe_b64encode(open(filename, "rb").read())])
    return fn_load_image


def imageToTensor(inputs, input_type=1):
    if input_type == 1:
        input_tensor = inputs
    elif input_type == 2:
        from image_utils import load_base64_tensor
        input_tensor = load_base64_tensor(inputs)
    else:
        raise ValueError("invalid input type '{}'".format(input_type))
    return input_tensor


def standardize(img):
    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean) / std
    return img


def load_image_with_PIL(infilename):
    img = Image.open(infilename)
    img = img.resize((64, 64))
    img.load()
    data = np.asarray(img, dtype=np.float32)
    data = standardize(data)
    return tf.convert_to_tensor(value=[data] * 128, dtype=tf.float32)


def predict_two_labels():
    IMAGE_DIR = r'./data/nsfw'
    input_type = 1
    image_loader = "yahoo"
    fn_load_image = load_image(input_type, image_loader)

    # 分类数
    classer_num = 2

    model = ResModel(num_blocks=[3, 4, 6, 3],
                     block_strides=[1, 2, 2, 2],
                     input_type=input_type,
                     output_classer=classer_num,
                     filter_depths=[32, 32, 128],
                     kernel_size=3)
    # model = getModel()
    count = 0
    for i in findAllFile(IMAGE_DIR):
        # print('predict for: ' + i)
        image = fn_load_image(i)
        imageTensor = imageToTensor(image, input_type)
        result = model(imageTensor).numpy().tolist()[0]
        # # 五分类模型
        # drawings, hentai, neutral, porn, sexy = result[0], result[1], result[2], result[3], result[4]
        # print("class=% drawings=%d hentai=%d neutral=%d porn=%d sexy=%d" %
        #       (IMAGE_DIR, drawings, hentai, neutral, porn, sexy))
        # 二分类模型
        sfw, nsfw = result[0], result[1]
        # print(result)
        if sfw >= 0.8:
            print("%s sfw=%.8f nsfw=%.8f" %
                  (i, sfw, nsfw))
            count += 1
        break
        # break
    print('count: %d' % count)


def predict_five_labels():
    IMAGE_DIR = r'./data/nsfw'

    # 载入模型
    model = tf.saved_model.load("models/1547856517")

    # 获取模型的签名函数
    infer = model.signatures["serving_default"]

    # model = getModel()
    count = 0
    for i in findAllFile(IMAGE_DIR):
        # print('predict for: ' + i)
        image = load_image_with_PIL(i)
        # imageTensor = imageToTensor(image, 2)
        # print(image.shape)
        rsp = infer(image)
        print(rsp.keys())
        prob, classes = rsp['probabilities'].numpy()[
            0], rsp['classes'].numpy()[0]
        print("pwd=%s" % i,
              "prob", prob, "classes", classes)
        if classes == 3:
            count += 1
        # break
    print('count: %d' % count)


if __name__ == '__main__':
    predict_five_labels()
