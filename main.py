import tensorflow as tf
from model import ResModel, getModel
from image_utils import create_tensorflow_image_loader
from image_utils import create_yahoo_image_loader
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


if __name__ == '__main__':
    IMAGE_DIR = r'./data/nsfw'
    input_type = 1
    image_loader = "yahoo"
    fn_load_image = load_image(input_type, image_loader)

    model = ResModel(input_type)
    # model = getModel()
    count = 0
    for i in findAllFile(IMAGE_DIR):
        # print('predict for: ' + i)
        image = fn_load_image(i)
        imageTensor = imageToTensor(image, input_type)
        result = model(imageTensor).numpy().tolist()[0]
        sfw, nsfw = result[0], result[1]
        # print(result)
        if sfw >= 0.8:
            print("%s sfw=%.8f nsfw=%.8f" %
                  (i, sfw, nsfw))
            count += 1
        # break
    print('count: %d' % count)
