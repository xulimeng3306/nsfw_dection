import tensorflow as tf
# import numpy as np
from predict import load_image

# 指定 SavedModel 的路径
saved_model_path = './models/1547856517/'

# 加载 SavedModel
model = tf.saved_model.load(saved_model_path)
print("=======================", model.signatures)
model = tf.function(model.signatures["predict"])
# 获取权重值
# with tf.compat.v1.Session() as sess:
#     # 初始化模型
#     sess.run(tf.compat.v1.global_variables_initializer())

#     # 获取权重值
#     weights_value = sess.run(model.variables)
#     print(weights_value)

loader = load_image(input_type=1, image_loader="yahoo")
img = loader('./data/sfw/0342077f2662a49aea8980b62c470dbb.jpg')
output = model(img)
print(output)

# 将权重保存为 .npy 文件
# npy_file_path = './models/1547856517/nsfw_five_classer-weights.npy'
# np.save(npy_file_path, weights_value)
# print("Weights saved to:", npy_file_path)
