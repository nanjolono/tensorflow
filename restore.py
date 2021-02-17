from __future__ import division, print_function, absolute_import

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import ImageUtil as imgUtil


# fetch trans data
fashion_mnist = keras.datasets.fashion_mnist
# load data from download
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# 恢复模型
restored_model = tf.keras.models.load_model('my_model.h5')
# 测试数据

# 读取图片转成灰度格式
img = Image.open('img_1.png').convert('L')

# resize的过程
if img.size[0] != 28 or img.size[1] != 28:
    img = img.resize((28, 28))

# 暂存像素值的一维数组
arr = []

for i in range(28):
    for j in range(28):
        # mnist 里的颜色是0代表白色（背景），1.0代表黑色
        pixel = 1.0 - float(img.getpixel((j, i))) / 255.0
        # pixel = 255.0 - float(img.getpixel((j, i))) # 如果是0-255的颜色值
        arr.append(pixel)

arr1 = np.array(arr).reshape((1, 28, 28, 1))
predictions = restored_model.predict(arr1)





excpetType = 3
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 1
num_cols = 1
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
plt.subplot(num_rows, 2 * num_cols, 2 * 0 + 1)
imgUtil.plot_image_sign(excpetType, predictions[0], excpetType, test_images)
plt.subplot(num_rows, 2 * num_cols, 2 * 0 + 2)
imgUtil.plot_value_array_sign(excpetType, predictions[0], excpetType)
plt.tight_layout()
plt.show()
