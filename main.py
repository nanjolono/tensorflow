from __future__ import division, print_function, absolute_import

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# fetch trans data
fashion_mnist = keras.datasets.fashion_mnist
# load data from download
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# fetch trans data to save variable
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# data's num and attribute.
print(train_images.shape)
print(len(train_labels))
print(test_images.shape)
print(len(test_labels))

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0
plt.figure(figsize=(10, 10))

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

# 将训练模型保存为文件
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
# 恢复模型
restored_model = tf.keras.models.load_model('my_model.h5')
# 测试数据
loss, acc = restored_model.evaluate(test_images, test_labels, verbose=2)
# 测试结果
print("Restored model, accuracy:{:5.2f}%".format(100 * acc))

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
print('预测结果打印')
print(predictions)


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


type = 4
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 1
num_cols = 1
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
plt.subplot(num_rows, 2 * num_cols, 2 * 0 + 1)
plot_image(type, predictions[0], test_labels, test_images)
plt.subplot(num_rows, 2 * num_cols, 2 * 0 + 2)
plot_value_array(type, predictions[0], test_labels)
plt.tight_layout()
plt.show()
