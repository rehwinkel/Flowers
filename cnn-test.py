from PIL import Image
import os
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def loadVecFromImg(filename):
	img = Image.open(filename)
	return (np.array(list(img.tobytes())) * (1 / 255)).reshape(img.width, img.height)

def saveImgFromVec(vector, filename):
	Image.fromarray((vector * 255).astype("uint8"), "L").save(filename)

def unpool(value, name='unpool'):
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat([out, tf.zeros_like(out)], i)
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out


data = np.zeros(0)
for i in range(1):
	vec = loadVecFromImg("train/img" + str(i) + ".jpg")
	data = np.append(data, vec)

session = tf.InteractiveSession()

input_layer = tf.reshape(data, [-1, 128, 128, 1])
print(input_layer)

conv1 = tf.layers.conv2d(inputs=input_layer, filters=192, kernel_size=[32, 32], padding="same", activation=tf.nn.relu)
print(conv1)

pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
print(pool1)

conv2 = tf.layers.conv2d(inputs=pool1, filters=384, kernel_size=[16, 16], padding="same", activation=tf.nn.relu)
print(conv2)

pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
print(pool2)

conv3 = tf.layers.conv2d(inputs=pool2, filters=768, kernel_size=[8, 8], padding="same", activation=tf.nn.relu)
print(conv3)

pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
print(pool3)

conv4 = tf.layers.conv2d(inputs=pool3, filters=768, kernel_size=[4, 4], padding="same", activation=tf.nn.relu)
print(conv4)

pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
print(pool4)

flat = tf.reshape(pool4, [-1, 8 * 8 * 768])
print(flat)

dense = tf.layers.dense(inputs=flat, units=8192, activation=tf.nn.relu)
print(dense)

end = tf.layers.dense(inputs=dense, units=128, activation=tf.nn.relu)
print(end)

r_dense = tf.layers.dense(inputs=end, units=8192, activation=tf.nn.relu)
print(r_dense)

r_dense2 = tf.layers.dense(inputs=r_dense, units=8*8*768, activation=tf.nn.relu)
print(r_dense2)

r_flat = tf.reshape(r_dense2, [-1, 8, 8, 768])
print(r_flat)

r_pool4 = unpool(r_flat)
print(r_pool4)

r_conv4 = tf.layers.conv2d_transpose(inputs=r_pool4, filters=768, kernel_size=[4, 4], padding="same", activation=tf.nn.relu)
print(r_conv4)

r_pool3 = unpool(r_conv4)
print(r_pool3)

r_conv3 = tf.layers.conv2d_transpose(inputs=r_pool3, filters=768, kernel_size=[8, 8], padding="same", activation=tf.nn.relu)
print(r_conv3)

r_pool2 = unpool(r_conv3)
print(r_pool2)

r_conv2 = tf.layers.conv2d_transpose(inputs=r_pool2, filters=384, kernel_size=[16, 16], padding="same", activation=tf.nn.relu)
print(r_conv2)

r_pool1 = unpool(r_conv2)
print(r_pool1)

r_conv1 = tf.layers.conv2d_transpose(inputs=r_pool1, filters=192, kernel_size=[32, 32], padding="same", activation=tf.nn.relu)
print(r_conv1)

r_conv = tf.layers.conv2d_transpose(inputs=r_conv1, filters=1, kernel_size=[32, 32], padding="same", activation=tf.nn.relu)
print(r_conv)

loss = tf.reduce_mean(tf.square(r_conv - input_layer))
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
session.run(tf.global_variables_initializer())

for i in range(100):
	xx, ls = session.run([train_op, loss], feed_dict={input_layer: np.array(data.reshape(-1, 128, 128, 1))})
	print('Epoch: {} - cost= {:.5f}'.format((i + 1), ls))