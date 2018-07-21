import numpy as np
import tensorflow as tf
import os
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def vectorizeImage(id):
	pic = Image.open("out/frame" + str(id) + ".jpg")
	gray = Image.new("L", pic.size)
	gray.paste(pic)
	gray_bytes = gray.tobytes()
	vec = np.zeros(len(gray_bytes))
	for b in range(len(gray_bytes)):
		vec[b] = gray_bytes[b] / 255
	return vec

def saveVectorAsImage(id, vector):
	pixels = [0] * 4096
	for v in range(len(vector)):
		#print(vector[v])
		pixels[v] = int(abs(vector[v]) * 255)
	pixels = np.array(pixels).reshape(64, 64)
	Image.fromarray(pixels.astype("uint8"), "L").save("gen/gen" + str(id) + ".jpg", "JPEG")

class NeuralNetwork:
	def __init__(self, layers):
		self.session = tf.InteractiveSession()

		self.inputs = tf.placeholder(tf.float32, shape=[None, layers[0]])
		self.outputs = tf.placeholder(tf.float32, shape=[None, layers[-1]])

		prev_layer_out = self.inputs
		for i in range(1, len(layers)-1):
			weights = tf.Variable(tf.truncated_normal([layers[i-1], layers[i]]))
			biases = tf.Variable(tf.zeros([layers[i]]))
			layer_outputs = tf.nn.leaky_relu(tf.matmul(prev_layer_out, weights) + biases)
			prev_layer_out = layer_outputs
		weights_last = tf.Variable(tf.truncated_normal([layers[-2], layers[-1]]))
		biases_last = tf.Variable(tf.zeros([layers[-1]]))

		self.logits = tf.nn.leaky_relu(tf.matmul(prev_layer_out, weights_last) + biases_last)

		self.cost_fun = 0.000000006 * tf.reduce_sum(tf.subtract(self.logits, self.outputs) * tf.subtract(self.logits, self.outputs))
		self.train_step = tf.train.GradientDescentOptimizer(0.000002).minimize(self.cost_fun)
		self.session.run(tf.global_variables_initializer())

	def train(self, in_data, out_data, iterations):
		for i in range(iterations):
			xx, loss = self.session.run([self.train_step, self.cost_fun], feed_dict={self.inputs: np.array(in_data), self.outputs: np.array(out_data)})
			print(loss)
			#for d in in_data:
				#print(str(i) + ":", d, self.feedRounded(d))

	def feed(self, in_data):
		return self.session.run(self.logits, feed_dict={self.inputs: np.array([in_data])})

	def feedRounded(self, in_data):
		fed = np.array(self.feed(in_data))
		s = fed.shape
		flat = fed.flatten()

		out = np.zeros(len(flat))
		for f in range(len(flat)):
			out[f] = round(flat[f], 4)
		return out.reshape(s)

images = []
for i in range(50):
	images.append(vectorizeImage(i))
images = np.array(images)

image_size = 4096
training_inputs = images.copy()#[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
training_outputs = images.copy()#[[0.0], [1.0], [1.0], [0.0]]

my_network = NeuralNetwork([image_size, 1024, 128, 1024, image_size])
my_network.train(training_inputs, training_outputs, 100)

saveVectorAsImage(0, my_network.feed(images[0]).flatten())
saveVectorAsImage(1, images[0].flatten())