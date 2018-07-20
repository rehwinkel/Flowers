import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

		self.cost_fun = 0.5 * tf.reduce_sum(tf.subtract(self.logits, self.outputs) * tf.subtract(self.logits, self.outputs))
		self.train_step = tf.train.GradientDescentOptimizer(0.05).minimize(self.cost_fun)
		self.session.run(tf.global_variables_initializer())

	def train(self, in_data, out_data, iterations):
		for i in range(iterations):
			xx, loss = self.session.run([self.train_step, self.cost_fun], feed_dict={self.inputs: np.array(in_data), self.outputs: np.array(out_data)})

	def feed(self, in_data):
		return self.session.run(self.logits, feed_dict={self.inputs: np.array([in_data])})

training_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
training_outputs = [[0.0], [1.0], [1.0], [0.0]]

my_network = NeuralNetwork([2, 4, 1])
my_network.train(training_inputs, training_outputs, 1000)
print(my_network.feed([0.0, 0.0]))
print(my_network.feed([1.0, 0.0]))
print(my_network.feed([0.0, 1.0]))
print(my_network.feed([1.0, 1.0]))