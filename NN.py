import numpy as np
import tensorflow as tf
from dataset.mnist import load_mnist

class DNN:
	def __init__(self, mode, trunc, save):
		self.weights = []
		self.biases = []
		self.input_dim = None
		self.output_dim = None
		self.hidden_dims = None
		self.actvs = None
		self.loss = None

		self._x = None
		self._t = None
		self._keep_prob = None

		self.logits = None
		self._log = {
			"accuracy" : [],
			"loss"     : []
			}

		self._sess = tf.Session()
		self.ckpt_dir = "./Save"
		self.ckpt_path = "./Save/model.ckpt"

		self.modes = {
			"train" : False,
			"eval"  : False
		}
		self.set_mode(mode)
		self.trunc = trunc
		self.is_save  = save

	def __del__(self):
		self._sess.close()

	def set_mode(self, mode):
		self.modes[mode] = True

	def is_trunc(self):
		if self.trunc is True:
			return True
		else:
			return False

	def save_session(self):
		saver = tf.train.Saver()
		saver.save(self._sess, self.ckpt_path)

	def has_ckpt(self):
		ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
		if ckpt:
			return True
		else:
			return False

	def restore_session(self, path):
		saver = tf.train.Saver()
		saver.restore(self._sess, path)

	def delete_session(self):
		self._sess.close()

	def init_layer_param(self, input_dim, output_dim, hidden_dims, actvs):
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.hidden_dims = hidden_dims
		self.actvs = actvs

	def init_loss_param(self, loss):
		self.loss = loss

	def init_weight(self, shape):
		initial = tf.truncated_normal(shape, stddev=0.01)
		return tf.Variable(initial)

	def init_bias(self, shape):
		initial = tf.zeros(shape)
		return tf.Variable(initial)

	def define_placeholder(self):
		#define variable for TensorFlow
		self._x = tf.placeholder(tf.float32, shape=[None, self.input_dim])
		self._t = tf.placeholder(tf.float32, shape=[None, self.output_dim])
		self._keep_prob = tf.placeholder(tf.float32)

	def set_dropout(self, src, keep_prob):
		return tf.nn.dropout(src, keep_prob)

	def set_layer_relu(self, src, src_dim, dst_dim):
		self.weights.append(self.init_weight([src_dim, dst_dim]))
		self.biases.append(self.init_bias([dst_dim]))

		return tf.nn.relu(tf.matmul(src, self.weights[-1]) + self.biases[-1])

	def set_layer_softmax(self, src, src_dim, dst_dim):
		self.weights.append(self.init_weight([src_dim, dst_dim]))
		self.biases.append(self.init_bias([dst_dim]))

		return tf.nn.softmax(tf.matmul(src, self.weights[-1]) + self.biases[-1])

	def infer(self, x, keep_prob):
		
		#dst = set_layer_leru(x, self.input_dim, self.hidden_dims[0])
		#dst = set_dropout(dst, keep_prob)

		#set hidden layer
		#src_dims = self.hidden_dims[  :-1]
		#dst_dims = self.hidden_dims[+1:  ]

		#set input & hidden layer
		src_dims = self.hidden_dims[:-1]
		src_dims.insert(0, self.input_dim)
		dst_dims = self.hidden_dims[:]

		for i, (src_dim, dst_dim) in enumerate(zip(src_dims, dst_dims)):
			src = x if i == 0 else dst
			dst =  self.set_layer(src, src_dim, dst_dim, "relu")
			#dst = self.set_layer_relu(src, src_dim, dst_dim)
			dst = self.set_dropout(dst, keep_prob)

		#set output layer
		y = self.set_layer(dst, self.hidden_dims[-1], self.output_dim, "softmax")
		#y = self.set_layer_softmax(dst, self.hidden_dims[-1], self.output_dim)

		return y
	
	def calc_mse(self, y, t):
		return tf.reduce_mean(tf.square(y - t))

	def calc_cross_entropy(self, y, t):
		return tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), reduction_indices=[1]))

	def train(self, loss):
		optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
		#optimizer = tf.train.GradientDescentOptimizer(0.01)
		train_step = optimizer.minimize(loss)
		return train_step

	def calc_loss(self, y, t):
		#call Loss class and set loss function
		loss = Loss(self.loss).switch_loss()
		return loss.calc(y, t)

	def set_layer(self, src, src_dim, dst_dim, actv_func):
		self.weights.append(self.init_weight([src_dim, dst_dim]))
		self.biases.append(self.init_bias([dst_dim]))

		#call Layer class and set acticate function
		layer = Layer(actv_func).switch_layer()
		return layer.set(src, self.weights[-1], self.biases[-1])

	def calc_accuracy(self, y, t):
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
		return accuracy

	def evaluate(self, x_test, t_test):
		if self.modes["train"] is True:
			acc_op = self.calc_accuracy(self._logits, self._t)

		if self.modes["eval"] is True:
			y = self.infer(self._x, self._keep_prob)
			loss_op = self.calc_loss(y, self._t)
			train_op = self.train(loss_op)

			acc_op = self.calc_accuracy(y, self._t)

			ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
			if self.should_restore() is True:
				print(ckpt.model_checkpoint_path)
				self.restore_session(ckpt.model_checkpoint_path)
			else:
				init = tf.global_variables_initializer()
				self._sess.run(init)

		accuracy = acc_op.eval(session=self._sess,
			feed_dict = 
			{
				self._x : x_test,
				self._t : t_test,
				self._keep_prob : 1.0
			}
		)
		return accuracy

	
	def confirm(self, x_test, t_test):
		#define model
		y = self.infer(self._x, self._keep_prob)
		#loss = DNN.calc_cross_entropy(y, t)
		loss_op = self.calc_loss(y, self._t)
		train_op = self.train(loss_op)

		#define calc accuracy
		acc_op = self.calc_accuracy(y, self._t)

		ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
		if self.has_checkpoint():
			print(ckpt.model_checkpoint_path)
			self.restore_session(ckpt.model_checkpoint_path)
		else:
			init = tf.global_variables_initializer()
			self._sess.run(init)

		accuracy = acc_op.eval(session=self._sess,
			feed_dict = 
			{
				self._x : x_test,
				self._t : t_test,
				self._keep_prob : 1.0
			}
		)
		return accuracy

	def should_restore(self):
		if self.trunc is False and self.has_ckpt() is True:
			return True
		else:
			return False

	def fit(self, x_train, t_train, prob, epochs, batch_size):
		#define model
		y = self.infer(self._x, self._keep_prob)
		#loss = DNN.calc_cross_entropy(y, t)
		loss_op = self.calc_loss(y, self._t)
		train_op = self.train(loss_op)

		#for restor
		self._logits = y

		#define calc accuracy
		acc_op = self.calc_accuracy(y, self._t)

		ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
		if self.should_restore():
			print(ckpt.model_checkpoint_path)
			self.restore_session(ckpt.model_checkpoint_path)
		else:
			init = tf.global_variables_initializer()
			self._sess.run(init)

		batchs = x_train.shape[0] // batch_size
		batchs = 20

		for epoch in range(epochs):
			for i in range(batchs):
				batch_mask = np.random.choice(x_train.shape[0], batch_size)
				x_batch = x_train[batch_mask]
				t_batch = t_train[batch_mask]

				train_op.run(session=self._sess, 
					feed_dict = 
					{
						self._x : x_batch,
						self._t : t_batch,
						self._keep_prob : prob
					}
				)
			
			loss = loss_op.eval(session=self._sess,
				feed_dict =
				{
					self._x : x_train,
					self._t : t_train,
					self._keep_prob : 1.0
				}
			)
			
			accuracy = acc_op.eval(session=self._sess,
				feed_dict =
					{
						self._x : x_train,
						self._t : t_train,
						self._keep_prob : 1.0
					}
			)
			self._log["loss"].append(loss)
			self._log["accuracy"].append(accuracy)
			print("epoch : ", epoch, "loss : ", loss, "accuracy : ", accuracy)

#Loss class is called by calc_loss from DNN class
class Loss:
	def __init__(self, loss):
		self.loss = loss

	def calc(self, y, t):
		pass

	def switch_loss(self):
		if   self.loss is "mse":
			return MSELoss()

		elif self.loss is "cross_entropy":
			return CrossEntropyLoss()
			
		else:
			return DefaultLoss()

class MSELoss(Loss):
	def __init__(self):
		pass

	def calc(self, y, t):
		return tf.reduce_mean(tf.square(y - t))

class CrossEntropyLoss(Loss):
	def __init__(self):
		pass

	def calc(self, y, t):
		return tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), reduction_indices=[1]))

class DefaultLoss(Loss):
	def __init__(self):
		pass

	def calc(self, y, t):
		return y


#Layer class is called by set_layer from DNN class
class Layer:
	def __init__(self, actv):
		self.actv = actv

	def set(self, src, weight, bias):
		pass

	def switch_layer(self):
		if   self.actv is "relu":
			return ReLULayer()

		elif self.actv is "softmax":
			return SoftmaxLayer()
			
		else:
			return None
		
class ReLULayer(Layer):
	def __init__(self):
		pass

	def set(self, src, weight, bias):
		return tf.nn.relu(tf.matmul(src, weight) + bias)

class SoftmaxLayer(Layer):
	def __init__(self):
		pass

	def set(self, src, weight, bias):
		return tf.nn.softmax(tf.matmul(src, weight) + bias)

def main():
	#layer configuration
	input_dim = 28 * 28
	output_dim = 10
	hidden_dims = [200,200,200]
	actvs = []

	#for test run without saved checkpoint
	#model = DNN(mode="train", trunc=True, save=False)
	model = DNN(mode="eval", trunc=False, save=False)

	#(should not use) for test run and destroy saved checkpoint
	#model = DNN(mode="train", trunc=True, save=True)

	#for resume saved run and save checkpoint
	#model = DNN(mode="train", trunc=False, save=True)

	#for for test run with saved checkpoint
	#model = DNN(mode="train", trunc=False, save=False)

	#initialize params
	model.init_layer_param(input_dim, output_dim, hidden_dims, actvs)
	model.init_loss_param("cross_entropy")

	model.define_placeholder()

	#for mnist
	(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
	train_size = x_train.shape[0]
	batch_size = 10

	#epoch
	keep_prob = 1.0
	epochs= 20
	batch_size = 10

	if model.modes["train"] is True:
		#epoch
		keep_prob = 1.0
		epochs= 20
		batch_size = 10

		model.fit(x_train, t_train, keep_prob, epochs, batch_size)
		accuracy = model.evaluate(x_test, t_test)
		print("evaluate : ", accuracy)

		if model.is_save is True:
			model.save_session()

	if model.modes["eval"] is True:
		accuracy = model.evaluate(x_test, t_test)
		print(accuracy)

if __name__ == '__main__':
	main()
