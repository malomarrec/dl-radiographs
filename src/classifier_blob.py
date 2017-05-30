import argparse
import json
import numpy as np
import time
from os import listdir
from scipy import ndimage
import tensorflow as tf

import math
import matplotlib.pyplot as plt


data_path = "../data/labels.json"

print("data path: ",data_path)
# print(out_path)


if "." in data_path:
	if data_path.split(".")[-1] != "json":
		raise ValueError("Invalid Extension")
else:
	data_path + ".json"

data = []

for line in open(data_path, 'r'):
	data.append(json.loads(line))	

#Extractng labels only
number_blobs = []
for dic in data:
	number_blobs.append(dic['nb_blobs'])
print(np.shape(number_blobs))
Y = np.asarray(number_blobs)
# print(type(Y))
Y -= 1 ## -1 since the labels are between 0 and 4 for softmax!


#Exctrating images
def get_img_list(path_to_dir, extension=".png"):
	"""
	Collects the filenames of all the files in a directory with a certaine extension
	Input: 
	path_to_dir: path to the directory
	extension
	Output:
	list of file names
	"""
	filenames = listdir(path_to_dir)
	return [filename for filename in filenames if filename.endswith(extension)]

tic = time.time()

path_data = "../data/radiographs/"

#Get file lists
data_list = get_img_list(path_data)

if __name__ == '__main__':
	X = np.array([ndimage.imread(path_data+img_name, flatten=False) for img_name in data_list])
	print(X.shape)

toc = time.time()

print(toc - tic)

n_train = 100


#Shuffle and split into train and test
indices = np.random.permutation(X.shape[0])
training_idx, test_idx = indices[:n_train], indices[n_train:]
X_train = X[training_idx,:]
Y_train = Y[training_idx]

# print('Shape X_train: ', np.shape(X_train))
# print('Shape Y_train: ', np.shape(Y_train))

indices = np.random.permutation(X.shape[0])
training_idx, test_idx = indices[:110], indices[110:]
X_test, X_train = X[training_idx,:], X[test_idx,:]
Y_test, Y_train = Y[training_idx], Y[test_idx]
print('Shape X_train: ', np.shape(X_train))
print('Shape X_test: ', np.shape(X_test))
print('Shape Y_train: ', np.shape(Y_train))
print('Shape Y_test: ', np.shape(Y_test))


#Model
tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, 691, 691, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)


#Parameters
FIRST_CONV_DEPTH = 32
SECOND_CONV_DEPTH = 64
THIRD_CONV_DEPTH = 128
REG_WEIGHT = 1e-3
LEARNING_RATE = 1e-4
BATCH_SIZE = 10



def conv_relu_batchnorm(input, is_training, filter_tuple, conv_depth, layer_name, regularizer=None):
	"""Creates graph for a conv-relu-batchnorm"""
	with tf.variable_scope(layer_name) as scope:
		conv = tf.layers.conv2d(
			X,
			FIRST_CONV_DEPTH,
			kernel_size = filter_tuple,
			strides = (2,2),
			padding='same',
			use_bias = True,
			kernel_initializer = tf.contrib.layers.xavier_initializer(),
			bias_initializer = tf.zeros_initializer(),
			kernel_regularizer = regularizer,
			name = 'conv')

		relu = tf.nn.relu(
			conv,
			name="relu"
		)

		batchnorm = tf.layers.batch_normalization(
			relu,
			axis=-1,
			momentum=0.99,
			epsilon=0.001,
			center=True,
			scale=True,
			beta_initializer=tf.zeros_initializer(),
			gamma_initializer=tf.ones_initializer(),
			moving_mean_initializer=tf.zeros_initializer(),
			moving_variance_initializer=tf.ones_initializer(),
			beta_regularizer=None,
			gamma_regularizer=None,
			training=is_training,
			trainable=True,
			name='batchnorm',
			reuse=None
		    )

	return batchnorm



def model_1(X,y,is_training, regularizer = None):
#  [conv-relu-conv-relu-pool]xN -> [affine]xM -> [softmax or SVM]   

	in_dim = X.shape[0]
	print(in_dim)

	# Two conv layers
	layer1_out = conv_relu_batchnorm(X, is_training, filter_tuple = (5,5), conv_depth = FIRST_CONV_DEPTH, layer_name = "layer1", regularizer=regularizer)
	layer2_out = conv_relu_batchnorm(layer1_out, is_training, filter_tuple = (3,3), conv_depth = SECOND_CONV_DEPTH, layer_name = "layer2", regularizer=regularizer)
	layer3_out = conv_relu_batchnorm(layer2_out, is_training, filter_tuple = (3,3), conv_depth = THIRD_CONV_DEPTH, layer_name = "layer3", regularizer=regularizer)

	#Maxpool
	mp3 = tf.nn.max_pool(layer3_out, ksize = (1, 2, 2, 1), strides = (1, 2, 2, 1), padding = 'VALID')
	print(mp3.shape)
	reshaped = tf.reshape(mp3, (-1, 173*173*32))
	#activation_fn=None forces a linear activation. Rempove that optionnal arg to make it a relu
	ll1 = tf.contrib.layers.fully_connected(reshaped,
		num_outputs = 5,
		activation_fn = None,
		weights_initializer = tf.contrib.layers.xavier_initializer(),
		weights_regularizer = regularizer,
		biases_initializer = tf.zeros_initializer()
		)

	return ll1

regularizer = tf.contrib.layers.l2_regularizer(scale=REG_WEIGHT)

logits = model_1(X,y,is_training, regularizer)

#Get regularization term
reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

#Get loss
total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = y)
total_loss += reg_term

mean_loss = tf.reduce_mean(total_loss)



#lr = tf.train.exponential_decay(starter_learning_rate, global_step, 5000, 0.96)
optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE)
train_step = optimizer.minimize(mean_loss)



def run_model(session, predict, loss_val, Xd, yd,
				epochs=1, batch_size=64, print_every=10,
				training=None, plot_losses=False):
	# have tensorflow compute accuracy
	correct_prediction = tf.equal(tf.argmax(predict,1), y)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# shuffle indices
	train_indicies = np.arange(Xd.shape[0])
	np.random.shuffle(train_indicies)

	training_now = training is not None

	# setting up variables we want to compute (and optimizing)
	# if we have a training function, add that to things we compute
	variables = [mean_loss,correct_prediction,accuracy]
	if training_now:
	    variables[-1] = training

	# counter 
	iter_cnt = 0
	losses = []
	for e in range(epochs):
	# keep track of losses and accuracy
		correct = 0#
		# make sure we iterate over the dataset once
	for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
	# generate indicies for the batch
		start_idx = (i*batch_size)%X_train.shape[0]
		idx = train_indicies[start_idx:start_idx+batch_size]

		# create a feed dictionary for this batch
		feed_dict = {X: Xd[idx],
		y: yd[idx],
		is_training: training_now }
		# get batch size
		actual_batch_size = yd[i:i+batch_size].shape[0]

		# have tensorflow compute loss and correct predictions
		# and (if given) perform a training step
		loss, corr , _ = session.run(variables,feed_dict=feed_dict)
		# aggregate performance stats
		losses.append(loss*actual_batch_size)
		correct += np.sum(corr)

		if training_now and (iter_cnt % print_every) == 0:
			print(iter_cnt,np.sum(corr)/float(actual_batch_size))
		iter_cnt += 1
		total_correct = correct/float(Xd.shape[0])

	total_loss = np.sum(losses)/float(Xd.shape[0])
	print(total_loss,total_correct,e+1)

	if plot_losses:
		plt.plot(losses)
		plt.grid(True)
		plt.title('Epoch {} Loss'.format(e+1))
		plt.xlabel('minibatch number')
		plt.ylabel('minibatch loss')
		plt.show()
	return losses, total_loss,total_correct

with tf.Session() as sess:
	with tf.device("/cpu:0"): #"/cpu:0" or "/gpu:0" 
		sess.run(tf.global_variables_initializer())
		print('Training')
		losses, loss, total_correct = run_model(sess,logits,mean_loss,X_train,Y_train,10,10,100,train_step,True)
		print('Validation')
		run_model(sess,logits,mean_loss,X_test,Y_test,1,64)





