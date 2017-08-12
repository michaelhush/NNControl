import numpy as np 
import tensorflow as tf
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # removes warning
import pickle 
import KNN_class


maxtime = 2
timestep = 0.1
num_steps = int(maxtime/timestep)

test_num_samples = 100

plot_num = 12


## testing data
xtestset = pickle.load(open('xtestset.p','rb'))
ytestset = pickle.load(open('ytestset.p','rb'))
yltestset = pickle.load(open('yltestset.p','rb'))
print ('loading complete')
## NN_pen_test
def NN_test():
	sess = tf.Session()
	NN_saver = tf.train.import_meta_graph('defaultNNmodel_save.meta') ## change key name for different NN_dict key
	NN_saver.restore(sess, "defaultNNmodel_save")
	[xNN, y_pred, yNN_] = tf.get_collection('defaultNNvars')
	NN_cost = tf.reduce_mean((yNN_ - y_pred)*(yNN_ - y_pred))
	test_NN_cost =  sess.run(NN_cost, feed_dict={xNN: xtestset, yNN_: ytestset})
	print("test_NN_cost:{}".format(test_NN_cost))
	## Save the plotting
	NN_plot = sess.run(y_pred, feed_dict={xNN: np.reshape(xtestset[plot_num], [1,num_steps])})
	NN_plot = np.reshape(NN_plot, [num_steps]) 
	pickle.dump(NN_plot, open('NN_plot.p','wb'))
	sess.close()

def Lin_test():
	sess = tf.Session()
	yl_ = tf.placeholder(tf.float32,[None,num_steps])
	yNN_ = tf.placeholder(tf.float32,[None,num_steps])
	lin_cost = tf.reduce_mean((yl_ - yNN_)*(yl_ - yNN_))
	test_lin_cost = sess.run(lin_cost, feed_dict={yl_: yltestset, yNN_: ytestset})
	print("test_Lin_cost:{}".format(test_lin_cost))
	sess.close()
	
def LSTM_test():
	input_size = 1
	output_size = 1
	## load trained NN
	sess=tf.Session()
	
	num_layer = 2
	state_size = 16

	batch_state = np.zeros((num_layer, 2, test_num_samples, state_size))
	plot_state = np.zeros((num_layer, 2, 1, state_size)) 

	LSTM_saver = tf.train.import_meta_graph('defaultLSTMmodel_save.meta') ## change key name for different NN_dict keys
	LSTM_saver.restore(sess, "defaultLSTMmodel_save")
	[x_in, init_state, Wout, bout, state_series, output_plot, y_out] = tf.get_collection('defaultLSTMvars')
	output_series = tf.reshape(tf.matmul(tf.reshape(state_series, [-1, state_size]), Wout) +bout , [test_num_samples, num_steps, output_size] )
	LSTM_cost = tf.reduce_mean((y_out - output_series)*(y_out - output_series))
	test_LSTM_cost =  sess.run(LSTM_cost, 
	feed_dict={x_in: np.reshape(xtestset,[test_num_samples, num_steps, 1]), y_out: np.reshape(ytestset, [test_num_samples, num_steps, 1]), init_state : batch_state })
	print("test_LSTM_cost:{}".format(test_LSTM_cost))
	## save the plotting
	LSTM_plot = np.squeeze(sess.run(output_plot, 
                                feed_dict = {x_in: np.reshape(xtestset[plot_num],[1,num_steps,input_size]), init_state: plot_state}))
	pickle.dump(LSTM_plot, open('LSTM_plot.p','wb'))
	sess.close()

def RNN_test():
	input_size = 1
	output_size = 1
	state_size = 2
	batch_state = np.zeros((test_num_samples, state_size))
	sess = tf.Session()
	RNN_saver = tf.train.import_meta_graph('defaultRNNmodel_save.meta')
	RNN_saver.restore(sess, "defaultRNNmodel_save")
	[x_in, init_state, output_series, cost, y_out] = tf.get_collection('defaultRNNvars')
	xtest = np.reshape(xtestset, [test_num_samples, num_steps, input_size])
	ytest = np.reshape(ytestset, [test_num_samples, num_steps, input_size])
	test_RNN_cost = sess.run(cost, feed_dict = {x_in: xtest, 
		y_out: ytest, init_state: batch_state})
	print('test_RNN_cost:.{}'.format(test_RNN_cost))
	## save the plotting (what is the cheapest way to produce the output)
	RNN_plot = sess.run(output_series, feed_dict= {x_in: xtest, y_out: ytest, init_state: batch_state})
	RNN_plot = np.array(RNN_plot)[:,plot_num]
	pickle.dump(RNN_plot, open('RNN_plot.p','wb'))
	sess.close()



def KNN_test():
	knnxtrainset = pickle.load(open("xtrainset.p","rb"))
	knnytrainset = pickle.load(open("ytrainset.p","rb"))
	print('knn training sample loading complete')
	## knn parameters -- this takes a long time to train comment out if not interested
	k = 5 ## number of neighbours
	norm = "Euclidean"
	knn = KNN_class.KNN(k, norm, knnxtrainset, knnytrainset, num_steps)
	knncost = knn.KNN_cost(xtestset, ytestset)
	print ("kNN cost:{}".format(knncost))
	## knn test plot
	neig = knn.kneighbours(xtestset[plot_num])
	knn_plot = knn.predictKNN(neig)
	pickle.dump(knn_plot, open('knn_plot.p','wb'))


## script for testing inidividual models
# KNN_test()
# NN_test()
# RNN_test()
# LSTM_test()
Lin_test()