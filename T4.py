# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 16:34:40 2017
The goal of this assignment is make the neural network convolutional.
@author: Yuting Kou
"""

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import os
import random

'''
1. reload data
'''
data_root = r'D:\\data\\Yuting\\Deep Learning' # Change me to store data elsewhere
pickle_file = os.path.join(data_root, 'notMNIST.pickle')

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)
  
image_size = 28
num_labels = 10

'''
2. reform data
'''
image_size = 28
num_labels = 10
num_channels = 1 # grayscale

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size,image_size,num_channels)).astype(np.float32)
  # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)  

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
  
'''
3. build small convoluted layers
Let's build a small network with two convolutional layers, followed by one 
fully connected layer. Convolutional networks are more expensive computationally, 
so we'll limit its depth and number of fully connected nodes.
uses convolutions with stride 2 to reduce the dimensionality
'''
batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

graph_cnn = tf.Graph()  
with graph_cnn.as_default():
    
    #input data
    tf_train_dataset = tf.placeholder(tf.float32,
                    shape = (batch_size,image_size,image_size,num_channels))
    tf_train_labels = tf.placeholder(tf.float32,shape=(batch_size, num_labels))
    tf_test_dataset = tf.constant(test_dataset)
    tf_valid_dataset = tf.constant(valid_dataset)
    
    # Variables.
    weights = [
            tf.Variable(tf.truncated_normal([patch_size,patch_size,num_channels,depth],stddev=0.1)),
            tf.Variable(tf.truncated_normal([patch_size,patch_size,depth,depth],stddev=0.1)),
            tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1)),
            tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
            ]
    biases = [
            tf.Variable(tf.zeros([depth])),
            tf.Variable(tf.constant(1.0, shape=[depth])),
            tf.Variable(tf.constant(1.0, shape=[num_hidden])),
            tf.Variable(tf.constant(1.0, shape=[num_labels]))
            ]
    
    # Model.
    def model(data):
        conv = tf.nn.conv2d(data, weights[0], [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + biases[0])
        conv = tf.nn.conv2d(hidden, weights[1], [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + biases[1])
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, weights[2]) + biases[2])
        return tf.matmul(hidden, weights[3]) + biases[3]
    
    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))  

#run
num_steps = 1001

with tf.Session(graph=graph_cnn) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
      print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

'''
Problem 2: max pooling
'''
graph_cnn_max = tf.Graph()  

with graph_cnn_max.as_default():
    
    #input data
    tf_train_dataset = tf.placeholder(tf.float32,
                    shape = (batch_size,image_size,image_size,num_channels))
    tf_train_labels = tf.placeholder(tf.float32,shape=(batch_size, num_labels))
    tf_test_dataset = tf.constant(test_dataset)
    tf_valid_dataset = tf.constant(valid_dataset)
    
    global_step = tf.Variable(0)
    
    # Variables.
    
    weights = [
            tf.Variable(tf.truncated_normal([patch_size,patch_size,num_channels,depth],stddev=0.1)),
            tf.Variable(tf.truncated_normal([patch_size,patch_size,depth,depth],stddev=0.1)),
            tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1)),
            tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
            ]
    biases = [
            tf.Variable(tf.zeros([depth])),
            tf.Variable(tf.constant(1.0, shape=[depth])),
            tf.Variable(tf.constant(1.0, shape=[num_hidden])),
            tf.Variable(tf.constant(1.0, shape=[num_labels]))
            ]
    
    # Model.
    def model(data):
        conv1 = tf.nn.conv2d(data, weights[0], [1, 1, 1, 1], padding='SAME')
        bias1 = tf.nn.relu(conv1 + biases[0])
        pool1 = tf.nn.max_pool(bias1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.conv2d(pool1, weights[1], [1, 1, 1, 1], padding='SAME')
        bias2 = tf.nn.relu(conv2 + biases[1])
        pool2 = tf.nn.max_pool(bias2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        shape = pool2.get_shape().as_list()
        reshape = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, weights[2]) + biases[2])
        return tf.matmul(hidden, weights[3]) + biases[3]
        
    # Training Computation
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = tf_train_labels))
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))  
    
#run
num_steps = 1001

with tf.Session(graph=graph_cnn_max) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
      print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))    
      
'''
Problem 4: find best performance of CNN
at L2 regulation
at learning rate decay
'''

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64
beta_regul = 1e-3
drop_out = 0.5

graph_mycnn = tf.Graph()

with graph_mycnn.as_default():
    
    #input data
    tf_train_dataset = tf.placeholder(tf.float32,
                    shape = (batch_size,image_size,image_size,num_channels))
    tf_train_labels = tf.placeholder(tf.float32,shape=(batch_size, num_labels))
    tf_test_dataset = tf.constant(test_dataset)
    tf_valid_dataset = tf.constant(valid_dataset)
    global_step = tf.Variable(0)
    
    
    # Variables.
    size = ((image_size - patch_size + 1)//2 - patch_size + 1)//2
    weights = [
            tf.Variable(tf.truncated_normal([patch_size,patch_size,num_channels,depth],stddev=0.1)),
            tf.Variable(tf.truncated_normal([patch_size,patch_size,depth,depth],stddev=0.1)),
            tf.Variable(tf.truncated_normal([size * size *depth, num_hidden], stddev=0.1)),
            tf.Variable(tf.truncated_normal([num_hidden, num_hidden], stddev=0.1)),
            tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
            ]
    biases = [
            tf.Variable(tf.zeros([depth])),
            tf.Variable(tf.constant(1.0, shape=[depth])),
            tf.Variable(tf.constant(1.0, shape=[num_hidden])),
            tf.Variable(tf.constant(1.0, shape=[num_hidden])),
            tf.Variable(tf.constant(1.0, shape=[num_labels]))
            ]
    
    # Model.
    def model(data,keep_prob=1):
        #C1: input 28*28
        conv1 = tf.nn.conv2d(data, weights[0], [1, 1, 1, 1], padding='VALID')
        bias1 = tf.nn.relu(conv1 + biases[0])
        #C2: input 24*24
        pool2 = tf.nn.avg_pool(bias1,[1,2,2,1],[1,2,2,1],padding = 'VALID')
        #C3: input 12*12
        conv3 = tf.nn.conv2d(pool2, weights[1], [1, 1, 1, 1], padding='VALID')
        bias3 = tf.nn.relu(conv3 + biases[1])
        #C4: input 8*8
        pool4 = tf.nn.avg_pool(bias3, [1,2,2,1],[1,2,2,1], padding='VALID')
        #C5: input 4*4
        shape = pool4.get_shape().as_list()
        reshape = tf.reshape(pool4, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden5 = tf.nn.relu(tf.matmul(reshape, weights[2]) + biases[2])
        #F6
        drop6 = tf.nn.dropout(hidden5,keep_prob)
        #
        hidden6 = tf.nn.relu(tf.matmul(drop6,weights[3])+biases[3])
        drop6 = tf.nn.dropout(hidden6,keep_prob)
        return tf.matmul(drop6, weights[4]) + biases[4]
    
    # Training Computation
    logits = model(tf_train_dataset,drop_out)
    loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = tf_train_labels))
    # Optimizer.
    learning_rate = tf.train.exponential_decay(0.05,  global_step, 1000, 0.85, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)
    #optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset,1))
    test_prediction = tf.nn.softmax(model(tf_test_dataset,1))  
    
#run
num_steps = 5001

with tf.Session(graph=graph_mycnn) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
      print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))