# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 14:33:49 2017
explore regulation tech

@author: Yuting Kou
"""

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import os
import matplotlib.pyplot as plt

'''
1. reload data
'''
data_root = r'D:\\data\\Deep Learning' # Change me to store data elsewhere
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
def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
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
Problem 1:
Introduce and tune L2 regularization for both logistic and neural network models. 

#logistic models#
'''
batch_size = 128

graph_l2 = tf.Graph()
with graph_l2.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    beta_regul = tf.placeholder(tf.float32)
  
    # Variables.
    weights = tf.Variable(
            tf.truncated_normal([image_size * image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))
    
  
    # Training computation.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    #add l2 regulation
    loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))+beta_regul*tf.nn.l2_loss(weights)
  
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
            tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

'''
training model
'''
num_steps = 3001

with tf.Session(graph=graph_l2) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels,beta_regul : 1e-3}
        _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_labels))
            print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

#tuning betas

def tuning_beta(regul_value,graph,name):
    for regul in regul_value:
      with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
          offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
          batch_data = train_dataset[offset:(offset + batch_size), :]
          batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
          feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, beta_regul : regul}
          _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        accuracy_val.append(accuracy(test_prediction.eval(), test_labels))
        
    plt.semilogx(regul_value, accuracy_val)
    plt.grid(True)
    plt.title('Test accuracy by regularization (%s)'% (name))
    plt.show()    
    
num_steps = 3001
regul_val = [pow(10, i) for i in np.arange(-4, -2, 0.1)]
accuracy_val = []
tuning_beta(regul_val,graph_l2,"logistic")    

'''
neural network reg
'''

num_hidden_nodes = 1024
batch_size = 128
graph_nn_l2 = tf.Graph()

with graph_nn_l2.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    beta_regul = tf.placeholder(tf.float32)
  
    # Variables.
    weights = [
            tf.Variable(tf.truncated_normal([image_size * image_size, num_hidden_nodes])),
            tf.Variable(tf.truncated_normal([num_hidden_nodes, num_labels]))]
    biases = [
            tf.Variable(tf.zeros([num_hidden_nodes])),
            tf.Variable(tf.zeros([num_labels]))]
            
  
    # Training computation.
    hidden_layer = tf.nn.relu(tf.matmul(tf_train_dataset, weights[0]) + biases[0])
    logits = tf.matmul(hidden_layer,weights[1])+biases[1]
    loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits) + \
            beta_regul*(tf.nn.l2_loss(weights[0])+tf.nn.l2_loss(weights[1])))
  
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    
    lay1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights[0]) + biases[0])
    valid_prediction = tf.nn.softmax(tf.matmul(lay1_valid, weights[1]) + biases[1])
    
    lay1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights[0]) + biases[0])
    test_prediction = tf.nn.softmax(tf.matmul(lay1_test, weights[1]) + biases[1])

'''
run
'''
num_steps = 3001

with tf.Session(graph=graph_nn_l2) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, beta_regul : 1e-3}
        _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_labels))
            print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

#tuning betas
num_steps = 3001
regul_val = [pow(10, i) for i in np.arange(-4, -2, 0.1)]
accuracy_val = []
tuning_beta(regul_val,graph_nn_l2,"1-layer net")    

'''
Problem 2
Let's demonstrate an extreme case of overfitting. Restrict your training data to just a few batches. What happens?

logistic 
'''
batch_size = 128
num_hidden_nodes = 1024
num_steps = 101
num_batches = 3

with tf.Session(graph=graph_nn_l2) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        #offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        offset = ((step % num_batches) * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, beta_regul : 1e-3}
        _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 2 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_labels))
            print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

'''
Problem 3
Introduce Dropout on the hidden layer of the neural network. Remember: Dropout should only be introduced during training, not evaluation, otherwise your evaluation results would be stochastic as well. TensorFlow provides nn.dropout() for that, but you have to make sure it's only inserted during training.
What happens to our extreme overfitting case?

NN
'''
batch_size = 128
num_hidden_nodes = 1024

graph_nn = tf.Graph()
with graph_nn.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
  
    # Variables.
    weights = [
            tf.Variable(tf.truncated_normal([image_size * image_size, num_hidden_nodes])),
            tf.Variable(tf.truncated_normal([num_hidden_nodes, num_labels]))]
    biases = [
            tf.Variable(tf.zeros([num_hidden_nodes])),
            tf.Variable(tf.zeros([num_labels]))]
         
    # Training computation.
      # Training computation.
    hidden_layer = tf.nn.relu(tf.matmul(tf_train_dataset, weights[0]) + biases[0])
    drop1 = tf.nn.dropout(hidden_layer, 0.5)
    logits = tf.matmul(drop1,weights[1])+biases[1]
    loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  
        
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    
    lay1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights[0]) + biases[0])
    valid_prediction = tf.nn.softmax(tf.matmul(lay1_valid, weights[1]) + biases[1])
    
    lay1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights[0]) + biases[0])
    test_prediction = tf.nn.softmax(tf.matmul(lay1_test, weights[1]) + biases[1])
    
num_steps = 3001
batch_size = 128

with tf.Session(graph=graph_nn) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
   
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels)) 
  
'''
Problem 4: Find Best Performance
'''
  
num_hidden_nodes1 = 1024
num_hidden_nodes2 = 100
batch_size = 128
graph_nn_l2_2 = tf.Graph()

with graph_nn_l2_2.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
   
    global_step = tf.Variable(0)

  
    # Variables.
    weights = [
            tf.Variable(tf.truncated_normal([image_size * image_size,num_hidden_nodes1],stddev=np.sqrt(2.0 / (image_size * image_size))) ),
            tf.Variable(tf.truncated_normal([num_hidden_nodes1, num_hidden_nodes2],stddev=np.sqrt(2.0 / (num_hidden_nodes1)))),
            tf.Variable(tf.truncated_normal([num_hidden_nodes2, num_labels],stddev=np.sqrt(2.0 / (num_hidden_nodes2))))]
    biases = [
            tf.Variable(tf.zeros([num_hidden_nodes1])),
            tf.Variable(tf.zeros([num_hidden_nodes2])),
            tf.Variable(tf.zeros([num_labels]))]
            
  
    # Training computation.
    hidden_layer1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights[0]) + biases[0])
    hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer1, weights[1]) + biases[1])
    logits = tf.matmul(hidden_layer2,weights[2])+biases[2]
    loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
          
  
    # Optimizer.
    #learning rate
    learning_rate = tf.train.exponential_decay(0.5,  global_step, 1000, 0.65, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step= global_step)
  
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    
    lay1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights[0]) + biases[0])
    lay2_valid = tf.nn.relu(tf.matmul(lay1_valid, weights[1]) + biases[1])
    valid_prediction = tf.nn.softmax(tf.matmul(lay2_valid, weights[2]) + biases[2])
    
    lay1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights[0]) + biases[0])
    lay2_test = tf.nn.relu(tf.matmul(lay1_test, weights[1]) + biases[1])
    test_prediction = tf.nn.softmax(tf.matmul(lay2_test, weights[2]) + biases[2])

'''
run
'''
num_steps = 9001

#display sample
import random
def disp_sample_dataset(dataset, labels):
  items = random.sample(range(len(labels)), 8)
  for i, item in enumerate(items):
    plt.subplot(2, 4, i+1)
    plt.axis('off')
    plt.title(pretty_labels[labels[item]])
    plt.imshow(dataset[item])   
    
pretty_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'}
   

with tf.Session(graph=graph_nn_l2_2) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_labels))
            print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
       
    disp_sample_dataset(test_dataset.reshape(-1,image_size,image_size), np.argmax(test_prediction.eval(), 1))
    plt.show()    
    

 
