####################
# IMPORTS
####################

from tensorflow.contrib.layers import *
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import datetime
import time
import sys

from osim.env import *
from osim.http.client import Client

import argparse
import math

####################
# SOME FUNCTIONS
####################

def transform_obs(obs_value):
	obs_value_transformed = np.asarray(obs_value)
	obs_value_transformed[242:] = obs_value_transformed[242:]/5
	obs_value_transformed = obs_value_transformed/10

	return obs_value_transformed

def visualize_layer_responses(what_to_plot, title):
    plt.hist(what_to_plot.flatten(), 50, density=True, facecolor='g', alpha=0.75)
    plt.title(title)
    plt.show()

####################
# INITIAL SETUP
####################

# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--log_path', action='store', default=".")
args = parser.parse_args()

# Load walking environment
env = L2M2019Env(visualize=False)
env.reset()

nb_actions = env.action_space.shape[0]

exponentially_decay_action = True
exp_action_decay_const = 0.9
exp_action_addition = 0.1

####################
# NETWORK CREATION
####################

x = tf.placeholder(tf.float32, shape=(None,) + env.observation_space.shape, name='x')
flattened_x = tf.contrib.layers.flatten(x)

weights_init = tf.contrib.layers.variance_scaling_initializer()
bias_init = tf.constant_initializer(0.0)

# ACTOR(s)

a_h1 = fully_connected(inputs=flattened_x, num_outputs=444, activation_fn=tf.nn.relu, weights_initializer=weights_init, weights_regularizer=None, biases_initializer=bias_init, scope='a_h1')
a_h2 = fully_connected(inputs=a_h1, num_outputs=444, activation_fn=tf.nn.relu, weights_initializer=weights_init, weights_regularizer=None, biases_initializer=bias_init, scope='a_h2')
a_h3 = fully_connected(inputs=a_h2, num_outputs=444, activation_fn=tf.nn.relu, weights_initializer=weights_init, weights_regularizer=None, biases_initializer=bias_init, scope='a_h3')

actor_heads = []
actor_heads_logits = []

if exponentially_decay_action:
	num_options = 2
else:
	num_options = 2
	action_choices = np.arange(num_options)/(num_options - 1)

for i in range(nb_actions):
	actor_heads_logits.append(fully_connected(inputs=a_h3, num_outputs=num_options, activation_fn=None, weights_initializer=weights_init, weights_regularizer=None, biases_initializer=bias_init, scope='a' + str(i)))
	actor_heads.append(tf.nn.softmax(actor_heads_logits[-1]))

# CRITIC

v_h1 = fully_connected(inputs=flattened_x, num_outputs=444, activation_fn=tf.nn.relu, weights_initializer=weights_init, weights_regularizer=None, biases_initializer=bias_init, scope='v_h1')
v_h2 = fully_connected(inputs=v_h1, num_outputs=444, activation_fn=tf.nn.relu, weights_initializer=weights_init, weights_regularizer=None, biases_initializer=bias_init, scope='v_h2')
v_h3 = fully_connected(inputs=v_h2, num_outputs=444, activation_fn=tf.nn.relu, weights_initializer=weights_init, weights_regularizer=None, biases_initializer=bias_init, scope='v_h3')

critic = fully_connected(inputs=v_h3, num_outputs=1, activation_fn=None, weights_initializer=weights_init, weights_regularizer=None, biases_initializer=bias_init, scope='vf')

saver = tf.train.Saver()

####################
# NETWORK TESTING
####################

sess = tf.Session()
sess.run(tf.global_variables_initializer())

obs_batch_shape = (-1,) + env.observation_space.shape
loaded_obs = np.load(args.log_path + "/saved_trajectories.npy")

visualize_layer_responses(sess.run(a_h1, feed_dict={x: np.reshape(loaded_obs, obs_batch_shape)}), "a_h1")
visualize_layer_responses(sess.run(a_h2, feed_dict={x: np.reshape(loaded_obs, obs_batch_shape)}), "a_h2")
visualize_layer_responses(sess.run(a_h3, feed_dict={x: np.reshape(loaded_obs, obs_batch_shape)}), "a_h3")
visualize_layer_responses(np.asarray(sess.run(actor_heads, feed_dict={x: np.reshape(loaded_obs, obs_batch_shape)})), "probas")

saver.restore(sess, args.log_path + "/model.ckpt")

visualize_layer_responses(sess.run(a_h1, feed_dict={x: np.reshape(loaded_obs, obs_batch_shape)}), "a_h1_trained")
visualize_layer_responses(sess.run(a_h2, feed_dict={x: np.reshape(loaded_obs, obs_batch_shape)}), "a_h2_trained")
visualize_layer_responses(sess.run(a_h3, feed_dict={x: np.reshape(loaded_obs, obs_batch_shape)}), "a_h3_trained")
visualize_layer_responses(np.asarray(sess.run(actor_heads, feed_dict={x: np.reshape(loaded_obs, obs_batch_shape)})), "probas_trained")

