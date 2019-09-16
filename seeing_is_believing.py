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

def transform_obs(obs_value, separate_tgtvelocity_field=False):
    obs_value_transformed = np.asarray(obs_value)

    if separate_tgtvelocity_field:
        corr_shape_vel_tgtfield = np.reshape(obs_value_transformed[:242], (2, 11, 11))
        center_value = corr_shape_vel_tgtfield[:, 5, 5]
        center_value_normalized = center_value/(np.sqrt(center_value[0]*center_value[0] + center_value[1]*center_value[1])*2)

        obs_value_transformed = np.hstack([obs_value_transformed[242:]/10, center_value_normalized[0], center_value_normalized[1]])

    else:
        obs_value_transformed[242:] = obs_value_transformed[242:]/5
        obs_value_transformed = obs_value_transformed/10
    
    return obs_value_transformed

class L2M2019EnvStanding(L2M2019Env):
    def step(self, action, project=True, obs_as_dict=True):
        observation, reward, done, info = super(L2M2019EnvStanding, self).step(action, project=project, obs_as_dict=obs_as_dict)

        reward = 0.1 #.d_reward['alive'] --> this isn't working :(
        # if not super(L2M2019EnvStanding, self).is_done() and (super(L2M2019EnvStanding, self).osim_model.istep >= super(L2M2019EnvStanding, self).spec.timestep_limit):
        #     reward += 10

        return observation, reward, done, info

####################
# INITIAL SETUP
####################

# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--store_trajectories', default=False, action='store_true')
parser.add_argument('--dontvisualize', default=True, action='store_false')
parser.add_argument('--how_many_episodes_to_run', default=3, type=int)
parser.add_argument('--log_path', action='store', default=".")
parser.add_argument('--gamma', default=0.995, type=float)
args = parser.parse_args()

# Load walking environment
env = L2M2019EnvStanding(visualize=args.dontvisualize, difficulty=0)
env.reset()

nb_actions = env.action_space.shape[0]
separate_tgtvelocity_field = True
num_options = 2

####################
# NETWORK CREATION
####################

if separate_tgtvelocity_field:
    x = tf.placeholder(tf.float32, shape=(None, 99), name='x')
    env_obs_shape = 99
else:
    x = tf.placeholder(tf.float32, shape=(None, 339), name='x')
    env_obs_shape = 339

weights_init = tf.contrib.layers.variance_scaling_initializer()
bias_init = tf.constant_initializer(0.0)

# ACTOR(s)

a_h1 = fully_connected(inputs=x, num_outputs=444, activation_fn=tf.nn.relu, weights_initializer=weights_init, weights_regularizer=None, biases_initializer=bias_init, scope='a_h1')
a_h2 = fully_connected(inputs=a_h1, num_outputs=444, activation_fn=tf.nn.relu, weights_initializer=weights_init, weights_regularizer=None, biases_initializer=bias_init, scope='a_h2')
a_h3 = fully_connected(inputs=a_h2, num_outputs=444, activation_fn=tf.nn.relu, weights_initializer=weights_init, weights_regularizer=None, biases_initializer=bias_init, scope='a_h3')

actor_heads = []
actor_heads_logits = []

action_choices = np.arange(num_options)/(num_options - 1)

for i in range(nb_actions):
	if num_options == 2:
		actor_heads_logits.append(tf.concat([fully_connected(inputs=a_h3, num_outputs=1, activation_fn=None, weights_initializer=weights_init, weights_regularizer=None, biases_initializer=bias_init, scope='a' + str(i)), tf.zeros((tf.shape(x)[0], 1))], axis=1))
	else:
		actor_heads_logits.append(fully_connected(inputs=a_h3, num_outputs=num_options, activation_fn=None, weights_initializer=weights_init, weights_regularizer=None, biases_initializer=bias_init, scope='a' + str(i)))
	actor_heads.append(tf.nn.softmax(actor_heads_logits[-1]))

# CRITIC

v_h1 = fully_connected(inputs=x, num_outputs=444, activation_fn=tf.nn.relu, weights_initializer=weights_init, weights_regularizer=None, biases_initializer=bias_init, scope='v_h1')
v_h2 = fully_connected(inputs=v_h1, num_outputs=444, activation_fn=tf.nn.relu, weights_initializer=weights_init, weights_regularizer=None, biases_initializer=bias_init, scope='v_h2')
v_h3 = fully_connected(inputs=v_h2, num_outputs=444, activation_fn=tf.nn.relu, weights_initializer=weights_init, weights_regularizer=None, biases_initializer=bias_init, scope='v_h3')

critic = fully_connected(inputs=v_h3, num_outputs=1, activation_fn=None, weights_initializer=weights_init, weights_regularizer=None, biases_initializer=bias_init, scope='vf')

saver = tf.train.Saver()

####################
# NETWORK TESTING
####################

sess = tf.Session()
saver.restore(sess, args.log_path + "/best_model.ckpt")

discounted_rewards_list = []
value_function_list = []
ep_length_list = []
rewards_list = []

if args.store_trajectories:
	saved_obs = []

for i in range(args.how_many_episodes_to_run):
	action_supplier = np.zeros(nb_actions)
	discounted_reward = 0
	total_reward = 0
	ep_length = 0
	net_gamma = 1

	obs = env.reset(obs_as_dict=False)
	obs = transform_obs(obs, separate_tgtvelocity_field=True)
	done = False

	value_function_list.append(sess.run(critic, feed_dict={x: np.reshape(obs, (1, env_obs_shape))})[0][0])

	while not done:
		if args.store_trajectories:
			saved_obs.append(obs)
		all_actions_probs = sess.run(actor_heads, feed_dict={x: np.reshape(obs, (1, env_obs_shape))})
		action_supplier = np.zeros(nb_actions)
		for h in range(nb_actions):
			action_supplier[h] = np.random.choice(action_choices, p=all_actions_probs[h][0])

		obs, reward, done, info = env.step(action_supplier, obs_as_dict=False)
		obs = transform_obs(obs, separate_tgtvelocity_field=True)
		
		ep_length += 1
		total_reward += reward
		discounted_reward += reward*net_gamma
		net_gamma *= args.gamma

	ep_length_list.append(ep_length)
	rewards_list.append(total_reward)
	discounted_rewards_list.append(discounted_reward)
	print("%3d Current episode reward: %0.2f"%(i+1, total_reward))

print("\nAverage reward from policy: %0.2f"%(np.mean(rewards_list)))
print("Average episode length from policy: %0.2f \n"%(np.mean(ep_length_list)))
print("Average discounted reward from policy: %0.2f"%(np.mean(discounted_rewards_list)))
print("Average beginning value function from policy: %0.2f"%(np.mean(value_function_list)))
print("Percentage discrepancy: %d%%"%(int((np.abs(np.mean(value_function_list)-np.mean(discounted_rewards_list))/np.mean(discounted_rewards_list))*100)))

if args.store_trajectories:
	np.save(args.log_path + "/saved_trajectories", np.asarray(saved_obs))



