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

def safe_entropy(logits): # function copied from openai baselines
    a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

def visualize_layer_responses(what_to_plot, title):
    plt.hist(what_to_plot.flatten(), 50, density=True, facecolor='g', alpha=0.75)
    plt.title(title)
    plt.show()

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

####################
# INITIAL SETUP
####################

# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--log_path', action='store', default=".")
args = parser.parse_args()

# Load walking environment
env = L2M2019Env(visualize=False, difficulty=0)
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

####################
# NETWORK TRAINING
####################

lr = 1e-4
gamma = 0.99
vf_coeff = 0.5
adv_lambda = 0.97
max_grad_norm = 0.5
entropy_coeff = 0.01

num_training_steps = 5500
num_rollout_steps = 100
reward_avg_num = 10
log_interval = 100

ADV = tf.placeholder(tf.float32, shape=(None, 1), name='adv')
TARG = tf.placeholder(tf.float32, shape=(None, 1), name='targ')
A = [tf.placeholder(tf.int32, shape=(None, 1), name='a'+str(i)) for i in range(nb_actions)]

pg_loss = tf.reduce_mean([tf.expand_dims(tf.nn.softmax_cross_entropy_with_logits_v2(logits=actor_heads_logits[i], labels=tf.one_hot(A[i], num_options)), axis=1)*ADV for i in range(nb_actions)])
entropy_loss = tf.reduce_mean([safe_entropy(actor_heads_logits[i]) for i in range(nb_actions)])
critic_loss = tf.reduce_mean(tf.square(critic - TARG))

total_loss = pg_loss + vf_coeff*critic_loss - entropy_coeff*entropy_loss

params = tf.trainable_variables()
grads = tf.gradients(total_loss, params)
grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

grads = list(zip(grads, params))
trainer = tf.train.AdamOptimizer(learning_rate=lr)
_train = trainer.apply_gradients(grads)

saver = tf.train.Saver()

learning_batch_actions = [np.zeros((num_rollout_steps, 1)) for i in range(nb_actions)]
learning_batch_input = np.zeros((num_rollout_steps, env_obs_shape))
learning_batch_delta_values = np.zeros((num_rollout_steps, 1))
learning_batch_adv_values = np.zeros((num_rollout_steps, 1))
learning_batch_targets = np.zeros((num_rollout_steps, 1))
helper_batch_dones = np.zeros((num_rollout_steps, 1))

episode_count = 0
ep_reward_count = 0
max_action_prob = 0
ep_rewards_list = []
ep_rewards_avged = []
timeout_fraction_list = []
timeout_fraction_avged = []
max_probability_stats2plot = []

obs = env.reset(obs_as_dict=False)
obs = transform_obs(obs, separate_tgtvelocity_field=separate_tgtvelocity_field)
action_supplier = np.zeros(nb_actions)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
state_value = sess.run(critic, feed_dict={x: np.reshape(obs, (1, env_obs_shape))})[0][0]

t_i = time.time()
t_interim = time.time()

for i in range(num_training_steps):
    if i%log_interval == 0:
        max_action_prob = 0
        print("-------------------")
        print("Training round %d"%i)

    for j in range(num_rollout_steps):

        all_actions_probs = sess.run(actor_heads, feed_dict={x: np.reshape(obs, (1, env_obs_shape))})
        action_supplier = np.zeros(nb_actions)
        for h in range(nb_actions):
            action_supplier[h] = np.random.choice(action_choices, p=all_actions_probs[h][0])
        
        for h in range(nb_actions):
            learning_batch_actions[h][j, 0] = action_supplier[h]
        
        learning_batch_input[j] = obs

        obs, reward, done, info = env.step(action_supplier, obs_as_dict=False)
        obs = transform_obs(obs, separate_tgtvelocity_field=separate_tgtvelocity_field)
        ep_reward_count += reward

        all_actions_probs_numpified = np.reshape(np.asarray(all_actions_probs), (22, 2))
        max_action_prob += np.amin(np.amax(all_actions_probs_numpified, axis=1))

        if done:
            helper_batch_dones[j, 0] = 1

            ep_rewards_list.append(ep_reward_count)
            ep_reward_count = 0
            episode_count += 1

            if env.is_done():
                timeout_fraction_list.append(0)
            else:
                timeout_fraction_list.append(1)

            one_step_target = reward
            obs = env.reset(obs_as_dict=False)
            obs = transform_obs(obs, separate_tgtvelocity_field=separate_tgtvelocity_field)
            new_state_value = sess.run(critic, feed_dict={x: np.reshape(obs, (1, env_obs_shape))})[0][0]
        else:
            helper_batch_dones[j, 0] = 0
            
            new_state_value = sess.run(critic, feed_dict={x: np.reshape(obs, (1, env_obs_shape))})[0][0]
            one_step_target = reward + gamma*new_state_value

        learning_batch_delta_values[j, 0] = one_step_target - state_value
        learning_batch_targets[j, 0] = state_value

        state_value = new_state_value

    if i%log_interval == 0:
        timeout_fraction_avged.append(np.mean(timeout_fraction_list[-reward_avg_num:]))
        max_probability_stats2plot.append(max_action_prob/num_rollout_steps)
        ep_rewards_avged.append(np.mean(ep_rewards_list[-reward_avg_num:]))

        print("Time for most current batch: " + str(datetime.timedelta(seconds=(time.time() - t_interim))))
        print("Timeout fraction over last %d episodes: %0.2f"%(reward_avg_num, timeout_fraction_avged[-1]))
        print("Average reward over last %d episodes: %0.2f"%(reward_avg_num, ep_rewards_avged[-1]))
        print("Average max probability in current rollout: %0.2f"%(max_probability_stats2plot[-1]))
        t_interim = time.time()
        sys.stdout.flush()

    # GAE (in PPO if less than 200 timesteps are remaining, just break out of the loop. lambda^200 will be close enought to 0)
    running_total = 0
    for j in range(num_rollout_steps):
        if helper_batch_dones[num_rollout_steps-1-j, 0] == 1: # after the episode has finished, deltas will all be zero
            running_total = 0
        running_total = learning_batch_delta_values[num_rollout_steps-1-j, 0]*(1-np.power(adv_lambda, j+1)) + gamma*adv_lambda*running_total
        learning_batch_adv_values[num_rollout_steps-1-j, 0] = running_total/(1-np.power(adv_lambda, j+1))
        learning_batch_targets[num_rollout_steps-1-j, 0] += learning_batch_adv_values[num_rollout_steps-1-j, 0]

    training_feeddict = {i:d for i, d in zip(A, learning_batch_actions)}
    training_feeddict[ADV] = learning_batch_adv_values
    training_feeddict[TARG] = learning_batch_targets
    training_feeddict[x] = learning_batch_input

    sess.run(_train, feed_dict=training_feeddict)

timeout_fraction_avged.append(np.mean(timeout_fraction_list[-reward_avg_num:]))
ep_rewards_avged.append(np.mean(ep_rewards_list[-reward_avg_num:]))

np.save(args.log_path + "/max_probability_stats2plot", np.asarray(max_probability_stats2plot))
np.save(args.log_path + "/timeout_fraction_avged", np.asarray(timeout_fraction_avged))
np.save(args.log_path + "/ep_rewards_avged", np.asarray(ep_rewards_avged))
saver.save(sess, args.log_path + "/model.ckpt")

print("Time for training to complete: " + str(datetime.timedelta(seconds=(time.time() - t_i))))

