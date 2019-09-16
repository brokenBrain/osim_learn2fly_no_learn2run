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
parser.add_argument('--log_path', action='store', default=".")
args = parser.parse_args()

# Load walking environment
env = L2M2019EnvStanding(visualize=False, difficulty=0)
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

action_choices = (np.arange(num_options)/(num_options - 1)).astype(int)

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

# TRAINING HYPER PARAMS

lr = 1e-4
vf_coeff = 1
kl_coeff = 0
gamma = 0.995
td_lambda = 0.97
max_grad_norm = 0.5
entropy_coeff = 0.0

adam_beta1 = 0.9
adam_beta2 = 0.999
adam_epsilon = 1e-6

ppo_dataset_size = 4096
num_ppo_iterations = 250
ppo_minibatch_size = 256
ppo_clip_parameter = 0.2
num_epochs_per_iteration = 10
kl_cutoff = (np.log(1 + ppo_clip_parameter) - np.log(1 - ppo_clip_parameter))/2 # note that this isn't really a penalty on kl but on log(rtheta)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(lr, global_step, int((ppo_dataset_size/ppo_minibatch_size)*num_epochs_per_iteration), 0.96, staircase=True)

load_path = "../outputs/adamepsi_1e6/0/2"

# TENSORFLOW LOSS SETTING UP

ADV = tf.placeholder(tf.float32, shape=(None, 1), name='adv')
TARG = tf.placeholder(tf.float32, shape=(None, 1), name='targ')
A = [tf.placeholder(tf.int32, shape=(None, 1), name='a'+str(i)) for i in range(nb_actions)]
OLD_PI = [tf.placeholder(tf.float32, shape=(None, 1), name='oldpi'+str(i)) for i in range(nb_actions)]

eps_diff = [tf.abs(1 - tf.divide(tf.reduce_sum(actor_heads[i]*tf.squeeze(tf.one_hot(A[i], num_options)), axis=1, keepdims=True), OLD_PI[i])) for i in range(nb_actions)]
r_theta_metrics = (tf.reduce_min(eps_diff), tf.reduce_mean(eps_diff), tf.reduce_max(eps_diff))

r_theta = [tf.divide(tf.reduce_sum(actor_heads[i]*tf.squeeze(tf.one_hot(A[i], num_options)), axis=1, keepdims=True), OLD_PI[i]) for i in range(nb_actions)]
ppo_loss = -tf.reduce_mean([tf.minimum(r_theta[i]*ADV, tf.clip_by_value(r_theta[i], 1-ppo_clip_parameter, 1+ppo_clip_parameter)*ADV) for i in range(nb_actions)])
entropy_loss = tf.reduce_mean([safe_entropy(actor_heads_logits[i]) for i in range(nb_actions)])
critic_loss = tf.reduce_mean(tf.square(critic - TARG))
extra_kl_penalty = tf.reduce_mean([tf.maximum(tf.cast(kl_cutoff*kl_cutoff, tf.float32), tf.square(tf.log(r_theta[i]))) for i in range(nb_actions)])

total_loss = ppo_loss + vf_coeff*critic_loss - entropy_coeff*entropy_loss + kl_coeff*extra_kl_penalty

params = tf.trainable_variables()
grads = tf.gradients(total_loss, params)
grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

grads = list(zip(grads, params))
trainer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=adam_beta1, beta2=adam_beta2, epsilon=adam_epsilon)
_train = trainer.apply_gradients(grads, global_step=global_step)

saver = tf.train.Saver(var_list=tf.trainable_variables())

learning_batch_actions = [np.zeros((ppo_dataset_size, 1)) for i in range(nb_actions)]
learning_batch_old_pis = [np.zeros((ppo_dataset_size, 1)) for i in range(nb_actions)]
learning_batch_input = np.zeros((ppo_dataset_size, env_obs_shape))
learning_batch_delta_values = np.zeros((ppo_dataset_size, 1))
learning_batch_adv_values = np.zeros((ppo_dataset_size, 1))
learning_batch_targets = np.zeros((ppo_dataset_size, 1))
helper_batch_dones = np.zeros((ppo_dataset_size, 1))
action_supplier = np.zeros(nb_actions)

ep_gamma = 1
ep_reward_count = 0
best_reward_sofar = -1
ep_discounted_reward = 0

vf_discrepancy = []
ep_rewards_list = []
ep_rewards_avged = []
timeout_fraction_list = []
timeout_fraction_avged = []
max_probability_stats2plot = []
maxmax_probability_stats2plot = []

all_action_probabilities_tracker = np.zeros((num_ppo_iterations, nb_actions))

obs = env.reset(obs_as_dict=False)
obs = transform_obs(obs, separate_tgtvelocity_field=separate_tgtvelocity_field)
action_supplier = np.zeros(nb_actions)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if load_path is not None:
    saver.restore(sess, load_path + "/model.ckpt")

state_value = sess.run(critic, feed_dict={x: np.reshape(obs, (1, env_obs_shape))})[0][0]

t_i = time.time()

for i in range(num_ppo_iterations):
    t_interim = time.time()

    episode_count = 0
    max_action_prob = 0
    maxmax_action_prob = 0
    ep_discounted_reward_list = []
    ep_discounted_reward_list_vf = []
    
    print("-------------------")
    print("PPO iteration %d"%i)

    for j in range(ppo_dataset_size):

        all_actions_probs = sess.run(actor_heads, feed_dict={x: np.reshape(obs, (1, env_obs_shape))})
        for h in range(nb_actions):
            action_supplier[h] = np.random.choice(action_choices, p=all_actions_probs[h][0])
            learning_batch_actions[h][j, 0] = action_supplier[h]
            learning_batch_old_pis[h][j, 0] = all_actions_probs[h][0][int(action_supplier[h])]
        
        learning_batch_input[j] = obs

        obs, reward, done, info = env.step(action_supplier, obs_as_dict=False)
        obs = transform_obs(obs, separate_tgtvelocity_field=separate_tgtvelocity_field)
        ep_discounted_reward += reward*ep_gamma
        ep_reward_count += reward
        ep_gamma *= gamma

        all_actions_probs_numpified = np.reshape(np.asarray(all_actions_probs), (nb_actions, 2))
        each_actions_maxprob = np.amax(all_actions_probs_numpified, axis=1)

        all_action_probabilities_tracker[i, :] += each_actions_maxprob
        maxmax_action_prob += np.amax(each_actions_maxprob)
        max_action_prob += np.amin(each_actions_maxprob)

        if done:
            helper_batch_dones[j, 0] = 1

            ep_discounted_reward_list.append(ep_discounted_reward)
            ep_rewards_list.append(ep_reward_count)
            ep_discounted_reward = 0
            ep_reward_count = 0
            episode_count += 1
            ep_gamma = 1

            if env.is_done():
                new_state_value = 0
                timeout_fraction_list.append(0)
            else:
                new_state_value = sess.run(critic, feed_dict={x: np.reshape(obs, (1, env_obs_shape))})[0][0]
                timeout_fraction_list.append(1)

            one_step_target = reward + gamma*new_state_value
            obs = env.reset(obs_as_dict=False)
            obs = transform_obs(obs, separate_tgtvelocity_field=separate_tgtvelocity_field)
            new_state_value = sess.run(critic, feed_dict={x: np.reshape(obs, (1, env_obs_shape))})[0][0]
            ep_discounted_reward_list_vf.append(new_state_value)
        else:
            helper_batch_dones[j, 0] = 0
            
            new_state_value = sess.run(critic, feed_dict={x: np.reshape(obs, (1, env_obs_shape))})[0][0]
            one_step_target = reward + gamma*new_state_value

        learning_batch_delta_values[j, 0] = one_step_target - state_value
        learning_batch_targets[j, 0] = state_value

        state_value = new_state_value

    # GAE
    running_total = 0
    for j in range(ppo_dataset_size):
        if helper_batch_dones[ppo_dataset_size-1-j, 0] == 1:
            running_total = 0
        running_total = learning_batch_delta_values[ppo_dataset_size-1-j, 0] + gamma*td_lambda*running_total
        learning_batch_adv_values[ppo_dataset_size-1-j, 0] = running_total
        learning_batch_targets[ppo_dataset_size-1-j, 0] += learning_batch_adv_values[ppo_dataset_size-1-j, 0]

    # LOGGING
    vf_discrepancy.append(int((np.abs(np.mean(ep_discounted_reward_list_vf)-np.mean(ep_discounted_reward_list))/np.mean(ep_discounted_reward_list))*100))
    timeout_fraction_avged.append(np.mean(timeout_fraction_list[-episode_count:]))
    maxmax_probability_stats2plot.append(maxmax_action_prob/ppo_dataset_size)
    max_probability_stats2plot.append(max_action_prob/ppo_dataset_size)
    ep_rewards_avged.append(np.mean(ep_rewards_list[-episode_count:]))

    print("Time for current PPO iteration: " + str(datetime.timedelta(seconds=(time.time() - t_interim))))
    print("Timeout fraction over last %d episodes: %0.2f"%(episode_count, timeout_fraction_avged[-1]))
    print("Average reward over last %d episodes: %0.2f"%(episode_count, ep_rewards_avged[-1]))
    print("Average max probability in current rollout: %0.2f"%(max_probability_stats2plot[-1]))
    print("Average maxmax probability in current rollout: %0.2f"%(maxmax_probability_stats2plot[-1]))
    print("Average discounted reward from policy: %0.2f"%(np.mean(ep_discounted_reward_list)))
    print("Average beginning value function from policy: %0.2f"%(np.mean(ep_discounted_reward_list_vf)))
    print("Percentage discrepancy: %d%%"%(vf_discrepancy[-1]))
    print("Current learning rate: %0.2e"%(sess.run(learning_rate)))

    if ep_rewards_avged[-1] >= best_reward_sofar:
        best_reward_sofar = ep_rewards_avged[-1]
        saver.save(sess, args.log_path + "/best_model.ckpt")
        print("best model saved in iteration %d"%i)

    t_interim = time.time()
    sys.stdout.flush()

    # PPO training
    num_grad_updates = int((ppo_dataset_size/ppo_minibatch_size)*num_epochs_per_iteration)
    rtheta_metric_diary = np.zeros((num_grad_updates, 3))
    for j in range(num_grad_updates):
        sampled_indices = np.random.choice(ppo_dataset_size, size=(ppo_minibatch_size), replace=False)

        training_feeddict = {k:d[sampled_indices] for k, d in zip(A, learning_batch_actions)}
        for k in range(nb_actions):
            training_feeddict[OLD_PI[k]] = learning_batch_old_pis[k][sampled_indices]

        training_feeddict[ADV] = learning_batch_adv_values[sampled_indices]
        training_feeddict[TARG] = learning_batch_targets[sampled_indices]
        training_feeddict[x] = learning_batch_input[sampled_indices]
        r_theta_metric_vals, _ = sess.run((r_theta_metrics, _train), feed_dict=training_feeddict)
        rtheta_metric_diary[j, 0] = r_theta_metric_vals[0]
        rtheta_metric_diary[j, 1] = r_theta_metric_vals[1]
        rtheta_metric_diary[j, 2] = r_theta_metric_vals[2]
    np.save(args.log_path + "/rtheta_metrics_" + str(i), np.asarray(rtheta_metric_diary))

np.save(args.log_path + "/all_action_probabilities_tracker", all_action_probabilities_tracker/ppo_dataset_size)
np.save(args.log_path + "/maxmax_probability_stats2plot", np.asarray(maxmax_probability_stats2plot))
np.save(args.log_path + "/max_probability_stats2plot", np.asarray(max_probability_stats2plot))
np.save(args.log_path + "/timeout_fraction_avged", np.asarray(timeout_fraction_avged))
np.save(args.log_path + "/ep_rewards_avged", np.asarray(ep_rewards_avged))
np.save(args.log_path + "/vf_discrepancy", np.asarray(vf_discrepancy))
saver.save(sess, args.log_path + "/final_model.ckpt")

print("Time for training to complete: " + str(datetime.timedelta(seconds=(time.time() - t_i))))

