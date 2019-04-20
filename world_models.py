import tensorflow as tf
import yaml
import numpy as np
import gym
import roboschool
import matplotlib.pyplot as plt
import pickle
import copy
import sys
import bandit_env
import tensorflow.contrib.slim as slim
from datetime import datetime
import os

# Wrapper reference : https://stackoverflow.com/questions/47745027/tensorflow-how-to-obtain-intermediate-cell-states-c-from-lstmcell-using-dynam
class Wrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(self, inner_cell):
        super(Wrapper, self).__init__()
        self._inner_cell = inner_cell
    @property
    def state_size(self):
        return self._inner_cell.state_size
    @property
    def output_size(self):
        return (self._inner_cell.state_size, self._inner_cell.output_size)
    def call(self, input, *args, **kwargs):
        output, next_state = self._inner_cell(input, *args, **kwargs)
        emit_output = (next_state, output)
        return emit_output, next_state

class agent():
    def __init__(self, state_size, action_size, reward_size, time_size):
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.time_size = time_size
        self.input_size = self.action_size + self.reward_size
        self.env_model_neurons = 50
        self.state_encoder_neurons = 50
        self.env_model_layers = 1

        self.architecture()

    def single_cell(self, shape):
        return Wrapper(tf.nn.rnn_cell.LSTMCell(shape, dtype=tf.float32, forget_bias=1.0, initializer=tf.contrib.layers.xavier_initializer(seed=2351), name='basic_lstm_cell'))
        # return tf.nn.rnn_cell.LSTMCell(shape, dtype=tf.float32, forget_bias=1.0, initializer=tf.contrib.layers.xavier_initializer(seed=2349), name='basic_lstm_cell')
        # return tf.nn.rnn_cell.LSTMCell(shape, dtype=tf.float32, forget_bias=1.0, initializer=tf.contrib.layers.xavier_initializer(seed=2349), name='basic_lstm_cell', state_is_tuple=True)

    def env_model(self, encoded_state, action, reuse_weights=False):
        with tf.variable_scope("env_model", reuse=reuse_weights):
            env_model_input = tf.concat([encoded_state, action], axis=2)
            lstm_layers = tf.nn.rnn_cell.MultiRNNCell([self.single_cell(self.env_model_neurons) for _ in range(self.env_model_layers)])
            batch_size = tf.shape(env_model_input)[0]
            self.init_state = lstm_layers.zero_state(batch_size=batch_size, dtype=tf.float32)
            lstm_out, self.env_model_rnn_hidden_state = tf.nn.dynamic_rnn(cell=lstm_layers, inputs=env_model_input, initial_state=self.init_state)
            lstm_output = lstm_out[1]
            self.env_model_cell_state = lstm_out[0].c
            next_encoded_state_1 = tf.contrib.layers.fully_connected(lstm_output, self.state_encoder_neurons, activation_fn=tf.nn.tanh,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(seed=2352),
                                              biases_initializer=None, trainable=True)
            next_encoded_state = tf.reshape(next_encoded_state_1, [tf.shape(encoded_state)[0], tf.shape(encoded_state)[1], self.state_encoder_neurons])
        return next_encoded_state

    def state_encoder(self, state, reuse_weights = False):
        with tf.variable_scope("state_encoder", reuse=reuse_weights):
            state_input_flat = tf.reshape(state, [-1, self.state_size])
            encoded_state_flat_1 = tf.contrib.layers.fully_connected(state_input_flat, self.state_encoder_neurons, activation_fn=tf.nn.tanh,
                                                                     weights_initializer=tf.contrib.layers.xavier_initializer(seed=2349),
                                                                     biases_initializer=None, trainable=True)
            encoded_state_flat_2 = tf.contrib.layers.fully_connected(encoded_state_flat_1, self.state_encoder_neurons, activation_fn=tf.nn.tanh,
                                                                     weights_initializer=tf.contrib.layers.xavier_initializer(seed=2350),
                                                                     biases_initializer=None, trainable=True)
            encoded_state = tf.reshape(encoded_state_flat_2, [tf.shape(state)[0], tf.shape(state)[1], self.state_encoder_neurons])
        return encoded_state

    def architecture(self):
        self.state_input = tf.placeholder(shape=[None, None, self.state_size], dtype=tf.float32)
        self.prev_state_input = tf.placeholder(shape=[None, None, self.state_size], dtype=tf.float32)
        self.target_v = tf.placeholder(shape=[None, None, 1], dtype=tf.float32)
        self.advantage = tf.placeholder(shape=[None, None, 1], dtype=tf.float32)
        self.prev_action_taken = tf.placeholder(shape=[None, None, self.action_size], dtype=tf.float32)
        self.action_taken = tf.placeholder(shape=[None, None, self.action_size], dtype=tf.float32)
        self.lr = tf.placeholder(tf.float32)

        encoded_state = self.state_encoder(self.state_input)
        prev_encoded_state = self.state_encoder(self.prev_state_input, reuse_weights=True)
        env_model_predicted_state = self.env_model(prev_encoded_state, self.prev_action_taken)
        controller_input = tf.concat([encoded_state, tf.stop_gradient(self.env_model_cell_state)], axis=2)
        controller_input_flat = tf.reshape(controller_input, [-1, self.state_encoder_neurons+self.state_encoder_neurons])
        with tf.variable_scope("Controller"):
            self.value_fn = tf.contrib.layers.fully_connected(controller_input_flat, 1, activation_fn=None,
                                                         weights_initializer = tf.contrib.layers.xavier_initializer(seed=2345),
                                                         biases_initializer= tf.contrib.layers.xavier_initializer(seed=2346), trainable=True)
            self.policy = tf.nn.softmax(tf.contrib.layers.fully_connected(controller_input_flat, self.action_size, activation_fn=None,
                                                                     weights_initializer = tf.contrib.layers.xavier_initializer(seed=2347),
                                                                     biases_initializer=tf.contrib.layers.xavier_initializer(seed=2348), trainable=True))

        self.target_v_flat = tf.reshape(self.target_v, [-1, 1],)
        self.advantage_flat = tf.reshape(self.advantage, [-1,1])
        self.action_taken_flat = tf.reshape(self.action_taken, [-1, self.action_size])
        value_loss = 0.5*tf.reduce_mean(tf.square(self.target_v_flat - self.value_fn))
        policy_loss = -tf.reduce_mean(tf.log(tf.reduce_sum(self.policy*self.action_taken_flat)+1e-9)*self.advantage_flat)
        entropy = -tf.reduce_mean(self.policy*tf.log(self.policy+1e-9))

        env_model_loss = 0.5*tf.reduce_mean(tf.square(env_model_predicted_state[:, 1:, :] - self.state_encoder(self.state_input[:, 1:, :], reuse_weights=True)))
        # all_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # env_model_vars = tf.trainable_variables(scope="env_model")
        # excluding_env_model_vars = [var for var in all_trainable_vars if var not in env_model_vars]
        controller_loss = policy_loss + value_loss - 0.05*entropy
        # loss = policy_loss + value_loss - 0.05*entropy + env_model_loss
        # self.loss_summary = tf.summary.scalar('train_controller_loss', controller_loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='adam_opt')
        self.train_op_2 = optimizer.minimize(env_model_loss)
        self.train_op_1 = optimizer.minimize(controller_loss)

def to_one_hot(val, vector_len):
    one_hot = np.zeros(vector_len)
    one_hot[val] = 1.0
    return one_hot

def train(agent):
    num_tasks = 25000
    num_episodes_per_task = 1
    max_len_episode = 200
    discount_factor = 0.9
    learning_rate = 0.0005

    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=None)
    init_vars = tf.global_variables_initializer()
    sess.run(init_vars)
    results_dir = './results_world_models_{0}'.format(datetime.now().strftime('%d_%m_%H_%M'))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    writer_op = tf.summary.FileWriter(results_dir + '/tf_graphs', sess.graph)

    rewards_plot = []
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    reward_size = 1
    num_actions = env.action_space.n

    for task in range(num_tasks):
        env_model_hidden_state_val = None
        task_states, task_time_steps, task_rewards, task_actions, task_state_values, task_advantages, task_target_v = [], [], [], [], [], [], []
        for episode in range(num_episodes_per_task):
            env.reset()
            time_step, done = 1, False
            state = env.state
            prev_state = np.zeros_like(state)
            reward, action, episode_reward = 0.0, 0, 0.0
            while(time_step <= max_len_episode and done == False):
                state_input = np.array([[state]])
                feed = {agent.state_input:state_input, agent.prev_state_input:np.array([[prev_state]]), agent.prev_action_taken: np.array([[to_one_hot(action, num_actions)]])}
                if(env_model_hidden_state_val != None):
                    feed[agent.init_state] = env_model_hidden_state_val
                policy, state_v_val, env_model_hidden_state_val = sess.run([agent.policy, agent.value_fn, agent.env_model_rnn_hidden_state], feed_dict=feed)
                policy = policy[0]
                state_v_val = state_v_val[0][0]
                task_state_values.append(state_v_val)
                action = np.random.choice(num_actions, p=policy)
                task_states.append(state)
                task_actions.append(to_one_hot(action, num_actions))
                prev_state = state
                state, reward, done, info = env.step(action)
                task_rewards.append(reward)
                episode_reward+=reward
                time_step +=1
            rewards_plot.append(np.sum(episode_reward))
        R = 0.0
        if(done == True):
            R = 0.0
        else:
            state_input = np.array([[state]])
            feed = {agent.state_input:state_input, agent.prev_state_input:np.array([[prev_state]]), agent.prev_action_taken: np.array([[to_one_hot(action, num_actions)]]), agent.init_state: env_model_hidden_state_val}
            state_v_val = sess.run([agent.value_fn], feed_dict=feed)
            state_v_val = state_v_val[0][0]
            R = state_v_val
        # Generalized Advantage Estimate ::: A = Sum((gamma^t)*delta), delta = r + gamma*V(s(t+1)) - V(s(t))
        task_general_adv = np.array(task_rewards) + discount_factor*np.array((task_state_values[1:]+[0.0])) - np.array(task_state_values)
        adv = 0.0
        for i in reversed(range(len(task_rewards))):
            R = task_rewards[i] + discount_factor*R
            adv = task_general_adv[i] + discount_factor*adv
            task_target_v.append(R)
            task_advantages.append(adv)
        task_target_v = np.flip(task_target_v)
        task_advantages = np.flip(task_advantages)

        state_input = np.array([task_states]).reshape((1, -1, state_size))
        prev_state_input = np.concatenate((np.zeros((1,1,state_size)), np.array([task_states]).reshape((1, -1, state_size))[:, :-1, :]), axis=1)
        task_advantages = np.hstack(task_advantages).reshape((1, -1, 1))
        task_target_v = np.hstack(task_target_v).reshape((1, -1, 1))
        task_actions = np.vstack(task_actions).reshape((1, -1, num_actions))
        task_prev_actions = np.concatenate((np.array([[[1, 0]]]), np.vstack(task_actions).reshape((1, -1, num_actions))[:, :-1, :]), axis=1)
        feed = {agent.state_input:state_input, agent.prev_state_input:prev_state_input, agent.action_taken: task_actions, agent.prev_action_taken:task_prev_actions, agent.advantage: task_advantages, agent.target_v: task_target_v, agent.lr: learning_rate}
        # _, _, loss_summary_val = sess.run([agent.train_op_1, agent.train_op_2, agent.loss_summary], feed_dict=feed)
        _, _ = sess.run([agent.train_op_2, agent.train_op_1], feed_dict=feed)
        # writer_op.add_summary(loss_summary_val, task)

        if ((task + 1) % 1000 == 0):
            print(task)
            cumsum_rewards = np.cumsum(np.insert(rewards_plot, 0, 0))
            smoothed_rewards = (cumsum_rewards[50:] - cumsum_rewards[:-50])/50.0
            fp = open(results_dir+'/rewards_{0}.pickle'.format(task+1), 'wb')
            pickle.dump(smoothed_rewards, fp)
            fp.close()
            plt.plot(smoothed_rewards)
            plt.savefig(results_dir+'/graph_task_{0}.png'.format(task+1))
            saver.save(sess, results_dir+'/models/itr_{0}.ckpt'.format(task+1))

    cumsum_rewards = np.cumsum(np.insert(rewards_plot, 0, 0))
    smoothed_rewards = (cumsum_rewards[50:] - cumsum_rewards[:-50]) / 50.0
    fp = open(results_dir+'/final_rewards.pickle', 'wb')
    pickle.dump(smoothed_rewards, fp)
    fp.close()
    saver.save(sess, results_dir+'/models/final.ckpt')
    sess.close()
    plt.plot(smoothed_rewards)
    plt.show()
    return

def main():
    # env = gym.make('correlatedbandit-v0')
    # env = gym.make('Ant-v2')
    # env = gym.make('RoboschoolAnt-v1')
    # env = gym.make('Humanoid-v2')
    env = gym.make('CartPole-v0')
    env.reset()
    num_actions = env.action_space.n
    state_size = env.observation_space.shape[0]
    reward_size = 1
    time_size = 1

    tf.reset_default_graph()
    meta_rl_agent = agent(state_size, num_actions, reward_size, time_size)
    train(meta_rl_agent)
    tf.reset_default_graph()
    # meta_rl_agent = agent(state_size, num_actions, reward_size, time_size)
    # test(meta_rl_agent)

    return

if __name__=='__main__':
    main()