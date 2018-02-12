"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools

# np.random.seed(1)
# tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions=4,
            n_features=2,
            learning_rate=0.01, 
            reward_decay=0.9,
            e_greedy=0.8,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            len_max = 1000,
            e_greedy_increment=None,
            output_graph=False,
            dueling = True
    ):

        self.no_fea = n_features
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.dueling = dueling
        self.len_max =len_max
        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        # n_features =2, devotes start and end. add 1, is the length.
        self.memory = np.zeros((self.memory_size, (n_features+1) * 2 + 2 + self.len_max)) # 128 is the length of indices

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []  # cost history

    # Two nets have the same structure but different parameters.
    # One with parameters eval_net_params, another with target_net_params.
    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            if self.dueling:
                # Dueling DQN
                with tf.variable_scope('Value'):
                    w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, 1], initializer=b_initializer, collections=c_names)
                    self.V = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Advantage'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.A = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Q'):
                    out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))  # Q = V(s) + A(s,a)
            else:
                with tf.variable_scope('Q'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    out = tf.matmul(l1, w2) + b2

            return out
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.no_fea], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        # self.inputarray = tf.Variable(tf.constant(0, shape=[None, self.no_fea]), dtype=tf.float32)
        # self.inputarray = self.make_input_row(self.s)
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 32, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers, 0 is mean, 0.3 is stddev

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.no_fea, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            # loss = [(q_target-q_eval)^2]/n where n = len(q_target)
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)


        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input, s_ denotes s' or s_{t+1}
        # self.inputarray_ = self.make_input_row(self.s_)

        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.no_fea, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, indices, s_,):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # change the tuple s to narray s
        s = np.asarray([x for xs in s for x in xs])
        s_ = np.array([x for xs in s_ for x in xs])
        # print 's, [a, r], s_', s, [a, r], s_
        transition = np.hstack((s, [a, r], indices, s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1
    def make_input_row(self, state):
        input_array = np.zeros(self.no_fea)
        print 'sssss',state
        state = state[0]
        print state
        # state[0][0] is the start point in the state, state[0][-1] is the end point, make the attention bar as 1.
        input_array[state[0]:state[-1]+1] = 1
        input_array = input_array[np.newaxis, :]
        return input_array


    def choose_action(self, state):
        # to have batch dimension when feed into tf placeholder
        short_state = np.zeros(1, dtype=[('start', np.float32), ('end', np.float32)])
        short_state['start'] = state['start']
        short_state['end'] = state['end']

        short_state = np.asarray([x for xs in short_state for x in xs])
        short_state = short_state[np.newaxis, :]
        # print state, state.shape,

        # make the intial input_array
        # input_array = self.make_input_row(state)


        if np.random.uniform() < self.epsilon:
            # forward feed the state and get q value for every actions

            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: short_state})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)  # q_next is q_target

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon. if epsilon_increment != None, the epsilon will gradually increase, the increment = epsilon_increment
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):

        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



