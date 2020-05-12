import tensorflow as tf
import numpy as np
'''
Revision profile: (1)Implementing the GPU accelerating 
                  (2)Defining a batches generator
Author: Michael
Date: 2019-3-29-22-33
'''


class Autoencoder(object):

    def __init__(self, n_layers, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(), ae_para=[0, 0]):
        self.n_layers = n_layers
        self.transfer = transfer_function
        self.in_keep_prob = 1 - ae_para[0]

        network_weights = self._initialize_weights()
        self.weights = network_weights
        self.sparsity_level = np.repeat([0.0001], self.n_layers[1]).astype(np.float32)
        self.sparse_reg = ae_para[1]
        self.epsilon = 1e-06

        # model
        # with tf.device('/gpu:0'):
        self.x = tf.placeholder(tf.float32, [None, self.n_layers[0]])  # 输入网络的样本维度为 m x n,m为样本个数，n为特征个数
        self.keep_prob = tf.placeholder(tf.float32)

        self.hidden_encode = []
        h = tf.nn.dropout(self.x, self.keep_prob)
        for layer in range(len(self.n_layers)-1):
            h = self.transfer(
                tf.add(tf.matmul(h, self.weights['encode'][layer]['w']),
                       self.weights['encode'][layer]['b']))
            self.hidden_encode.append(h)  # 每个list是128行6列的矩阵，128为minbatch大小，6为特征

        self.hidden_recon = []
        for layer in range(len(self.n_layers)-1):
            h = tf.nn.tanh(
                tf.add(tf.matmul(h, self.weights['recon'][layer]['w']),
                       self.weights['recon'][layer]['b']))
            self.hidden_recon.append(h)
        self.reconstruction = self.hidden_recon[-1]

        # cost
        self.sparse_cost = 0
        if self.sparse_reg == 0:
            self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        else:
            self.sparse_cost = self.sparse_reg * self.kl_divergence(self.sparsity_level, self.hidden_encode[-1])
            self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))+self.sparse_cost

        self.optimizer = optimizer.minimize(self.cost)

    def _initialize_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        # Encoding network weights
        encoder_weights = []
        for layer in range(len(self.n_layers)-1):
            w = tf.Variable(
                initializer((self.n_layers[layer], self.n_layers[layer + 1]),
                            dtype=tf.float32))
            b = tf.Variable(
                tf.zeros([self.n_layers[layer + 1]], dtype=tf.float32))
            encoder_weights.append({'w': w, 'b': b})
        # Recon network weights
        recon_weights = []
        for layer in range(len(self.n_layers)-1, 0, -1):
            w = tf.Variable(
                initializer((self.n_layers[layer], self.n_layers[layer - 1]),
                            dtype=tf.float32))
            b = tf.Variable(
                tf.zeros([self.n_layers[layer - 1]], dtype=tf.float32))
            recon_weights.append({'w': w, 'b': b})
        all_weights['encode'] = encoder_weights
        all_weights['recon'] = recon_weights
        return all_weights

    def kl_divergence(self, p, p_hat):
        return tf.reduce_mean(p * tf.log(tf.clip_by_value(p, 1e-8, tf.reduce_max(p)))
                              - p * tf.log(tf.clip_by_value(p_hat, 1e-8, tf.reduce_max(p_hat)))
                              + (1 - p) * tf.log(tf.clip_by_value(1-p, 1e-8, tf.reduce_max(1-p)))
                              - (1 - p) * tf.log(tf.clip_by_value(1-p_hat, 1e-8, tf.reduce_max(1-p_hat))))

    def partial_fit(self):
        return self.cost, self.optimizer, self.hidden_encode[-1]  # self.sparse_cost

    def calc_total_cost(self):
        return self.cost

    def transform(self):
        return self.hidden_encode[-1]

    def reconstruct(self):
        return self.hidden_recon[-1]

    def setNewX(self, x):
        self.hidden_encode = []
        h = tf.nn.dropout(x, self.keep_prob)
        for layer in range(len(self.n_layers) - 1):
            h = self.transfer(
                tf.add(tf.matmul(h, self.weights['encode'][layer]['w']),
                       self.weights['encode'][layer]['b']))
            self.hidden_encode.append(h)

    '''
    该函数从之前的代码中修改得到，代码中的输入矩阵X为m行n列，其中n为特征个数，m为样本个数
    '''
    @staticmethod
    def random_mini_batches(X, mini_batch_size=64):
        m = X.shape[0]  # 矩阵列数为样本个数
        mini_batches = []
        # Step 1: Shuffle
        permutation = list(np.random.permutation(m))
        shuffled__x = X[permutation, :]

        # Step 2: Partition
        num_complete_minibatches = int(m / mini_batch_size)  # 如果不套int，单独靠np.floor是float64
        for k in range(0, num_complete_minibatches):  # range也符合起始位终止位的调用规则
            mini_batch__x = shuffled__x[(k * mini_batch_size):((k + 1) * mini_batch_size), :]
            mini_batches.append(mini_batch__x)

        if m % mini_batch_size != 0:
            mini_batch__x = shuffled__x[(num_complete_minibatches * mini_batch_size):(m + 1), :]
            mini_batches.append(mini_batch__x)
        return mini_batches

