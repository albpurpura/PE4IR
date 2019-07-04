"""
    Author: Alberto Purpura
    Copyright: (C) 2019-2020 <http://www.dei.unipd.it/ 
    Department of Information Engineering> (DEI), <http://www.unipd.it/ University of Padua>, Italy
    License: <http://www.apache.org/licenses/LICENSE-2.0 Apache License, Version 2.0>
"""

import tensorflow as tf
import numpy as np

"""
This is the model file to use probabilistic WE.
"""


class Model:

    def __init__(self, config):
        self.config = config

        self.msize1 = config['data1_maxlen']
        self.msize2 = config['data2_maxlen']
        self.psize1 = config['data1_psize']
        self.psize2 = config['data2_psize']

        self.training = tf.placeholder(tf.bool, name='training')
        # self.X1 = tf.placeholder(tf.int32, name='X1',
        #                          shape=(None, self.msize1))
        # self.X2 = tf.placeholder(tf.int32, name='X2',
        #                          shape=(None, self.msize2))
        self.X1_len = tf.placeholder(tf.int32, name='X1_len', shape=(None,))
        self.X2_len = tf.placeholder(tf.int32, name='X2_len', shape=(None,))
        self.Y = tf.placeholder(tf.int32, name='Y', shape=(None,))

        self.dpool_index = tf.placeholder(tf.int32, name='dpool_index',
                                          shape=(None, self.msize1, self.msize2, 3))

        self.batch_size = tf.shape(self.Y)[0]

        # self.embedding = tf.get_variable('embedding',
        #                                  initializer=config['embedding'],
        #                                  dtype=tf.float32, trainable=False)
        #
        # self.embed1 = tf.nn.embedding_lookup(self.embedding, self.X1)
        # self.embed2 = tf.nn.embedding_lookup(self.embedding, self.X2)
        #
        # tf.add_to_collection('explain_input', self.embed1)
        # tf.add_to_collection('explain_input', self.embed2)
        #
        # self.cross = tf.einsum('abd,acd->abc', self.embed1, self.embed2)
        self.cross = tf.placeholder(tf.float32, name='sim_matrix', shape=(None, self.msize1, self.msize2))
        self.cross_img = tf.expand_dims(self.cross, 3)

        self.conv1 = tf.layers.conv2d(self.cross_img, filters=8, kernel_size=[3, 3], padding='SAME',
                                      activation=tf.nn.relu, kernel_initializer=tf.glorot_normal_initializer(seed=0))

        # dynamic pooling
        self.conv1_expand = tf.gather_nd(self.conv1, self.dpool_index)
        stride1 = self.msize1 / self.psize1
        stride2 = self.msize2 / self.psize2

        suggestion1 = self.msize1 / stride1
        suggestion2 = self.msize2 / stride2

        if suggestion1 != self.psize1 or suggestion2 != self.psize2:
            print("DynamicMaxPooling Layer can not "
                  "generate ({} x {}) output feature map,"
                  "please use ({} x {} instead.)"
                  .format(self.psize1, self.psize2,
                          suggestion1, suggestion2))
            exit()

        self.pool1 = tf.nn.max_pool(self.conv1_expand,
                                    [1, stride1, stride2, 1],
                                    [1, stride1, stride2, 1], 'VALID')

        with tf.variable_scope('fc1'):
            # self.fc1 = tf.nn.relu(tf.contrib.layers.linear(
            #     tf.reshape(self.pool1, [-1, config['data1_psize'] * config['data2_psize'] * 8]), 20))
            pool1_w_do = tf.layers.dropout(self.pool1, rate=0.5, seed=0, training=self.training)
            self.fc1 = tf.layers.dropout(tf.nn.relu(tf.contrib.layers.linear(
                tf.reshape(pool1_w_do, [-1, self.pool1.shape[1] * self.pool1.shape[2] * 8]), 128)), seed=0, rate=0.5,
                training=self.training)

        self.pred = tf.contrib.layers.linear(self.fc1, 1)
        tf.add_to_collection('explain_output', self.pred)

        pos = tf.strided_slice(self.pred, [0], [self.batch_size], [2])
        neg = tf.strided_slice(self.pred, [1], [self.batch_size], [2])

        self.loss = tf.reduce_mean(tf.maximum(1.0 + neg - pos, 0.0))

        self.train_model = tf.train.AdamOptimizer(learning_rate=5e-3).minimize(self.loss)

        self.saver = tf.train.Saver(max_to_keep=20)

    @staticmethod
    def dynamic_pooling_index(len1, len2, max_len1, max_len2, compress_ratio1=1, compress_ratio2=1):
        def dpool_index_(batch_idx, len1_one, len2_one, max_len1, max_len2):
            '''
            TODO: Here is the check of sentences length to be positive.
            To make sure that the lenght of the input sentences are positive.
            if len1_one == 0:
                print("[Error:DynamicPooling] len1 = 0 at batch_idx = {}".format(batch_idx))
                exit()
            if len2_one == 0:
                print("[Error:DynamicPooling] len2 = 0 at batch_idx = {}".format(batch_idx))
                exit()
            '''
            if len1_one == 0:
                stride1 = max_len1
            else:
                stride1 = 1.0 * max_len1 / len1_one

            if len2_one == 0:
                stride2 = max_len2
            else:
                stride2 = 1.0 * max_len2 / len2_one

            idx1_one = [int(i / stride1) for i in range(max_len1)]
            idx2_one = [int(i / stride2) for i in range(max_len2)]
            mesh1, mesh2 = np.meshgrid(idx1_one, idx2_one)
            index_one = np.transpose(np.stack([np.ones(mesh1.shape) * batch_idx,
                                               mesh1, mesh2]), (2, 1, 0))
            return index_one

        index = []
        dpool_bias1 = dpool_bias2 = 0
        if max_len1 % compress_ratio1 != 0:
            dpool_bias1 = 1
        if max_len2 % compress_ratio2 != 0:
            dpool_bias2 = 1
        cur_max_len1 = max_len1 // compress_ratio1 + dpool_bias1
        cur_max_len2 = max_len2 // compress_ratio2 + dpool_bias2
        for i in range(len(len1)):
            index.append(dpool_index_(i, len1[i] // compress_ratio1,
                                      len2[i] // compress_ratio2, cur_max_len1, cur_max_len2))
        return np.array(index)

    @staticmethod
    def init_step(sess):
        sess.run(tf.global_variables_initializer())

    def train_step(self, sess, feed_dict):
        feed_dict[self.dpool_index] = self.dynamic_pooling_index(feed_dict[self.X1_len], feed_dict[self.X2_len],
                                                                 self.config['data1_maxlen'],
                                                                 self.config['data2_maxlen'])
        _, loss = sess.run([self.train_model, self.loss], feed_dict=feed_dict)
        return loss

    def test_step(self, sess, feed_dict):
        feed_dict[self.dpool_index] = self.dynamic_pooling_index(feed_dict[self.X1_len], feed_dict[self.X2_len],
                                                                 self.config['data1_maxlen'],
                                                                 self.config['data2_maxlen'])
        pred = sess.run(self.pred, feed_dict=feed_dict)
        return pred

    def eval_step(self, sess, node, feed_dict):
        feed_dict[self.dpool_index] = self.dynamic_pooling_index(feed_dict[self.X1_len], feed_dict[self.X2_len],
                                                                 self.config['data1_maxlen'],
                                                                 self.config['data2_maxlen'])
        node_value = sess.run(node, feed_dict=feed_dict)
        return node_value
