from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np

# See: https://arxiv.org/pdf/1409.4842.pdf
class GoogLeNet:

    def __init__(self):
        self.num_labels = 14
        self.NAME = "GoogLeNet"

        #training params
        self.batch_size = 64
        self.learning_rate = 0.1
        self.weight_decay = 0.001

        self.graph = tf.get_default_graph()

        self.lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

        return None

    def construct_graph(self, x, y):
        with self.graph.as_default():
            model = self.conv(x, filters=64, kernel_size=7, stride=2, name='conv1_k7_s2')
            model = self.max_pool(model, pool_size=3, stride=2, name="maxpool1_p3_s2")
            model = tf.nn.local_response_normalization(input=model, alpha=0.0001, beta=0.75)
            model = self.conv(model, filters=64, kernel_size=1, stride=1, name='conv2_k1_s1')
            model = self.conv(model, filters=192, kernel_size=3, stride=1, name='conv2_k3_s1')
            model = tf.nn.local_response_normalization(input=model, alpha=0.0001, beta=0.75)
            model = self.max_pool(model, pool_size=3, stride=2, name='maxpool2_p3_s2')

            model = self._inception_module(model, filters=[64, 96, 128, 16, 32, 32],
                                           name='inception3a')

            model = self._inception_module(model, filters=[128, 128, 192, 32, 96, 64],
                                           name='inception3b')
            model = self.max_pool(model, pool_size=3, stride=2, name='maxpool3_p3_s2')
            model = self._inception_module(model, filters=[192, 96, 208, 16, 48, 64],
                                           name='inception4a')
            model = self._inception_module(model, filters=[160, 112, 224, 24, 64, 64],
                                           name='inception4b')
            model = self._inception_module(model, filters=[128, 128, 256, 24, 64, 64],
                                           name='inception4c')
            model = self._inception_module(model, filters=[112, 144, 288, 32, 64, 64],
                                           name='inception4d')
            model = self._inception_module(model, filters=[256, 160, 320, 32, 128, 128],
                                           name='inception4e')
            model = self.max_pool(model, pool_size=3, stride=2, name='maxpool4_p3_s2')
            model = self._inception_module(model, filters=[256, 160, 320, 32, 128, 128],
                                           name='inception5a')
            model = self._inception_module(model, filters=[384, 192, 384, 48, 128, 128],
                                           name='inception5b')
            model = self.avg_pool(model, pool_size=7, stride=1, name='avgpool5_p7_s1')

            logits = self.fully_connected(model)
            self.ys_pred = tf.nn.sigmoid(logits, name='prediction')

            with tf.name_scope('loss'):
                total_labels = tf.to_float(tf.multiply(self.batch_size, self.num_labels))
                num_positive_labels = tf.count_nonzero(y, dtype=tf.float32)
                num_negative_labels = tf.subtract(total_labels, num_positive_labels)
                Bp = tf.divide(total_labels, num_positive_labels)
                Bn = tf.divide(total_labels, num_negative_labels)
                # The loss function
                cross_entropy = -tf.reduce_sum((tf.multiply(Bp,  y * tf.log(self.ys_pred + 1e-9))) + 
                                               (tf.multiply(Bn, (1-y) * tf.log(1-self.ys_pred + 1e-9))),  
                                               name="cross_entropy")

                self.loss = cross_entropy # + l2 * self.weight_decay

            # Training the network with Adam using standard parameters.
            self.train_step = tf.train.AdamOptimizer(
                learning_rate=self.lr,
                beta1=0.9,
                beta2=0.999).minimize(self.loss)

    # Define some wrapper functions for brevity/readability
    def conv(self, inputs, filters, kernel_size, stride, name, padding='SAME',
             activation=tf.nn.relu):
        return tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=[kernel_size, kernel_size],
            strides=stride,
            padding=padding,
            activation=activation,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
            name=name)

    def max_pool(self, inputs, pool_size, stride, name):
        return tf.layers.max_pooling2d(
            inputs=inputs,
            pool_size=[pool_size, pool_size],
            strides=stride,
            padding='SAME',
            name=name)
        
    def avg_pool(self, inputs, pool_size, stride, name):
        return tf.layers.average_pooling2d(
            inputs=inputs,
            pool_size=[pool_size, pool_size],
            strides=stride,
            padding='VALID',
            name=name)
    
    def fully_connected(self, inputs):
        dropout = tf.layers.dropout(inputs, rate=1 - self.keep_prob, training=self.is_training)
        # Need to reshape dropout to 2D tensor for FC layer, multiply the dimensions excluding
        # batch size
        new_shape = int(np.prod(self._get_tensor_shape(dropout)[1:]))
        dropout = tf.reshape(dropout, [-1, new_shape])
        return tf.layers.dense(dropout, self.num_labels)

    def _get_tensor_shape(self, tensor):
        return tensor.get_shape().as_list()

    def _inception_module(self, inputs, filters, name):
        if len(filters) != 6:
            raise ValueError('Invalid filters input')
        # From left to right in the graph @ https://arxiv.org/pdf/1409.4842.pdf fig.3
        with tf.name_scope(name):
            conv1_k1_s1 = self.conv(inputs, filters=filters[0], kernel_size=1, stride=1,
                                    name=name + '_conv1_k1_s1')
            conv2_k1_s1 = self.conv(inputs, filters=filters[1], kernel_size=1, stride=1,
                                    name=name + '_conv2_k1_s1')
            conv3_k3_s1 = self.conv(conv2_k1_s1, filters=filters[2], kernel_size=3, stride=1,
                                    name=name + '_conv3_k3_s1')
            conv4_k1_s1 = self.conv(inputs, filters=filters[3], kernel_size=1, stride=1,
                                    name=name + '_conv4_k1_s1')
            conv5_k5_s1 = self.conv(conv4_k1_s1, filters=filters[4], kernel_size=5, stride=1,
                                    name=name + '_conv5_k5_s1')
            pool1_p3_s1 = self.max_pool(inputs, pool_size=3, stride=1, name=name + '_pool1_p3_s1')
            conv6_k1_s1 = self.conv(pool1_p3_s1, filters=filters[5], kernel_size=1, stride=1,
                                    name=name + '_conv6_k1_s1')
            tensor_list = [conv1_k1_s1, conv3_k3_s1, conv5_k5_s1, conv6_k1_s1]
            return tf.concat(tensor_list, axis=3, name=name + '_merge')
