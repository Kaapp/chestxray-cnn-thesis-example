from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from models.inception_resnet import inception_resnet_v2

slim = tf.contrib.slim

# See: https://arxiv.org/pdf/1409.4842.pdf
class Network:

    def __init__(self):
        self.num_labels = 14
        self.NAME = "InceptionResNet"

        #training params
        self.batch_size = 24
        self.learning_rate = 0.001
        self.weight_decay = 0.001

        self.graph = tf.get_default_graph()

        self.lr = tf.placeholder(tf.float32, shape=[], name='LR')
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

        return None
    
    def construct_base_graph(self, x):
        self.graph = tf.get_default_graph()

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            model, _ = inception_resnet_v2.inception_resnet_v2(x, scope='InceptionResnetV2',
                                                               create_aux_logits=False)
            return model

    def add_logits_layer(self, model, y):
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            with tf.variable_scope('InceptionResnetV2_Retrain', values=[model], reuse=False):
                with slim.arg_scope([slim.batch_norm, slim.dropout],
                                    is_training=self.is_training):
                    with tf.variable_scope('Logits'):
                        model = slim.dropout(model, self.keep_prob, scope='Dropout_0b')
                        logits = slim.fully_connected(model, self.num_labels, activation_fn=None)
                        logits = tf.squeeze(logits)
                        self.ys_pred = tf.nn.sigmoid(logits, name='predictions')
                    with tf.variable_scope('Loss'):
                        '''
                        total_labels = tf.to_float(tf.multiply(self.batch_size, self.num_labels))
                        num_positive_labels = tf.count_nonzero(y, dtype=tf.float32)
                        num_negative_labels = tf.subtract(total_labels, num_positive_labels)
                        Bp = tf.divide(total_labels, num_positive_labels)
                        Bn = tf.divide(total_labels, num_negative_labels)

                        cross_entropy = -tf.reduce_sum((tf.multiply(Bp,  y * tf.log(self.ys_pred + 1e-9))) + 
                                        (tf.multiply(Bn, (1-y) * tf.log(1-self.ys_pred + 1e-9))), 
                                        name="cross_entropy")
                        '''
                        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
                        #tf.reduce_sum((-y * tf.log(self.ys_pred)) - ((1-y) * tf.log(1-self.ys_pred)))
                        self.loss = tf.reduce_sum(cross_entropy)
                    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "InceptionResnetV2_Retrain")
                    with tf.variable_scope('Train_step'):
                        self.train_step = tf.train.AdamOptimizer(
                            learning_rate=self.lr, 
                            beta1=0.9,
                            beta2=0.999).minimize(self.loss)
                    vars_to_init = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "InceptionResnetV2_Retrain")
                    return vars_to_init
