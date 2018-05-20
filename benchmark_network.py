"""
Adapted from SciKitLearn example @ http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

=======================================
Receiver Operating Characteristic (ROC)
=======================================

Example of Receiver Operating Characteristic (ROC) metric to evaluate
classifier output quality.

ROC curves typically feature true positive rate on the Y axis, and false
positive rate on the X axis. This means that the top left corner of the plot is
the "ideal" point - a false positive rate of zero, and a true positive rate of
one. This is not very realistic, but it does mean that a larger area under the
curve (AUC) is usually better.

The "steepness" of ROC curves is also important, since it is ideal to maximize
the true positive rate while minimizing the false positive rate.

Multiclass settings
-------------------

ROC curves are typically used in binary classification to study the output of
a classifier. In order to extend ROC curve and ROC area to multi-class
or multi-label classification, it is necessary to binarize the output. One ROC
curve can be drawn per label, but one can also draw a ROC curve by considering
each element of the label indicator matrix as a binary prediction
(micro-averaging).

Another evaluation measure for multi-class classification is
macro-averaging, which gives equal weight to the classification of each
label.

.. note::

    See also :func:`sklearn.metrics.roc_auc_score`,
             :ref:`sphx_glr_auto_examples_model_selection_plot_roc_crossval.py`.

    """
#print(__doc__)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('TkAgg')


import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

#import tensorflow as tf
import numpy as np
import GoogLeNet as Model
import DataHandler
import tensorflow as tf
import os.path
import re
import random
from operator import itemgetter
from sklearn.metrics import roc_auc_score

from tensorflow.python import debug as tf_debug
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


network_types = [('ce_all', 'CE + All'), ('ce_last_only', 'CE + Last'), ('wce_all', 'WCE + All'), ('wce_last_only', 'WCE + Last')]



def get_num_trainable_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    return total_parameters

# Initialise network values
network = Model.Network()

# Get our list of files and their labels, and create our placeholders to feed
data = DataHandler.DataHandler()
features_placeholder = tf.placeholder(tf.string, shape=[None])
labels_placeholder = tf.placeholder(tf.float32, shape=[None, len(data.GROUND_TRUTHS)])
# Create a dataset from our placeholders
dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# Map the filenames to the actual image data
dataset = dataset.map(data.image_parse_function)
# Split the dataset into batches depending on the network's specified batch size.
dataset = dataset.batch(network.batch_size)

# Create an iterator for our datasets
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, dataset.output_types, dataset.output_shapes)

test_features, test_labels = data.get_dataset('testing')
test_iterator = dataset.make_initializable_iterator()


# Get our final image data and label from the iterator, pass it to the network and let
# the network build it's graph, followed by the summary ops
(x, y) = iterator.get_next()
model = network.construct_base_graph(x)

# Start a session
with tf.Session(graph=network.graph) as sess:
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # Create a saver so we can save/load model checkpoints after epochs
    saver = tf.train.Saver()

    # Add the logits layer and restore all weights from checkpoint
    vars_to_initialize = network.add_logits_layer(model, y)

    for network_type, network_type_display_name in network_types:
        # Look for existing ckpt file else initialise!
        available_ckpts = [int(re.match(r"(?:[a-zA-Z]*_)([0-9]*)(?:\.ckpt\.txt)", f).group(1)) 
                        for f in os.listdir('./checkpoints_archive/{0}_{1}/'.format(network.NAME, network_type)) 
                        if f.endswith('.ckpt.txt')]
        if len(available_ckpts) > 0:
            # Sort the list of checkpoint numbers in descending order so first entry is latest
            available_ckpts.sort(reverse=True)
            print('Restoring from epoch {0}'.format(available_ckpts[0]))
            

            # Print the current models number of training params
            print("Total training params: %.1fM" % (get_num_trainable_params() / 1e6))

            # Overwrite the values with the checkpoint values
            saver = tf.train.Saver()
            saver.restore(sess, './checkpoints_archive/{0}_{2}/{0}_{1}.ckpt'.format(network.NAME, available_ckpts[0], network_type))

            # Get the iterator handles to feed for train/val/test
            test_handle = sess.run(test_iterator.string_handle())
                
            sess.run(test_iterator.initializer, feed_dict={features_placeholder: test_features, labels_placeholder: test_labels})
            # Begin benchmark
            output_data = {}
            for pathology in data.GROUND_TRUTHS:
                output_data[pathology] = []

            while True:
                try:
                    # Fill each list with predictions and truths for that pathology
                    predictions, truths = sess.run([network.ys_pred, y],
                                                feed_dict={
                                                    handle: test_handle,
                                                    network.lr: network.learning_rate,
                                                    network.is_training: False,
                                                    network.keep_prob: 1.0
                                                    #,network.current_step: batches_completed
                                                })
                    
                    for index, prediction in enumerate(predictions):
                        for pathology_index, pathology_name in enumerate(data.GROUND_TRUTHS):
                            output_data[pathology_name].append(
                                (predictions[index][pathology_index], truths[index][pathology_index])
                            )
                except tf.errors.OutOfRangeError:
                    break
            
            mean_auroc = 0
            for pathology in output_data:
                # Sort list by prediction descending
                output_data[pathology].sort(key=itemgetter(0), reverse=True)
                preds, truths = zip(*output_data[pathology])
                auroc = roc_auc_score(truths, preds)
                mean_auroc += auroc
                print('{0}: {1}'.format(pathology, auroc))




            # Compute ROC curve and ROC area for each class
            n_classes = 14
            fpr = dict()
            tpr = dict()
            thresholds = dict()
            roc_auc = dict()
            for pathology in output_data:
                preds, truths = zip(*output_data[pathology])
                preds = np.asarray(preds)
                truths = np.asarray(truths)
                fpr[pathology], tpr[pathology], thresholds[pathology] = roc_curve(truths, preds)
                roc_auc[pathology] = auc(fpr[pathology], tpr[pathology])


            # Plot all ROC curves
            plt.figure()


            lines = [('blue', 'solid'), ('black', 'solid'), ('red', 'solid'), ('green', 'solid'), ('purple', 'solid'),
                        ('blue', 'dashed'), ('black', 'dashed'), ('red', 'dashed'), ('green', 'dashed'), ('purple', 'dashed'),
                        ('blue', 'dotted'), ('black', 'dotted'), ('red', 'dotted'), ('green', 'dotted'), ('purple', 'dotted')]
            for i, (color, linestyle) in zip(fpr, lines):
                plt.plot(fpr[i], tpr[i], color=color, lw=2, ls=linestyle, label=i.replace('_', ' '))

            plt.plot([0, 1], [0, 1], color='gray', ls='dotted', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('{0} ({1})'.format(network.NAME, network_type_display_name))
            plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
            plt.savefig('../dissertation/graphs/{0}_{1}.pdf'.format(network.NAME, network_type), bbox_inches="tight")
            plt.savefig('../dissertation/graphs/{0}_{1}.png'.format(network.NAME, network_type), bbox_inches="tight")
            plt.close()




    else:
        print('No Checkpoint found, initialising global variables to defaults.')
        # Initialise our global vars (W and b)
        exit(1)