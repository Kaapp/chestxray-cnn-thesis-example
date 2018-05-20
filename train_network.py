from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import tensorflow as tf
import numpy as np
import InceptionResNet as Model
import DataHandler
import tensorflow as tf
import os.path
import re
import random
from operator import itemgetter
from sklearn.metrics import roc_auc_score

from preprocessing import inception_preprocessing as augment

from tensorflow.python import debug as tf_debug
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

NUM_EPOCHS = 30
VALIDATION_SET_SIZE = 10000

def augment_images(dataset):
    def random_flipping_wrapper(image, label):
        return data.random_flipping(image, label, network.is_training)

    augmented = dataset.map(random_flipping_wrapper)

    return dataset.concatenate(augmented)

def get_num_trainable_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    return total_parameters

def add_summary_ops(ground_truth):
    # We will round our networks predictions such that >50% presence is a positive, <=50% presence is negative
    thresholds = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    # For inference, we will display the actual percentages
    p, _ = tf.metrics.precision_at_thresholds(labels=ground_truth, predictions=network.ys_pred, thresholds=thresholds)
    r, _ = tf.metrics.recall_at_thresholds(labels=ground_truth, predictions=network.ys_pred, thresholds=thresholds)
    # Using F1 because false negative and false positive are equally bad in medicine
    precision = tf.reduce_mean(p)
    recall = tf.reduce_mean(r)
    f1 = 2 * precision * recall / (precision + recall)

    with tf.name_scope("summaries"):
        tf.summary.scalar("loss", network.loss)
        # Plotting learning rate forces us to feed learning rate even when we don't train.
        tf.summary.scalar("learning_rate", network.lr)
        tf.summary.scalar("precision", precision)
        tf.summary.scalar("recall", recall)
        tf.summary.scalar("f1_score", f1)       
        network.summary_op = tf.summary.merge_all()
    
    return p, r, f1

# Initialise network values
network = Model.Network()

# Get our list of files and their labels, and create our placeholders to feed
data = DataHandler.DataHandler()
train_features, train_labels = data.get_dataset('training')
val_features, val_labels = data.get_dataset('validation')
VALIDATION_SET_SIZE = len(val_features)
features_placeholder = tf.placeholder(tf.string, shape=[None])
labels_placeholder = tf.placeholder(tf.float32, shape=[None, len(data.GROUND_TRUTHS)])
# Create a dataset from our placeholders
dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# Map the filenames to the actual image data
dataset = dataset.map(data.image_parse_function)

# If we're training, perform image augmentations and concat datasets
dataset = augment_images(dataset)

# Set image bounds to [-1, 1] from [0, 1]
dataset = dataset.map(data.finalise_images)

# Split the dataset into batches depending on the network's specified batch size.
dataset = dataset.batch(network.batch_size)

# Create an iterator for our datasets
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, dataset.output_types, dataset.output_shapes)
train_iterator = dataset.make_initializable_iterator()
val_iterator = dataset.make_initializable_iterator()

test_features, test_labels = data.get_dataset('testing')
test_iterator = dataset.make_initializable_iterator()

# Get our final image data and label from the iterator, pass it to the network and let
# the network build it's graph, followed by the summary ops
(x, y) = iterator.get_next()
model = network.construct_base_graph(x)

# Create our summary file writer so we can track our progress on TensorBoard
train_writer = tf.summary.FileWriter('./train_logs/' + network.NAME + '/train', network.graph)
val_writer = tf.summary.FileWriter('./train_logs/' + network.NAME + '/val', network.graph)

# Start a session
with tf.Session(graph=network.graph) as sess:
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # Create a saver so we can save/load model checkpoints after epochs
    saver = tf.train.Saver()
    batches_completed = 0
    epochs_completed = 0

    # Look for existing ckpt file else initialise!
    available_ckpts = [int(re.match(r"(?:[a-zA-Z]*_)([0-9]*)(?:\.ckpt\.txt)", f).group(1)) 
                       for f in os.listdir('./checkpoints/' + network.NAME + '/') 
                       if f.endswith('.ckpt.txt')]
    if len(available_ckpts) > 0:
        # Sort the list of checkpoint numbers in descending order so first entry is latest
        available_ckpts.sort(reverse=True)
        print('Restoring from epoch {0}'.format(available_ckpts[0]))

        # If we're loading the base pretrained model
        if int(available_ckpts[0]) == 0:
            # Load the pretrained base
            saver.restore(sess, './checkpoints/{0}/{0}_{1}.ckpt'.format(network.NAME, available_ckpts[0]))
            # Add the new logits layer and default initialise
            vars_to_initialize = network.add_logits_layer(model, y)
            vars_to_initialize = tf.variables_initializer(vars_to_initialize)
            sess.run(vars_to_initialize)
            # Need to create a new saver with the added logits layer, replace the old one.
            saver = tf.train.Saver()
        else:
            # Add the logits layer and restore all weights from checkpoint
            vars_to_initialize = network.add_logits_layer(model, y)
            # Init the new vars
            for var in vars_to_initialize:
                sess.run(var.initializer)
            # Need to create a new saver with the added logits layer, replace the old one.
            saver = tf.train.Saver()
            # Overwrite the values with the checkpoint values            
            saver.restore(sess, './checkpoints/{0}/{0}_{1}.ckpt'.format(network.NAME, available_ckpts[0]))

        # load epoch and batch values from old model
        with open('./checkpoints/{0}/{0}_{1}.ckpt.txt'.format(network.NAME, available_ckpts[0])) as info_file:
            values = info_file.read().splitlines()
            if len(values) == 4:
                batches_completed = int(values[1])
                epochs_completed = int(values[3])
        
    else:
        print('No Checkpoint found, initialising global variables to defaults.')
        # Initialise our global vars (W and b)
        sess.run(tf.global_variables_initializer())

    # Add summary ops
    p, r, f1 = add_summary_ops(y)


    # Initialise our local vars (for calculating precision/recall/f1 during training)
    sess.run(tf.local_variables_initializer())

    # Print the current models number of training params
    print("Total training params: %.1fM" % (get_num_trainable_params() / 1e6))

    # Get the iterator handles to feed for train/val/test
    train_handle = sess.run(train_iterator.string_handle())
    val_handle = sess.run(val_iterator.string_handle())
    test_handle = sess.run(test_iterator.string_handle())

    no_improvement_last_epoch = False
    old_loss = 2**32 - 1 # A large number in case this is our first run

    # Compute for NUM_EPOCHS
    while epochs_completed < NUM_EPOCHS:
        print("Beginning epoch {0}".format(epochs_completed))

        print("Shuffling training dataset...")
        train_dataset = list(zip(train_features, train_labels))
        random.shuffle(train_dataset)
        train_features, train_labels = zip(*train_dataset)
        
        print("Shuffling validation dataset...")
        val_dataset = list(zip(val_features, val_labels))
        random.shuffle(val_dataset)
        val_features, val_labels = zip(*val_dataset)

        # Initialise our iterators with data (this also resets them to the beginning of their dataset)
        sess.run(train_iterator.initializer, feed_dict={features_placeholder: train_features, labels_placeholder: train_labels, network.is_training: True})
        sess.run(val_iterator.initializer, feed_dict={features_placeholder: val_features, labels_placeholder: val_labels, network.is_training: False})
        
        while True:
            try:
                # Every 1000 batches, also trace runtime statistics for debugging memory usage/compute time
                if batches_completed % 1000 == 0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _, loss, prediction, summary, _x, _y, _p, _r, _f1 = sess.run([network.train_step, network.loss, network.ys_pred, network.summary_op, x, y, p, r, f1 ],
                                                                                 feed_dict={
                                                                                     handle: train_handle,
                                                                                     network.lr: network.learning_rate,
                                                                                     network.is_training: True,
                                                                                     network.keep_prob: 0.8
                                                                                     #,network.current_step: batches_completed
                                                                                 },
                                                                                 options=run_options,
                                                                                 run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'batch{0}'.format(batches_completed))
                    train_writer.add_summary(summary, global_step=batches_completed)
                # else just train normally
                else:
                    _, loss, prediction, summary, _x, _y, _p, _r, _f1 = sess.run([network.train_step, network.loss, network.ys_pred, network.summary_op, x, y, p, r, f1 ],
                                                                                 feed_dict={ 
                                                                                     handle: train_handle,
                                                                                     network.lr: network.learning_rate, 
                                                                                     network.is_training: True, 
                                                                                     network.keep_prob: 0.8
                                                                                     #,network.current_step: batches_completed
                                                                                 })
                    train_writer.add_summary(summary, global_step=batches_completed)
                # Also run a validation batch every 20 batches for TensorBoard
                if batches_completed % 20 == 0:
                    loss, prediction, summary, _x, _y, _p, _r, _f1 = sess.run([network.loss, network.ys_pred, network.summary_op, x, y, p, r, f1 ],
                                                                              feed_dict={ 
                                                                                  handle: val_handle,
                                                                                  network.lr: network.learning_rate, 
                                                                                  network.is_training: False, 
                                                                                  network.keep_prob: 1.0
                                                                                  #,network.current_step: batches_completed
                                                                              })
                    val_writer.add_summary(summary, global_step=batches_completed)
                
                
                batches_completed = batches_completed + 1
            # If we ran out of data, that's the end of our epoch
            except tf.errors.OutOfRangeError:
                break

        
        # After our epoch, calculate mean loss over full validation set
        sess.run(val_iterator.initializer, feed_dict={ features_placeholder: val_features, labels_placeholder: val_labels, network.is_training: False })
        total_loss = 0
        while True:
            try:
                loss, _preds, _y, _x, _p, _r, _f1 = sess.run([network.loss, network.ys_pred, y, x, p, r, f1], 
                                                             feed_dict={
                                                                 handle: val_handle,
                                                                 network.lr: network.learning_rate,
                                                                 network.is_training: False,
                                                                 network.keep_prob: 1.0
                                                                 #,network.current_step: batches_completed
                                                             })
                total_loss += loss
            # run predictions until validation set is exhausted
            except tf.errors.OutOfRangeError:
                break

        # Compare the test to the previous models test, either drop learning rate or stop early if no improvement
        mean_loss = total_loss / VALIDATION_SET_SIZE
        
        epochs_completed = epochs_completed + 1

        print("Mean loss after {0} epochs: {1}".format(epochs_completed, mean_loss))

        try:
            # Try to read old loss from previous checkpoint
            with open('./checkpoints/' + network.NAME + '/' + network.NAME + '_%d.ckpt.txt' % (epochs_completed - 1), mode='r') as file:
                file_data = file.read().splitlines()
                old_loss = float(file_data[0])
                old_learning_rate = float(file_data[2])
        except:
            # Must be first checkpoint
            pass

        print("Previous loss was: {0}, loss delta for epoch: {1}".format(old_loss, mean_loss - old_loss))
        # If we didn't improve
        if mean_loss >= old_loss:
            # and we just dropped the learning rate last epoch
            if no_improvement_last_epoch:
                # Stop training early
                print("We're done! Best model was after {0} epochs at {1} mean loss.".format((epochs_completed - 2), old_loss))
                break
            else: # Decay learning rate by factor of 10, and take the previous weights
                network.learning_rate = network.learning_rate * 0.1 
                print("Setting learning rate to {0} and restoring previous checkpoint {1}".format(network.learning_rate, epochs_completed - 1))
                saver.restore(sess, './checkpoints/' + network.NAME + '/' + network.NAME + '_%d.ckpt' % (epochs_completed - 1))
                mean_loss = old_loss
                
                # If we still don't improve next time after lowering learning rate
                no_improvement_last_epoch = True
        else:        
            no_improvement_last_epoch = False 

        print("Saving new checkpoint and updating metadata...")
        # Save this model as a new checkpoint
        file_name = './checkpoints/' + network.NAME + '/' + network.NAME + '_%d.ckpt' % epochs_completed
        save_path = saver.save(sess, file_name)
        
        # also save current learning rate and global step in an associated text file!
        with open('./checkpoints/' + network.NAME + '/' + network.NAME + '_%d.ckpt.txt' % epochs_completed, mode='w') as out_file:
            out_file.write('{0}\n{1}\n{2}\n{3}'.format(mean_loss, batches_completed, network.learning_rate, epochs_completed))
        
        
        
        sess.run(test_iterator.initializer, feed_dict={features_placeholder: test_features, labels_placeholder: test_labels, network.is_training: False})
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


        print('Mean Auroc: {0}'.format(mean_auroc / network.num_labels))






train_writer.close()
val_writer.close()
