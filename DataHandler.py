import csv
import hashlib
import re
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.util import compat

class DataHandler:

    # 112120 images
    # 70% training, 10% validation, 20% testing
    # ~78484 training, ~11212 validation, ~22424 training
    # 75712 training, 10812 validation, 25596 training <- actual splits.
    def __init__(self, multi_label=True):
        
        self.TOTAL_IMAGES = 112120
        self.training_percentage = 70
        self.validation_percentage = 10
        self.testing_percentage = 20
        self.multi_label = multi_label

        if multi_label:
            self.GROUND_TRUTHS = ['Cardiomegaly','Emphysema','Effusion','Hernia','Infiltration',
                              'Mass','Nodule','Atelectasis','Pneumothorax','Pleural_Thickening',
                              'Pneumonia','Fibrosis','Edema','Consolidation']
            self.image_list = self.create_multilabel_label_dict()
        else:
            self.GROUND_TRUTHS = ['Pathology', 'No Pathology']
            self.image_list = self.create_singlelabel_label_dict()
        
        return None

    def create_multilabel_label_dict(self):
        '''
        1. create mapping filename -> dataset using the txt file so x = { "001.png": "testing", etc } O(n)
        2. create normal list by iterating the csv line by line but check mapping to tell which data set. O(n)
        3. for train/val set we need to hash to get approx split. -> O(2n) creation.
        '''
        image_list = {
            'training': [],
            'validation': [],
            'testing': []
        } 
        
        file_mapping = {}
        with open('./train_val_list.txt') as file:
            train_files = file.read().splitlines()
            for file_name in train_files: 
                file_mapping[file_name] = 1

        with open('./test_list.txt') as file:
            test_files = file.read().splitlines()
            for file_name in test_files:
                file_mapping[file_name] = 0

        first_line = True
        with open('../data/Data_Entry_2017.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            
            for row in reader:
                if first_line:
                    first_line = False
                    continue
                # row[0] = filename
                # row[1] = ground truths
                file_name = row[0]
                try: 
                    if file_mapping[file_name] == 1:
                        # Train/validation set, need to hash to split
                        percentage_hash = self.get_percentage_hash(row[0])
                        if percentage_hash < 12.5: # 10% of total data is 12.5% of remaining data
                            image_list['validation'].append((file_name, self.new_y_array(row[1])))
                        else:
                            image_list['training'].append((file_name, self.new_y_array(row[1])))
                    else:
                        image_list['testing'].append((file_name, self.new_y_array(row[1])))
                except KeyError:
                    pass
                    
        return image_list

    def create_singlelabel_label_dict(self):
        return []

    def get_percentage_hash(self, file_name):
        # Hash only the patient number so that multiple images from the same patient
        # compute the same hash so they will be placed in the same subset.
        big_number = 2 ** 27 - 1 #~134M
        file_name = re.sub("_[0-9]{3}\.png", "", file_name)
        file_name_hashed = hashlib.sha1(compat.as_bytes(file_name)).hexdigest()
        # Bring hash in range [1-big_number], multiply by factor to set range [0-100]
        percentage_hash = ((int(file_name_hashed, 16) % (big_number + 1)) * (100.0 / big_number))
        return percentage_hash

    def new_y_array(self, truth_string):
        array = np.zeros(len(self.GROUND_TRUTHS), dtype=np.float32)
    
        if self.multi_label:
            labels_array = truth_string.split('|')

            for label in labels_array:
                try:
                    label_index = self.GROUND_TRUTHS.index(label)
                    array[label_index] = 1
                except ValueError:
                    pass #do nothing, it's No Finding which we encode as all zeros
                
        return array

    def image_parse_function(self, filename, label):
        image_string = tf.read_file('../data/images/multi-label/' + filename)
        image_decoded = tf.image.decode_png(image_string, channels=1)
        #image_resized = tf.image.resize_images(image_decoded, [224,224])
        #image_resized = tf.image.resize_images(image_decoded, [331, 331])
        image_resized = tf.image.resize_images(image_decoded, [299, 299])
        image_rgb = tf.image.grayscale_to_rgb(image_resized)
        image_float = tf.image.convert_image_dtype(image_rgb, dtype=tf.float32)
        return image_float, label

    def add_random_brightness(self, image, label, is_training=False):
        image = tf.cond(is_training, 
                        lambda: tf.image.random_brightness(image, max_delta=32.0 / 255.0),
                        lambda: image)
        # Force clamping of image values
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    def random_flipping(self, image, label, is_training=False):
        image = tf.cond(is_training, 
                        lambda: tf.image.flip_left_right(image), 
                        lambda: image)
        return image, label

    def random_scaling_and_crop(self, image, label, is_training=False):
        def distort_image(image):
            original_dims = tf.shape(image)[0:2]
            distorted_bbox = tf.image.sample_distorted_bounding_box(tf.shape(image),
                                                                    min_object_covered=0.85,
                                                                    use_image_if_no_bounding_boxes=True)
            bbox_begin, bbox_size, _ = distorted_bbox
            image = tf.slice(image, bbox_begin, bbox_size)
            image = tf.image.resize_images(image, size=original_dims)

        image = tf.cond(is_training, 
                        lambda: distort_image(image),
                        lambda: image)
        return image, label

    def finalise_images(self, image, label):
        # Rescales images from [0,1] to [-1,1]
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image, label

    def get_dataset(self, data_type='training', num_examples=0):
        if num_examples < 0:
            raise ValueError('Invalid num_examples: %d' % num_examples)
        size = len(self.image_list[data_type])
        features = []
        labels = []

        if num_examples == 0 or num_examples >= size:
            for feature, label in self.image_list[data_type]:
                #for feature, label in reverse(self.image_list[data_type]):
                features.append(feature)
                labels.append(label)
        else:
            for index in range(num_examples):
                #for index in range(num_examples - 1, -1, -1):
                feature, label = self.image_list[data_type][index]
                features.append(feature)
                labels.append(label)

        return features, labels

    def get_pathology_counts(self, data_type='validation'):
        image_dict = {}
        pathology_dict = {
            'multi-label': []
        }
        with open('./' + data_type + '_images.txt') as file:
            images = file.read().splitlines()
            for image in images:
                image_dict[image] = 1
        with open('../data/Data_Entry_2017.csv') as file:
            first_line = True
            reader = csv.reader(file)
            for row in reader:
                if first_line:
                    first_line = False
                    continue
                # row[0] = filename
                # row[1] = ground truths
                if row[0] in image_dict:
                    labels = row[1].split('|')
                    if len(labels) > 1:
                        pathology_dict['multi-label'].append(row[0])
                    else:
                        if labels[0] not in pathology_dict:
                            pathology_dict[labels[0]] = []
                        pathology_dict[labels[0]].append(row[0])
            
        return pathology_dict
                            
        