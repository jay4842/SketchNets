import tensorflow as tf
import numpy as np
import cv2
import time
import os
import sys

# some image loader helpers
def getOptimizer(cfgs, learning_rate):
    type_    = cfgs['train']['optimizer']
    momentum = cfgs['train']['momentum']

    if(type_ == 'adam'):
        return tf.train.AdamOptimizer(learning_rate=learning_rate)
    if(type_ == 'momentum'):
        return tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
    if(type_ == 'gradientDescent'):
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    if(type_ == 'RMSProp'):
        return tf.train.RMSPropOptimizer(learning_rate=learning_rate) 
# make a session with some config settings as well
# - you can modify this if you want to add anymore config settings
# - additionally the gpu fraction is .4 by default
def get_session(gpu_fraction=0.4):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''
    num_threads = 2
    # gives error that has to deal with the version of tensorflow, and the cudNN version as well
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    #return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    config = tf.ConfigProto() #allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement=False
    config.gpu_options.per_process_gpu_memory_fraction=gpu_fraction
    config.intra_op_parallelism_threads=num_threads
    #config.log_device_placement=True
    sess = tf.Session(config=config)
    return sess

# for loading a single test image
def get_test_image(cfgs, image_path, w=128, h=256, standardize=True, channel_swap=None):
    if(cfgs['data']['root_paths']):
        image_path = image_path.replace('\\', '/')
        image = cv2.imread(image_path)
        label = image_path.split('/')[-1].split(cfgs['data']['label_seperator'])[0]
    else:
        image = cv2.imread(cfgs['data']['data_start'] + image_path)
        label = image_path.split(cfgs['data']['label_seperator'])[0]

    image = cv2.resize(image, (w, h))

    if(standardize):
        image = (image - np.mean(image)) / (np.std(image))
    
    if(channel_swap is not None):
        image = image[:,:,channel_swap]
    # Soem more label stuff
    label = cfgs['data']['classes'].index(label)
    bin_label = [0 for x in range(len(cfgs['data']['classes']))]
    bin_label[label] = 1

    return image, bin_label
# return an image batch, usually for training
def get_images(cfgs, batch_paths, ids, w=128, h=256, augment=True, standardize=True, channel_swap=None):
    images = []
    labels = []
    for idx, b in enumerate(ids):
        path_ = batch_paths[b]
        path_ = path_.replace('\\','/')
        if(cfgs['data']['root_paths']):
            image = cv2.imread(path_)
            label = path_.split('/')[-1].split(cfgs['data']['label_seperator'])[0]
        # else add the base path to it
        else:
            image = cv2.imread(cfgs['data']['data_start'] + path_)
            label = path_.split(cfgs['data']['label_seperator'])[0]

        # resize it
        image = cv2.resize(image, (w,h))

        # sometimes flip the image
        # - If more augmentation is needed,
        #   add additional lines here
        if (augment and np.random.random() > 0.5):
            image = np.fliplr(image)
        

        # normalize the image
        #image = normalize(image)
        if(standardize):
            image = (image - np.mean(image)) / (np.std(image))
        # default is none but sometimes we might want to swap the channels
        if(channel_swap is not None):
            image = image[:,:,channel_swap]

        images.append(image)

        label = cfgs['data']['classes'].index(label)
        bin_label = [0 for x in range(len(cfgs['data']['classes']))]
        bin_label[label] = 1
        labels.append(bin_label)

    return images, labels

# A get batch helper
# - This function is a helper to call get_images
# - needs a path array and an ids array that holds all the indicies in the paths array
def get_batch(cfgs, paths, ids, batch_size=5, standardize=False, w=128, h=256):
	batch_ids = np.random.choice(ids,batch_size)  
	return get_images(cfgs, paths, batch_ids, standardize=standardize, w=w, h=h)
# file system helpers
