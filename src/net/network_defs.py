# This file will contain ways to define a model
# - example: resNet 20 and AlexNet will be here
# - more will be added as we progress with testing
import tensorflow as tf
import numpy as np
import src.util.model_helpers as mh

def ResNet_20(cfgs, inputs, image_size, train=True):
    wave_type = cfgs['model']['wavelet']
    data_type = cfgs['data']['type']
    pooling = cfgs['model']['pooling']

    with tf.variable_scope('ResNet'):
        if(data_type == 'MNIST'):
            # we need to reshape our inputs
            inputs = tf.reshape(inputs, [-1,28,28,1])
            inputs = tf.image.grayscale_to_rgb(inputs)
            inputs = tf.image.resize_nearest_neighbor(inputs, (image_size[1], image_size[0]))
        #
        in_shape = inputs.get_shape().as_list()
        print(in_shape)
        # 
        # the initial [3 x 3 x 16] convolution
        inputs = mh.conv2d_fixed_padding(inputs=inputs, filters=16, kernel_size=3, strides=1)
        print(inputs.get_shape().as_list())
        # now the three blocks
        block_2 = mh.block_layer(inputs, filters=16, num_blocks=3, strides=1, training=train, name='conv2_x')
        print(block_2.get_shape().as_list())

        block_3 = mh.block_layer(block_2, filters=32, num_blocks=3, strides=2, training=train, name='conv3_x')
        print(block_3.get_shape().as_list())

        block_4 = mh.block_layer(block_3, filters=64, num_blocks=3, strides=2, training=train, name='conv4_x')
        print(block_4.get_shape().as_list())
        # apply another pool here

        with tf.variable_scope('pool_out'):
            shape = block_4.get_shape().as_list()
            #axes = [1, 2]
            #pool_out = tf.reduce_mean(block_4, axes)
            pool_out = mh.pool(block_4, pool=pooling, wavelet=wave_type) # for now this is the one we will mess with
            print(pool_out.get_shape().as_list())
            return pool_out

def ResNet_18(cfgs, inputs, image_size, train=True):
    print('RESNET_18')
    wave_type = cfgs['model']['wavelet']
    data_type = cfgs['data']['type']
    pooling = cfgs['model']['pooling']

    with tf.variable_scope('ResNet'):
        if(data_type == 'MNIST'):
            # we need to reshape our inputs
            inputs = tf.reshape(inputs, [-1,28,28,1])
            inputs = tf.image.grayscale_to_rgb(inputs)
            inputs = tf.image.resize_nearest_neighbor(inputs, (image_size[1], image_size[0]))
        #
        in_shape = inputs.get_shape().as_list()
        print(in_shape)
        # 
        # the initial [3 x 3 x 16] convolution
        conv_1 = mh.conv2d_fixed_padding(inputs=inputs, filters=16, kernel_size=7, strides=2)
        print(conv_1.get_shape().as_list())
        # now the four blocks
        block_1 = mh.block_layer(conv_1, filters=32, num_blocks=2, strides=1, training=train, name='conv1_x')
        print(block_1.get_shape().as_list())

        block_2 = mh.block_layer(block_1, filters=64, num_blocks=2, strides=2, training=train, name='conv2_x')
        print(block_2.get_shape().as_list())

        block_3 = mh.block_layer(block_2, filters=128, num_blocks=2, strides=2, training=train, name='conv3_x')
        print(block_3.get_shape().as_list())

        block_4 = mh.block_layer(block_3, filters=128, num_blocks=2, strides=1, training=train, name='conv4_x')
        print(block_4.get_shape().as_list())
        # apply another pool here

        with tf.variable_scope('pool_out'):
            shape = block_4.get_shape().as_list()
            #axes = [1, 2]
            #pool_out = tf.reduce_mean(block_4, axes)
            pool_out = mh.pool(block_4, pool=pooling, wavelet=wave_type) # for now this is the one we will mess with
            print(pool_out.get_shape().as_list())
            return pool_out

# This will be our shallow net
# - Using more standard networks is more well standard
# - another note: this is alexNet_v2
'''
From: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/alexnet.py

      net = layers.conv2d(inputs, 64, [11, 11], 1, padding='VALID', scope='conv1') 64 ->  64
      net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool1')                   64 -> 32
      net = layers.conv2d(net, 192, [5, 5], scope='conv2')
      net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool2')                   32 -> 16
      net = layers.conv2d(net, 384, [3, 3], scope='conv3') 
      net = layers.conv2d(net, 384, [3, 3], scope='conv4')
      net = layers.conv2d(net, 256, [3, 3], scope='conv5')
      net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool5')                   16 -> 8
''' 
# Modified the initial convolution, the stride of 4 is too much, changed to 2
# - also might cut features in half, not sure yet
def AlexNet(cfgs, inputs, image_size):
    with tf.variable_scope('AlexNet'):
        wave_type = cfgs['model']['wavelet']
        data_type = cfgs['data']['type']
        pooling = cfgs['model']['pooling']
        if(data_type == 'MNIST'):
            # we need to reshape our inputs
            inputs = tf.reshape(inputs, [-1,28,28,1])
            inputs = tf.image.grayscale_to_rgb(inputs)
            inputs = tf.image.resize_nearest_neighbor(inputs, (image_size[1], image_size[0]))
        #
        #in_shape = inputs.get_shape().as_list()
        # rest of the net here
        conv_1 = mh.conv2d(inputs, 16, kernel=11, stride=1, padding='VALID', name='conv_1')
        pool_1 = mh.pool(conv_1, pooling, wave_type)
        conv_2 = mh.conv2d(pool_1, 32, kernel=3, name='conv_2')
        pool_2 = mh.pool(conv_2, pooling, wave_type)
        conv_3 = mh.conv2d(pool_2, 64, kernel=3, name='conv_3')
        conv_4 = mh.conv2d(conv_3, 64, kernel=3, name='conv_4')
        conv_5 = mh.conv2d(conv_4, 128, kernel=3, name='conv_5')
        pool_3 = mh.pool(conv_5, pooling, wave_type)

        return pool_3
#   #Done

# sketch nets now

# helper to get network def based on string input
def get_model_function(model_type):
    models = {}
    models['base'] = ResNet_20
    # more later
    return models[model_type]
