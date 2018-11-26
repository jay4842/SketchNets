# This file will contain ways to define a model
# - example: resNet 20 and AlexNet will be here
# - more will be added as we progress with testing
import tensorflow as tf
import numpy as np
import src.util.model_helpers as mh

def ResNet_20(args, inputs, image_size, type_, pooling='wave', train=True):
    with tf.variable_scope('ResNet'):
        if(type_ == 'MNIST'):
            # we need to reshape our inputs
            inputs = tf.reshape(inputs, [-1,28,28,1])
            inputs = tf.image.grayscale_to_rgb(inputs)
            inputs = tf.image.resize_images(inputs, (image_size[1], image_size[0]))
        #
        in_shape = inputs.get_shape().as_list()
        print(in_shape)
        # 
        # the initial [3 x 3 x 16] convolution
        inputs = conv2d_fixed_padding(inputs=inputs, filters=16, kernel_size=3, strides=1)
        print(inputs.get_shape().as_list())
        # here a pool would be used

        print(inputs.get_shape().as_list())
        # now the three blocks
        block_2 = block_layer(inputs, filters=16, num_blocks=3, strides=1, training=train, name='conv2_x')
        print(block_2.get_shape().as_list())

        block_3 = block_layer(block_2, filters=32, num_blocks=3, strides=2, training=train, name='conv3_x')
        print(block_3.get_shape().as_list())

        block_4 = block_layer(block_3, filters=64, num_blocks=3, strides=2, training=train, name='conv4_x')
        print(block_4.get_shape().as_list())
        # apply another pool here

        with tf.variable_scope('pool_out'):
            shape = block_4.get_shape().as_list()
            #axes = [1, 2]
            #pool_out = tf.reduce_mean(block_4, axes)
            pool_out = net_h.pool(block_4, pooling='avg') # for now this is the one we will mess with
            print(pool_out.get_shape().as_list())
            return pool_out