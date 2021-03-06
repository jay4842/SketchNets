# This file will contain ways to define a model
# - example: resNet 20 and AlexNet will be here
# - more will be added as we progress with testing
import tensorflow as tf
import numpy as np
import src.util.model_helpers as mh

def ResNet_20(cfgs, inputs, image_size, train=True, transform_done=False):
    wave_type = cfgs['model']['wavelet']
    data_type = cfgs['data']['type']
    pooling = cfgs['model']['pooling']

    with tf.variable_scope('ResNet'):
        if(data_type == 'MNIST' and not transform_done):
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

def ResNet_18(cfgs, inputs, image_size, train=True,transform_done=False):
    print('RESNET_18')
    wave_type = cfgs['model']['wavelet']
    data_type = cfgs['data']['type']
    pooling = cfgs['model']['pooling']

    with tf.variable_scope('ResNet'):
        if(data_type == 'MNIST' and not transform_done):
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

''' 
# Modified the initial convolution, the stride of 4 is too much, changed to 2
# - also might cut features in half, not sure yet
def AlexNet(cfgs, inputs, image_size, transform_done=False):
    with tf.variable_scope('AlexNet'):
        wave_type = cfgs['model']['wavelet']
        data_type = cfgs['data']['type']
        pooling = cfgs['model']['pooling']
        if(data_type == 'MNIST' and not transform_done):
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

# Sketch net 1
# - first create dwt by feeding the inputs tensor to a dwt transform
#   - This will be based on: wave_type.
# - Feed the four outputs into a cnn. 
# - Return that output.
def sketch_net_1(cfgs, inputs, image_size):
    with tf.variable_scope('Sketch_Net_1'):
        wave_type = cfgs['model']['wavelet']
        data_type = cfgs['data']['type']
        # return a backend based on the dataset that is currently being used
        cnn_backend = get_model_function(data_type, 'base')
        # convert mnist data to d2 image vs 1d image
        with tf.variable_scope('process'):
            if(data_type == 'MNIST'):
                # we need to reshape our inputs
                inputs = tf.reshape(inputs, [-1,28,28,1])
                inputs = tf.image.grayscale_to_rgb(inputs)
                inputs = tf.image.resize_nearest_neighbor(inputs, (image_size[1], image_size[0]))
            #
            ll, lh, hl, hh = mh.dwt(inputs, wave_type, name='dwt')
            inputs = tf.concat([ll, lh, hl, hh], axis=1)
        # return the output of the cnn backend, 
        # additionally we have also performed the image transform on 
        # MNIST so we do not need to do it again.
        return cnn_backend(cfgs, inputs, image_size, transform_done=True)

# sketch net 2
# - first create dwt by feeding the inputs tensor to a dwt transform
# - Feed ll to a deeper network.
# - feed lh, hl, and hh to they're own shallow networks.
# - return the sum of the four networks
def sketch_net_2(cfgs, inputs, image_size):
    with tf.variable_scope('Sketch_Net_2'):
        wave_type = cfgs['model']['wavelet']
        data_type = cfgs['data']['type']
        deep_cnn = ResNet_20
        shallow_cnn = AlexNet
        # return a backend based on the dataset that is currently being used
        with tf.variable_scope('process'):
            if(data_type == 'MNIST'):
                # we need to reshape our inputs
                inputs = tf.reshape(inputs, [-1,28,28,1])
                inputs = tf.image.grayscale_to_rgb(inputs)
                inputs = tf.image.resize_nearest_neighbor(inputs, (image_size[1], image_size[0]))
            #
            ll, lh, hl, hh = mh.dwt(inputs, wave_type, name='dwt')
        
        ll = deep_cnn(cfgs, ll, image_size, transform_done=True)
        lh = shallow_cnn(cfgs, lh, image_size, transform_done=True)
        hl = shallow_cnn(cfgs, hl, image_size, transform_done=True)
        hh = shallow_cnn(cfgs, hh, image_size, transform_done=True)

        # now sum them
        return tf.add_n([ll,lh,hl,hh], name='sum')
# end of sketch_2

# sketch 3
# - Perform dwt on initial image creating ll, lh, hl, hh
# - feed ll to a deep network
# - concat 1h, hl, hh, feed to a shallow network
# - return the sum of the two networks
def sketch_net_3(cfgs, inputs, image_size):
    with tf.variable_scope('Sketch_Net_3'):
        wave_type = cfgs['model']['wavelet']
        data_type = cfgs['data']['type']
        deep_cnn = ResNet_20
        shallow_cnn = AlexNet
        # return a backend based on the dataset that is currently being used
        with tf.variable_scope('process'):
            if(data_type == 'MNIST'):
                # we need to reshape our inputs
                inputs = tf.reshape(inputs, [-1,28,28,1])
                inputs = tf.image.grayscale_to_rgb(inputs)
                inputs = tf.image.resize_nearest_neighbor(inputs, (image_size[1], image_size[0]))
            # dwt
            ll, lh, hl, hh = mh.dwt(inputs, wave_type, name='dwt')
        
        deep = deep_cnn(cfgs, ll, image_size, transform_done=True)
        shal = shallow_cnn(cfgs, tf.concat([lh,hl, hh], axis=-1), image_size, transform_done=True)

        return tf.add_n([deep,shal], name='sum')
# end of sketch net 3

# sketch 4
# - Perform dwt on initial image creating ll, lh, hl, hh
# - perform another dwt on ll creating dwt' = [ll', lh', hl', hh']
# - upsample dwt' to match lh, hl, and hh
# - concat all, [ll', lh', hl', hh', lh, hl, hh] and feed to a normal cnn
# - return normal cnn
def sketch_net_4(cfgs, inputs, image_size):
    with tf.variable_scope('Sketch_Net_4'):
        wave_type = cfgs['model']['wavelet']
        data_type = cfgs['data']['type']
        deep_cnn = ResNet_20
        # return a backend based on the dataset that is currently being used
        with tf.variable_scope('process'):
            if(data_type == 'MNIST'):
                # we need to reshape our inputs
                inputs = tf.reshape(inputs, [-1, 28, 28, 1])
                inputs = tf.image.grayscale_to_rgb(inputs)
                inputs = tf.image.resize_nearest_neighbor(inputs, (image_size[1], image_size[0]))
            # dwt processes
            ll_1, lh_1, hl_1, hh_1 = mh.dwt(inputs, wave_type, name='dwt_1')
            ll_2, lh_2, hl_2, hh_2 = mh.dwt(ll_1, wave_type, name='dwt_2')
            dwt_2 = tf.concat([ll_2, lh_2, hl_2, hh_2], axis=-1)
            dwt_2 = tf.image.resize_nearest_neighbor(dwt_2, (image_size[1]//2, image_size[0]//2))
            conv_in = tf.concat([dwt_2, lh_1, hl_1, hh_1], axis=-1)
        
        return deep_cnn(cfgs, conv_in, image_size, transform_done=True)
    


        
        
# helper to get network def based on string input
# - will add the other models as I go along
def get_model_function(data_type, model_type):
    models = {}
    models['base'] = ResNet_20
    
    # more later
    return models[model_type]
