import tensorflow as tf
import numpy as np

import src.util.wavelets as wa
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

# Batch normalization
def batch_norm(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=_BATCH_NORM_DECAY, updates_collections=None, \
           epsilon=_BATCH_NORM_EPSILON, scale=True, scope=name)

# Helper to return a wavelet to use for a transform
def get_wavefn(type_='haar'):
    if(type_ =='haar'):
        return wa.haar
    elif(type_ =='db2'):
        return wa.db2
    elif(type_=='db3'):
        return wa.db3
    elif(type_=='db4'):
        return wa.db4
    else:
        return wa.db1 # same as haar but is the default

# A helper to return a type of pooling
# - supports max, avg, and wavelet pooling
# - stride is something that will be fixed later
#   mainly for wavelets, ex: a stride of 4 means two wavelet calls
def pool(X, pool='max', wavelet='haar', stride=2, name='pooling'):
    if(pool == 'wave'):
        wavelet = get_wavefn(wavelet)
        return wa.dwt(X, wavelet)[0, 0, :, :, :, :]# LxLy (approx)
    if(pool == 'avg'):
        return tf.nn.avg_pool(X, ksize=[1, 2, 2, 1], strides=[1, stride, stride, 1], \
           padding='SAME', name=name)
    if(pool == 'max'):
        return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, stride, stride, 1], \
                padding='SAME', name=name)
# end of pool helper

# The dwt function
# - uses wavelet library to preform the transform.
# - I am not sure If I need to keep the batch norm after but I think we do need it.
def dwt(x, wave_type, name='dwt'):
    with tf.variable_scope(name):
        x = batch_norm(x, name='bn_input')

        #[w_pass, h_pass, b, height, width, channel]
        in_shape = x.get_shape().as_list()
        x_dwt = wa.dwt(x,get_wavefn(type_=wave_type)) # This guy gets all of them boys
        
        ll = tf.image.resize_nearest_neighbor(x_dwt[0, 0, :, :, :, :], (in_shape[1], in_shape[2]))
        ll = tf.reshape(ll, [-1, in_shape[1], in_shape[2], in_shape[-1]])
        ll = batch_norm(ll, name='ll_bn')

        lh = tf.image.resize_nearest_neighbor(x_dwt[0, 1, :, :, :, :], (in_shape[1], in_shape[2]))
        lh = tf.reshape(lh, [-1, in_shape[1], in_shape[2], in_shape[-1]])
        lh = batch_norm(lh, name='lh_bn')

        hl = tf.image.resize_nearest_neighbor(x_dwt[1, 0, :, :, :, :], (in_shape[1], in_shape[2]))
        hl = tf.reshape(hl, [-1, in_shape[1], in_shape[2], in_shape[-1]])
        hl = batch_norm(hl, name='hl_bn')

        hh = tf.image.resize_nearest_neighbor(x_dwt[1, 1, :, :, :, :], (in_shape[1], in_shape[2]))
        hh = tf.reshape(hh, [-1, in_shape[1], in_shape[2], in_shape[-1]])
        hh = batch_norm(hh, name='hh_bn')

        return ll, lh, hl, hh
# end of dwt



# Start of tensorflow helpers # # # # 

# general helpers
def get_dim(x):
    shape = x.get_shape().as_list()
    dim = 1
    for d in shape[1:]:
        dim *= d
    return dim

def lrelu(x):
    with tf.variable_scope('lrelu') as scope:
        if x.dtype is not tf.complex64:
            return tf.nn.leaky_relu(x)
        else:
            return x

def relu(x):
    with tf.variable_scope('relu') as scope:
        if x.dtype is not tf.complex64:
            return tf.nn.relu(x)
        else:
            return x

def delist(net):
    if type(net) is list: # if the value is a list delist it
        net = tf.concat(net,-1,name = 'cat')
    return net

def conv2d(net, filters, kernel = 3, stride = 1, dilation_rate = 1, activation = relu, 
            padding = 'SAME', trainable = True, name = None, reuse = None):
    # define the output
    net = tf.layers.conv2d(delist(net),filters,kernel,stride,padding,dilation_rate = dilation_rate, 
            activation = activation,trainable = trainable, name = name, reuse = reuse)
    return net

# RESNET HELPERS
# From resNet tensorflow
# - they use tf.layers batch norm instead of tf.nn. 
def batch_norm(inputs, training, data_format='channels_last'):
    """Performs a batch normalization using a standard set of parameters."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    return tf.layers.batch_normalization( inputs=inputs, axis=3, momentum=_BATCH_NORM_DECAY,
     epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=training, fused=True)

# TODO
# - Fixed padding
# From the tensorflow ResNet implementation.
def fixed_padding(inputs, kernel_size): 
    with tf.variable_scope('fixed_padding'):
        # channels are last for me
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return padded_inputs
# end of that guy

# TODO
# - conv_2d fixed padding
def conv2d_fixed_padding(inputs, filters, kernel_size, strides, name='conv_2d_fixed_padding', data_format='channels_last'): # I shouldn't have to change this
    with tf.variable_scope(name):
        if strides > 1:
            inputs = fixed_padding(inputs, kernel_size)

        return tf.layers.conv2d( inputs=inputs, filters=filters, kernel_size=kernel_size, 
            strides=strides, padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),data_format=data_format)

# TODO
# - Make projection shortcut a function
def projection_shortcut(inputs, filters_out, strides):
    with tf.variable_scope('projection_shortcut'):
        return conv2d_fixed_padding(inputs=inputs, filters=filters_out, kernel_size=1, strides=strides)

# Net building blocks here
# NOTE: 
# - the building block residual_block is a version 1 block that is simplified.
def res_block_v1(inputs, filters, filters_out, training, projection_shortcut, strides, name, data_format='channels_last'):
    with tf.variable_scope(name):
        '''
        - setup shortcut
        - conv_2d_fixed_padding
        - batch_norm
        - activation
        - conv_2d_fixed_padding
        - batch_norm
        - add shortcut
        - activation
        '''
        shortcut = inputs
        if projection_shortcut is not None:
            shortcut = projection_shortcut(inputs, filters_out, strides)
            shortcut = batch_norm(inputs=shortcut, training=training)

        # now the two convolutions
        # conv_layer 1
        inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides, name='conv_2d_1')
        inputs = batch_norm(inputs=inputs, training=training)
        inputs = tf.nn.relu(inputs)

        # conv_layer 2
        inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=1, name='conv_2d_2')
        inputs = batch_norm(inputs=inputs, training=training)
        inputs += shortcut
        inputs = tf.nn.relu(inputs)

        return inputs
# a bit different than the previous implementation (cleaner than the older one)

def res_block_v2(inputs, filters, filters_out, training, projection_shortcut, strides, name, data_format='channels_last'):
    # TODO add the res bock v2 that is mentioned in their paper
    with tf.variable_scope(name):
        shortcut = inputs
        inputs = batch_norm(inputs, training, data_format)
        inputs = tf.nn.relu(inputs)

        # projection shortcut?
        if(projection_shortcut is not None):
            shortcut = projection_shortcut(shortcut)

        
# also from tensorflow ResNet
def block_layer(inputs, filters, num_blocks, strides, training, name, bottleneck=False,
                 block_function=res_block_v1, data_format='channels_last'):
    with tf.variable_scope(name):
        # This guy will create one layer of blocks
        # if there is a bottle neck the out features is 4x the input
        filters_out = filters * 4 if bottleneck else filters

        # projection and striding is only applied to the first layer
        inputs = block_function(inputs, filters, filters_out, training, projection_shortcut, strides, 'block_0', data_format)
        
        # Now add blocks using our block_function
        for _ in range(1, num_blocks):
            inputs = block_function(inputs, filters, filters_out, training, None, 1, 'block_{}'.format(_), data_format)

        return tf.identity(inputs, name)

# an inference function to get logits and predictions.
# - uses a tf.dense layer as a fully connected layer
def inference(inputs, classes):
    # flatten the input, get our classes
    num_classes = classes
    dim = get_dim(inputs)

    inputs = tf.reshape(inputs, [-1, dim])
    # feed through a tf.dense_layer
    inputs = tf.layers.dense(inputs=inputs, units=num_classes)
    logits = tf.identity(inputs, 'logits')
    probs = tf.nn.softmax(logits, name='probs')
    return logits, probs

# END OF RESNET HELPERS
