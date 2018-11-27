import tensorflow as tf
import numpy as np
from glob import glob
import time
from termcolor import colored, cprint
from tensorflow.examples.tutorials.mnist import input_data
import yaml
import os

import src.net.model as net
import src.util.util as util

# simple testing flow
def test(cfgs=None, save_dir=None):
    cprint('\n' + 'Testing...\n', 'cyan', attrs=['bold'])
    tf.reset_default_graph()
	# open cfgs if not passed
    if(cfgs is None):
        file = open('wave_cfg.yaml')
        cfgs = yaml.load(file)
        file.close()
	# get dataset type
    type_ = cfgs['data']['type']
    # make and setup model
    with tf.variable_scope('network'):
        model = net.Model(cfgs, name='Model')
        model.define_model()
    # setup dataset vars
    image_size = [cfgs['data']['image_w'],cfgs['data']['image_h']]
    mnist = None
    if(type_ == 'MNIST'):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    if(type_ == 'MNIST'):
        test_images = mnist.test.images
        test_labels = mnist.test.labels
    else:
        test_dir = cfgs['data']['data_start'] + 'test/'
        test_images = glob(test_dir + '*.png')

    tests = len(test_images)
    # setup session
    sess = util.get_session(cfgs['gpu_limit'])
    # restore
    checkpoint = tf.train.latest_checkpoint(save_dir)
    saver = tf.train.Saver()
    saver.restore(sess,checkpoint)
    # keep track of accuracies  and root mean square errors
    accs  = []
    rmses = []
    for idx in range(tests):
        if type_ == 'MNIST':
            net_in = [test_images[idx]]
            net_lab = [test_labels[idx]]
        else:
            net_in, net_lab = util.get_test_image(cfgs, test_images[idx], w=image_size[0], h=image_size[1])
            net_in = [net_in]
            net_lab = [net_lab]
        acc_, rmse_ = sess.run([model.accuracy, model.rmse],feed_dict={model.inputs: net_in, model.labels: net_lab})
        accs.append(acc_)
        rmses.append(rmse_)
        print('\r{}/{}'.format(idx,tests), end='')
    
    # calculate average scores
    out_accuracy = np.sum(accs) / len(accs)
    out_rmse = np.sum(rmses) / len(rmses)
    line = 'Test accuracy = {:.4f} Test RMSE = {:.4f}'.format(out_accuracy, out_rmse)
    cprint('\n' + line, 'magenta', attrs=['bold'])

    # write the result to a file
    with open(save_dir + 'test_out.txt', 'w') as file:
        file.write(line + '\n')
        file.close()
    
    sess.close()
    return out_accuracy, out_rmse
