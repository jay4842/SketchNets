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
# a train flow that will rely on cfgs for setting it up
def train(cfgs=None, save_dir=None):
    print('Train')
    with tf.device('/device:GPU:0'):
        tf.reset_default_graph()
        val_accs  = []
        val_rmses = []
		
        # Open our config file and load it using the yaml api
        if(cfgs is None):
            file = open('wave_cfg.yaml')
            cfgs = yaml.load(file)
            file.close()
		
        type_ = cfgs['data']['type']
        pooling = cfgs['model']['pooling']
        model_type = cfgs['model']['model_type']
        epochs = cfgs['train']['num_epochs']
        batch_size = cfgs['data']['batch_size']
        
        # create a dict to keep track of epoch results
        train_results = {}
        for idx in range(0,epochs):
            train_results['epoch_{}'.format(idx)] = {}
            train_results['epoch_{}'.format(idx)]['acc'] = []
            train_results['epoch_{}'.format(idx)]['rmse'] = []

        save = 'model_save/'
        # check our save if its there or not
        if(save_dir == None):
            save_dir = save

        # Select our network
        # - this will also tack on an infrence as well
        with tf.variable_scope('network'):
            model = net.Model(cfgs, name='Model')
            model.define_model()
        # set up the input tensors
        with tf.variable_scope('input'):
            # other images
            # setup the datasets
            #mnist
            mnist = None # mnist is set to none as a placeholder
            if(type_ == 'MNIST'):
                iterations = 60000 // cfgs['data']['batch_size'] # get iterations based on dataset size
                mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
            else:
                iterations = 50000 // cfgs['data']['batch_size'] # this infers that the set is cifar-10

        epoch_size = iterations

    if(cfgs['debug'] or cfgs['model_debug']): input('->')
    save_interval = cfgs['train']['save_interval'] # set out save interval
    #if(save_interval == -1): 
    #    save_interval =  int(epoch_size/5)# this is a placeholder...
    save_interval = iterations
    sess = util.get_session(cfgs['gpu_limit'])

    # setup our image paths if we are not using MNIST
    if not(cfgs['data']['type'] == 'MNIST'): # get paths and make ids array that correlates with it
        image_paths = glob(cfgs['data']['data_start']+'train/*.png')
        train_ids = list(range(0,len(image_paths)))
        np.random.shuffle(train_ids) # shuffle the ids. Randomizes them

    cprint('training....', 'green', attrs=['bold'])
    cprint('Save:      {}'.format(save_dir), 'cyan', attrs=['bold'])
    cprint('Dataset:   {}'.format(type_), 'green', attrs=['underline', 'bold'])
    print('Model:      {}'.format(colored('{}'.format(model_type), 'yellow') ))
    print('wavelet:    {}'.format(colored('{}'.format(cfgs['model']['wavelet']), 'yellow')))
    print('pooling:    {}'.format(colored('{}'.format(pooling), 'yellow')))
    print('epochs:     {}'.format(colored('{}'.format(epochs), 'yellow')))
    print('iterations: {}'.format(colored('{}'.format(iterations), 'yellow')))
    print('batch size: {}\n'.format(colored('{}'.format(batch_size), 'yellow')))

    with sess:
        # get starting time
        start_time = time.time()
        # initialize variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # starting up...
        print('training...')
        # base learning rate, it could change if you wanted to add that feature
        base_lr = cfgs['train']['learning_rate']
        # open our output text file. We want to log everything for making plots later
        with open(save_dir + 'train_out.txt', 'w+') as out_file:
            for e in range(epochs): # iterate
                    for idx in range(iterations):
                        '''
                        load image
                        feed to network, update network[training only]
                        print result
                        log result to out_file
                        '''
                        if(type_ == 'MNIST'):
                            images_in, batch_labels = mnist.train.next_batch(cfgs['data']['batch_size'])
                        else:
                            images_in, batch_labels = util.get_batch(cfgs, image_paths, train_ids, batch_size=batch_size, w=cfgs['data']['image_w'], h=cfgs['data']['image_h'])

                        _, _loss, _acc, _rmse = sess.run([model.train_op, model.reduce_loss, model.accuracy, model.rmse], feed_dict={model.inputs: images_in, model.labels: batch_labels, model.learning_rate: base_lr})
                        train_results['epoch_{}'.format(e)]['acc'].append(_acc)
                        train_results['epoch_{}'.format(e)]['rmse'].append(_rmse)
                        text = '[T] Epoch [{}/{}] | [{}/{}] TRAINING loss : {:.4f} TRAINING accuracy : {:.4f}'.format(e,cfgs['train']['num_epochs']-1, idx, epoch_size, _loss, _acc)
                        print('\r' + text, end='')
                        out_file.write(text + '\n')
                        if(idx % save_interval == 0):
                            if(type_ == 'MNIST'):
                                images_in, batch_labels = mnist.train.next_batch(cfgs['data']['batch_size'])
                            else:
                                images_in, batch_labels = util.get_batch(cfgs, image_paths, train_ids, batch_size=batch_size, w=cfgs['data']['image_w'], h=cfgs['data']['image_h'])

                            _loss, _acc, _rmse = sess.run([model.reduce_loss, model.accuracy, model.rmse], feed_dict={model.inputs: images_in, model.labels: batch_labels})
                            
                            line = '[V] Epoch [{}/{}] | [{}/{}] VAL loss : {:.4f} VAL accuracy : {:.4f} VAL RMSE : {:.3f}'.format(e,cfgs['train']['num_epochs']-1, idx, epoch_size, _loss, _acc, _rmse)
                            cprint('\r ' + line, 'yellow', attrs=['bold'])
                            out_file.write('{}\n'.format(line))
                            val_accs.append(_acc)
                            val_rmses.append(_rmse)
            out_file.close() # CLOSE FILE
    # save network after completion
    saver = tf.train.Saver()
    saver.save(sess, save_dir + 'epoch_{}_model.ckpt'.format(e),global_step=epoch_size) # save our model state
    elapsed_time = time.time() - start_time
    time_out = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    sess.close()
    # get averages of training results, [not validation]
    train_acc = []
    train_rmse = []
    for idx in range(0,epochs):
        _acc = sum(train_results['epoch_{}'.format(idx)]['acc']) / len(train_results['epoch_{}'.format(idx)]['acc'])
        _rmse = sum(train_results['epoch_{}'.format(idx)]['rmse']) / len(train_results['epoch_{}'.format(idx)]['rmse'])
        train_acc.append(_acc)
        train_rmse.append(_rmse)

    elapsed_time = time.time() - start_time
    return elapsed_time, val_accs, val_rmses, train_acc, train_rmse
