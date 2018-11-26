import tensorflow as tf
import numpy as np
from glob import glob
import time
from termcolor import colored, cprint
from tensorflow.examples.tutorials.mnist import input_data
import yaml
import os

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
            if(model_type == 'sketch_1'):
                print('sketch_1')

            if(model_type == 'sketch_2'):
                print('sketch_2')
                    
            if(model_type == 'sketch_3'):
                print('sketch_3')
                    
            if(model_type == 'sketch_4'):
                print('sketch_4')
                    
            if(model_type == 'sketch_5'):
                print('sketch_5')

            if(model_type == 'sketch_6'):
                print('sketch_6')
                
            if(model_type == 'base'):
                print('base')
        
        # set up the input tensors
        with tf.variable_scope('input'):
            # other images
            image_size = [cfgs['data']['image_w'],cfgs['data']['image_h']]

            #mnist
            mnist = None
            if(type_ == 'MNIST'):
                iterations = 60000 // cfgs['data']['batch_size']
                mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
                x_ = tf.placeholder(tf.float32, [None, 784], name='image')
                labels = tf.placeholder(tf.float32, [None, 10], name='label')
            else:
                iterations = 50000 // cfgs['data']['batch_size']
                x_ = tf.placeholder(tf.float32,[None, image_size[1], image_size[0], cfgs['data']['num_channels']],name='image') #
                labels = tf.placeholder(tf.float32, [None, cfgs['data']['num_classes']] ,name='label')
		
        epoch_size = iterations

        # loss section
        with tf.variable_scope('loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.to_int64(labels), logits=[probs])
            arg_logit = tf.argmax(probs, -1)
            arg_label = tf.argmax(tf.to_int64(labels),-1)
            correct = tf.equal(arg_logit, arg_label)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            _op, rmse = tf.metrics.mean_squared_error(arg_label, arg_logit)

            reduce_loss = tf.reduce_mean(loss)
            tf.summary.scalar('total_loss', reduce_loss)
            tf.summary.scalar('accuracy', accuracy)

        # optimizer
        with tf.variable_scope('train'):
            # first get our optimizer
            learning_rate = tf.placeholder('float', [])
            opt = util.getOptimizer(cfgs, learning_rate)
            train_op = opt.minimize(reduce_loss)
    
        merged_summary = tf.summary.merge_all() # tensor board summary
    if(cfgs['debug'] or cfgs['model_debug']): input('->')
    save_interval = cfgs['train']['save_interval'] # set out save interval
    #if(save_interval == -1): 
    #    save_interval =  int(epoch_size/5)# this is a placeholder...
    save_interval = iterations
    sess = util.get_session(cfgs['gpu_limit'])

    # setup our image paths if we are not using MNIST
    if not(cfgs['data']['type'] == 'MNIST'):
        image_paths = glob(cfgs['data']['data_start']+'train/*.png')
        train_ids = list(range(0,len(image_paths)))
        np.random.shuffle(train_ids)

    cprint('training....', 'green', attrs=['bold'])
    cprint('Save:      {}'.format(save_dir), 'cyan', attrs=['bold'])
    cprint('Dataset:   {}'.format(type_), 'green', attrs=['underline', 'bold'])
    print('Model:      {}'.format(colored('{}'.format(model_type), 'yellow') ))
    print('arch:       {}'.format(colored('{}'.format(arch), 'yellow')))
    print('convs:      {}'.format(colored('{}'.format(conv_layers), 'yellow')))
    print('wavelet:    {}'.format(colored('{}'.format(cfgs['model']['wavelet']), 'yellow')))
    print('pooling:    {}'.format(colored('{}'.format(pooling), 'yellow')))
    print('epochs:     {}'.format(colored('{}'.format(epochs), 'yellow')))
    print('iterations: {}'.format(colored('{}'.format(iterations), 'yellow')))
    print('batch size: {}\n'.format(colored('{}'.format(batch_size), 'yellow')))

    with sess:
        start_time = time.time()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print('training...')
        class_names = cfgs['data']['classes']
        base_lr = cfgs['train']['learning_rate']
        with open(save_dir + 'train_out.txt', 'w+') as out_file:
            for e in range(epochs):
                    validation_text = ''
                    for idx in range(iterations):
                        if(type_ == 'MNIST'):
                            images_in, batch_labels = mnist.train.next_batch(cfgs['data']['batch_size'])
                        else:
                            images_in, batch_labels = util.get_batch(cfgs, image_paths, train_ids, batch_size=batch_size, w=cfgs['data']['image_w'], h=cfgs['data']['image_h'])

                        _, _loss, _acc, _rmse, summ = sess.run([train_op, reduce_loss, accuracy, rmse, merged_summary], feed_dict={x_: images_in, labels: batch_labels, learning_rate: base_lr})
                        train_results['epoch_{}'.format(e)]['acc'].append(_acc)
                        train_results['epoch_{}'.format(e)]['rmse'].append(_rmse)
                        text = '\rEpoch [{}/{}] | [{}/{}] TRAINING loss : {:.4f} TRAINING accuracy : {:.4f}'.format(e,cfgs['train']['num_epochs']-1, idx, epoch_size, _loss, _acc)
                        print(text, end='')

                        if(idx % save_interval == 0):
                            if(type_ == 'MNIST'):
                                images_in, batch_labels = mnist.train.next_batch(cfgs['data']['batch_size'])
                            else:
                                images_in, batch_labels = util.get_batch(cfgs, image_paths, train_ids, batch_size=batch_size, w=cfgs['data']['image_w'], h=cfgs['data']['image_h'])

                            _loss, _acc, _rmse, summ = sess.run([reduce_loss, accuracy, rmse, merged_summary], feed_dict={x_: images_in, labels: batch_labels})
                            
                            line = 'Epoch [{}/{}] | [{}/{}] VAL loss : {:.4f} VAL accuracy : {:.4f} VAL RMSE : {:.3f}'.format(e,cfgs['train']['num_epochs']-1, idx, epoch_size, _loss, _acc, _rmse)
                            cprint('\r ' + line, 'yellow', attrs=['bold'])
                            out_file.write('{}\n'.format(line))
                            val_accs.append(_acc)
                            val_rmses.append(_rmse)
        
        saver = tf.train.Saver()
        saver.save(sess, save_dir + 'epoch_{}_model.ckpt'.format(e),global_step=epoch_size) # save our model state
        elapsed_time = time.time() - start_time
        time_out = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    
    out_file.close()
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