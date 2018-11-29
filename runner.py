import yaml
import os
import time
import tensorflow as tf
import argparse
import sys
from termcolor import colored, cprint

# my imports
from src.train import train
from src.test import test
from src.config import make_config_file
from src.downloader import get_cifar_10
import src.util as util
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# This will call all types of test runs for our testing only
# - There will be a custom runner definition for single runs as well
# - This will use the run_single function as a helper
#
# Only has to make small changes to run multiple test cases
def run_all(cfgs, py = 'python3'):
    cprint('Starting up...', 'magenta', attrs=['bold'])
    # base runner
    run_baseline(cfgs, py=py)

    # runner for MNIST
    waves = ['haar', 'db2', 'db3', 'db4']
    nets = ['sketch_1', 'sketch_2', 'sketch_3', 'sketch_4', 'sketch_5', 'sketch_6']
    pools = ['wave', 'avg', 'max']
    # run every pooling method
    for pool in pools:
        # run every wavelet
        for w in waves:
            # run every net
            for net in nets:
                cmd = '{} runner.py --model_type={} --arch=3 \
		        --conv_layers=2 --wavelet={} --pooling={} --dataset=MNIST \
		        --epochs=15 --batch_size=100'.format(py, net,w, pool)
                os.system(cmd)
                time.sleep(2)#'''

    # runner for CIFAR 9/23/18 error where images for testing did not load right
    for pool in pools:
        # run every wavelet
        for w in waves:
            # run every net
            for net in nets:
                cmd = '{} runner.py --model_type={} --arch=4 \
		        --conv_layers=3 --wavelet={} --pooling={} --dataset=cifar-10 \
		        --epochs=30 --batch_size=100 --image_w=64 --image_h=64'.format(py, net,w, pool)
                os.system(cmd)
                time.sleep(2)#'''

# end of that guy
def run_baseline(cfgs, py = 'python'):
    pools = ['avg', 'max', 'wave']
    waves = ['haar', 'db2', 'db3', 'db4']
    nets = ['base']
    # run every pooling method
    for pool in pools:
        # run every wavelet
        for w in waves:
            # run every net
            for net in nets:
                if((pool == 'avg' or pool == 'max') and (w == 'haar')):
                    cmd = '{} runner.py --model_type={} --arch=3 \
                    --conv_layers=2 --wavelet={} --pooling={} --dataset=MNIST \
                    --epochs=15 --batch_size=100'.format(py, net,w, pool)
                    os.system(cmd)
                elif pool == 'wave':
                    cmd = '{} runner.py --model_type={} --arch=3 \
                    --conv_layers=2 --wavelet={} --pooling={} --dataset=MNIST \
                    --epochs=15 --batch_size=100'.format(py, net,w, pool)
                    os.system(cmd)
                time.sleep(2)#'''
    # runner for CIFAR
    for pool in pools:
        # run every wavelet
        for w in waves:
            # run every net
            for net in nets:
                if((pool == 'avg' or pool == 'max') and (w == 'haar')):
                    cmd = '{} runner.py --model_type={}\
                    --wavelet={} --pooling={} --dataset=cifar-10 \
                    --epochs=30 --batch_size=100'.format(py, net,w, pool)
                    os.system(cmd)
                elif pool == 'wave':
                    cmd = '{} runner.py --model_type={} \
                    --wavelet={} --pooling={} --dataset=cifar-10 \
                    --epochs=30 --batch_size=100'.format(py, net,w, pool)
                os.system(cmd)
                time.sleep(2)

# This can be customized by either a config file or params
def run_single(cfgs):
    #util.send_email_update('jay4842@gmail.com', 'test', subject='network test complete',importance='low')
    #input('->')
    cprint('Starting up...', 'magenta', attrs=['bold'])
    # set up some config stuff for the run
    type_ = cfgs['data']['type']
    pooling = cfgs['model']['pooling']
    fc = cfgs['model']['fc']
    arch = cfgs['model']['arch']
    conv_layers = cfgs['model']['conv_layers']
    model_type = cfgs['model']['model_type']

    # This guy will first call a training flow
    # - That will return accs and rmse for validations
    save_ = '{}/{}/{}/{}_{}/'.format(cfgs['save_dir'], type_, pooling, model_type, cfgs['model']['wavelet']) 
    cprint('\n' + '{}'.format(save_), 'green', attrs=['bold'])
    os.makedirs(save_, exist_ok=True)

    time_out, acc_out, rmse_out, train_acc_out, train_rmse_out = train(cfgs=cfgs, save_dir=save_) 
    # After that run a testing run using the model save from the previous train
    out_accuracy, out_rmse = test(cfgs=cfgs, save_dir=save_)
    # format our time
    time_out = time.strftime("%H:%M:%S", time.gmtime(time_out))
    cprint('Accuracy:  {:.4f} RMSE: {:.4f}  Time:  {}'.format(out_accuracy, out_rmse, time_out), 'magenta', attrs=['bold'])
    lines = ''
    with open(save_ + 'output.txt', 'w') as file:
        line = 'Accuracy:  {:.4f} RMSE: {:.4f}  Time:  {}\n'.format(out_accuracy, out_rmse, time_out)
        file.write(line)
        lines += line
        # Write the _rmse and acc scores during validation
        file.write('Validation Accuracy scores:\n[')
        for line in range(len(acc_out)):
            file.write('{}'.format(acc_out[line]))
            if(line < len(acc_out) - 1):
                file.write(',')
        file.write(']\n')

        file.write('Validation RMSE scores:\n[')
        for line in range(len(rmse_out)):
            file.write('{}'.format(rmse_out[line]))
            if(line < len(rmse_out) - 1):
                file.write(',')
        file.write(']\n')
        # training results
        file.write('Training Accuracy scores:\n[')
        for line in range(len(train_acc_out)):
            file.write('{}'.format(train_acc_out[line]))
            if(line < len(train_acc_out) - 1):
                file.write(',')
        file.write(']\n')

        file.write('Training RMSE scores:\n[')
        for line in range(len(train_rmse_out)):
            file.write('{}'.format(train_rmse_out[line]))
            if(line < len(train_rmse_out) - 1):
                file.write(',')
        file.write(']\n')

        file.close()

        #util.send_email_update('jay4842@gmail.com', lines, subject='{} {} complete'.format(type_, model_type),importance='low')
    
    return out_accuracy, out_rmse, time_out
    # Print and return
# Done
 
# 
# Args here, It is a lot but there is a discription for each parameter that can be used.
# - If you prefer using a config file, call the program like so:
#   > python runner.py --use_cfg_file=True
parser = argparse.ArgumentParser(description='')
parser.add_argument('--runs', dest='runs', type=int ,default=50, help='The number of train test runs.')
parser.add_argument('--epochs', dest='epochs', type=int, default=1, help='The number of epochs preformed while training [For small datasets like mnist and cifar-10 use 1]')
parser.add_argument('--dataset', dest='dataset', default='MNIST', help='Which dataset you want to use for the current run. [MNIST, CIFAR-10, IMAGE_NET]')
parser.add_argument('--model_type', dest='model_type', default=' ', help='Which type of network to be used. [vgg16, simple, ResNet, sketch_1, sketch_2, sketch_3, sketch_4, sketch_5, sketch_6]')
parser.add_argument('--pooling', dest='pooling', default='wave', help='Which type of pooling to use [wave, max, avg]')
parser.add_argument('--wavelet', dest='wavelet', default='haar', help='Which wavelet type to use [haar, db2, db3, db4]')
parser.add_argument('--start_features', dest='start_features', type=int, default=16, help='Some networks have customizable features at the start, set this to change that. [16, 32 have been tested]')
parser.add_argument('--conv_layers', dest='conv_layers', type=int, default=4, help='Sets how many convolutions are in a single block.')
parser.add_argument('--dropout', dest='dropout', type=float,default=.9, help='Sets the dropout rate in fully connected layers.')
parser.add_argument('--arch', dest='arch', type=int ,default=5, help='How many convolution blocks are in a network.')
parser.add_argument('--batch_size', dest='batch_size',type=int, default=50, help='Set the batch size for your model. [For wavelet pooling use 15 or below]')
parser.add_argument('--optimizer', dest='optimizer', default='adam', help='Set the type of optimizer. [adam, momentum, RMSProp, and gradientDescent]')
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.0001, help='Set the learning rate to be used by the optimizer.')
parser.add_argument('--iterations', dest='iterations', type=int, default=1000, help='Set how many iterations that are in one epoch.')
parser.add_argument('--gpu_limit', dest='gpu_limit',type=float, default=.5, help='Set how much GPU memory you would like to use. [.5 is recommended]')
parser.add_argument('--momentum', dest='momentum', type=float, default=.9, help='Sets the momentum when using the momentum optimizer or others that use the value.')
parser.add_argument('--image_w', dest='image_w', type=int ,default=32, help='What width you want to use for your images')
parser.add_argument('--image_h', dest='image_h', type=int ,default=32, help='What height you want to use for your images')
parser.add_argument('--num_channels', dest='num_channels', type=int, default=3, help='How many channels there are in our input images.')
parser.add_argument('--num_classes', dest='num_classes', type=int, default=10, help='Set how many classes there are in your dataset [used by fully connected layers]')
parser.add_argument('--standardize', dest='standardize', type=bool, default=False, help='Apply per image standardization or not in the data loader class. [does not apply to MNIST]')
parser.add_argument('--debug', dest='debug', type=bool, default=False, help='Turn general debugging on or off.')
parser.add_argument('--model_debug', dest='model_debug', type=bool, default=False, help='Turn model specific debugging on or off.')
parser.add_argument('--dataset_path', dest='dataset_path', default='', help='This is if the dataset you want to use is not local. [example imageNet will not be local]')
parser.add_argument('--download_dataset', dest='download_dataset', type=bool, default=False, help='Set to true if you would like to download the dataset, best used along with the --dataset flag.')
parser.add_argument('--use_cfg_file', dest='use_cfg_file', type=bool, default=False, help='If you want to use the config file instead of arguments set this flag to True.')
parser.add_argument('--run_mnist', dest='run_mnist', type=bool, default=False, help='If set to True this will run all the sketch networks on the mnist dataset using default configurations. [Best user with the --use_cfg_file flag]')
parser.add_argument('--run_cifar', dest='run_cifar', type=bool, default=False, help='If set to True this will run all the sketch networks on the cifar-10 dataset using default configurations. [Best user with the --use_cfg_file flag]')
parser.add_argument('--run_image_net', dest='run_image_net', type=bool, default=False, help='If set to True this will run all the sketch networks on the image_net_2012 dataset using default configurations. [Best user with the --use_cfg_file flag]')
parser.add_argument('--run_network_tests', dest='run_network_tests', type=bool, default=False, help='Run all tests using one network config.')
parser.add_argument('--run_baseline', dest='run_baseline', type=bool, default=False, help='Run baseline tests on CIFAR and MNIST')
args = parser.parse_args()

if __name__ == '__main__':
    cprint('Working...', 'cyan', attrs=['bold'])
    
    if not(os.path.exists('data/cifar-10/')):
        get_cifar_10()

    # If we want to load all of our configs from a yaml file
    if(args.use_cfg_file):
        file = open('config.yaml')
        cfgs = yaml.load(file)
        file.close()
    # or if we just want to make it using our args
    else:
        cfgs = make_config_file(args)

    #print(args)
    #print('\n{}'.format(cfgs))
    #input('->')
    # Runners here
    if(args.run_network_tests):
        run_all(cfgs)
    
    elif(args.run_baseline):
        run_baseline(cfgs)
    #
    elif not (cfgs['model']['model_type'] == ' '):
        run_single(cfgs)
    
    else:
        text_1 = 'You must assign a model type using the model flag:'
        text_2 = 'python wavelet_runner.py --model_type=sketch_1'
        text_3 = '\nFor more help use the -h flag.'
        cprint(text_1, 'yellow')
        cprint(text_2, 'yellow', attrs=['bold'])
        cprint(text_3, 'grey', attrs=['bold', 'underline'])