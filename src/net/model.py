import tensorflow as tf
import src.net.network_defs as nets
import src.util.model_helpers as mh
import src.util.util as util
# This model skeleton will be used by other model types
# - define model input placeholder
# - define model output
# - define model loss
# - define model optimizer
class Model:
    # constructor
    def __init__(self, cfgs, name='model_base'):
        self.cfgs = cfgs
        self.name = name
        self.num_classes = cfgs['data']['num_classes']
        self.image_size = [cfgs['data']['image_w'], cfgs['data']['image_h']]
        # image size now
        if(cfgs['data']['type'] == 'MNIST'):
            self.inputs = tf.placeholder(tf.float32, [None,784], name='inputs')
            self.labels = tf.placeholder(tf.float32, [None,10], name='labels')
        else:
            self.inputs = tf.placeholder(tf.float32,[None, self.image_size[1], self.image_size[0], cfgs['data']['num_channels']],name='inputs')
            self.labels = tf.placeholder(tf.float32, [None, cfgs['data']['num_classes']] ,name='labels')
        # set model function based on model type in cfgs, if model_function param in define model is given a function this will
        #  not be used.
        self.model_function = nets.get_model_function(self.cfgs['data']['type'], self.cfgs['model']['model_type'])
    # defines self.output and layers in network
    # - you only need to change the model_function param, this will allow you to use
    #   whatever model you want.
    # - also I want everything to be the same, so the only thing that will change is the model output
    def define_model(self, model_function=None):
        with tf.variable_scope('model'):
            self.model_def = self.model_function if model_function is None else model_function
            self.output = self.model_def(self.cfgs, self.inputs, self.image_size)
            print(self.output)
            _, self.probs = mh.inference(self.output, self.num_classes)

        with tf.variable_scope('loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.to_int64(self.labels), logits=[self.probs])
            arg_logit = tf.argmax(self.probs, -1)
            arg_label = tf.argmax(tf.to_int64(self.labels),-1)
            correct = tf.equal(arg_logit, arg_label)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            _op, self.rmse = tf.metrics.mean_squared_error(arg_label, arg_logit)

            self.reduce_loss = tf.reduce_mean(loss)
            
        with tf.variable_scope('train'):
            self.learning_rate = tf.placeholder('float', [])
            self.opt = util.getOptimizer(self.cfgs, self.learning_rate)
            self.train_op = self.opt.minimize(self.reduce_loss)
    # end of model