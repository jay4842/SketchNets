import tensorflow as tf

# This model skeleton will be used by other model types
# - define model input placeholder
# - define model output
# - define model loss
# - define model optimizer
class Model:
    # constructor
    def __init__(self, name='model_base'):
        self.name = name
        self.inputs = tf.placeholder(tf.float32, [None,None], name='inputs')
        self.labels = tf.placeholder(tf.float32, [None,None], name='labels')
    # defines self.output and layers in network
    def define_model(self, model_function):
        print('placeholder, define model here')
        self.output = model_function
    # placeholder to define loss and optimizer as well
    def define_loss(self):
        self.loss = 0
        self.opt = None # must be defined
        print('placeholder, define loss here. opt must be defined as well')
    