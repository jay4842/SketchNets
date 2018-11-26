# make our config file based on the arguments
def make_config_file(args):
    sub_cfgs = {}
    # first the base configs
    sub_cfgs['runs'] = args.runs
    sub_cfgs['gpu_limit'] = args.gpu_limit
    sub_cfgs['debug'] = args.debug
    sub_cfgs['model_debug'] = args.model_debug
    sub_cfgs['save_dir'] = 'final_model_saves'
    # data configs
    sub_cfgs['data'] = {}
    sub_cfgs['data']['data_start'] = 'data/' + args.dataset + '/' if (args.dataset_path == '') else args.dataset_path
    sub_cfgs['data']['type'] = args.dataset
    sub_cfgs['data']['image_w'] = args.image_w
    sub_cfgs['data']['image_h'] = args.image_h
    sub_cfgs['data']['label_seperator'] = '+' # dont make this an argument just yet
    sub_cfgs['data']['image_seperator'] = '!'
    sub_cfgs['data']['num_channels'] = args.num_channels
    sub_cfgs['data']['num_classes'] = args.num_classes
    sub_cfgs['data']['train_split'] =.7 # This is always the split to use
    sub_cfgs['data']['batch_size'] = args.batch_size
    sub_cfgs['data']['classes'] = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    sub_cfgs['data']['root_paths'] = True
    sub_cfgs['data']['aug'] = False
    sub_cfgs['data']['standardize'] = args.standardize
    sub_cfgs['data']['list'] = {}
    sub_cfgs['data']['list']['train'] = sub_cfgs['data']['data_start'] + 'train.lst'
    sub_cfgs['data']['list']['test'] = sub_cfgs['data']['data_start'] + 'test.lst'

    # model configs
    sub_cfgs['model'] = {}
    sub_cfgs['model']['model_type'] = args.model_type
    sub_cfgs['model']['pooling'] = args.pooling
    sub_cfgs['model']['wavelet'] = args.wavelet
    sub_cfgs['model']['dropout'] = args.dropout
    sub_cfgs['model']['save'] = True # save output or not
    sub_cfgs['model']['start_features'] = args.start_features
    sub_cfgs['model']['arch'] = args.arch
    sub_cfgs['model']['conv_layers'] = args.conv_layers
    sub_cfgs['model']['fc'] = 2 # This is harder to customize

    # finally train specific stuff
    sub_cfgs['train'] = {}
    sub_cfgs['train']['optimizer'] = args.optimizer
    sub_cfgs['train']['learning_rate'] = args.learning_rate
    sub_cfgs['train']['momentum'] = args.momentum
    sub_cfgs['train']['save_interval'] = 100
    sub_cfgs['train']['iterations'] = args.iterations
    sub_cfgs['train']['num_epochs'] = args.epochs

    return sub_cfgs
