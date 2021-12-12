class DefaultConfigs(object):
    # actual parameters
    seed = 666
    weight_decay = 1e-4
    num_classes = 2
    resume = False
    pretrained = True
    model = 'resnet18'
    # hyper parameters
    gpus = "1"
    batch_size = 10
    max_iter = 4000
    is_hsv = False
    depth_size = 64
    enable_tensorboard = True

    tgt_best_model_name = 'RFMetaFAS_ICM_to_O.pth.tar'

    # source data information
    src1_data = 'casia'
    src1_train_num_frames = 1

    src2_data = 'replay'
    src2_train_num_frames = 1

    src3_data = 'msu'
    src3_train_num_frames = 1

    # target data information
    tgt_data = 'oulu'
    tgt_test_num_frames = 2

    # dev data information
    dev_data = 'merged'
    dev_test_num_frames = 1

    # paths information
    checkpoint_path = './' + tgt_data + '_checkpoint/' + model + '/DGFANet/'
    best_model_path = './' + tgt_data + '_checkpoint/' + model + '/best_model/'
    logs = './logs/'

config = DefaultConfigs()
