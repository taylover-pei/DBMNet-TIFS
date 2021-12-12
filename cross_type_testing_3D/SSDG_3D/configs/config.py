class DefaultConfigs(object):
    # actual parameters
    seed = 666
    weight_decay = 1e-4
    num_classes = 2
    resume = False
    pretrained = True
    model = 'resnet18'
    # hyper parameters
    gpus = "0"
    batch_size = 10
    norm_flag = True
    max_iter = 4000
    is_hsv = False
    enable_tensorboard = True

    tgt_best_model_name = 'model_best_0.19175_2.pth.tar'

    # source data information
    src1_data = 'oulu'
    src1_train_num_frames = 1

    src2_data = 'replay'
    src2_train_num_frames = 1

    src3_data = 'msu'
    src3_train_num_frames = 1

    src4_data = 'casia'
    src4_train_num_frames = 1

    # target data information
    tgt_data = 'hkbu'
    tgt_test_num_frames = 2

    # dev data
    dev_data = 'merged'
    dev_test_num_frames = 1

    # paths information
    checkpoint_path = './' + tgt_data + '_checkpoint/' + model + '/DGFANet/'
    best_model_path = './' + tgt_data + '_checkpoint/' + model + '/best_model/'
    logs = './logs/'

config = DefaultConfigs()
