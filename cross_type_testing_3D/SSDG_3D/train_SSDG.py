from utils.utils import save_checkpoint, AverageMeter, Logger, accuracy, mkdirs, adjust_learning_rate, time_to_str
from utils.evaluate import eval
from utils.get_loader import get_dataset
from models.DG_ResNet18 import DG_model, Discriminator
from loss.hard_triplet_loss import HardTripletLoss
from loss.AdLoss import Real_AdLoss
import random
import numpy as np
from configs.config import config
from datetime import datetime
import time
from timeit import default_timer as timer

import os
import torch
import torch.nn as nn
import torch.optim as optim 

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.cuda.manual_seed(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = 'cuda'

def train():
    mkdirs()
    src1_train_dataloader_fake, src1_train_dataloader_real, \
    src2_train_dataloader_fake, src2_train_dataloader_real, \
    src3_train_dataloader_fake, src3_train_dataloader_real, \
    src4_train_dataloader_fake, src4_train_dataloader_real, \
    dev_dataloader, tgt_dataloader = get_dataset()

    best_model_ACC = 0.0
    best_model_HTER = 1.0
    best_model_AUC = 0.0

    # 0:loss, 1:top-1, 2:EER, 3:HTER, 4:AUC, 5:threshold
    valid_args = [np.inf, 0, 0, 0, 0, 0, 0]

    loss_classifier = AverageMeter()
    classifer_top1 = AverageMeter()

    net = DG_model().to(device)
    ad_net_real = Discriminator().to(device)

    log = Logger()
    log.open(config.logs + config.tgt_data + '_log_train_' + config.model + '_SSDG.txt', mode='a')
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
    log.write('** start training target model! **\n')
    log.write(
        '--------|---------------- VALID ------------------|--- classifier ---|------- Current Best -----|--------------|\n')
    log.write(
        '  iter  |   loss    top-1   HTER     AUC    thr   |   loss   top-1   |   top-1   HTER    AUC    |    time      |\n')
    log.write(
        '---------------------------------------------------------------------------------------------------------------|\n')
    start = timer()
    criterion = {
        'softmax': nn.CrossEntropyLoss().cuda(),
        'triplet': HardTripletLoss(margin=0.1, hardest=False).cuda()
    }
    optimizer_dict = [
        {"params": filter(lambda p: p.requires_grad, net.parameters()), "lr": 0.01},
        {"params": filter(lambda p: p.requires_grad, ad_net_real.parameters()), "lr": 0.01},
    ]
    optimizer = optim.SGD(optimizer_dict, lr=0.1, momentum=0.9, weight_decay=5e-4)
    init_param_lr = []
    for param_group in optimizer.param_groups:
        init_param_lr.append(param_group["lr"])

    iter_per_epoch = 10

    src1_train_iter_real = iter(src1_train_dataloader_real)
    src1_iter_per_epoch_real = len(src1_train_iter_real)
    src2_train_iter_real = iter(src2_train_dataloader_real)
    src2_iter_per_epoch_real = len(src2_train_iter_real)
    src3_train_iter_real = iter(src3_train_dataloader_real)
    src3_iter_per_epoch_real = len(src3_train_iter_real)
    src4_train_iter_real = iter(src4_train_dataloader_real)
    src4_iter_per_epoch_real = len(src4_train_iter_real)
    src1_train_iter_fake = iter(src1_train_dataloader_fake)
    src1_iter_per_epoch_fake = len(src1_train_iter_fake)
    src2_train_iter_fake = iter(src2_train_dataloader_fake)
    src2_iter_per_epoch_fake = len(src2_train_iter_fake)
    src3_train_iter_fake = iter(src3_train_dataloader_fake)
    src3_iter_per_epoch_fake = len(src3_train_iter_fake)
    src4_train_iter_fake = iter(src4_train_dataloader_fake)
    src4_iter_per_epoch_fake = len(src4_train_iter_fake)

    max_iter = config.max_iter
    epoch = 1
    if (config.resume == True):
        checkpoint = torch.load(config.tgt_best_model_name)
        net.load_state_dict(checkpoint["state_dict"])
        epoch = checkpoint['epoch']
        print('\n**epoch: ', epoch)

    for iter_num in range(max_iter+1):
        if (iter_num % src1_iter_per_epoch_real == 0):
            src1_train_iter_real = iter(src1_train_dataloader_real)
        if (iter_num % src2_iter_per_epoch_real == 0):
            src2_train_iter_real = iter(src2_train_dataloader_real)
        if (iter_num % src3_iter_per_epoch_real == 0):
            src3_train_iter_real = iter(src3_train_dataloader_real)
        if (iter_num % src4_iter_per_epoch_real == 0):
            src4_train_iter_real = iter(src4_train_dataloader_real)
        if (iter_num % src1_iter_per_epoch_fake == 0):
            src1_train_iter_fake = iter(src1_train_dataloader_fake)
        if (iter_num % src2_iter_per_epoch_fake == 0):
            src2_train_iter_fake = iter(src2_train_dataloader_fake)
        if (iter_num % src3_iter_per_epoch_fake == 0):
            src3_train_iter_fake = iter(src3_train_dataloader_fake)
        if (iter_num % src4_iter_per_epoch_fake == 0):
            src4_train_iter_fake = iter(src4_train_dataloader_fake)
        if (iter_num != 0 and iter_num % iter_per_epoch == 0):
            epoch = epoch + 1
        param_lr_tmp = []
        for param_group in optimizer.param_groups:
            param_lr_tmp.append(param_group["lr"])

        net.train(True)
        ad_net_real.train(True)
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, epoch, init_param_lr)

        src1_img_real, src1_depth_img_real, src1_label_real = src1_train_iter_real.next()
        src1_img_real = src1_img_real.cuda()
        src1_depth_img_real = src1_depth_img_real.cuda()
        src1_label_real = src1_label_real.cuda()
        input1_real_shape = src1_img_real.shape[0]

        src2_img_real, src2_depth_img_real, src2_label_real = src2_train_iter_real.next()
        src2_img_real = src2_img_real.cuda()
        src2_depth_img_real = src2_depth_img_real.cuda()
        src2_label_real = src2_label_real.cuda()
        input2_real_shape = src2_img_real.shape[0]

        src3_img_real, src3_depth_img_real, src3_label_real = src3_train_iter_real.next()
        src3_img_real = src3_img_real.cuda()
        src3_depth_img_real = src3_depth_img_real.cuda()
        src3_label_real = src3_label_real.cuda()
        input3_real_shape = src3_img_real.shape[0]

        src4_img_real, src4_depth_img_real, src4_label_real = src4_train_iter_real.next()
        src4_img_real = src4_img_real.cuda()
        src4_depth_img_real = src4_depth_img_real.cuda()
        src4_label_real = src4_label_real.cuda()
        input4_real_shape = src4_img_real.shape[0]

        src1_img_fake, src1_depth_img_fake, src1_label_fake = src1_train_iter_fake.next()
        src1_img_fake = src1_img_fake.cuda()
        src1_depth_img_fake = src1_depth_img_fake.cuda()
        src1_label_fake = src1_label_fake.cuda()
        input1_fake_shape = src1_img_fake.shape[0]

        src2_img_fake, src2_depth_img_fake, src2_label_fake = src2_train_iter_fake.next()
        src2_img_fake = src2_img_fake.cuda()
        src2_depth_img_fake = src2_depth_img_fake.cuda()
        src2_label_fake = src2_label_fake.cuda()
        input2_fake_shape = src2_img_fake.shape[0]

        src3_img_fake, src3_depth_img_fake, src3_label_fake = src3_train_iter_fake.next()
        src3_img_fake = src3_img_fake.cuda()
        src3_depth_img_fake = src3_depth_img_fake.cuda()
        src3_label_fake = src3_label_fake.cuda()
        input3_fake_shape = src3_img_fake.shape[0]

        src4_img_fake, src4_depth_img_fake, src4_label_fake = src4_train_iter_fake.next()
        src4_img_fake = src4_img_fake.cuda()
        src4_depth_img_fake = src4_depth_img_fake.cuda()
        src4_label_fake = src4_label_fake.cuda()
        input4_fake_shape = src4_img_fake.shape[0]

        input_data = torch.cat([src1_img_real, src2_img_real, src3_img_real, src4_img_real, src1_img_fake, src2_img_fake, src3_img_fake, src4_img_fake], dim=0)
        source_label = torch.cat([src1_label_real, src2_label_real, src3_label_real, src4_label_real,
                                  src1_label_fake, src2_label_fake, src3_label_fake, src4_label_fake], dim=0)

        classifier_label_out, feature = net(input_data)
        feature_real = feature.narrow(0, 0, input1_real_shape+input2_real_shape+input3_real_shape+input4_real_shape)
        discriminator_out_real = ad_net_real(feature_real)

        real_domain_label_1 = torch.LongTensor(input1_real_shape, 1).fill_(1).cuda()
        real_domain_label_2 = torch.LongTensor(input2_real_shape, 1).fill_(1).cuda()
        real_domain_label_3 = torch.LongTensor(input3_real_shape, 1).fill_(1).cuda()
        real_domain_label_4 = torch.LongTensor(input4_real_shape, 1).fill_(1).cuda()
        fake_domain_label_1 = torch.LongTensor(input1_fake_shape, 1).fill_(0).cuda()
        fake_domain_label_2 = torch.LongTensor(input2_fake_shape, 1).fill_(2).cuda()
        fake_domain_label_3 = torch.LongTensor(input3_fake_shape, 1).fill_(3).cuda()
        fake_domain_label_4 = torch.LongTensor(input4_fake_shape, 1).fill_(4).cuda()
        source_domain_label = torch.cat([real_domain_label_1, real_domain_label_2, real_domain_label_3, real_domain_label_4,
                                         fake_domain_label_1, fake_domain_label_2, fake_domain_label_3, fake_domain_label_4], dim=0).view(-1)
        triplet = criterion["triplet"](feature, source_domain_label)


        real_shape_list = []
        real_shape_list.append(input1_real_shape)
        real_shape_list.append(input2_real_shape)
        real_shape_list.append(input3_real_shape)
        real_shape_list.append(input4_real_shape)
        real_adloss = Real_AdLoss(discriminator_out_real, criterion["softmax"], real_shape_list)

        cls_loss = criterion["softmax"](classifier_label_out.narrow(0, 0, input_data.size(0)), source_label)

        total_loss = 1 * cls_loss + 1 * triplet  + 0.5 * real_adloss 
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_classifier.update(cls_loss.item())
        acc = accuracy(classifier_label_out.narrow(0, 0, input_data.size(0)), source_label, topk=(1,))
        classifer_top1.update(acc[0])

        print('\r', end='', flush=True)
        print(
            '  %4.1f  |  %6.3f  %6.3f  %6.3f  %6.3f  %4.3f  |  %6.3f  %6.3f  |  %6.3f  %6.3f  %6.3f | %s'
            % (
                (iter_num+1) / iter_per_epoch,
                valid_args[0], valid_args[6], valid_args[3] * 100, valid_args[4] * 100, valid_args[5],
                loss_classifier.avg, classifer_top1.avg,
                float(best_model_ACC), float(best_model_HTER * 100), float(best_model_AUC * 100),
                time_to_str(timer() - start, 'min'))
            , end='', flush=True)

        if (iter_num != 0 and (iter_num+1) % iter_per_epoch == 0):
            # 0:loss, 1:top-1, 2:EER, 3:HTER， 4:AUC, 5:threshold, 6:acc_at_threshold
            valid_args = eval(dev_dataloader, net)
            # judge model according to HTER
            is_best = valid_args[3] <= best_model_HTER
            best_model_HTER = min(valid_args[3], best_model_HTER)
            threshold = valid_args[5]
            if (valid_args[3] <= best_model_HTER):
                best_model_ACC = valid_args[6]
                best_model_AUC = valid_args[4]
            save_list = [epoch, valid_args, best_model_HTER, best_model_ACC, threshold]

            save_checkpoint(save_list, is_best, net)
            print('\r', end='', flush=True)
            log.write(
                '  %4.1f  |  %6.3f  %6.3f  %6.3f  %6.3f  %4.3f  |  %6.3f  %6.3f  |  %6.3f  %6.3f  %6.3f | %s'
                % (
                (iter_num+1) / iter_per_epoch,
                valid_args[0], valid_args[6], valid_args[3] * 100, valid_args[4] * 100, valid_args[5],
                loss_classifier.avg, classifer_top1.avg,
                float(best_model_ACC), float(best_model_HTER * 100), float(best_model_AUC * 100),
                time_to_str(timer() - start, 'min')
                ))
            log.write('\n')
            time.sleep(0.01)

if __name__ == '__main__':
    train()
