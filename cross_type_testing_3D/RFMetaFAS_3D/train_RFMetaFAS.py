import random
import numpy as np
from configs.config import config
from datetime import datetime
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils.get_loader import get_dataset
from models.DG_ResNet18_Meta import DGFANet
from utils.evaluate import eval
from utils.utils import AverageMeter, adjust_learning_rate, mkdirs, Logger, save_checkpoint, time_to_str, zero_param_grad, accuracy
from timeit import default_timer as timer
from collections import OrderedDict
from tensorboardX import SummaryWriter

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
    src1_train_dataloader_real, src1_train_dataloader_fake, \
    src2_train_dataloader_real, src2_train_dataloader_fake, \
    src3_train_dataloader_real, src3_train_dataloader_fake, \
    src4_train_dataloader_real, src4_train_dataloader_fake, \
    dev_dataloader, tgt_dataloader = get_dataset()

    # 0:HTER, 1:ACER, 2:AUC
    best_model_ACC = 0.0
    best_model_HTER = 1.0
    best_model_AUC = 0.0

    # 0:loss, 1:top-1, 2:EER, 3:HTER, 4:AUC, 5:threshold
    valid_args = [np.inf, 0, 0, 0, 0, 0, 0]

    loss_classifier = AverageMeter()
    loss_depth = AverageMeter()
    classifer_top1 = AverageMeter()

    net = DGFANet().to(device)

    if(config.enable_tensorboard):
        tblogger = SummaryWriter(comment='_' + config.tgt_data)

    log = Logger()
    log.open(config.logs + config.tgt_data + '_log_train_' + config.model + '_RFMetaFAS.txt', mode='a')
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
        'mse_loss': nn.MSELoss().cuda(),
        'l1_loss': nn.L1Loss().cuda()
    }
    optimizer_dict = [
        {"params": filter(lambda p: p.requires_grad, net.parameters()), "lr": 0.01},
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
        checkpoint = torch.load(config.best_model_path + config.tgt_best_model_name)
        net.load_state_dict(checkpoint["state_dict"])
        epoch = checkpoint['epoch']
        print('\n**epoch: ',epoch)

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
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, epoch, init_param_lr)

        # ============ one batch extraction ============#
        src1_img_real, src1_depth_img_real, src1_label_real = src1_train_iter_real.next()
        src1_img_real = src1_img_real.cuda()
        src1_depth_img_real = src1_depth_img_real.cuda()
        src1_label_real = src1_label_real.cuda()

        src2_img_real, src2_depth_img_real, src2_label_real = src2_train_iter_real.next()
        src2_img_real = src2_img_real.cuda()
        src2_depth_img_real = src2_depth_img_real.cuda()
        src2_label_real = src2_label_real.cuda()

        src3_img_real, src3_depth_img_real, src3_label_real = src3_train_iter_real.next()
        src3_img_real = src3_img_real.cuda()
        src3_depth_img_real = src3_depth_img_real.cuda()
        src3_label_real = src3_label_real.cuda()

        src4_img_real, src4_depth_img_real, src4_label_real = src4_train_iter_real.next()
        src4_img_real = src4_img_real.cuda()
        src4_depth_img_real = src4_depth_img_real.cuda()
        src4_label_real = src4_label_real.cuda()

        src1_img_fake, src1_depth_img_fake, src1_label_fake = src1_train_iter_fake.next()
        src1_img_fake = src1_img_fake.cuda()
        src1_depth_img_fake = src1_depth_img_fake.cuda()
        src1_label_fake = src1_label_fake.cuda()

        src2_img_fake, src2_depth_img_fake, src2_label_fake = src2_train_iter_fake.next()
        src2_img_fake = src2_img_fake.cuda()
        src2_depth_img_fake = src2_depth_img_fake.cuda()
        src2_label_fake = src2_label_fake.cuda()

        src3_img_fake, src3_depth_img_fake, src3_label_fake = src3_train_iter_fake.next()
        src3_img_fake = src3_img_fake.cuda()
        src3_depth_img_fake = src3_depth_img_fake.cuda()
        src3_label_fake = src3_label_fake.cuda()

        src4_img_fake, src4_depth_img_fake, src4_label_fake = src4_train_iter_fake.next()
        src4_img_fake = src4_img_fake.cuda()
        src4_depth_img_fake = src4_depth_img_fake.cuda()
        src4_label_fake = src4_label_fake.cuda()

        # ============ one batch collection ============#
        domain_img1 = torch.cat([src1_img_real, src1_img_fake], dim=0)
        domain_depth_img1 = torch.cat([src1_depth_img_real, src1_depth_img_fake], dim=0)
        domain_label1 = torch.cat([src1_label_real, src1_label_fake], dim=0)

        domain_img2 = torch.cat([src2_img_real, src2_img_fake], dim=0)
        domain_depth_img2 = torch.cat([src2_depth_img_real, src2_depth_img_fake], dim=0)
        domain_label2 = torch.cat([src2_label_real, src2_label_fake], dim=0)

        domain_img3 = torch.cat([src3_img_real, src3_img_fake], dim=0)
        domain_depth_img3 = torch.cat([src3_depth_img_real, src3_depth_img_fake], dim=0)
        domain_label3 = torch.cat([src3_label_real, src3_label_fake], dim=0)

        domain_img4 = torch.cat([src4_img_real, src4_img_fake], dim=0)
        domain_depth_img4 = torch.cat([src4_depth_img_real, src4_depth_img_fake], dim=0)
        domain_label4 = torch.cat([src4_label_real, src4_label_fake], dim=0)

        # ============ doamin list augmentation ============#
        domain_img_list = [domain_img1, domain_img2, domain_img3, domain_img4]
        domain_depth_img_list = [domain_depth_img1, domain_depth_img2, domain_depth_img3, domain_depth_img4]
        domain_label_list = [domain_label1, domain_label2, domain_label3, domain_label4]

        domain_list = list(range(len(domain_img_list)))
        random.shuffle(domain_list)
        meta_train_list = domain_list[:3]
        meta_test_list = domain_list[3:]

        # ============ meta training ============#
        Loss_dep_train = 0.0
        Loss_cls_train = 0.0

        adapted_state_dicts = []

        for index in meta_train_list:
            img_meta = domain_img_list[index]
            depth_GT_meta = domain_depth_img_list[index]
            label_meta = domain_label_list[index]

            batchindex = list(range(len(img_meta)))
            random.shuffle(batchindex)

            img_random = img_meta[batchindex, :]
            depthGT_random = depth_GT_meta[batchindex, :]
            label_random = label_meta[batchindex]

            feature, cls_out, depth_map = net(img_random)
            cls_loss = criterion['softmax'](cls_out, label_random)
            depth_loss = criterion['mse_loss'](depth_map, depthGT_random)

            Loss_dep_train += depth_loss
            Loss_cls_train += cls_loss

            zero_param_grad(net.feature_embedder.parameters())
            grads_feature_embedder = torch.autograd.grad(cls_loss, net.feature_embedder.parameters(), create_graph=True)

            fast_weights_FeatEmbder = net.feature_embedder.cloned_state_dict()

            lr = 0.0001
            adapted_params = OrderedDict()
            for (key, val), grad in zip(net.feature_embedder.named_parameters(), grads_feature_embedder):
                adapted_params[key] = val - lr * grad
                fast_weights_FeatEmbder[key] = adapted_params[key]

            adapted_state_dicts.append(fast_weights_FeatEmbder)

        # ============ meta testing ============#
        Loss_dep_test = 0.0
        Loss_cls_test = 0.0
        index = meta_test_list[0]
        img_meta = domain_img_list[index]
        depth_GT_meta = domain_depth_img_list[index]
        label_meta = domain_label_list[index]

        batchindex = list(range(len(img_meta)))
        random.shuffle(batchindex)

        img_random = img_meta[batchindex, :]
        depthGT_random = depth_GT_meta[batchindex, :]
        label_random = label_meta[batchindex]

        feature, cls_out, depth_map = net(img_random)
        Loss_dep_test += criterion['mse_loss'](depth_map, depthGT_random)

        for n_src in range(len(meta_train_list)):
            a_dict = adapted_state_dicts[n_src]
            feature, cls_out, depth_map = net(img_random, a_dict)
            cls_loss = criterion['softmax'](cls_out, label_random)
            Loss_cls_test += cls_loss
            acc = accuracy(cls_out, label_random, topk=(1,))
            classifer_top1.update(acc[0])

        Loss_meta_train = Loss_cls_train + 5 * Loss_dep_train
        Loss_meta_test = Loss_cls_test + 5 * Loss_dep_test

        total_loss = Loss_meta_train + Loss_meta_test

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_classifier.update(Loss_cls_test.item())
        loss_depth.update(Loss_dep_test.item())

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

            if (config.enable_tensorboard):
                info = {
                    'Valid_loss': valid_args[0],
                    'Valid_top_1': valid_args[1],
                    'Valid_EER': valid_args[2],
                    'Valid_HTER': valid_args[3],
                    'Valid_AUC': valid_args[4],
                    'Valid_threshold': valid_args[5],
                    'Valid_ACC': valid_args[6]
                }
                for tag, value in info.items():
                    tblogger.add_scalar(tag, value, epoch)

            save_checkpoint(save_list, is_best, net)
            print('\r', end='', flush=True)
            log.write(
                '  %4.1f  |  %6.3f  %6.3f  %6.3f  %6.3f  %4.3f  |  %6.3f  %6.3f  |  %6.3f  %6.3f  %6.3f | %s'
                % (
                (iter_num+1) / iter_per_epoch,
                valid_args[0], valid_args[6], valid_args[3] * 100, valid_args[4] * 100, valid_args[5],
                loss_classifier.avg, classifer_top1.avg,
                float(best_model_ACC), float(best_model_HTER * 100), float(best_model_AUC * 100),
                time_to_str(timer() - start, 'min'),
                ))
            log.write('\n')
            time.sleep(0.01)
    if(config.enable_tensorboard):
        tblogger.close()

if __name__ == '__main__':
    train()










