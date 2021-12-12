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
from models.DG_ResNet18_Meta_two import DGFANet
from utils.utils import AverageMeter, adjust_learning_rate, mkdirs, Logger, accuracy, save_checkpoint, time_to_str, zero_param_grad
from utils.evaluate import eval_two_branch
from loss.hard_triplet_loss import HardTripletLoss
from timeit import default_timer as timer
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
    src1_train_dataloader_fake, src1_train_dataloader_real, \
    src2_train_dataloader_fake, src2_train_dataloader_real, \
    src3_train_dataloader_fake, src3_train_dataloader_real, \
    dev_dataloader, tgt_dataloader, src1_valid_dataloader, src2_valid_dataloader, src3_valid_dataloader = get_dataset()

    best_model_ACC = 0.0
    best_model_HTER = 1.0
    best_model_AUC = 0.0

    # 0:loss, 1:top-1, 2:EER, 3:HTER, 4:AUC, 5:threshold
    valid_args = [np.inf, 0, 0, 0, 0, 0, 0]

    loss_classifier = AverageMeter()
    loss_triplet = AverageMeter()
    loss_depth = AverageMeter()
    classifer_top1 = AverageMeter()

    net = DGFANet().to(device)

    if(config.enable_tensorboard):
        tblogger = SummaryWriter(comment='_' + config.tgt_data)

    log = Logger()
    log.open(config.logs + config.tgt_data + '_log_train_' + config.model + '_DBMNet.txt', mode='a')
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
        'triplet': HardTripletLoss(margin=0.1, hardest=False).cuda(),
        'l1_loss': nn.L1Loss().cuda(),
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
    src1_train_iter_fake = iter(src1_train_dataloader_fake)
    src1_iter_per_epoch_fake = len(src1_train_iter_fake)
    src2_train_iter_fake = iter(src2_train_dataloader_fake)
    src2_iter_per_epoch_fake = len(src2_train_iter_fake)
    src3_train_iter_fake = iter(src3_train_dataloader_fake)
    src3_iter_per_epoch_fake = len(src3_train_iter_fake)
    src1_valid_iter = iter(src1_valid_dataloader)
    src2_valid_iter = iter(src2_valid_dataloader)
    src3_valid_iter = iter(src3_valid_dataloader)
    src1_valid, src1_valid_label, _, _ = src1_valid_iter.next()
    src1_valid = src1_valid.cuda()
    src1_valid_label = src1_valid_label.cuda()
    src2_valid, src2_valid_label, _, _ = src2_valid_iter.next()
    src2_valid = src2_valid.cuda()
    src2_valid_label = src2_valid_label.cuda()
    src3_valid, src3_valid_label, _, _ = src3_valid_iter.next()
    src3_valid = src3_valid.cuda()
    src3_valid_label = src3_valid_label.cuda()

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
        if (iter_num % src1_iter_per_epoch_fake == 0):
            src1_train_iter_fake = iter(src1_train_dataloader_fake)
        if (iter_num % src2_iter_per_epoch_fake == 0):
            src2_train_iter_fake = iter(src2_train_dataloader_fake)
        if (iter_num % src3_iter_per_epoch_fake == 0):
            src3_train_iter_fake = iter(src3_train_dataloader_fake)
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

        # ============ doamin list augmentation ============#
        domain_img_list = [domain_img1, domain_img2, domain_img3]
        domain_depth_img_list = [domain_depth_img1, domain_depth_img2, domain_depth_img3]
        domain_label_list = [domain_label1, domain_label2, domain_label3]

        domain_valid_list = [src1_valid, src2_valid, src3_valid]
        domain_valid_label_list = [src1_valid_label, src2_valid_label, src3_valid_label]

        domain_list = list(range(len(domain_img_list)))
        random.shuffle(domain_list)
        meta_train_list = domain_list[:2]
        meta_test_list = domain_list[2:]

        # ============ meta training ============#
        Loss_meta_train = 0.0
        Loss_domain_FeatEmbder = 0.0
        Loss_domain_DepthEstmator = 0.0

        adapted_state_dicts = []
        depth_adapted_state_dicts = []
        for index in meta_train_list:
            best_acc_featembdder = 0.0
            best_acc_depthestimator = 0.0
            # prepare data
            img_meta = domain_img_list[index]
            depth_GT_meta = domain_depth_img_list[index]
            label_meta = domain_label_list[index]
            batchindex = list(range(len(img_meta)))
            random.shuffle(batchindex)
            img_random_train = img_meta[batchindex, :]
            depthGT_random_train = depth_GT_meta[batchindex, :]
            label_random_train = label_meta[batchindex]   

            img_random_val = domain_valid_list[index]
            label_random_val = domain_valid_label_list[index]

            
            # forward
            feature, cls_out, depth_map, depth_cls_out = net(img_random_train)
            cls_loss = criterion['softmax'](cls_out, label_random_train)
            triplet_loss = criterion['triplet'](feature, label_random_train)
            depth_cls_loss = criterion['softmax'](depth_cls_out, label_random_train)
            depth_loss = criterion['mse_loss'](depth_map, depthGT_random_train)
            Loss_domain_FeatEmbder = config.lambda_cls * cls_loss + config.lambda_triplet * triplet_loss
            Loss_domain_DepthEstmator = config.lambda_depth_cls * depth_cls_loss + config.lambda_depth_reg * depth_loss

            lr = 0.0001
            # for feature embedder
            zero_param_grad(net.feature_embedder.parameters())
            grads_feature_embedder = torch.autograd.grad(config.lambda_cls * cls_loss, net.feature_embedder.parameters(), create_graph=True)
            fast_weights_FeatEmbder = net.feature_embedder.cloned_state_dict()
            fast_weights_FeatEmbder_new = list(map(lambda p: p[1] - lr * p[0], zip(grads_feature_embedder, net.feature_embedder.parameters())))
            for (key, val), param_new in zip(net.feature_embedder.named_parameters(), fast_weights_FeatEmbder_new):
                fast_weights_FeatEmbder[key] = param_new

            # for depth estimator
            zero_param_grad(net.depth_estimator.parameters())
            grads_depth_estmator = torch.autograd.grad(config.lambda_depth_cls * depth_cls_loss, net.depth_estimator.parameters(), create_graph=True)
            fast_weights_DepthEstmator = net.depth_estimator.cloned_state_dict()
            fast_weights_DepthEstmator_new = list(map(lambda p: p[1] - lr * p[0], zip(grads_depth_estmator, net.depth_estimator.parameters())))
            for (key, val), param_new in zip(net.depth_estimator.named_parameters(), fast_weights_DepthEstmator_new):
                fast_weights_DepthEstmator[key] = param_new

            ##### Evaluation to select #####
            net.eval()
            feature, cls_out, depth_map, depth_cls_out = net(img_random_val, fast_weights_FeatEmbder, fast_weights_DepthEstmator)
            net.train(True)
            acc_val_featembdder = accuracy(cls_out, label_random_val, topk=(1,))[0]
            if(acc_val_featembdder >= best_acc_featembdder):
                best_acc_featembdder = acc_val_featembdder
                fast_weights_FeatEmbder_final = fast_weights_FeatEmbder
                Loss_domain_FeatEmbder_final = Loss_domain_FeatEmbder
            acc_val_depthestimator = accuracy(depth_cls_out, label_random_val, topk=(1,))[0]
            if(acc_val_depthestimator >= best_acc_depthestimator):
                best_acc_depthestimator = acc_val_depthestimator
                fast_weights_DepthEstmator_final = fast_weights_DepthEstmator   
                Loss_domain_DepthEstmator_final = Loss_domain_DepthEstmator  
            for k in range(1, config.step):
                # forward
                feature, cls_out, depth_map, depth_cls_out = net(img_random_train, fast_weights_FeatEmbder, fast_weights_DepthEstmator)
                cls_loss = criterion['softmax'](cls_out, label_random_train)
                triplet_loss = criterion['triplet'](feature, label_random_train)
                depth_cls_loss = criterion['softmax'](depth_cls_out, label_random_train)
                depth_loss = criterion['mse_loss'](depth_map, depthGT_random_train)
                Loss_domain_FeatEmbder = config.lambda_cls * cls_loss + config.lambda_triplet * triplet_loss
                Loss_domain_DepthEstmator = config.lambda_depth_cls * depth_cls_loss + config.lambda_depth_reg * depth_loss

                # for feature embedder
                zero_param_grad(net.feature_embedder.parameters())
                grads_feature_embedder = torch.autograd.grad(config.lambda_cls * cls_loss, fast_weights_FeatEmbder_new, create_graph=True)
                fast_weights_FeatEmbder = net.feature_embedder.cloned_state_dict()
                fast_weights_FeatEmbder_new = list(map(lambda p: p[1] - lr * p[0], zip(grads_feature_embedder, fast_weights_FeatEmbder_new)))
                for (key, val), param_new in zip(net.feature_embedder.named_parameters(), fast_weights_FeatEmbder_new):
                    fast_weights_FeatEmbder[key] = param_new

                # for depth estimator
                zero_param_grad(net.depth_estimator.parameters())
                grads_depth_estmator = torch.autograd.grad(config.lambda_depth_cls * depth_cls_loss, fast_weights_DepthEstmator_new, create_graph=True)
                fast_weights_DepthEstmator = net.depth_estimator.cloned_state_dict()
                fast_weights_DepthEstmator_new = list(map(lambda p: p[1] - lr * p[0], zip(grads_depth_estmator, fast_weights_DepthEstmator_new)))
                for (key, val), param_new in zip(net.depth_estimator.named_parameters(), fast_weights_DepthEstmator_new):
                    fast_weights_DepthEstmator[key] = param_new

                ##### Evaluation to select #####
                net.eval()
                feature, cls_out, depth_map, depth_cls_out = net(img_random_val, fast_weights_FeatEmbder, fast_weights_DepthEstmator)
                net.train(True)
                acc_val_featembdder = accuracy(cls_out, label_random_val, topk=(1,))[0]
                if(acc_val_featembdder >= best_acc_featembdder):
                    best_acc_featembdder = acc_val_featembdder
                    fast_weights_FeatEmbder_final = fast_weights_FeatEmbder
                    Loss_domain_FeatEmbder_final = Loss_domain_FeatEmbder
                acc_val_depthestimator = accuracy(depth_cls_out, label_random_val, topk=(1,))[0]
                if(acc_val_depthestimator >= best_acc_depthestimator):
                    best_acc_depthestimator = acc_val_depthestimator
                    fast_weights_DepthEstmator_final = fast_weights_DepthEstmator   
                    Loss_domain_DepthEstmator_final = Loss_domain_DepthEstmator 
                    
            adapted_state_dicts.append(fast_weights_FeatEmbder_final)
            depth_adapted_state_dicts.append(fast_weights_DepthEstmator_final)
            Loss_meta_train += Loss_domain_FeatEmbder_final + Loss_domain_DepthEstmator_final
        # ============ meta testing ============#
        Loss_dep_test = 0.0
        Loss_dep_cls_test = 0.0
        Loss_cls_test = 0.0
        Loss_triplet_test = 0.0
        index = meta_test_list[0]
        img_meta = domain_img_list[index]
        depth_GT_meta = domain_depth_img_list[index]
        label_meta = domain_label_list[index]

        batchindex = list(range(len(img_meta)))
        random.shuffle(batchindex)

        img_random_test = img_meta[batchindex, :]
        depthGT_random_test = depth_GT_meta[batchindex, :]
        label_random_test = label_meta[batchindex]

        for n_src in range(len(meta_train_list)):

            a_dict_cls = adapted_state_dicts[n_src]
            a_dict_depth = depth_adapted_state_dicts[n_src]
            if (n_src == 0):
                index = meta_train_list[1]
            else:
                index = meta_train_list[0]
            img_meta_train = domain_img_list[index]
            label_meta_train = domain_label_list[index]
            depthGT_random_train = domain_depth_img_list[index]
            batchindex_train = list(range(len(img_meta_train)))
            random.shuffle(batchindex_train)
            img_random_train = img_meta_train[batchindex_train, :]
            label_random_train = label_meta_train[batchindex_train]
            depthGT_random_train = depthGT_random_train[batchindex_train, :]
            img_random = torch.cat([img_random_test, img_random_train], dim=0)
            label_random = torch.cat([label_random_test, label_random_train], dim=0)
            depthGT = torch.cat([depthGT_random_test, depthGT_random_train], dim=0)
            feature, cls_out, depth_map, depth_cls_out = net(img_random, a_dict_cls, a_dict_depth)

            # for cls branch
            cls_loss = criterion['softmax'](cls_out.narrow(0, 0, img_random_test.size(0)), label_random.narrow(0, 0, img_random_test.size(0)))
            Loss_cls_test += cls_loss
            Loss_triplet_test += criterion['triplet'](feature, label_random)
            # for depth branch
            depth_cls_loss = criterion['softmax'](depth_cls_out.narrow(0, 0, img_random_test.size(0)), label_random.narrow(0, 0, img_random_test.size(0)))
            Loss_dep_cls_test += depth_cls_loss
            depth_loss = criterion['mse_loss'](depth_map, depthGT)
            Loss_dep_test += depth_loss

            cls_out = (cls_out + depth_cls_out) / 2.0
            acc = accuracy(cls_out, label_random, topk=(1,))
            classifer_top1.update(acc[0])

        Loss_meta_test = config.lambda_cls * Loss_cls_test + config.lambda_triplet * Loss_triplet_test + config.lambda_depth_cls * Loss_dep_cls_test + config.lambda_depth_reg * Loss_dep_test 
        total_loss = Loss_meta_train + Loss_meta_test

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        Loss_cls_test_all = (Loss_cls_test+Loss_dep_cls_test) / 2.0
        loss_triplet.update(Loss_triplet_test.item())
        loss_classifier.update(Loss_cls_test_all.item())
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
            valid_args = eval_two_branch(dev_dataloader, net)
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
                time_to_str(timer() - start, 'min')
                ))
            log.write('\n')
            time.sleep(0.01)
    if(config.enable_tensorboard):
        tblogger.close()

if __name__ == '__main__':
    train()










