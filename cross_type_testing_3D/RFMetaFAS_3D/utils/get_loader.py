import os
from configs.config import config
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils.dataset import YunpeiDataset
from utils.utils import sample_frames

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.cuda.manual_seed(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def get_dataset():
    print('Load Source Data')
    print('Source Data: ', config.src1_data)
    src1_train_data_fake = sample_frames(flag=0, num_frames=config.src1_train_num_frames, dataset_name=config.src1_data)
    src1_train_data_real = sample_frames(flag=1, num_frames=config.src1_train_num_frames, dataset_name=config.src1_data)
    print('Source Data: ', config.src2_data)
    src2_train_data_fake = sample_frames(flag=0, num_frames=config.src2_train_num_frames, dataset_name=config.src2_data)
    src2_train_data_real = sample_frames(flag=1, num_frames=config.src2_train_num_frames, dataset_name=config.src2_data)
    print('Source Data: ', config.src3_data)
    src3_train_data_fake = sample_frames(flag=0, num_frames=config.src3_train_num_frames, dataset_name=config.src3_data)
    src3_train_data_real = sample_frames(flag=1, num_frames=config.src3_train_num_frames, dataset_name=config.src3_data)

    print('Source Data: ', config.src4_data)
    src4_train_data_fake = sample_frames(flag=0, num_frames=config.src4_train_num_frames, dataset_name=config.src4_data)
    src4_train_data_real = sample_frames(flag=1, num_frames=config.src4_train_num_frames, dataset_name=config.src4_data)


    print('Load Dev Data')
    print('Dev Data: ', config.dev_data)
    dev_test_data = sample_frames(flag=2, num_frames=config.dev_test_num_frames, dataset_name=config.dev_data)

    print('Load Target Data')
    print('Target Data: ', config.tgt_data)
    tgt_test_data = sample_frames(flag=3, num_frames=config.tgt_test_num_frames, dataset_name=config.tgt_data)

    src1_train_dataloader_fake = DataLoader(YunpeiDataset(src1_train_data_fake, train=True),
                                            batch_size=config.batch_size, shuffle=True)
    src1_train_dataloader_real = DataLoader(YunpeiDataset(src1_train_data_real, train=True),
                                            batch_size=config.batch_size, shuffle=True)
    src2_train_dataloader_fake = DataLoader(YunpeiDataset(src2_train_data_fake, train=True),
                                            batch_size=config.batch_size, shuffle=True)
    src2_train_dataloader_real = DataLoader(YunpeiDataset(src2_train_data_real, train=True),
                                            batch_size=config.batch_size, shuffle=True)
    src3_train_dataloader_fake = DataLoader(YunpeiDataset(src3_train_data_fake, train=True),
                                            batch_size=config.batch_size, shuffle=True)
    src3_train_dataloader_real = DataLoader(YunpeiDataset(src3_train_data_real, train=True),
                                            batch_size=config.batch_size, shuffle=True)
    src4_train_dataloader_fake = DataLoader(YunpeiDataset(src4_train_data_fake, train=True),
                                            batch_size=config.batch_size, shuffle=True)
    src4_train_dataloader_real = DataLoader(YunpeiDataset(src4_train_data_real, train=True),
                                            batch_size=config.batch_size, shuffle=True)
    tgt_dataloader = DataLoader(YunpeiDataset(tgt_test_data, train=False), batch_size=config.batch_size, shuffle=False)
    dev_dataloader = DataLoader(YunpeiDataset(dev_test_data, train=False), batch_size=config.batch_size, shuffle=False)
    return src1_train_dataloader_fake, src1_train_dataloader_real, \
           src2_train_dataloader_fake, src2_train_dataloader_real, \
           src3_train_dataloader_fake, src3_train_dataloader_real, \
           src4_train_dataloader_fake, src4_train_dataloader_real, \
           dev_dataloader, tgt_dataloader








