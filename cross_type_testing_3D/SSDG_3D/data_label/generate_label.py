import os
import json
import sys
import glob
import pandas as pd

data_dir = '$root/mtcnn_processed_data/'

def _3dmad_process():
    all_final_json = []
    label_save_dir = './3dmad/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_all = open(label_save_dir + 'all_label.json', 'w')
    dataset_path = data_dir + '3dmad_256/'
    path_list = glob.glob(dataset_path + '**/*.png', recursive=True)
    path_list.sort()
    for i in range(len(path_list)):
        flag = path_list[i].split('/')[-3]
        if (flag == 'session01' or flag == 'session02'):
            label = 1
        else:
            label = 0
        dict = {}
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label
        all_final_json.append(dict)
    print('\n3DMAD: ', len(path_list))
    print('3DMAD(all): ', len(all_final_json))
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()

def hkbu_process():
    all_final_json = []
    label_save_dir = './hkbu/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_all = open(label_save_dir + 'all_label.json', 'w')
    dataset_path = data_dir + 'hkbu_256/'
    path_list = glob.glob(dataset_path + '**/*.png', recursive=True)
    path_list.sort()
    for i in range(len(path_list)):
        flag = path_list[i].split('/')[-3]
        if (flag == 's1' or flag == 's2'):
            label = 1
        else:
            label = 0
        dict = {}
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label
        all_final_json.append(dict)
    print('\nHKBU_MARs: ', len(path_list))
    print('HKBU_MARs(all): ', len(all_final_json))
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()


def msu_process():
    test_list = []
    for line in open('/home/kai/shiliang/data/fas/mtcnn_dataset_preprocess/msu_256/test_sub_list.txt', 'r'):
        test_list.append(line[0:2])
    train_list = []
    for line in open('/home/kai/shiliang/data/fas/mtcnn_dataset_preprocess/msu_256/train_sub_list.txt', 'r'):
        train_list.append(line[0:2])
    valid_list = ['53', '54', '55']

    print(test_list)
    print(train_list)
    print(valid_list)
    dev_final_json = []
    valid_final_json = []
    real_final_json = []
    fake_final_json = []
    label_save_dir = './msu/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_dev = open(label_save_dir + 'dev_label.json', 'w')
    f_valid = open(label_save_dir + 'valid_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    dataset_path = data_dir + 'msu_256/'
    path_list = glob.glob(dataset_path + '**/*.png', recursive=True)
    path_list.sort()
    for i in range(len(path_list)):
        # print(path_list[i])
        flag = path_list[i].find('/real/')
        if(flag != -1):
            label = 1
        else:
            label = 0
        dict = {}
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label
        video_num = path_list[i].split('/')[-2].split('_')[0]
        if(video_num in valid_list):
            valid_final_json.append(dict)
        elif (video_num in train_list):
            if(label == 1):
                real_final_json.append(dict)
            else:
                fake_final_json.append(dict)
        else:
            dev_final_json.append(dict)
            
    print('\nMSU: ', len(path_list))
    print('MSU(dev): ', len(dev_final_json))
    print('MSU(valid): ', len(valid_final_json))
    print('MSU(real): ', len(real_final_json))
    print('MSU(fake): ', len(fake_final_json))
    json.dump(dev_final_json, f_dev, indent=4)
    f_dev.close()
    json.dump(valid_final_json, f_valid, indent=4)
    f_valid.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()


def casia_process():
    dev_final_json = []
    valid_final_json = []
    real_final_json = []
    fake_final_json = []
    label_save_dir = './casia/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_dev = open(label_save_dir + 'dev_label.json', 'w')
    f_valid = open(label_save_dir + 'valid_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    dataset_path = data_dir + 'casia_256/'
    path_list = glob.glob(dataset_path + '**/*.png', recursive=True)
    path_list.sort()
    for i in range(len(path_list)):
        flag = path_list[i].split('/')[-2]
        if (flag == '1' or flag == '2' or flag == 'HR_1'):
            label = 1
        else:
            label = 0
        dict = {}
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label
        if(path_list[i].find('/test_release/30/') != -1 or path_list[i].find('/test_release/29/') != -1 or path_list[i].find('/test_release/28/') != -1):
            valid_final_json.append(dict)
        elif(path_list[i].find('/test_release/') != -1):
            dev_final_json.append(dict)
        else: 
            if (label == 1):
                real_final_json.append(dict)
            else:
                fake_final_json.append(dict)
            
    print('\nCasia: ', len(path_list))
    print('Casia(dev): ', len(dev_final_json))
    print('Casia(valid): ', len(valid_final_json))
    print('Casia(real): ', len(real_final_json))
    print('Casia(fake): ', len(fake_final_json))
    json.dump(dev_final_json, f_dev, indent=4)
    f_dev.close()
    json.dump(valid_final_json, f_valid, indent=4)
    f_valid.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()


def replay_process():
    valid_final_json = []
    dev_final_json = []
    real_final_json = []
    fake_final_json = []
    label_save_dir = './replay/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_valid = open(label_save_dir + 'valid_label.json', 'w')
    f_dev = open(label_save_dir + 'dev_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    dataset_path = data_dir + 'replay_256/'
    path_list = glob.glob(dataset_path + '**/*.png', recursive=True)
    path_list.sort()
    for i in range(len(path_list)):
        flag = path_list[i].find('/real/')
        if (flag != -1):
            label = 1
        else:
            label = 0
        dict = {}
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label

        if (path_list[i].find('/replayattack-train/') != -1 or path_list[i].find('/replayattack-test/') != -1):
            # 原训练和测试集用于新训练集，原验证集（除少数用于选step数外）用于新测试集
            if (label == 1):
                real_final_json.append(dict)
            else:
                fake_final_json.append(dict)
        elif(path_list[i].find('/replayattack-devel/') != -1):
            if(path_list[i].find('_client003_') != -1 or path_list[i].find('_client005_') != -1 or path_list[i].find('_client010_') != -1):
                valid_final_json.append(dict)
            else:
                dev_final_json.append(dict)
            
        
    print('\nReplay: ', len(path_list))
    print('Replay(valid): ', len(valid_final_json))
    print('Replay(dev): ', len(dev_final_json))
    print('Replay(real): ', len(real_final_json))
    print('Replay(fake): ', len(fake_final_json))
    json.dump(valid_final_json, f_valid, indent=4)
    f_valid.close()
    json.dump(dev_final_json, f_dev, indent=4)
    f_dev.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()

def oulu_process():
    valid_final_json = []
    dev_final_json = []
    real_final_json = []
    fake_final_json = []
    label_save_dir = './oulu/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_valid = open(label_save_dir + 'valid_label.json', 'w')
    f_dev = open(label_save_dir + 'dev_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    dataset_path = data_dir + 'oulu_256/'
    path_list = glob.glob(dataset_path + '**/*.png', recursive=True)
    path_list.sort()
    for i in range(len(path_list)):
        flag = int(path_list[i].split('/')[-2].split('_')[-1])
        if (flag == 1):
            label = 1
        else:
            label = 0
        dict = {}
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label
        
        if (path_list[i].find('/Train_files/') != -1 or path_list[i].find('/Test_files/') != -1):
            if (label == 1):
                real_final_json.append(dict)
            else:
                fake_final_json.append(dict)
        elif(path_list[i].find('/Dev_files/') != -1):
            if(path_list[i].find('_21_') != -1):
                valid_final_json.append(dict)
            else:
                dev_final_json.append(dict)
            
        
    print('\nOulu: ', len(path_list))
    print('Oulu(valid): ', len(valid_final_json))
    print('Oulu(dev): ', len(dev_final_json))
    print('Oulu(real): ', len(real_final_json))
    print('Oulu(fake): ', len(fake_final_json))
    json.dump(valid_final_json, f_valid, indent=4)
    f_valid.close()
    json.dump(dev_final_json, f_dev, indent=4)
    f_dev.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()


def merge_dev():
    f_merge = open('merged_dev_label.json', 'w')
    merging_sets = ['oulu', 'msu', 'replay', 'casia']
    merged_json = []
    for dataset in merging_sets:
        filename = dataset + '/dev_label.json'
        data = pd.read_json(filename)
        merged_json += [{'photo_path': data['photo_path'][i], 'photo_label': int(data['photo_label'][i])} for i in range(len(data['photo_path']))]

    json.dump(merged_json, f_merge, indent=4)
    f_merge.close()

if __name__=="__main__":

    _3dmad_process()
    hkbu_process()
    msu_process()
    casia_process()
    replay_process()
    oulu_process()
    merge_dev()
