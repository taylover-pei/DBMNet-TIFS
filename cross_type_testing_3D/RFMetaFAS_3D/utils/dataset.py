import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import Image
import sys
sys.path.append("..")
from configs.config import config
import cv2

def OriImg_loader(path):
    RGBimg = Image.open(path).convert('RGB')
    HSVimg = Image.open(path).convert('HSV')
    return RGBimg, HSVimg

def DepthImg_loader(path, imgsize=config.depth_size):
    img = Image.open(path)
    re_img = img.resize((imgsize, imgsize), resample=Image.BICUBIC)
    return re_img

class YunpeiDataset(Dataset):
    def __init__(self, data_pd, transforms=None, train=True):
        self.is_hsv = config.is_hsv
        self.train = train
        self.photo_path = data_pd['photo_path'].tolist()
        self.photo_label = data_pd['photo_label'].tolist()
        self.photo_belong_to_video_ID = data_pd['photo_belong_to_video_ID'].tolist()
        self.depth_transforms = T.Compose([T.ToTensor()])
        if transforms is None:
            if not train:
                self.transforms = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.photo_path)

    def __getitem__(self, item):
        if self.train:
            img_path = self.photo_path[item]
            depth_path = img_path.split('.png')[0] + '_depth.jpg'
            label = self.photo_label[item]
            if (self.is_hsv):
                ori_rgbimg, ori_hsvimg = OriImg_loader(img_path)
                ori_rgbimg = self.transforms(ori_rgbimg)
                ori_hsvimg = self.transforms(ori_hsvimg)
                img = torch.cat([ori_rgbimg, ori_hsvimg], dim=0)
            else:
                img = Image.open(img_path)
                img = self.transforms(img)

            depth_img = DepthImg_loader(depth_path)
            depth_img = self.depth_transforms(depth_img)
            return img, depth_img, label
        else:
            img_path = self.photo_path[item]
            label = self.photo_label[item]
            videoID = self.photo_belong_to_video_ID[item]
            if(self.is_hsv):
                ori_rgbimg, ori_hsvimg = OriImg_loader(img_path)
                ori_rgbimg = self.transforms(ori_rgbimg)
                ori_hsvimg = self.transforms(ori_hsvimg)
                img = torch.cat([ori_rgbimg, ori_hsvimg], dim=0)
            else:
                img = Image.open(img_path)
                img = self.transforms(img)
            return img, label, videoID, img_path
