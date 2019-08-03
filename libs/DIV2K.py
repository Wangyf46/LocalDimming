import os
import sys
sys.path.insert(0, '/home/wangyf/codes/LocalDimming')
import cv2
import ipdb
import torch
from torch.utils.data import Dataset

from libs.utils  import *

class DIV2K(Dataset):
    def __init__(self, args):
        self.args = args
        self.img_dir = os.path.join(args.path, 'DIV2K_valid_HR_aug')
        self.name_list = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        img_file =  os.path.join(self.img_dir, name)
        Iin = cv2.imread(img_file).astype('float32')                 # numpy-RGB: uint8[0-255]-->float32[0.0-255.0]

        BL = LocalDimming(Iin, self.args)                            # numpy-float32-[0.0-255.0]
        LD = get_LD(BL, Iin, self.args)                              # numpy-float32-[0.0-255.0]


        Icp = get_Icp(Iin, LD, self.args.cp)                         # float32
        Iout = get_Iout(Icp, LD)                                     # float32

        Iin_transform = torch.from_numpy(Iin.transpose((2, 0, 1)))
        LD_transform = torch.from_numpy(LD)
        Icp_transform = torch.from_numpy(Icp.transpose((2, 0, 1)))
        Iout_transform = torch.from_numpy(Iout.transpose((2, 0, 1)))

        return  Iin_transform, LD_transform, Icp_transform, Iout_transform, name



