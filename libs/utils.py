import sys
sys.path.insert(0, '/home/wangyf/codes/LocalDimming')
import ipdb
import cv2
import numpy as np

from libs.BL import *
from libs.LD import *
from libs.cp import *


def LocalDimming(Iin, args):                      # numpy-float32-[0.0, 255.0]
    h = args.base_size[0] / args.block_size[0]
    w = args.base_size[1] / args.block_size[1]
    BL = np.zeros(args.block_size, dtype='float32')
    B, G, R = cv2.split(Iin)
    R_G = np.where(R > G, R, G)
    gray = np.where(R_G > B, R_G, B)              # float32
    for i in range(args.block_size[0]):
        x1 = int(h * i)
        x2 = int(h * (i + 1))
        for j in range(args.block_size[1]):
            y1 = int(w * j)
            y2 = int(w * (j + 1))
            block = gray[x1:x2, y1:y2]
            BL[i][j] = get_BL(block, means=args.bl)
    # BL= np.uint8(BL)                             # uint8
    return BL


def get_BL(block, means='max'):
    if means == 'max':
        BL = np.max(block)
    elif means == 'avg':
        BL = np.mean(block)
    elif means == 'LUT':
        I_max = np.max(block)
        I_avg = np.mean(block)
        diff = I_max - I_avg
        BL = I_avg + 0.50 * (diff + diff ** 2 / 255)
    elif means == 'PSNR':
        BL = psnr(block)
    elif means == 'CDF':
        BL = cdf(block)
    BL = np.where(BL < 0, 0, BL)
    BL = np.where(BL > 255, 255, BL)
    return BL


def get_LD(BL, Iin, args):
    if args.ld == 'BMA':
        LD = getLSF_bma(BL, args.BMA).astype('float32')
    elif args.ld == 'LUT':
        LD = getLSF_lut(BL, Iin, args).astype('float32')    ## block effect
    return LD


def get_Icp(Iin, LD, means='Linear'):
    if means == 'linear':
        Icp = getCP_linear(Iin, LD)
    elif means == 'unlinear':
        Icp = getCP_unlinear(Iin, LD)    ## block effect
    elif means == '2steps':
        Icp = getCP_2steps(Iin, LD)
    elif means == 'log':
        Icp = getCP_log(Iin, LD)
    return Icp



## TODO: rgb == yuv == Icp
def get_Iout(Icp, LD):
    Iout = Icp * LD[:, :,np.newaxis] / 255.0
    return Iout











