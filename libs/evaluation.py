import sys
sys.path.insert(0, '/home/wangyf/codes/LocalDimming')
import ipdb
import cv2

import torch
import torch.nn.functional as F
from math import exp
import numpy as np

from libs.transform import *

def get_PSNR(target, ref):
    MSE = np.mean((target - ref) ** 2)
    PSNR = 20 * np.log10(255 / np.sqrt(MSE))
    return PSNR



def get_SSIM(target, ref):
    target = np.float32(target)
    ref = np.float32(ref)

    k1, k2 = 0.01, 0.03
    L = 255
    c1, c2 = (k1*L)**2, (k2*L)**2
    c3 = c2/2
    alpha, beta, gamma = 1, 1, 1

    ux, uy = np.mean(target), np.mean(ref)
    vx, vy = np.var(target), np.var(ref)     # variance
    sdx, sdy = np.std(target), np.std(ref)   # sd
    Covariance = np.sum(target * ref) / (target.shape[0] * target.shape[1] * 3) - ux * uy

    Luminance = (2*ux*uy + c1) / (ux**2 + uy**2 + c1)
    Contrast = (2*sdx*sdy + c2) / (vx + vy + c2)
    Structure = (Covariance + c3) / (sdx*sdy + c3)

    SSIM = (Luminance**alpha) * (Contrast**beta) * (Structure**gamma)
    return  SSIM


## RGB-->XYX--->LAB
def get_ColorDifference(target, ref):
    B1, G1, R1 = cv2.split(target)
    R1_gamma, G1_gamma, B1_gamma = gamma(R1/255.0, G1/255.0, B1/255.0)
    X1, Y1, Z1 = rgbToxyz(R1_gamma, G1_gamma, B1_gamma)
    L1, a1, b1 = xyzTolab(X1, Y1, Z1)

    B2, G2, R2 = cv2.split(ref)
    R2_gamma, G2_gamma, B2_gamma = gamma(R2 / 255.0, G2 / 255.0, B2 / 255.0)
    X2, Y2, Z2 = rgbToxyz(R2_gamma, G2_gamma, B2_gamma)
    L2, a2, b2 = xyzTolab(X2, Y2, Z2)

    CD = np.mean(np.sqrt((L2-L1)**2 + (a2-a1)**2 + (b1-b2)**2))
    return CD



def get_H(Y):
    Height, Width = Y.shape
    p = 0.1 * Height * Width
    q = 0.9 * Height * Width
    Y_max = np.max(Y)
    Y_min = np.min(Y)
    total = 0
    for H_10 in range(Y_min, Y_max+1):
        num = np.sum(Y == H_10)
        total += num
        if total > p:
            break
    for H_90 in range(Y_min, Y_max + 1):
        num = np.sum(Y == H_90)
        total += num
        if total > q:
            break

    return H_10, H_90



## CR
def get_Contrast(Iin, Iout):
    Bin, Gin, Rin = cv2.split(Iin)
    Yin, _, _ = rgbToyuv(Rin, Gin, Bin)
    Hin_10, Hin_90 = get_H(np.uint8(Yin))

    Bout, Gout, Rout = cv2.split(Iout)
    Yout, _, _ = rgbToyuv(Rout, Gout, Bout)
    Hout_10, Hout_90 = get_H(np.uint8(Yout))

    if Hin_10 != 0:
        cr1 = Hin_90 / Hin_10
    else:
        cr1 = 1000

    if Hout_10 != 0:
        cr2 = Hout_90 / Hout_10
    else:
        cr2 = 1000

    return cr1, cr2