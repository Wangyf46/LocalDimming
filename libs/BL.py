import sys
sys.path.insert(0, '/home/wangyf/codes/LocalDimming')
import numpy as np


def cdf(block):
    Height, Width = block.shape
    block = np.uint8(block)
    I_max = np.max(block)
    I_min = np.min(block)
    total = 0
    k = 0.9
    threshold = k * Height * Width
    for BL in range(I_min, I_max+1):
        num = np.sum(block == BL)
        total += num
        if total > threshold:
            break
    return BL



def psnr(block):
    Height, Width = block.shape
    Rpsnr = 30
    Emse = 65.025
    Etse = Emse * Height * Width
    I_max = np.max(block)

    ## Ic init
    n = 0.7
    # n = 0.8
    # n = 0.9
    gamma = 2.2
    Ic = (n ** (1.0 / gamma)) * I_max

    Etse_m = 0
    while(1):
        if Etse_m <= Etse:
            for level in range(Ic, I_max+1):
                Etse_m += np.sum(block == level) * ((level-Ic)**2)
            Ic = Ic -1
        else:
            break
    BL = (Ic / 255) ** gamma
    return BL



def gauss(block):
    Height, Width = block.shape
    I_avg = np.mean(block)
    delta = np.sum((block - I_avg) ** 2) / (Height * Width)
    BL = (delta ** 0.5) * 1.5 + I_avg
    return BL