import os
import sys
sys.path.insert(0, '/home/wangyf/codes/LocalDimming')
import cv2
import time
import argparse
import ipdb
import numpy as np
from torch.utils.data import DataLoader

from libs.DIV2K import DIV2K
from libs.evaluation import *
from libs.transform import *


parse = argparse.ArgumentParser()
parse.add_argument('--path', type=str, default='/home/wangyf/datasets/')
parse.add_argument('--base_size', type=int, default=[1080, 1920])
parse.add_argument('--block_size', type=int, default=[9, 16])
parse.add_argument('--bl', type=str, default='LUT')
parse.add_argument('--ld', type=str, default='BMA')
parse.add_argument('--cp', type=str, default='unlinear')
parse.add_argument('--BMA', type=int, default=4)
parse.add_argument('--bz', default=1)
args = parse.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main():
    DATE = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    dir_LD = os.path.join('output', DATE, 'LD')
    # dir_Icp = os.path.join('output', DATE, 'Icp')
    # dir_Iout = os.path.join('output', DATE, 'Iout')
    if not os.path.isdir(dir_LD):
        os.makedirs(dir_LD)
    # if not os.path.isdir(dir_Icp):
    #     os.makedirs(dir_Icp)
    # if not os.path.isdir(dir_Iout):
    #     os.makedirs(dir_Iout)
    # f = open(os.path.join('output', DATE, args.bl + '-' + args.ld + '-' + args.cp + '-record.txt'), 'w')

    listDataset = DIV2K(args)

    test_loader = DataLoader(listDataset,
                              batch_size=args.bz)

    psnrs, ssims, cds, cr1s, cr2s, num = 0, 0, 0, 0, 0, 0
    for i_batch, (Iin_transform, LD_transform, Icp_transform, Iout_transform, name) in enumerate(test_loader):
        # Iin = np.uint8(Iin_transform.squeeze(0).numpy().transpose((1, 2, 0)))
        LD = np.uint8(LD_transform.squeeze(0).numpy())
        # Icp = np.uint8(Icp_transform.squeeze(0).numpy().transpose((1, 2, 0)))
        # Iout = np.uint8(Iout_transform.squeeze(0).numpy().transpose((1, 2, 0)))

        LD_name = os.path.join(dir_LD, name[0])
        # Icp_name = os.path.join(dir_Icp, name[0])
        # Iout_name = os.path.join(dir_Iout, name[0])
        cv2.imwrite(LD_name, LD)
        print(LD_name)
        # cv2.imwrite(Icp_name, Icp)
        # cv2.imwrite(Iout_name, Iout)


    #     psnr = get_PSNR(Iin, Iout)
    #     ssim = get_SSIM(Iin, Iout)           # 0.0-1.0
    #     cd = get_ColorDifference(Iin, Iout)  # 0.0
    #     cr1, cr2 = get_Contrast(Iin, Iout)
    #
    #     psnrs += psnr
    #     ssims += ssim
    #     cds += cd
    #     cr1s += cr1
    #     cr2s += cr2
    #     num += 1
    #
    #     print_str = 'Index: [{0}]  '.format(name[0])
    #     print_str += 'PSNR: {0}  '.format(psnr)
    #     print_str += 'SSIM: {0}  '.format(ssim)
    #     print_str += 'CD: {0}  '.format(cd)
    #     print_str += 'CR: [{0}/{1}]\t'.format(cr1, cr2)
    #
    #     ## 打印到文件
    #     print(print_str)
    #     print(print_str, file=f)
    #
    # print(psnrs/num, ssims/num, cds/num, cr1s/num, cr2s/num)
    # print(psnrs / num, ssims / num, cds / num, cr1s / num, cr2s / num, file=f)


if __name__ == '__main__':
    main()
