import sys
sys.path.insert(0, '/home/wangyf/codes/LocalDimming')

import cv2
import numpy as np



def getLSF_lut(Iin, BL_init, args):
    h = args.base_size[0] / args.block_size[0]
    w = args.base_size[1] / args.block_size[1]
    LD = np.zeros(args.base_size)
    for i in range(args.block_size[0]):
        x1 = int(h * i)
        x2 = int(h * (i + 1))
        for j in range(args.block_size[1]):
            y1 = int(w * j)
            y2 = int(w * (j + 1))
            Iin_max = np.max(Iin[x1:x2, y1:y2])
            for m in range(x1, x2):
                for n in range(y1, y2):
                    if Iin[m][n] > 2 * BL_init[i][j] - Iin_max:
                        LD[m][n] = BL_init[i][j]
                    else:
                        LD[m][n] = Iin_max + Iin[m][n] / 2.0
    LD = np.where(LD < 0, 0, LD)
    LD = np.where(LD > 255, 255, LD)
    return LD



def getLSF_bma(BL_init, K):
    ## TODO
    a_R, b_R, c_R, d_R = 0.4, 0.2, 0.2, 0.2
    a_G, b_G, c_G, d_G = 0.38, 0.15, 0.12, 0.1
    a_B, b_B, c_B, d_B = 0.38, 0.11, 0.08, 0.06

    for t in range(K):
        Height, Width = BL_init.shape
        BL_mirr = np.zeros((Height + 2, Width + 2))

        # mid
        for i in range(Height):
            for j in range(Width):
                BL_mirr[i + 1][j + 1] = BL_init[i][j]

        # Up and Down
        for j in range(Width):
            BL_mirr[0][j + 1] = BL_init[0][j]
            BL_mirr[Height + 1][j + 1] = BL_init[Height - 1][j]

        # Left and Right
        for i in range(Height):
            BL_mirr[i + 1][0] = BL_init[i][0]
            BL_mirr[i + 1][Width + 1] = BL_init[i][Width - 1]

        # four
        BL_mirr[0][0] = BL_init[0][0]
        BL_mirr[0][Width + 1] = BL_init[0][Width - 1]
        BL_mirr[Height + 1][0] = BL_init[Height - 1][0]
        BL_mirr[Height + 1][Width + 1] = BL_init[Height - 1][Width - 1]

        # RED Local
        BL_blur = BL_mirr
        BL_blur[1][1] = a_R * BL_mirr[1][1] + b_R * BL_mirr[1][2] + \
                        c_R * BL_mirr[2][1] + d_R * BL_mirr[2][2]
        BL_blur[1][Width] = a_R * BL_mirr[1][Width] + b_R * BL_mirr[1][Width - 1] + \
                            c_R * BL_mirr[2][Width] + d_R * BL_mirr[2][Width - 1]
        BL_blur[Height][1] = a_R * BL_mirr[Height][1] + b_R * BL_mirr[Height][2] + \
                             c_R * BL_mirr[Height - 1][1] + d_R * BL_mirr[Height - 1][2]
        BL_blur[Height][Width] = a_R * BL_mirr[Height][Width] + b_R * BL_mirr[Height][Width - 1] + \
                                 c_R * BL_mirr[Height - 1][Width] + d_R * BL_mirr[Height - 1][Width - 1]

        # Green Local(left-right)
        for i in range(2, Height):
            BL_blur[i][1] = a_G * BL_mirr[i][1] + b_G * (BL_mirr[i - 1][1] + BL_mirr[i + 1][1]) + \
                            c_G * BL_mirr[i][2] + d_G * (BL_mirr[i - 1][2] + BL_mirr[i + 1][2])

            BL_blur[i][Width] = a_G * BL_mirr[i][Width] + b_G * (BL_mirr[i - 1][Width] + BL_mirr[i + 1][Width]) + \
                                c_G * BL_mirr[i][Width - 1] + d_G * (
                                            BL_mirr[i - 1][Width - 1] + BL_mirr[i + 1][Width - 1])

        # Green Local(up-down)
        for j in range(2, Width):
            BL_blur[1][j] = a_G * BL_mirr[1][j] + b_G * (BL_mirr[1][j - 1] + BL_mirr[1][j + 1]) + \
                            c_G * BL_mirr[2][j] + d_G * (BL_mirr[2][j - 1] + BL_mirr[2][j + 1])
            BL_blur[Height][j] = a_G * BL_mirr[Height][j] + b_G * (
                        BL_mirr[Height][j - 1] + BL_mirr[Height][j + 1]) + \
                                 c_G * BL_mirr[Height - 1][j] + d_G * (
                                             BL_mirr[Height - 1][j - 1] + BL_mirr[Height - 1][j + 1])

        # BLUE block
        for i in range(2, Height):
            for j in range(2, Width):
                BL_blur[i][j] = a_B * BL_mirr[i][j] + b_B * (BL_mirr[i][j - 1] + BL_mirr[i][j + 1]) + \
                                c_B * (BL_mirr[i - 1][j] + BL_mirr[i + 1][j]) + \
                                d_B * (BL_mirr[i - 1][j - 1] + BL_mirr[i - 1][j + 1] + BL_mirr[i + 1][j - 1] +
                                       BL_mirr[i + 1][j + 1])
        h, w = BL_blur.shape
        BL_blur_2x = cv2.resize(BL_blur, (2 * w, 2 * h))
        BL_init = BL_blur_2x

    LD = cv2.resize(BL_blur_2x, (1920, 1080))
    LD = np.where(LD < 0, 0, LD)
    LD = np.where(LD > 255, 255, LD)
    return LD

