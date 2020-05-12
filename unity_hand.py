# coding=utf-8
"""
Author: Michael
Department: SEU Hand Group
Date:2019/7/23
Profile: Control Program for Unity Hand

脚本完成的任务：
1.根据输入用户名读取训练的矩阵
2.进行手势预测，输出值的滤波
3.将控制信号经串口发送给Unity 上位机

Tip：
串口波特率115200 后期将测试把圆圈游戏也改为115200（之前为230400）
与机器手控制相比，不使用自由度竞争机制和手腕限位
发送数据为-128~128，其与robot hand不同,与circle_game相同

所有协议都依据circle_game

"""

from __future__ import print_function
import sys
import numpy as np
import serial
import os
from Lib.brain.Online_Predict import Online_Predict
from Lib.peakdet.peakdet import peakdet

# sys.path.append("..")  # 将环境变量往上切一级
# import Lib.openmyo.open_myo as myo

if __name__ == '__main__':
    hnd = Online_Predict()
    wirles = serial.Serial('/dev/ttyUSB0', 230400, timeout=5)
    wrist_fe = 0
    wrist_spin = 0
    hand_open = 0
    hand_open_hist = []
    grasp_lock = False
    delay  = 0
    try:
        while True:  # time out就是超时的意思！
            hnd.myo_device.services.waitForNotifications(1)
            prediction = hnd.filt()
            """
            输出数据
            """
            # print(prediction)
            if prediction[0, 0] < 0:
                wrist_spin = int(prediction[0, 0] * 100 + 255)
            else:
                wrist_spin = int(prediction[0, 0] * 100)

            if prediction[0, 1] < 0:
                wrist_fe = int(prediction[0, 1] * 100 + 255)
            else:
                wrist_fe = int(prediction[0, 1] * 100)

            if prediction[0, 2] < 0:
                hand_open = int(prediction[0, 2] * 100 + 255)
            else:
                hand_open = int(prediction[0, 2] * 100)

            """
            抓握锁定监测，测试用，实际没有使用，本来是想给残疾人用的
            """
            # delay += 1
            # hand_open_hist.append(prediction[0, 2])  # 函数约10ms执行周期，记200个数据表示获取最近2s的手势数据。
            # if len(hand_open_hist) > 100:
            #     hand_open_hist.pop(0)
            # if delay == 70:
            #     delay = 0
            #     maxtab, mintab = peakdet(np.array(hand_open_hist), delta=0.6)  # 这里的代码可以优化，通过检查峰谷的个数，距离，自动纠正步长，在这之前还能再滤波一下
            #     if np.shape(maxtab)[0] == 2: # 进到这里面 表示读取到了两连握
            #         hand_open_hist = []
            #         if grasp_lock == True:  # 默认False,这个判断句为监测上一次是否还是True
            #             print("解锁") # 连续两个True 解锁
            #             hnd.myo_device.services.vibrate(1)  # short vibration
            #             hnd.myo_device.services.vibrate(1)  # short vibration
            #             grasp_lock = False
            #         else: # 表明上一次为False,即解锁状态
            #             print("上锁")
            #             hnd.myo_device.services.vibrate(1)  # short vibration
            #             grasp_lock = True

            strr = b'\xfa\x03\x04' + bytes([wrist_spin]) + bytes([wrist_fe]) + bytes([hand_open]) + b'\xfb'
            wirles.write(strr)
            sys.stdout.write("\r{0}".format([wrist_fe, wrist_spin, hand_open]) + "     ")
            sys.stdout.flush()  # 直接输出

    except KeyboardInterrupt:  # 用户中断执行，如果进入了except，有finally肯定会进入.
        pass
    finally:
        pass
