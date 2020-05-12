# coding=utf-8
"""
Author: Michael
Department: SEU Hand Group
Date:2019/7/23
Profile: Online Prediction of AEN

脚本完成的任务：
1.根据输入用户名读取训练的矩阵
2.进行手势预测，输出值的滤波


Tip：
被robot_hand,unity hand,circle_game 共用
2019.8.2
尝试增加幅值最大衰减算法，以防止出现记录到过大的幅值。应该统计预测值在细分区间出现的个数，根据出现的罕见程度进行自动区分，自动衰减不是很好。
2019.8.10 增加衰减函数 考虑统计方法
衰减函数为y=e^(-0.02*t) + 0.2 10s从1下降到0.367
"""
from __future__ import print_function
import sys
import numpy as np
import serial
import os
import time
from math import log,exp
sys.path.append("..")  # 将环境变量往上切一级
import Lib.openmyo.open_myo as myo

class Online_Predict(object):  # 树莓派版本，无波形显示
    def __init__(self):
        self.U_T = []
        self.emg_series = np.zeros((8, 20))
        self.y_hat = np.zeros((1, 3))
        self.posture_history = [0 for x in range(10)]  # 均值滤波滑窗
        self.altitude_max = np.ones((3, 2))  # 分别表示三个自由度的极大值和极小值,第一列极大，第二列极小
        self.altitude_count = np.ones((3, 2)) * time.time()  # 初始化也要加入时间
        self.scale = np.ones((3,2))
        while True:
            self.usr_name = input("用户名（一卡通号或身份证号后6位）:")
            # self.usr_name = '189286'
            if len(self.usr_name) != 6:  # 如果未输入6位身份号码
                # print("Please enter the valid information, *** Last 6 digits of school ID or personal ID ***")
                print("警告!请正确输入用户名!*** 一卡通号或身份证号后6位 ***")
                os.system('pause')  # 按任意键退出暂停
                os.system('cls')  # 清屏
                continue  # 退出循环 重新输入
            else:
                if os.path.isdir(os.path.join(os.path.abspath('.'), 'Data', self.usr_name)) is True:  # 如果存在该用户的数据
                    self.target_addr = os.path.join(os.path.abspath('.'), 'Data', self.usr_name)
                    self.report = np.loadtxt(os.path.join(self.target_addr, 'slidWin_Data', 'report.out'))
                    if np.size(self.report) == 1:
                        print("警告！该用户仅完成了一轮训练，请增加训练次数再尝试识别！")
                        break
                    else:  # 训练次数达到两次以上，允许训练
                        print("提示！开始用户 %s 的在线识别！" % self.usr_name)
                        break
                else:
                    print("警告！未录入过该用户，请检查用户名！")
                    continue
        """
        读取网络参数
        """
        # 输入PCA前的肌电归一化参数
        self.raw_layer_SS_mean = np.loadtxt(os.path.join(self.target_addr, 'Scaler_Factors', 'raw_layer_SS_mean.out'))
        self.raw_layer_SS_scale = np.loadtxt(os.path.join(self.target_addr, 'Scaler_Factors', 'raw_layer_SS_scale.out'))
        # PCA矩阵
        for i in range(8):
            U_T = np.loadtxt(os.path.join(self.target_addr, 'PCA_Martix', 'PCAM_Ch%d.out' % i))  # 加载主成分矩阵  20行3列
            self.U_T.append(U_T)  # 加载主成分矩阵  每个20行2列
        # 输入AEN前的PCA归一化参数
        self.pca_layer_MAS_scale = np.loadtxt(os.path.join(self.target_addr, 'Scaler_Factors', 'pca_layer_MAS_scale.out'))
        # AEN矩阵
        self.w_ae = np.loadtxt(os.path.join(self.target_addr, 'AEN_Matrix', 'w_ae.out'))
        self.b_ae = np.loadtxt(os.path.join(self.target_addr, 'AEN_Matrix', 'b_ae.out'))
        # 输入Reg_NN前的归一化参数
        self.aen_layer_SS_mean = np.loadtxt(os.path.join(self.target_addr, 'Scaler_Factors', 'aen_layer_SS_mean.out'))
        self.aen_layer_SS_scale = np.loadtxt(os.path.join(self.target_addr, 'Scaler_Factors', 'aen_layer_SS_scale.out'))
        # Reg_NN矩阵
        self.weight_input_MLPReg = np.loadtxt(os.path.join(self.target_addr, 'RegNN_Matrix', 'weight_input_MLPReg.out'))  # 加载回归神经网络权重矩阵6行20列
        self.weight_output_MLPReg = np.loadtxt(os.path.join(self.target_addr, 'RegNN_Matrix', 'weight_output_MLPReg.out'))  # 20x3
        self.bias_input_MLPReg = np.loadtxt(os.path.join(self.target_addr, 'RegNN_Matrix', 'bias_input_MLPReg.out'))
        self.bias_output_MLPReg = np.loadtxt(os.path.join(self.target_addr, 'RegNN_Matrix', 'bias_output_MLPReg.out'))
        """
        MYO初始化
        """
        myo_mac_addr = myo.get_myo()
        print("MAC address: %s" % myo_mac_addr)
        self.myo_device = myo.Device()
        self.myo_device.services.sleep_mode(1)  # never sleep
        # myo_device.services.set_leds([128, 128, 255], [128, 128, 255])  # purple logo and bar LEDs)
        self.myo_device.services.vibrate(1)  # short vibration
        fw = self.myo_device.services.firmware()
        print("Firmware version: %d.%d.%d.%d" % (fw[0], fw[1], fw[2], fw[3]))
        batt = self.myo_device.services.battery()
        print("Battery level: %d" % batt)
        self.myo_device.services.emg_filt_notifications()
        self.myo_device.services.set_mode(myo.EmgMode.FILT, myo.ImuMode.OFF, myo.ClassifierMode.OFF)
        self.myo_device.add_emg_event_handler(lambda emg: self.predict(emg))

    def predict(self, emg):  # 将实时肌电信号转为预测信号，8行1列，行数为空间电极个数，列数为历史采样点个数
        self.emg_series = np.roll(self.emg_series, -1, axis=1)  # 初始化时定义了一个8行20列的零矩阵，这句表示滚动数据
        # print(self.emg_series.shape)
        self.emg_series[:, -1] = np.array(emg).T  # 上一步对数据滚动，这一步对末列赋值，最后达到滑窗效果

        preproce_val = np.zeros((1, 24))  # 8通道3主元
        val = None
        for i in range(8):
            val = np.dot((self.emg_series[i, :] - self.raw_layer_SS_mean[i]) / self.raw_layer_SS_scale[i], self.U_T[i])
            preproce_val[:, 3 * i: 3 * i + 3] = val / self.pca_layer_MAS_scale[i]

        # 自编码器网络 当时代入自编码器的神经元顺序是：
        feature = np.maximum(0, np.dot(preproce_val, self.w_ae) + self.b_ae.reshape(1, 6))  # maximum表示relu函数
        feature_s = np.true_divide(feature - self.aen_layer_SS_mean.T, self.aen_layer_SS_scale)
        # 回归神经网络
        y_0 = np.maximum(0, np.dot(feature_s, self.weight_input_MLPReg) + self.bias_input_MLPReg.reshape(1, -1))  # 隐层输出，maximum作为relu来用
        self.y_hat = np.dot(y_0, self.weight_output_MLPReg) + self.bias_output_MLPReg.reshape(1, 3)
        return self.y_hat

    def filt(self):
        prediction = np.zeros((1, 3))  # 三个自由度的滤波后输出值
        scale = 1 # 初始值，否则有可能变为NaN
        """
        数据归一化
        """
        for i in range(3):
            if self.y_hat[0, i] > 1:
                self.y_hat[0, i] = 1
            elif self.y_hat[0, i] < -1:
                self.y_hat[0, i] = -1

            if -0.2 < self.y_hat[0, i] < 0.2:  # 阈值去除干扰
                self.y_hat[0, i] = 0
            # 自适应幅值
            if self.y_hat[0, i] > self.scale[i, 0]:  # 找各个自由度的最大幅值,正负分开存储
                self.altitude_max[i, 0] = self.y_hat[0, i]
                self.altitude_count[i, 0] = time.time()
            elif self.y_hat[0, i] < - self.scale[i, 1]:  # 存储的是负值的绝对值
                self.altitude_max[i, 1] = -self.y_hat[0, i]  # 存储的是负值的绝对值
                self.altitude_count[i, 1] = time.time()

            self.scale[i, 0] = exp(log(self.altitude_max[i, 0]) - 0.02 * (time.time() - self.altitude_count[i, 0])) + 0.2
            self.scale[i, 1] = exp(log(self.altitude_max[i, 1]) - 0.02 * (time.time() - self.altitude_count[i, 1])) + 0.2

            if self.y_hat[0, i] > 0:
                prediction[0, i] = self.y_hat[0, i] / self.scale[i, 0]
            elif self.y_hat[0, i] <= 0:
                prediction[0, i] = self.y_hat[0, i] / self.scale[i, 1]

        """
        窗口滤波
        """
        self.posture_history.append(prediction)  # 建立滑窗
        if len(self.posture_history) > 9:
            self.posture_history.pop(0)  # 删除最老的历史记录
        prediction = np.mean(np.array(self.posture_history), axis=0)
        '''
        sys.stdout.write("\r[{0:<8.1f}{1:<8.1f}{2:<8.1f}]  [{3:<8.1f}{4:<8.1f}{5:<8.1f}]  [{6:<8.1f}{7:<8.1f}{8:<8.1f}]\n".
                         format(-self.scale[0, 1] * 100, prediction[0, 0] * 100, self.scale[0, 0] * 100,
                                -self.scale[1, 1] * 100, prediction[0, 1] * 100, self.scale[1, 0] * 100,
                                -self.scale[2, 1] * 100, prediction[0, 2] * 100, self.scale[2, 0] * 100))
        sys.stdout.flush()  # 直接输出
        '''
        return prediction


if __name__ == '__main__':
    hnd = Online_Predict()
    wirles = serial.Serial('/dev/ttyUSB0', 115200, timeout=5)

    wrist_fe = 0
    wrist_spin = 0
    hand_open = 0
    spin_dist = 0
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

            strr = b'\xfa\x03\x04' + bytes([wrist_spin]) + bytes([wrist_fe]) + bytes([hand_open]) + b'\xfb'
            wirles.write(strr)
            sys.stdout.write("\r{0}".format([wrist_fe, wrist_spin, hand_open]) + "     ")
            sys.stdout.flush()  # 直接输出

    except KeyboardInterrupt:  # 用户中断执行，如果进入了except，有finally肯定会进入.
        pass
    finally:
        pass
