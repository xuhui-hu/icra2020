# coding=utf-8
"""
Author: Michael
Department: SEU Hand Group
Date:2019/10/1
Profile: Data_Collect for HG_AEN

脚本完成的任务：
1.用户数据记录
以下功能转移至HG_AEN，因为在树莓派下运行过慢，转移至电脑端：
2.窗口划分，对窗口特征进行标准归一化，然后以追加方式存入slidWin_Date（归一化数据），report存储每次追加的矩阵长度
3.主成分分析，变换矩阵U_T存入PCA_Matrix，随训练集的追加写入实时更新
4.归一化参数存入Scaler_Factors，保存为raw_layer_SS_mean和raw_layer_SS_scale
Improvements:
1.增加交互，记录用户编号
2.存储所有历史数据，按比例划分训练集和测试集合。没有采用随机打乱数据样本的策略
原因在于，虽然目前样本间是没有相关性的，因为特征中已经包含了时间特征，但在生成标签时，需要样本间的顺序才能顺利提取波峰波谷
3.具体训练集测试集在HG_AEN划分，之后进行自编码器网络训练
4. 使用新的底层读取MYO数据（利用树莓派的板载蓝牙）

Tip：首先需要先运行Data_Collect_and_Preprocess采集数据
由于数据采集是通过蓝牙搜索手环的方法，因此暂时无法区分两个手环同时通电时，选择哪个手环的问题
预处理过程的关键参数
滑动窗口长度 20  滑动步长 1
提取主成分个数  3

关于处理离线数据：
    运行 sudo python Collect_and_Preprocess.py

"""
import sys
import time
import os  # 用于新建路径（文件夹）
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import Lib.openmyo.open_myo as myo



class Collect_and_Preprocess(object):
    def __init__(self):  # 其实也是一个方法，不过是初始化方法
        # 确认录入信息正确的循环语句
        # print("---", collect_flag)
        while True:
            # self.usr_name = input("User_name (last 6 digits of school ID or personal ID):")
            self.usr_name = input("用户名（一卡通号或身份证号后6位）:")
            # self.usr_name = "189286"
            if len(self.usr_name) != 6:  # 如果未输入6位身份号码
                # print("Please enter the valid information, *** Last 6 digits of school ID or personal ID ***")
                print("警告!请正确输入用户名!*** 一卡通号或身份证号后6位 ***")
                os.system('pause')  # 按任意键退出暂停
                os.system('cls')  # 清屏
                continue  # 退出循环 重新输入
            else:
                # 建立数据集目录 格式例如：Data/189286/2019-6-5-21-38-openmyo.out
                source_addr = os.path.abspath('.')  # 这是程序当前路径
                target_addr = os.path.join(source_addr,'Data') # 添加一个数据集的路径
                if not os.path.isdir(target_addr):
                    os.mkdir(r'%s' % target_addr)
                target_addr = os.path.join(source_addr,'Data', self.usr_name) # 添加一个含有用户名的路径
                if os.path.isdir(target_addr):  # 如果已经录入过该用户的数据，则仅作提醒，不新建路径。
                    print("用户 %s 已经录入过数据，将自动追加在历史数据后" % self.usr_name)
                else:  # else表示未录入过数据
                    os.mkdir(r'%s' % target_addr)  # 目录需要一级一级建立，否则会找不到底层目录
                    break

        self.t = time.localtime(time.time())  # 只要时分秒不变，后面的三个值怎么赋值都不影响
        self.target_addr = os.path.join(os.path.abspath('.'), 'Data',self.usr_name,
										'%d-%d-%d-%d-%d-%d-MYO-openmyo.txt' % (
										self.t.tm_year, self.t.tm_mon, self.t.tm_mday,
										self.t.tm_hour, self.t.tm_min, self.t.tm_sec))  # 文件存取路径
        myo_mac_addr = myo.get_myo()
        print("MAC address: %s" % myo_mac_addr,flush=True)
        self.myo_device = myo.Device()
        self.myo_device.services.sleep_mode(1)  # never sleep
        # myo_device.services.set_leds([128, 128, 255], [128, 128, 255])  # purple logo and bar LEDs)
        self.myo_device.services.vibrate(1)  # short vibration
        fw = self.myo_device.services.firmware()
        print("Firmware version: %d.%d.%d.%d" % (fw[0], fw[1], fw[2], fw[3]),flush=True)
        batt = self.myo_device.services.battery()
        print("Battery level: %d" % batt,flush=True)
        self.myo_device.services.emg_filt_notifications()
        self.myo_device.services.set_mode(myo.EmgMode.FILT, myo.ImuMode.OFF, myo.ClassifierMode.OFF)
        self.myo_device.add_emg_event_handler(lambda emg: self.emg_collect(emg))

    def emg_collect(self, emg):  # 类名可以用作函数来调用的方法,做继承，覆盖了原方法。
        sys.stdout.write("\r{0}".format(emg)+"     ")
        sys.stdout.flush()  # 直接输出
        with open(self.target_addr, 'a+') as f:
            for comp in emg:
                f.write(str(comp) + ',')
            f.write('\n')


if __name__ == '__main__':
    coll_and_proc = Collect_and_Preprocess()
    try:
        print("提示！正在采集数据，按Crtl和C的组合键退出数据采集")  # 对上一句交互命令的换行
        while True:  # time out就是超时的意思！
            coll_and_proc.myo_device.services.waitForNotifications(1)
    except KeyboardInterrupt:  # 用户中断执行，如果进入了except，有finally肯定会进入.
        print("\n提示 ！用户自行退出! ")
    finally:
        print("数据采集结束！")
		






