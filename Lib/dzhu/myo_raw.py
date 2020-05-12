# coding=utf-8
'''
	Original by dzhu
		https://github.com/dzhu/myo-raw

	Edited by Fernando Cosentino
		http://www.fernandocosentino.net/pyoconnect

		MyNote:

'''

from __future__ import print_function

import enum
import re
# import struct
import sys
import threading
import time

import serial
from serial.tools.list_ports import comports

from lib.dzhu.common import *

try:
    import pygame  # 导入pygame这个模块
    from pygame.locals import *  # This module contains various constants used by Pygame

    HAVE_PYGAME = True
except ImportError:
    HAVE_PYGAME = False


def multichr(ords):
    if sys.version_info[0] >= 3:
        return bytes(ords)
    else:
        return ''.join(map(chr, ords))


def multiord(b):
    if sys.version_info[0] >= 3:
        return list(b)
    else:
        return map(ord, b)


class Arm(enum.Enum):
    UNKNOWN = 0
    RIGHT = 1
    LEFT = 2


class XDirection(enum.Enum):
    UNKNOWN = 0
    X_TOWARD_WRIST = 1
    X_TOWARD_ELBOW = 2


class Pose(enum.Enum):
    REST = 0
    FIST = 1
    WAVE_IN = 2
    WAVE_OUT = 3
    FINGERS_SPREAD = 4
    THUMB_TO_PINKY = 5
    UNKNOWN = 255


class Packet(object):
    def __init__(self, ords):
        self.typ = ords[0]
        self.cls = ords[2]
        self.cmd = ords[3]
        self.payload = multichr(ords[4:])

    def __repr__(self):
        return 'Packet(%02X, %02X, %02X, [%s])' % \
               (self.typ, self.cls, self.cmd,
                ' '.join('%02X' % b for b in multiord(self.payload)))


class BT(object):
    ''' Implements the non-Myo-specific details of the Bluetooth protocol. '''

    def __init__(self, tty):
        self.ser = serial.Serial(port=tty, baudrate=9600, dsrdtr=1, timeout=0)
        self.buf = []
        self.lock = threading.Lock()
        self.handlers = []  # 里面存放了一个函数
        # self.delay = 0

    # internal data-handling methods
    def recv_packet(self, timeout=None):  # 作者注释：timeout默认为none，一定有返回值
        t0 = time.time()
        self.ser.timeout = None
        while timeout is None or time.time() < t0 + timeout:
            # print(timeout, time.time())
            if timeout is not None:
                self.ser.timeout = t0 + timeout - time.time()

            c = self.ser.read()  # timeout为none，c应该有值,串口返回的数据，接收完毕才会跳转到下一句吧

            if not c: return None  # 如果c为none，返回

            ret = self.proc_byte(ord(c))

            if ret:
                if ret.typ == 0x80:
                    self.handle_event(ret)  # 把ret代入那个函数
                return ret


    def recv_packets(self, timeout=.5):  # 这个函数一直没有被调用
        res = []
        t0 = time.time()
        while time.time() < t0 + timeout:
            p = self.recv_packet(t0 + timeout - time.time())
            if not p: return res
            res.append(p)
        return res

    def proc_byte(self, c):
        if not self.buf:
            if c in [0x00, 0x80, 0x08, 0x88]:
                self.buf.append(c)
            return None
        elif len(self.buf) == 1:
            self.buf.append(c)
            self.packet_len = 4 + (self.buf[0] & 0x07) + self.buf[1]
            return None
        else:
            self.buf.append(c)
            # print(self.buf)

        if self.packet_len and len(self.buf) == self.packet_len:
            p = Packet(self.buf)

            # self.delay += 1
            # if self.delay == 10:
            #     self.delay = 0
            #     print(p)

            self.buf = []
            return p
        return None

    def handle_event(self, p):
        for h in self.handlers:
            h(p)

    def add_handler(self, h):
        self.handlers.append(h)

    def remove_handler(self, h):  # 形参
        try:
            self.handlers.remove(h)
        except ValueError:
            pass

    def wait_event(self, cls, cmd):  # h!!!!!!!!
        res = [None]

        def h(p):
            if p.cls == cls and p.cmd == cmd:
                res[0] = p

        self.add_handler(h)  # 其实就是为了调用h

        while res[0] is None:
            self.recv_packet()
        self.remove_handler(h)
        return res[0]

    ## specific BLE commands
    def connect(self, addr):
        return self.send_command(6, 3, pack('6sBHHHH', multichr(addr), 0, 6, 6, 64, 0))

    def get_connections(self):
        return self.send_command(0, 6)

    def discover(self):
        return self.send_command(6, 2, b'\x01')

    def end_scan(self):
        return self.send_command(6, 4)

    def disconnect(self, h):
        return self.send_command(3, 0, pack('B', h))

    def read_attr(self, con, attr):
        self.send_command(4, 4, pack('BH', con, attr))
        return self.wait_event(4, 5)

    def write_attr(self, con, attr, val):
        self.send_command(4, 5, pack('BHB', con, attr, len(val)) + val)
        return self.wait_event(4, 1)

    def send_command(self, cls, cmd, payload=b'', wait_resp=True):
        s = pack('4B', 0, len(payload), cls, cmd) + payload
        self.ser.write(s)

        while True:
            p = self.recv_packet()  # timeout默认为none，一定有返回值

            # no timeout, so p won't be None
            if p.typ == 0: return p  # send_command返回的是这个值

            # not a response: must be an event
            self.handle_event(p)  # 正常通信不会到这一步。


class MyoRaw(object):
    '''Implements the Myo-specific communication protocol.'''

    def __init__(self, tty=None):  # 属性：默认None，也有可能是键入的其他设备名，默认参数写法
        if tty is None:
            tty = self.detect_tty()
            print(tty)
        if tty is None:
            raise ValueError('Myo dongle not found!')  # 异常处理

        self.bt = BT(tty)
        self.conn = None
        self.emg_handlers = []
        self.imu_handlers = []
        self.arm_handlers = []
        self.pose_handlers = []
        self.emg = (0,0,0,0,0,0,0,0)

    def detect_tty(self):

        for p in comports():
            if re.search(r'PID=2458:0*1', p[2]):
                print('using device:', p[0])
                return p[0]

        return None

    def run(self, timeout=None):
        return self.bt.recv_packet(timeout)

    def connect(self):
        ## stop everything from before
        self.bt.end_scan()  # Packet(00, 06, 04, [81 01])
        self.bt.disconnect(0)  # Packet(00, 03, 00, [00 86 01])
        self.bt.disconnect(1)  # Packet(00, 03, 00, [01 86 01])
        self.bt.disconnect(2)  # Packet(00, 03, 00, [02 86 01])

        ## start scanning
        print('scanning...')
        self.bt.discover()  # Packet(00, 06, 02, [00 00])

        while True:
            p = self.bt.recv_packet()
            print('scan response:', p)

            if p.payload.endswith(b'\x06\x42\x48\x12\x4A\x7F\x2C\x48\x47\xB9\xDE\x04\xA9\x01\x00\x06\xD5'):
                addr = list(multiord(p.payload[2:8]))
                break
        self.bt.end_scan()  # Packet(00, 06, 04, [00 00])

        ## connect and wait for status event
        conn_pkt = self.bt.connect(addr)  # Packet(00, 06, 03, [00 00 00])可能是蓝牙连接的意思。找到了特定的臂环
        # -------------------2017.2.11上面都看完了--------------------------------------------------------
        self.conn = multiord(conn_pkt.payload)[-1]  # 这是唯一给conn赋值的地方，为0

        self.bt.wait_event(3, 0)  # 不懂什么意思,每次特定addr的读写都要wait

        ## get firmware version
        fw = self.read_attr(0x17)  # firmware
        _, _, _, _, v0, v1, v2, v3 = unpack('BHBBHHHH', fw.payload)
        print('firmware version: %d.%d.%d.%d' % (v0, v1, v2, v3))  # 固件版本1.5.1970.2

        self.old = (v0 == 0)  # v0=1，返回false

        if self.old:  # 这个条件一直没进去，old指的是老版本
            ## don't know what these do; Myo Connect sends them, though we get data
            ## fine without them
            self.write_attr(0x19, b'\x01\x02\x00\x00')
            self.write_attr(0x2f, b'\x01\x00')
            self.write_attr(0x2c, b'\x01\x00')
            self.write_attr(0x32, b'\x01\x00')
            self.write_attr(0x35, b'\x01\x00')

            ## enable EMG data
            self.write_attr(0x28, b'\x01\x00')
            ## enable IMU data
            self.write_attr(0x1d, b'\x01\x00')

            ## Sampling rate of the underlying EMG sensor, capped to 1000. If it's
            ## less than 1000, emg_hz is correct. If it is greater, the actual
            ## framerate starts dropping inversely. Also, if this is much less than
            ## 1000, EMG data becomes slower to respond to changes. In conclusion,
            ## 1000 is probably a good value.
            C = 1000
            emg_hz = 50
            ## strength of low-pass filtering of EMG data
            emg_smooth = 100

            imu_hz = 50

            ## send sensor parameters, or we don't get any data
            self.write_attr(0x19, pack('BBBBHBBBBB', 2, 9, 2, 1, C, emg_smooth, C // emg_hz, imu_hz, 0, 0))
        else:
            name = self.read_attr(0x03)
            print('device name: %s' % name.payload)

            ## enable IMU data
            self.write_attr(0x1d, b'\x01\x00')
            ## enable on/off arm notifications
            self.write_attr(0x24, b'\x02\x00')

            # self.write_attr(0x19, b'\x01\x03\x00\x01\x01')
            self.start_raw()

        ## add data handlers
        def handle_data(p):
            if (p.cls, p.cmd) != (4, 5): return

            c, attr, typ = unpack('BHB', p.payload[:4])
            pay = p.payload[5:]

            if attr == 0x27:
                vals = unpack('8HB', pay)
                ## not entirely sure what the last byte is, but it's a bitmask that
                ## seems to indicate which sensors think they're being moved around or
                ## something
                emg = vals[:8]
                '''
                将肌电数据传出，revised by hxh
                '''
                self.emg = emg
                # print(emg)
                moving = vals[8]
                self.on_emg(emg, moving)
            elif attr == 0x1c:
                vals = unpack('10h', pay)
                quat = vals[:4]
                acc = vals[4:7]
                gyro = vals[7:10]
                self.on_imu(quat, acc, gyro)
            elif attr == 0x23:
                typ, val, xdir, _, _, _ = unpack('6B', pay)

                if typ == 1:  # on arm
                    self.on_arm(Arm(val), XDirection(xdir))
                elif typ == 2:  # removed from arm
                    self.on_arm(Arm.UNKNOWN, XDirection.UNKNOWN)
                elif typ == 3:  # pose
                    self.on_pose(Pose(val))
                    '''
                    我自己加的，用于发送数据
                    '''
                    # print(val)
                    # gesture = "%c" % val
                    # wirles.write(gesture)


            else:
                print('data with unknown attr: %02X %s' % (attr, p))

        self.bt.add_handler(handle_data)

    def write_attr(self, attr, val):
        if self.conn is not None:
            self.bt.write_attr(self.conn, attr, val)

    def read_attr(self, attr):
        if self.conn is not None:
            return self.bt.read_attr(self.conn, attr)
        return None

    def disconnect(self):
        if self.conn is not None:
            self.bt.disconnect(self.conn)

    def start_raw(self):
        '''Sending this sequence for v1.0 firmware seems to enable both raw data and
        pose notifications.
        '''

        self.write_attr(0x28, b'\x01\x00')
        # self.write_attr(0x19, b'\x01\x03\x01\x01\x00')
        self.write_attr(0x19, b'\x01\x03\x01\x01\x01')  # 这个是使能是否发送5个手势。

    def mc_start_collection(self):
        '''Myo Connect sends this sequence (or a reordering) when starting data
        collection for v1.0 firmware; this enables raw data but disables arm and
        pose notifications.
        '''

        self.write_attr(0x28, b'\x01\x00')
        self.write_attr(0x1d, b'\x01\x00')
        self.write_attr(0x24, b'\x02\x00')
        self.write_attr(0x19, b'\x01\x03\x01\x01\x01')
        self.write_attr(0x28, b'\x01\x00')
        self.write_attr(0x1d, b'\x01\x00')
        self.write_attr(0x19, b'\x09\x01\x01\x00\x00')
        self.write_attr(0x1d, b'\x01\x00')
        self.write_attr(0x19, b'\x01\x03\x00\x01\x00')
        self.write_attr(0x28, b'\x01\x00')
        self.write_attr(0x1d, b'\x01\x00')
        self.write_attr(0x19, b'\x01\x03\x01\x01\x00')

    def mc_end_collection(self):
        '''Myo Connect sends this sequence (or a reordering) when ending data collection
        for v1.0 firmware; this reenables arm and pose notifications, but
        doesn't disable raw data.
        '''

        self.write_attr(0x28, b'\x01\x00')
        self.write_attr(0x1d, b'\x01\x00')
        self.write_attr(0x24, b'\x02\x00')
        self.write_attr(0x19, b'\x01\x03\x01\x01\x01')
        self.write_attr(0x19, b'\x09\x01\x00\x00\x00')
        self.write_attr(0x1d, b'\x01\x00')
        self.write_attr(0x24, b'\x02\x00')
        self.write_attr(0x19, b'\x01\x03\x00\x01\x01')
        self.write_attr(0x28, b'\x01\x00')
        self.write_attr(0x1d, b'\x01\x00')
        self.write_attr(0x24, b'\x02\x00')
        self.write_attr(0x19, b'\x01\x03\x01\x01\x01')

    def vibrate(self, length):
        if length in xrange(1, 4):
            ## first byte tells it to vibrate; purpose of second byte is unknown
            self.write_attr(0x19, pack('3B', 3, 1, length))

    def add_emg_handler(self, h):
        self.emg_handlers.append(h)

    def add_imu_handler(self, h):
        self.imu_handlers.append(h)

    def add_pose_handler(self, h):
        self.pose_handlers.append(h)

    def add_arm_handler(self, h):
        self.arm_handlers.append(h)

    '''
    增加删除handler机制
    '''

    def delet_emg_handler(self, h):
        self.emg_handlers.remove(h)

    def on_emg(self, emg, moving):
        for h in self.emg_handlers:
            h(emg, moving)

    def on_imu(self, quat, acc, gyro):
        for h in self.imu_handlers:
            h(quat, acc, gyro)

    def on_pose(self, p):
        for h in self.pose_handlers:
            h(p)

    def on_arm(self, arm, xdir):
        for h in self.arm_handlers:
            h(arm, xdir)
    # 在这个基础上增加串口发送数据


if __name__ == '__main__':  # 不输入命令行参数
    last_vals = None

    w, h = 1200, 400
    scr = pygame.display.set_mode((w, h))  # scr = pygame.Surface pygame object for representing images


    def plot(scr, vals):  # 窗体对象和八路肌电信号组成的list
        DRAW_LINES = True  # 修改True或者False可以改变绘图风格

        global last_vals  # 申明全局变量 这样可以在函数中对该变量的赋值进行变化
        if last_vals is None:
            last_vals = vals
            return

        D = 5
        scr.scroll(-D)  # 图像左移5个像素点Move the image by dx pixels right and dy pixels down,dx dy can be negative
        scr.fill((0, 0, 0), (w - D, 0, w, h))  # 填充固定区域（黑色）（矩形的左上和右下坐标）（（1200-5，0）（1200，400））移动后图像有滞留，刷成黑色
        for i, (u, v) in enumerate(zip(last_vals, vals)):  # 他是怎么画线的？
            if DRAW_LINES:
                pygame.draw.line(scr, (0, 255, 0),  # 绿色数据线
                                 (w - D, int(h / 8 * (i + 1 - u))),  # 起始坐标
                                 (w, int(h / 8 * (i + 1 - v))))  # 终点坐标
                pygame.draw.line(scr, (255, 255, 255),  # 白色坐标轴线
                                 (w - D, int(h / 8 * (i + 1))),  # 起点坐标
                                 (w, int(h / 8 * (i + 1))))  # 终点坐标
            else:
                c = int(255 * max(0, min(1, v)))
                scr.fill((c, c, c), (w - D, i * h / 8, D, (i + 1) * h / 8 - i * h / 8));

        pygame.display.flip()  # 刷新整个画面
        last_vals = vals


    def proc_emg(emg, moving, times=[]):  # 绘制曲线，发送数据函数。定义一个list的time，最上面调用了time模块
        if HAVE_PYGAME:
            ## update pygame display

            plot(scr, [e / 2000. for e in emg])  # 把emg的每一个元素除以2000，然后重组为一个list
        else:
            print(emg)  # emg是一个8个元素的list

        ## print framerate of received data
        times.append(time.time())  # 打印最近20个时间戳，体现了函数调用频率
        if len(times) > 20:  # 20是元素个数
            # print((len(times) - 1) / (times[-1] - times[0]))
            times.pop(0)


    '''
    以下是我加的一段代码。增加串口通信
    '''
    # wirles = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.5)

    # 参数要么是None（无命令行参数），要么是sys.argv的第二个元素（第一个命令行参数，键入的其他设备名）
    m = MyoRaw(sys.argv[1] if len(sys.argv) >= 2 else None)  # 命令行传入参数为设备名，不传表示自动检测，可以理解为串口配置

    # 假设命令行未传入参数
    m.connect()  # 进行连接，也是MyoRaw的一个方法，把数据赋值给emg和moving.
    # 只调用了一次connect
    m.add_emg_handler(proc_emg)  # 采集信息，MyoRaw的一个方法，只是为了调用？
    m.add_arm_handler(lambda arm, xdir: print('arm', arm, 'xdir', xdir))  # 识别信息
    m.add_pose_handler(lambda p: print('pose', p))  # 把函数都赋值给一个list了
    try:
        while True:  # time out就是超时的意思！
            m.run()  # 这个数据可能要调试。同样是一个方法，理解成一个函数。0会有延迟，2正常，不知道怎么调,不写也ok

            if HAVE_PYGAME:
                for ev in pygame.event.get():
                    if ev.type == QUIT or (ev.type == KEYDOWN and ev.unicode == 'q'):  # 如果按关闭按钮或者键盘上的q
                        raise KeyboardInterrupt()
                    elif ev.type == KEYDOWN:  # K_1就是数字键1,ev.unicode ='q'就是字母键q
                        if K_1 <= ev.key <= K_3:  # 数字1键震动时间最短，3时间最长
                            m.vibrate(ev.key - K_0)
                        if K_KP1 <= ev.key <= K_KP3:
                            m.vibrate(ev.key - K_KP0)
    # 感觉ctrl+C是强制关闭，跟代码无关
    except KeyboardInterrupt:  # 用户中断执行，如果进入了except，有finally肯定会进入.
        pass
    finally:
        m.disconnect()
        print()
