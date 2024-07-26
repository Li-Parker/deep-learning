import carla
import random
import numpy as np
import cv2
from carla_base import Carla
# 数据集相关配置
import torch
from config_mine import TEST_DATASET_SIZE, TRAIN_DATASET_SIZES, WIDTH, HEIGHT, DATASET_PATH


class Data_collect(Carla):
    def __init__(self):
        # 继承Carla
        super().__init__(True, False)
        # 用于存储图片数据
        self.x_data = []
        # 用于存储转向数据
        self.y_data = []
        self.data_collect()

    def data_collect(self):
        running = True
        vehicle = self.vehicle
        count = 0
        dataset_size = TEST_DATASET_SIZE + TRAIN_DATASET_SIZES
        while running:
            if self.frame is not None:
                # 当车辆遇到交通灯时，设置交通灯为绿
                if vehicle.is_at_traffic_light():
                    traffic_light = vehicle.get_traffic_light()
                    traffic_light.set_state(carla.TrafficLightState.Green)
                # 显示实时结果
                cv2.imshow(' ', cv2.resize(self.frame, (WIDTH, HEIGHT)))
                cv2.waitKey(50)
                vehicle_data = vehicle.get_control()
                steer = vehicle_data.steer
                self.collect_data(steer)
                count += 1

                if count >= dataset_size:
                    running = False

                print('f[{count}/{dataset_size}]')
        # 将结果用npy格式存储
        np.save(DATASET_PATH + 'y_train.npy', np.array(self.y_data), allow_pickle=True)
        np.save(DATASET_PATH + 'x_train.npy', np.array(self.x_data), allow_pickle=True)

    def collect_data(self, steer):
        frame = self.frame
        frame = cv2.resize(frame, (200, 66))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.x_data.append(frame)
        self.y_data.append([steer])


if __name__ == '__main__':
    Data_collect()
