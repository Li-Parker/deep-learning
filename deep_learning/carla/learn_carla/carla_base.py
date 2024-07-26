#此脚本的作用是创建一个基本的carla仿真环境
import atexit

import carla
import random
import cv2
import numpy as np


class Carla:
    def __init__(self,autodrive,random_spawn_point):
        self.frame = None
        self.autopilot=autodrive
        self.spawn_point=random_spawn_point
        self.vehicle=None
        self.base_environment()

    def base_environment(self):
        #首先创建连接
        client=carla.Client('localhost',2000)
        #也可以使用client.load_word('map')可加载不同的地图
        world=client.get_world()
        try:
            world=client.load_world('Town05')
            client.set_timeout(100.0)
        except RuntimeError:
            print('loading world Town05')
            #设置time_out
            client.set_timeout(100.0)


        #从蓝图中拿出tesla model3
        blueprint_library=world.get_blueprint_library()
        vehicle=blueprint_library.filter('model3')[0]
        #设置model3的初始位置
        if self.spawn_point==False:
            spawn_point=random.choice(world.get_map().get_spawn_points())
        else:
            spawn_point=world.get_map().get_spawn_points()[self.spawn_point]
        self.vehicle=world.spawn_actor(vehicle,spawn_point)
        #设置model3的驾驶模式
        self.vehicle.set_autopilot(self.autopilot)

        # 设置camera的放置位置以及camera的属性
        blueprint=blueprint_library.find('sensor.camera.rgb')
        blueprint.set_attribute('image_size_x',f'{1280}')
        blueprint.set_attribute('image_size_y',f'{720}')
        blueprint.set_attribute('fov','70')
        transform=carla.Transform(carla.Location(x=0.4,z=1.2))

        sensor=world.spawn_actor(blueprint,transform,attach_to=self.vehicle)
        sensor.listen(self.data_process)
        # 因为model3是随机生成的地点，所以需要先给个视角好定位，当然可以在仿真的环境中一直保持这个视角
        #只要在while True中添加下面的代码即可
        spectator = world.get_spectator()
        transform = self.vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(x=-10, z=10),
                                                carla.Rotation(pitch=-40)))
        # while True:
        #     world.tick()
        #atexit模块的主要作用是在程序即将结束之间执行的代码。atexit模块使用register函数用于注册程序退出时的回调函数。
        #在程序退出的时候清除sensor
        def destroy():
            sensor.destroy()
        atexit.register(destroy)
    #sensor的回调函数，主要进行一些简单的数据处理
    def data_process(self,image):
        #frombuffer将data以流的形式读入转化成ndarray对象
        image=np.frombuffer(image.raw_data,dtype=np.dtype('uint8'))
        image=image.reshape(720,1280,4)
        self.frame=cv2.cvtColor(image,cv2.COLOR_BGRA2BGR)

if __name__ == '__main__':
    Carla(True,False)
