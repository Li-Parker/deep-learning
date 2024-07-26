# 导入所需的库
import numpy as np
import cv2

if __name__ == '__main__':
    # from matplotlib import pyplot as plt

    # 读取输入图像
    img = cv2.imread('D:\Desktop\pythonProject\deep_learning\carla\data_set\Town01_long\\routes_town01_11_05_20_55_58\\rgb_front\\0058.png')

    # 定义蒙版
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # 定义矩形
    rect = (150, 50, 500, 470)

    # 应用grabCut方法提取前景
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 20, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    # 显示提取出的前景图像

    # plt.imshow(img),plt.colorbar(),plt.show()
    cv2.imshow('前景图像', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
