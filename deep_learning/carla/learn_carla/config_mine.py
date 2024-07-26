import torch

DATASET_PATH = 'data/'
TRAIN_DATASET_SIZES = 800
TEST_DATASET_SIZES = 200
TRAIN_DATASET_SIZE = 320
TEST_DATASET_SIZE = 80
# 图像尺寸配置
WIDTH = 1280
HEIGHT = 900

SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 20
LEARNING_RATE = 0.0002
NUM_EPOCH = 50
MODEL_PATH = 'data/'