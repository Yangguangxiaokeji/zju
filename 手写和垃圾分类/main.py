###############################################################################
# 提交/系统测试用 main.py：仅包含模型加载与 predict()，不含任何训练代码。
# 与 notebook 生成版等价：只保留 cell-27 / cell-28 的内容。
###############################################################################
import os
import sys
import cv2
import numpy as np
from easydict import EasyDict

import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore import context
from mindspore.train.serialization import load_checkpoint

# 保证无论运行器的工作目录在哪，都能找到同目录下的 src_mindspore
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from src_mindspore.mobilenetv2 import MobileNetV2Backbone, mobilenet_v2  # 模型定义脚本

os.environ['GLOG_v'] = '2'  # Log Level = Error
has_gpu = (os.system('command -v nvidia-smi') == 0)
print('Excuting with', 'GPU' if has_gpu else 'CPU', '.')
context.set_context(mode=context.GRAPH_MODE, device_target='GPU' if has_gpu else 'CPU')

# 系统测试部分标签与该处一致，请不要改动
inverted = {0: 'Plastic Bottle', 1: 'Hats', 2: 'Newspaper', 3: 'Cans', 4: 'Glassware', 5: 'Glass Bottle', 6: 'Cardboard', 7: 'Basketball',
            8: 'Paper', 9: 'Metalware', 10: 'Disposable Chopsticks', 11: 'Lighter', 12: 'Broom', 13: 'Old Mirror', 14: 'Toothbrush',
            15: 'Dirty Cloth', 16: 'Seashell', 17: 'Ceramic Bowl', 18: 'Paint bucket', 19: 'Battery', 20: 'Fluorescent lamp', 21: 'Tablet capsules',
            22: 'Orange Peel', 23: 'Vegetable Leaf', 24: 'Eggshell', 25: 'Banana Peel'}

config = EasyDict({
    "num_classes": 26,
    "reduction": 'mean',
    "image_height": 224,
    "image_width": 224,
})


class GlobalPooling(nn.Cell):
    def __init__(self, reduction='mean'):
        super(GlobalPooling, self).__init__()
        if reduction == 'max':
            self.mean = ms.ops.ReduceMax(keep_dims=False)
        else:
            self.mean = ms.ops.ReduceMean(keep_dims=False)

    def construct(self, x):
        x = self.mean(x, (2, 3))
        return x


class MobileNetV2Head(nn.Cell):
    # 结构必须与 train_main.py 训练时完全一致（has_bias=True），否则加载 checkpoint 时参数对不上
    def __init__(self, input_channel=1280, hw=7, num_classes=1000, reduction='mean', activation="None"):
        super(MobileNetV2Head, self).__init__()
        if reduction:
            self.flatten = GlobalPooling(reduction)
        else:
            self.flatten = nn.Flatten()
            input_channel = input_channel * hw * hw
        self.dense = nn.Dense(input_channel, num_classes, has_bias=True)
        if activation == "Sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "Softmax":
            self.activation = nn.Softmax()
        else:
            self.need_activation = False

    def construct(self, x):
        x = self.flatten(x)
        x = self.dense(x)
        if self.need_activation:
            x = self.activation(x)
        return x


# 加载训练得到的最佳模型（时间戳目录与 best epoch 见训练日志）
backbone = MobileNetV2Backbone()
head = MobileNetV2Head(input_channel=backbone.out_channels, num_classes=config.num_classes, reduction=config.reduction)
network = mobilenet_v2(backbone, head)

model_path = './results/ckpt_mobilenetv2_all_uniform_lr-20261109-172445/mobilenetv2_all-4.ckpt'
load_checkpoint(model_path, net=network)


def image_process(image):
    """Precess one image per time.

    Args:
        image: shape (H, W, C)
    """
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    image = (np.array(image) - mean) / std
    image = image.transpose((2, 0, 1))
    img_tensor = Tensor(np.array([image], np.float32))
    return img_tensor


def predict(image):
    """
    加载模型和模型预测
    主要步骤:
        1.图片处理,此处尽量与训练模型数据处理一致
        2.用加载的模型预测图片的类别
    :param image: PIL 读取的图片对象，数据类型是 np.array，shape (H, W, C)
    :return: string, 模型识别图片的类别,
            包含 'Plastic Bottle','Hats','Newspaper','Cans'等共 26 个类别
    """
    image = cv2.resize(image, (256, 256))
    image = image[16:240, 16:240]
    image = image_process(image)

    logits = network(image)
    pred = np.argmax(logits.asnumpy(), axis=1)[0]

    return inverted[pred]
