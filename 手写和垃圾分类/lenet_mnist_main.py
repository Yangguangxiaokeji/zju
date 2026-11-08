###############################################################################
# 测试用 lenet_mnist_main.py：仅包含模型加载与测试集评测，不含任何训练代码。
# 训练请跑 lenet_mnist.ipynb / lenet_mnist.py，跑完把最佳权重路径填到 MODEL_PATH。
###############################################################################
import os

import mindspore as ms
import mindspore.context as context
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV

from mindspore import nn
from mindspore.train import Model
from mindspore.train.serialization import load_checkpoint

os.environ['GLOG_v'] = '2'  # Log Level = Error
has_gpu = (os.system('command -v nvidia-smi') == 0)
print('Excuting with', 'GPU' if has_gpu else 'CPU', '.')
context.set_context(mode=context.GRAPH_MODE, device_target='GPU' if has_gpu else 'CPU')

DATA_PATH = 'datasets/5fbdf571c06d3433df85ac65-momodel/MNIST/'

# 填写说明：跑完训练后，把日志里 "Chosen checkpoint is ..." 那一行的路径复制过来。
# ckpt 目录名带时间戳，每次重训都会变，形如：
#   ./results/ckpt_lenet-20261107-195005/lenet-best.ckpt
MODEL_PATH = './results/ckpt_lenet-<填入时间戳>/lenet-best.ckpt'


class LeNet5(nn.Cell):
    # 结构必须与 lenet_mnist.py 训练时完全一致（fc1/fc2 后带 ReLU），否则加载 checkpoint 时参数对不上
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, pad_mode='valid')
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(400, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 10)

    def construct(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def create_test_dataset(data_dir, batch_size=32, resize=(32, 32),
                        rescale=1/(255*0.3081), shift=-0.1307/0.3081):
    # 预处理链必须与训练时逐字一致，否则准确率没有可比性
    data_test = os.path.join(data_dir, 'test')
    print(data_test)
    ds = ms.dataset.MnistDataset(data_test, num_parallel_workers=4)
    ds = ds.map(input_columns=["image"], operations=[CV.Resize(resize), CV.Rescale(rescale, shift), CV.HWC2CHW()], num_parallel_workers=4)
    ds = ds.map(input_columns=["label"], operations=C.TypeCast(ms.int32), num_parallel_workers=4)
    # drop_remainder=False：10000 / 32 = 312.5，保留最后 16 张，评满整个测试集
    ds = ds.batch(batch_size, drop_remainder=False)

    return ds


def evaluate(data_dir):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            '找不到 checkpoint: {}\n'
            '请把训练日志里 "Chosen checkpoint is ..." 的路径填到 MODEL_PATH'.format(MODEL_PATH))

    ds_test = create_test_dataset(data_dir)
    net = LeNet5()
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    load_checkpoint(MODEL_PATH, net=net)
    model = Model(net, loss, metrics={'acc', 'loss'})
    metric = model.eval(ds_test, dataset_sink_mode=False)

    print('MODEL_PATH :', MODEL_PATH)
    print('测试集批次数:', ds_test.get_dataset_size())
    print('准确率: {:.4f}   损失: {:.4f}'.format(metric['acc'], metric['loss']))

    return metric


evaluate(DATA_PATH)
