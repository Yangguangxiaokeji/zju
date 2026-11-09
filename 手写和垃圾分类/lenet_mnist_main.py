###############################################################################
# 测试用 lenet_mnist_main.py：仅包含模型加载与测试集评测，不含任何训练代码。
# 训练请跑 lenet_mnist.ipynb / lenet_mnist.py，跑完把最佳权重路径填到 MODEL_PATH。
###############################################################################
import os
import argparse
import shutil
import struct

import numpy as np

import mindspore as ms
import mindspore.context as context
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV

from mindspore import nn
from mindspore.train import Model
from mindspore.train.serialization import load_checkpoint

os.environ['GLOG_v'] = '2'  # Log Level = Error
has_gpu = shutil.which('nvidia-smi') is not None
print('Excuting with', 'GPU' if has_gpu else 'CPU', '.')
context.set_context(mode=context.GRAPH_MODE, device_target='GPU' if has_gpu else 'CPU')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, 'datasets/5fbdf571c06d3433df85ac65-momodel/MNIST')

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


def read_mnist_dataset(data_path, indices=None):
    """Windows 中文路径使用 IDX 读取，其他路径使用原生 MNIST 读取器。"""
    options = {'num_parallel_workers': 4}
    if indices is None:
        options['shuffle'] = False
    else:
        options['sampler'] = ms.dataset.SubsetSampler(np.asarray(indices).tolist())
    if os.name != 'nt' or data_path.isascii():
        return ms.dataset.MnistDataset(data_path, **options)
    prefix = 't10k' if os.path.basename(os.path.normpath(data_path)) == 'test' else 'train'
    with open(os.path.join(data_path, prefix + '-images-idx3-ubyte'), 'rb') as source:
        magic, count, height, width = struct.unpack('>IIII', source.read(16))
        images = np.frombuffer(source.read(), dtype=np.uint8)
    with open(os.path.join(data_path, prefix + '-labels-idx1-ubyte'), 'rb') as source:
        label_magic, label_count = struct.unpack('>II', source.read(8))
        labels = np.frombuffer(source.read(), dtype=np.uint8)
    if magic != 2051 or label_magic != 2049 or count != label_count:
        raise ValueError('invalid MNIST IDX headers')
    if images.size != count * height * width or labels.size != count:
        raise ValueError('incomplete MNIST IDX data')
    images = images.reshape(count, height, width, 1)
    return ms.dataset.NumpySlicesDataset((images, labels), column_names=['image', 'label'], **options)


def create_test_dataset(data_dir, batch_size=100, resize=(32, 32),
                        rescale=1/(255*0.3081), shift=-0.1307/0.3081):
    # 缩放、归一化与通道顺序与训练脚本一致。
    data_test = os.path.join(data_dir, 'test')
    print(data_test)
    ds = read_mnist_dataset(data_test)
    ds = ds.map(input_columns=["image"], operations=[CV.Resize(resize), CV.Rescale(rescale, shift), CV.HWC2CHW()], num_parallel_workers=4)
    ds = ds.map(input_columns=["label"], operations=C.TypeCast(ms.int32), num_parallel_workers=4)
    # 保留不足一批的样本，改变 batch_size 时也能评测完整测试集。
    ds = ds.batch(batch_size, drop_remainder=False)

    return ds


def evaluate(data_dir=DATA_PATH, model_path=None, batch_size=100):
    model_path = model_path or MODEL_PATH
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            '找不到 checkpoint: {}\n'
            '请把训练日志里 "Chosen checkpoint is ..." 的路径填到 MODEL_PATH，'
            '或使用 --model-path 指定权重'.format(model_path))

    ds_test = create_test_dataset(data_dir, batch_size=batch_size)
    net = LeNet5()
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    load_checkpoint(model_path, net=net)
    net.set_train(False)
    model = Model(net, loss, metrics={'acc', 'loss'})
    metric = model.eval(ds_test, dataset_sink_mode=False)

    print('MODEL_PATH :', model_path)
    print('测试集批次数:', ds_test.get_dataset_size())
    print('准确率: {:.4f}   损失: {:.4f}'.format(metric['acc'], metric['loss']))

    return metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained LeNet5 checkpoint on MNIST test data')
    parser.add_argument('--model-path', default=MODEL_PATH)
    parser.add_argument('--data-path', default=DATA_PATH)
    parser.add_argument('--batch-size', type=int, default=100)
    args = parser.parse_args()
    evaluate(args.data_path, args.model_path, args.batch_size)
