###############################################################################
# 重要: 请务必把任务(jobs)中需要保存的文件存放在 results 文件夹内
# Important : Please make sure your files are saved to the 'results' folder
# in your jobs
###############################################################################
import os
import json
import shutil
import struct
import time
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mindspore as ms
import mindspore.context as context
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV

from mindspore import nn
from mindspore.train import Model
from mindspore.train.serialization import load_checkpoint, save_checkpoint

os.environ['GLOG_v'] = '2'
has_gpu = shutil.which('nvidia-smi') is not None
print('Excuting with', 'GPU' if has_gpu else 'CPU', '.')
context.set_context(mode=context.GRAPH_MODE, device_target='GPU' if has_gpu else 'CPU')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
DATA_PATH = os.path.join(PROJECT_ROOT, 'datasets/5fbdf571c06d3433df85ac65-momodel/MNIST')
TRAIN_CONFIG = {
    'lr': 0.01,
    'momentum': 0.9,
    'num_epochs': 10,
    'batch_size': 32,
    'eval_batch_size': 100,
    'val_ratio': 0.1,
    'seed': 42,
}


def get_split_indices(sample_count, val_ratio=0.1, seed=42):
    """固定划分训练集与验证集，官方测试集不参与选模型。"""
    if not 0 < val_ratio < 1:
        raise ValueError('val_ratio must be between 0 and 1')
    val_count = int(round(sample_count * val_ratio))
    if not 0 < val_count < sample_count:
        raise ValueError('training and validation splits must both contain samples')
    indices = np.random.RandomState(seed).permutation(sample_count)
    return indices[val_count:], indices[:val_count]


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


def create_dataset(data_dir, training=True, batch_size=32, resize=(32, 32),
                   rescale=1/(255*0.3081), shift=-0.1307/0.3081, buffer_size=10000,
                   split=None, indices=None):
    split = split or ('train' if training else 'test')
    if split not in ('train', 'val', 'test'):
        raise ValueError('split must be train, val or test')
    if split == 'val' and indices is None:
        raise ValueError('validation indices must come from the training split')
    data_path = os.path.join(data_dir, 'test' if split == 'test' else 'train')
    if not os.path.isdir(data_path):
        raise FileNotFoundError('找不到 MNIST 数据目录: {}'.format(data_path))
    ds = read_mnist_dataset(data_path, indices)
    ds = ds.map(input_columns=["image"], operations=[CV.Resize(resize), CV.Rescale(rescale, shift), CV.HWC2CHW()], num_parallel_workers=4)
    ds = ds.map(input_columns=["label"], operations=C.TypeCast(ms.int32), num_parallel_workers=4)
    if split == 'train':
        ds = ds.shuffle(buffer_size=buffer_size)
    ds = ds.batch(batch_size, drop_remainder=(split == 'train'))
    return ds


def create_train_val_datasets(data_dir, batch_size=32, eval_batch_size=100, val_ratio=0.1, seed=42):
    raw = read_mnist_dataset(os.path.join(data_dir, 'train'))
    train_indices, val_indices = get_split_indices(raw.get_dataset_size(), val_ratio, seed)
    ds_train = create_dataset(data_dir, batch_size=batch_size, split='train', indices=train_indices)
    ds_val = create_dataset(data_dir, batch_size=eval_batch_size, split='val', indices=val_indices)
    return ds_train, ds_val, train_indices, val_indices


def show_samples(data_dir, save_path=None):
    save_path = save_path or os.path.join(PROJECT_ROOT, 'results/mnist_samples.png')
    iterator = create_dataset(data_dir, batch_size=4, split='test').create_dict_iterator(output_numpy=True)
    batch = next(iterator)
    iterator.stop()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(5, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(batch['image'][i, 0], cmap='gray')
        ax.set_title('Number: {}'.format(batch['label'][i]))
        ax.axis('off')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


class LeNet5(nn.Cell):
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

def save_training_curve(history, curve_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history['train_loss'], label='train loss')
    ax1.plot(epochs, history['val_loss'], label='val loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss curve (LeNet5)')
    ax1.legend()
    ax2.plot(epochs, np.asarray(history['val_acc']) * 100, label='val acc', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation accuracy (LeNet5)')
    ax2.legend()
    fig.savefig(curve_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def train(data_dir, lr=0.01, momentum=0.9, num_epochs=10, ckpt_name="lenet",
          batch_size=32, eval_batch_size=100, val_ratio=0.1, seed=42, ckpt_dir=None):
    if num_epochs < 1:
        raise ValueError('num_epochs must be positive')
    ms.set_seed(seed)
    ms.dataset.config.set_seed(seed)
    ds_train, ds_val, train_indices, val_indices = create_train_val_datasets(
        data_dir, batch_size, eval_batch_size, val_ratio, seed)
    steps_per_epoch = ds_train.get_dataset_size()
    if steps_per_epoch == 0:
        raise ValueError('training split contains fewer samples than batch_size')
    ckpt_dir = ckpt_dir or os.path.join(PROJECT_ROOT, 'results/ckpt_lenet' + time.strftime('-%Y%m%d-%H%M%S'))
    os.makedirs(ckpt_dir, exist_ok=True)
    curve_path = os.path.join(ckpt_dir, 'training_curve.png')
    history_path = os.path.join(ckpt_dir, 'training_history.json')
    final_path = os.path.join(ckpt_dir, '{}-best.ckpt'.format(ckpt_name))
    np.savez_compressed(os.path.join(ckpt_dir, 'split_indices.npz'), train=train_indices, val=val_indices)
    print('checkpoints will be saved to', ckpt_dir)
    print('train samples: {}, val samples: {}, steps per epoch: {}'.format(
        len(train_indices), len(val_indices), steps_per_epoch))
    print('config: epochs={}, lr={}, momentum={}, batch_size={}, eval_batch_size={}, seed={}'.format(
        num_epochs, lr, momentum, batch_size, eval_batch_size, seed))

    net = LeNet5()
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = nn.Momentum(net.trainable_params(), lr, momentum)
    train_step = nn.TrainOneStepCell(nn.WithLossCell(net, loss), opt)
    eval_model = Model(net, loss, metrics={'acc', 'loss'})
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_acc, best_loss, best_epoch = -1.0, float('inf'), 0

    for epoch in range(1, num_epochs + 1):
        train_step.set_train()
        epoch_start = time.time()
        losses = []
        for step, batch in enumerate(ds_train.create_dict_iterator(), start=1):
            value = float(train_step(batch['image'], batch['label']).asnumpy())
            if not np.isfinite(value):
                raise FloatingPointError('non-finite loss at epoch {}, step {}'.format(epoch, step))
            losses.append(value)
            if step == 1 or step % 100 == 0 or step == steps_per_epoch:
                print('epoch: {}, step: {}/{}, loss: {:.6f}, elapsed: {:.1f}s'.format(
                    epoch, step, steps_per_epoch, value, time.time() - epoch_start), flush=True)
        train_seconds = time.time() - epoch_start
        net.set_train(False)
        metric = eval_model.eval(ds_val, dataset_sink_mode=False)
        epoch_loss = float(np.mean(losses))
        val_loss, val_acc = float(metric['loss']), float(metric['acc'])
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print('epoch: {}, time cost: {:.2f}, train loss: {:.6f}, val acc: {:.6f}, val loss: {:.6f}'.format(
            epoch, train_seconds, epoch_loss, val_acc, val_loss), flush=True)
        save_checkpoint(net, os.path.join(ckpt_dir, '{}-{}.ckpt'.format(ckpt_name, epoch)))
        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc, best_loss, best_epoch = val_acc, val_loss, epoch
            save_checkpoint(net, final_path)
        record = {
            'config': {'lr': lr, 'momentum': momentum, 'num_epochs': num_epochs,
                       'batch_size': batch_size, 'eval_batch_size': eval_batch_size,
                       'val_ratio': val_ratio, 'seed': seed},
            'train_samples': len(train_indices), 'val_samples': len(val_indices),
            'steps_per_epoch': steps_per_epoch, 'history': history,
            'best_epoch': best_epoch, 'best_val_acc': best_acc, 'best_val_loss': best_loss,
            'checkpoint_path': final_path, 'curve_path': curve_path,
        }
        with open(history_path, 'w', encoding='utf-8', errors='strict', newline='\n') as output:
            json.dump(record, output, ensure_ascii=False, indent=2)

    save_training_curve(history, curve_path)
    print('curve saved to', curve_path)
    print('history saved to', history_path)
    print('best epoch: {}, best val acc: {:.6f}, best val loss: {:.6f}'.format(
        best_epoch, best_acc, best_loss))
    print('Chosen checkpoint is', final_path)
    return final_path


def evaluate(data_dir, checkpoint_path, batch_size=100):
    ds_test = create_dataset(data_dir, batch_size=batch_size, split='test')
    net = LeNet5()
    load_checkpoint(checkpoint_path, net=net)
    net.set_train(False)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    model = Model(net, loss, metrics={'acc', 'loss'})
    metric = model.eval(ds_test, dataset_sink_mode=False)
    metric = {name: float(value) for name, value in metric.items()}
    print('test acc: {:.6f}, test loss: {:.6f}, checkpoint: {}'.format(
        metric['acc'], metric['loss'], checkpoint_path))
    with open(os.path.join(os.path.dirname(checkpoint_path), 'test_metrics.json'),
              'w', encoding='utf-8', errors='strict', newline='\n') as output:
        json.dump(metric, output, ensure_ascii=False, indent=2)
    return metric


def infer(data_dir, checkpoint_path, save_path=None):
    save_path = save_path or os.path.join(os.path.dirname(checkpoint_path), 'prediction_examples.png')
    iterator = create_dataset(data_dir, batch_size=4, split='test').create_dict_iterator()
    batch = next(iterator)
    iterator.stop()
    net = LeNet5()
    load_checkpoint(checkpoint_path, net=net)
    net.set_train(False)
    preds = np.argmax(net(batch['image']).asnumpy(), axis=1)
    images, labels = batch['image'].asnumpy(), batch['label'].asnumpy()
    fig, axes = plt.subplots(2, 2, figsize=(6, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i, 0], cmap='gray')
        ax.set_title('prediction: {}, truth: {}'.format(preds[i], labels[i]),
                     color='blue' if preds[i] == labels[i] else 'red')
        ax.axis('off')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('prediction examples saved to', save_path)
    return save_path


if __name__ == '__main__':
    print('开始训练...')
    CKPT = train(DATA_PATH, **TRAIN_CONFIG)
    evaluate(DATA_PATH, CKPT, batch_size=TRAIN_CONFIG['eval_batch_size'])
    infer(DATA_PATH, CKPT)
    print('training is ok!!!')
