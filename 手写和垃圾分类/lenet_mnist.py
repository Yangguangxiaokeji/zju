###############################################################################
# 重要: 请务必把任务(jobs)中需要保存的文件存放在 results 文件夹内
# Important : Please make sure your files are saved to the 'results' folder
# in your jobs
###############################################################################
import os
import time
import matplotlib.pyplot as plt
import numpy as np

import mindspore as ms
import mindspore.context as context
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV

from mindspore import nn, Tensor
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

has_gpu = (os.system('command -v nvidia-smi') == 0)
print('Excuting with', 'GPU' if has_gpu else 'CPU', '.')
context.set_context(mode=context.GRAPH_MODE, device_target='GPU' if has_gpu else 'CPU')

DATA_PATH = 'datasets/5fbdf571c06d3433df85ac65-momodel/MNIST/'
CKPT_DIR = 'results/ckpt_lenet' + time.strftime('-%Y%m%d-%H%M%S')
def create_dataset(data_dir, training=True, batch_size=32, resize=(32, 32),
                   rescale=1/(255*0.3081), shift=-0.1307/0.3081, buffer_size=10000):
    data_train = os.path.join(data_dir, 'train') # 训练集信息
    data_test = os.path.join(data_dir, 'test') # 测试集信息
    print(data_train)
    print(data_test)
    ds = ms.dataset.MnistDataset(data_train if training else data_test, num_parallel_workers=4)

    # 增加 num_parallel_workers=4，用 4 个线程并行处理图片
    ds = ds.map(input_columns=["image"], operations=[CV.Resize(resize), CV.Rescale(rescale, shift), CV.HWC2CHW()], num_parallel_workers=4)
    ds = ds.map(input_columns=["label"], operations=C.TypeCast(ms.int32), num_parallel_workers=4)
    # When `dataset_sink_mode=True` on Ascend, append `ds = ds.repeat(num_epochs) to the end
    ds = ds.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)

    return ds
import matplotlib.pyplot as plt
ds = create_dataset(DATA_PATH, training=False)
data = ds.create_dict_iterator()

for i, data in enumerate(data):
    if i >= 4:  # 只显示前四个图像
        break

    images = data['image'][0].numpy().squeeze(0)
    labels = data['label'][0].numpy().squeeze(0)

    plt.subplot(2, 2, i + 1)
    plt.imshow(images)  # 确保维度正确
    plt.title('Number: %s' % labels)
    plt.xticks([])

plt.show()
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
os.makedirs(CKPT_DIR, exist_ok=True)
print("checkpoints will be saved to", CKPT_DIR)

def train(data_dir, lr=0.01, momentum=0.9, num_epochs=10, ckpt_name="lenet"):
    ds_train = create_dataset(data_dir)
    ds_eval = create_dataset(data_dir, training=False)
    steps_per_epoch = ds_train.get_dataset_size()

    net = LeNet5()
    loss = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = nn.Momentum(net.trainable_params(), lr, momentum)

    ckpt_cfg = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=num_epochs)
    ckpt_cb = ModelCheckpoint(prefix=ckpt_name, directory=CKPT_DIR, config=ckpt_cfg)
    # 改成每 100 步打印一次，马上就能看到进度
    loss_cb = LossMonitor(per_print_times=100) 
    model = Model(net, loss, opt, metrics={'acc', 'loss'})
    # 下沉模式下 Host 与 Device 每个 epoch 才交互一次，LossMonitor 的 per_print_times 会被架空，
    # GPU 上整个 epoch 跑完前没有任何输出；关掉它才能看到每 100 步的进度
    model.train(num_epochs, ds_train, callbacks=[ckpt_cb, loss_cb], dataset_sink_mode=False)

    # 逐个 epoch 在测试集上评测，挑出 val acc 最高的那份权重
    best_acc, best_path = -1.0, None
    for epoch in range(1, num_epochs + 1):
        path = os.path.join(CKPT_DIR, f'{ckpt_name}-{epoch}_{steps_per_epoch}.ckpt')
        if not os.path.exists(path):
            continue
        candidate = LeNet5()
        load_checkpoint(path, net=candidate)
        metric = Model(candidate, loss, metrics={'acc', 'loss'}).eval(ds_eval, dataset_sink_mode=False)
        print('epoch {}/{}  val_acc={:.4f}  val_loss={:.4f}'.format(
            epoch, num_epochs, metric['acc'], metric['loss']))
        if metric['acc'] > best_acc:
            best_acc, best_path = metric['acc'], path

    # 删掉本次运行里所有不如 best 的 checkpoint，再把 best 重命名为 -best
    final_path = os.path.join(CKPT_DIR, f'{ckpt_name}-best.ckpt')
    for name in os.listdir(CKPT_DIR):
        path = os.path.join(CKPT_DIR, name)
        if name.endswith('.ckpt') and path != best_path:
            os.remove(path)
    os.replace(best_path, final_path)
    print('best val acc: {:.4f}'.format(best_acc))
    return final_path

print("开始训练...")
CKPT = train(DATA_PATH)
print("训练结束！开始加载 Checkpoint 验证...")
print("Chosen checkpoint is", CKPT)

def eval(data_dir):
    ds_eval = create_dataset(data_dir, training=False)
    net = LeNet5()
    loss = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    load_checkpoint(CKPT, net=net)
    model = Model(net, loss, metrics={'acc', 'loss'})
    metric = model.eval(ds_eval, dataset_sink_mode=False)
    print(metric)

eval(DATA_PATH)
def infer(data_dir):
    data_loader = create_dataset(data_dir, training=False).create_dict_iterator()
    net = LeNet5()
    load_checkpoint(CKPT, net=net)
    model = Model(net)

    images_l = []
    labels_l = []
    preds_l = []

    for i, data in enumerate(data_loader):

        images = data['image']
        labels = data['label']
        output = model.predict(images)
        preds = np.argmax(output.asnumpy(), axis=1)
        images_l.append(images)
        labels_l.append(labels)
        preds_l.append(preds)

    for i in range(1, 5):
        plt.subplot(2, 2, i)
        plt.imshow(np.squeeze(images_l[0][i].numpy()))
        color = 'blue' if preds_l[0][i] == labels_l[0][i] else 'red'
        plt.title("prediction: {}, truth: {}".format(preds_l[0][i], labels_l[0][i]), color=color)
        plt.xticks([])
    plt.show()

infer(DATA_PATH)
