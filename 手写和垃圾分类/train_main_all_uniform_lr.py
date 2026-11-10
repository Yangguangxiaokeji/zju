###############################################################################
# 第二阶段训练脚本：全网微调（fine-tune）
#
# 背景：
#   stage-1（train_main.py）冻结 backbone、只在缓存特征上训 head。
#   缓存特征只有一份随机增强抽签，head 两三个 epoch 就把它背下来
#   （train loss -> 0.004，val acc 钉死在 0.877），这是当前精度天花板。
#
# 本脚本在 stage-1 最佳 checkpoint 的基础上解冻整个网络，用真实图片
# （每个 epoch 都是新的随机增强）以小学习率微调少量 epoch。
#
# 验证集基线保护（不保证未见测试集分数不下降）：
#   1. 权重起点 = stage-1 最佳 ckpt（测试 91.54 的那个），不是从头训；
#   2. 训练前先在 val 上评估一次 stage-1 权重，作为 epoch 0 基线，
#      并参与 best 选择 => 最终选出的 ckpt 的 val acc 数学上 >= stage-1；
#   3. backbone_lr_max=0.002、head_lr_max=0.002，两组按相同进度 cosine 退火；
#      backbone 小步调整预训练特征，head 使用相对较大的学习率；
#   4. 若 best epoch = 0，说明微调没带来收益，直接继续用 stage-1 ckpt，
#      零损失退出。
#
# 使用方法（云端离线任务）：
#   1. 新建【GPU】离线任务（全网微调含 backbone 前向+反向，CPU 容器会非常慢），
#      勾选：train_main_all_uniform_lr.py、src_mindspore/、datasets/，
#      以及 stage-1 最佳权重 results/ckpt_mobilenetv2-20261109-153207/mobilenetv2-5.ckpt
#      （若勾选界面无法展开 results 勾选单个文件，则先把该 ckpt 复制到项目根目录
#        改名 stage1_best.ckpt，并把下方 config.stage1_ckpt 改成 './stage1_best.ckpt'；
#        不要勾选整个 results 目录，历史 ckpt 与特征缓存太大）；
#   2. 任务结束后，产物回到 results/ckpt_mobilenetv2_all-<时间戳>/，
#      按日志打印的 best epoch，把 main.py 的 model_path 指向
#        ./results/ckpt_mobilenetv2_all-<时间戳>/mobilenetv2_all-<best>.ckpt
#
# 重要: 请务必把任务(jobs)中需要保存的文件存放在 results 文件夹内
###############################################################################
import math
import numpy as np
import os
import shutil
import sys
import time
from matplotlib import pyplot as plt
from easydict import EasyDict

import mindspore as ms
from mindspore import context
from mindspore import nn
from mindspore import Tensor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, save_checkpoint

# 路径引导：保证离线任务里无论 cwd 在哪都能找到 src_mindspore（与 train_main.py 一致）
_base = os.path.dirname(os.path.abspath(__file__))
_candidates = [_base, os.path.dirname(_base), os.getcwd(), os.path.dirname(os.getcwd())]
PROJECT_ROOT = next((c for c in _candidates if os.path.isdir(os.path.join(c, 'src_mindspore'))), _base)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
print('project root:', PROJECT_ROOT)

from src_mindspore.dataset import create_dataset  # 数据处理脚本
from src_mindspore.mobilenetv2 import MobileNetV2Backbone, mobilenet_v2  # 模型定义脚本

os.environ['GLOG_v'] = '2'  # Log Level = Error
has_gpu = shutil.which('nvidia-smi') is not None
print('Excuting with', 'GPU' if has_gpu else 'CPU', '.')
if not has_gpu:
    print('警告: 全网微调在 CPU 上极慢，强烈建议改用 GPU 离线任务！')
context.set_context(mode=context.GRAPH_MODE, device_target='GPU' if has_gpu else 'CPU')

# 垃圾分类数据集标签，以及用于标签映射的字典。
index = {'00_00': 0, '00_01': 1, '00_02': 2, '00_03': 3, '00_04': 4, '00_05': 5, '00_06': 6, '00_07': 7,
         '00_08': 8, '00_09': 9, '01_00': 10, '01_01': 11, '01_02': 12, '01_03': 13, '01_04': 14,
         '01_05': 15, '01_06': 16, '01_07': 17, '02_00': 18, '02_01': 19, '02_02': 20, '02_03': 21,
         '03_00': 22, '03_01': 23, '03_02': 24, '03_03': 25}
inverted = {0: 'Plastic Bottle', 1: 'Hats', 2: 'Newspaper', 3: 'Cans', 4: 'Glassware', 5: 'Glass Bottle', 6: 'Cardboard', 7: 'Basketball',
            8: 'Paper', 9: 'Metalware', 10: 'Disposable Chopsticks', 11: 'Lighter', 12: 'Broom', 13: 'Old Mirror', 14: 'Toothbrush',
            15: 'Dirty Cloth', 16: 'Seashell', 17: 'Ceramic Bowl', 18: 'Paint bucket', 19: 'Battery', 20: 'Fluorescent lamp', 21: 'Tablet capsules',
            22: 'Orange Peel', 23: 'Vegetable Leaf', 24: 'Eggshell', 25: 'Banana Peel'}

# 微调超参：统一学习率、少 epoch、cosine 退火，epoch-0 验证集基线参与选择
config = EasyDict({
    "num_classes": 26,
    "reduction": 'mean',
    "image_height": 224,
    "image_width": 224,
    "batch_size": 24,
    "eval_batch_size": 10,
    "epochs": 8,               # 微调 epoch 不宜多，多了容易在小数据集上过拟合
    "backbone_lr_max": 0.002,  # 与 head 使用相同学习率
    "head_lr_max": 0.002,      # 与 backbone 使用相同学习率
    "decay_type": 'cosine',    # 退火到 0，末程平滑收敛
    "warmup_steps": 0,
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "class_index": index,
    "dataset_path": "./datasets/5fbdf571c06d3433df85ac65-momodel/garbage_26x100",
    "stage1_ckpt": './results/ckpt_mobilenetv2-20261109-153207/mobilenetv2-5.ckpt',  # stage-1 最佳权重，需在离线任务中勾选该文件（或复制到根目录改名 stage1_best.ckpt 后改此路径）
    "save_ckpt_path": './results/ckpt_mobilenetv2_all_uniform_lr',
    "curve_path": './results/training_curve_all_uniform_lr.png',
})


def build_lr(total_steps, lr_init=0.0, lr_end=0.0, lr_max=0.1, warmup_steps=0, decay_type='cosine'):
    """生成学习率数组（与 train_main.py 相同实现）。"""
    lr_init, lr_end, lr_max = float(lr_init), float(lr_end), float(lr_max)
    decay_steps = total_steps - warmup_steps
    lr_all_steps = []
    inc_per_step = (lr_max - lr_init) / warmup_steps if warmup_steps else 0
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + inc_per_step * (i + 1)
        else:
            if decay_type == 'cosine':
                cosine_decay = 0.5 * (1 + math.cos(math.pi * (i - warmup_steps) / decay_steps))
                lr = (lr_max - lr_end) * cosine_decay + lr_end
            elif decay_type == 'square':
                frac = 1.0 - float(i - warmup_steps) / (total_steps - warmup_steps)
                lr = (lr_max - lr_end) * (frac * frac) + lr_end
            else:
                lr = lr_max
        lr_all_steps.append(lr)
    return lr_all_steps


def build_optimizer(network, step_size):
    """为 backbone 和 head 配置同步衰减的分组学习率。"""
    total_steps = config.epochs * step_size
    backbone_lrs = build_lr(total_steps, lr_max=config.backbone_lr_max,
                            warmup_steps=config.warmup_steps, decay_type=config.decay_type)
    head_lrs = build_lr(total_steps, lr_max=config.head_lr_max,
                        warmup_steps=config.warmup_steps, decay_type=config.decay_type)

    group_params = []
    for subnet, lrs in ((network.backbone, backbone_lrs), (network.head, head_lrs)):
        params = subnet.trainable_params()
        # 保持未分组 Momentum 的默认规则：BN 的 beta/gamma 不做权重衰减。
        decay_params = [p for p in params if 'beta' not in p.name and 'gamma' not in p.name]
        no_decay_params = [p for p in params if 'beta' in p.name or 'gamma' in p.name]
        for params, weight_decay in ((decay_params, config.weight_decay), (no_decay_params, 0.0)):
            if params:
                group_params.append({'params': params, 'lr': Tensor(lrs, ms.float32),
                                     'weight_decay': weight_decay})
    group_params.append({'order_params': network.trainable_params()})
    opt = nn.Momentum(group_params, learning_rate=config.head_lr_max,
                      momentum=config.momentum, weight_decay=config.weight_decay)
    return opt, backbone_lrs, head_lrs


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
    # 结构必须与 train_main.py / stage-1 ckpt 完全一致（has_bias=True），
    # 否则 load_checkpoint 按名字匹配时参数对不上
    def __init__(self, input_channel=1280, hw=7, num_classes=1000, reduction='mean', activation="None"):
        self.need_activation = True
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


def build_network():
    """构建整网并加载 stage-1 最佳权重，然后解冻所有参数。"""
    if not os.path.exists(config.stage1_ckpt):
        raise FileNotFoundError(
            "找不到 stage-1 权重: {}\n"
            "离线任务只同步勾选的文件——请在建任务时勾选该 ckpt 文件；\n"
            "若无法勾选 results 内的单个文件，可将其复制到项目根目录改名\n"
            "stage1_best.ckpt，并把 config.stage1_ckpt 改为 './stage1_best.ckpt'。".format(config.stage1_ckpt))

    backbone = MobileNetV2Backbone()
    head = MobileNetV2Head(input_channel=backbone.out_channels,
                           num_classes=config.num_classes,
                           reduction=config.reduction)
    network = mobilenet_v2(backbone, head)
    # stage-1 ckpt 是 combined 网络存的，backbone.* 与 head.* 参数名一一对应
    load_checkpoint(config.stage1_ckpt, net=network)
    print('stage-1 checkpoint loaded:', config.stage1_ckpt)

    # 全网微调：解冻所有参数
    for param in network.get_parameters():
        param.requires_grad = True
    # BatchNorm 运行统计量由前向计算更新，不参与梯度和权重衰减。
    for _, cell in network.cells_and_names():
        if isinstance(cell, nn.BatchNorm2d):
            cell.moving_mean.requires_grad = False
            cell.moving_variance.requires_grad = False
    return network


def train_all():
    train_dataset = create_dataset(config=config)
    eval_dataset = create_dataset(config=config, training=False)
    step_size = train_dataset.get_dataset_size()
    if step_size == 0:
        raise ValueError("train dataset is empty, check config.dataset_path and batch_size")

    network = build_network()
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    eval_model = Model(network, loss_fn, metrics={'acc', 'loss'})

    opt, backbone_lrs, head_lrs = build_optimizer(network, step_size)
    train_net = nn.WithLossCell(network, loss_fn)
    train_one_step = nn.TrainOneStepCell(train_net, opt)
    train_one_step.set_train()
    print('training batches per epoch:', step_size)
    print('learning rate maxima: backbone={}, head={}, decay={}'.format(
        config.backbone_lr_max, config.head_lr_max, config.decay_type))

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    # epoch 0：stage-1 基线（未做任何更新），参与 best 选择 => 保底不降
    network.set_train(False)
    base_result = eval_model.eval(eval_dataset, dataset_sink_mode=False)
    history['train_loss'].append(0.0)
    history['val_loss'].append(base_result['loss'])
    history['val_acc'].append(base_result['acc'])
    print("epoch: 0 (stage-1 baseline), val acc: {}, val loss: {}".format(
        base_result['acc'], base_result['loss']))

    step = 0
    for epoch in range(config.epochs):
        network.set_train()
        epoch_start = time.time()
        losses = []
        for batch_step, batch in enumerate(train_dataset.create_dict_iterator(), start=1):
            loss = train_one_step(batch['image'], batch['label'])
            loss_value = float(loss.asnumpy())
            losses.append(loss_value)
            if batch_step == 1 or batch_step % 10 == 0 or batch_step == step_size:
                print('epoch: {}, step: {}/{}, loss: {:.6f}, backbone lr: {:.8e}, head lr: {:.8e}, elapsed: {:.1f}s'.format(
                    epoch + 1, batch_step, step_size, loss_value, backbone_lrs[step], head_lrs[step],
                    time.time() - epoch_start),
                    flush=True)
            step += 1
        network.set_train(False)

        epoch_seconds = time.time() - epoch_start
        epoch_loss = float(np.mean(np.array(losses)))
        val_result = eval_model.eval(eval_dataset, dataset_sink_mode=False)
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_result['loss'])
        history['val_acc'].append(val_result['acc'])

        print("epoch: {}, time cost: {}, train loss: {}, val acc: {}, val loss: {}".format(
            epoch + 1, epoch_seconds, epoch_loss, val_result['acc'], val_result['loss']))
        save_checkpoint(network, os.path.join(config.save_ckpt_path, "mobilenetv2_all-{}.ckpt".format(epoch + 1)))

    # plot curves（x 轴 0 = stage-1 基线）
    plt.switch_backend('Agg')
    epochs_range = range(len(history['val_acc']))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs_range, history['train_loss'], label='train loss')
    ax1.plot(epochs_range, history['val_loss'], label='val loss')
    ax1.set_xlabel('Epoch (0 = stage-1 baseline)')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss curve (fine-tune)')
    ax1.legend()
    ax2.plot(epochs_range, history['val_acc'], label='val acc', color='green')
    ax2.set_xlabel('Epoch (0 = stage-1 baseline)')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation accuracy (fine-tune)')
    ax2.legend()
    plt.savefig(config.curve_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    best_epoch = int(np.argmax(history['val_acc']))
    print("curve saved to", config.curve_path)
    print("best epoch: {}, best val acc: {}".format(best_epoch, max(history['val_acc'])))
    return history, best_epoch


EXPERIMENT_NAME = "全网统一学习率"

def append_training_record(history, best_epoch, run_ts):
    """Append this successful fine-tuning run to results/训练轨迹.md."""
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    trace_path = os.path.join(results_dir, '训练轨迹.md')
    checkpoint_name = (
        config.stage1_ckpt if best_epoch == 0
        else os.path.join(config.save_ckpt_path, "mobilenetv2_all-{}.ckpt".format(best_epoch))
    )
    checkpoint_path = os.path.relpath(checkpoint_name, results_dir).replace(os.sep, '/')
    curve_path = os.path.relpath(config.curve_path, results_dir).replace(os.sep, '/')
    checkpoint_exists = os.path.isfile(checkpoint_name)
    checkpoint_record = (
        '- 最佳 checkpoint：[{}]({})'.format(os.path.basename(checkpoint_name), checkpoint_path)
        if checkpoint_exists else
        '- 最佳 checkpoint：{}（文件不存在，请核对训练输出）'.format(checkpoint_path)
    )
    checkpoint_dir = os.path.relpath(config.save_ckpt_path, results_dir).replace(os.sep, '/')
    lines = [
        '',
        '',
        '## {} 训练记录 {}'.format(EXPERIMENT_NAME, run_ts.lstrip('-')),
        '',
        '- checkpoint 文件夹：[{}]({}/)'.format(checkpoint_dir, checkpoint_dir),
        checkpoint_record,
        '- 最佳轮次：{}'.format(best_epoch),
        '- 最佳验证准确率：{:.6f}'.format(float(history['val_acc'][best_epoch])),
        '- 对应验证损失：{:.6f}'.format(float(history['val_loss'][best_epoch])),
        '- 训练曲线：[查看 loss 与验证准确率曲线]({})'.format(curve_path),
        '',
        '| 超参数 / 配置 | 本次取值 |',
        '| --- | --- |',
    ]
    for name, value in config.items():
        if name != 'class_index':
            value_text = str(value).replace('|', '\\|').replace('\n', ' ')
            lines.append('| {} | {} |'.format(name, value_text))
    needs_header = not os.path.exists(trace_path) or os.path.getsize(trace_path) == 0
    with open(trace_path, 'a', encoding='utf-8', errors='strict', newline='\n') as output:
        if needs_header:
            output.write('# 训练轨迹\n\n记录训练超参数、checkpoint 文件夹、验证结果和训练曲线。每次成功训练后追加记录。\n')
        output.write('\n'.join(lines) + '\n')
    print('training record saved to', trace_path)
    return trace_path


if __name__ == '__main__':
    run_ts = time.strftime('-%Y%m%d-%H%M%S')
    config.save_ckpt_path += run_ts
    config.curve_path = os.path.join(config.save_ckpt_path, os.path.basename(config.curve_path))
    os.makedirs(config.save_ckpt_path)
    print("checkpoints will be saved to", config.save_ckpt_path)

    history, best_epoch = train_all()
    append_training_record(history, best_epoch, run_ts)

    if best_epoch == 0:
        print("微调未超过 stage-1 基线：请继续使用 stage-1 权重", config.stage1_ckpt)
    else:
        CKPT = 'mobilenetv2_all-{}.ckpt'.format(best_epoch)
        print("Chosen checkpoint is", CKPT)
        print("请把 main.py 的 model_path 指向: {}/{}".format(config.save_ckpt_path, CKPT))
    print('training is ok!!!')
