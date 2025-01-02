# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
这是YOLOv5模型的一部分实现，主要涉及检测（Detect）和分割（Segment）头部的定义
以及基础模型（BaseModel）和检测模型（DetectionModel）的定义。

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

"""
Detect 类：
用于YOLOv5的检测头部，负责生成最终的检测结果。
__init__ 方法初始化检测层，包括锚点、卷积层等。
forward 方法执行前向传播，生成检测框和置信度。
_make_grid 方法用于生成网格和锚点网格。
Segment 类：
继承自 Detect 类，用于YOLOv5的分割头部。
增加了分割掩码的生成部分。
forward 方法除了生成检测框外，还生成分割掩码。

BaseModel 类：
YOLOv5的基础模型类，提供了前向传播、性能分析、模型融合等功能。
forward 和 _forward_once 方法实现了单尺度推理和训练。
fuse 方法用于融合卷积层和批量归一化层。
info 方法用于打印模型信息。

DetectionModel 类：
继承自 BaseModel 类，用于构建YOLOv5的检测模型。
__init__ 方法初始化模型，读取配置文件并构建模型结构。
forward 方法支持标准前向传播和增强推理。
_forward_augment 方法用于增强推理，通过不同尺度和翻转进行预测。
_descale_pred 和 _clip_augmented 方法用于处理增强推理后的预测结果。
"""



class Detect(nn.Module):
    """ YOLOv5的检测头部 负责生成边界框和类别预测。"""

    stride = None  # 步长，构建时计算。即特征图相对于输入图像的缩放比例。它在构建时会根据模型的配置进行计算。
    dynamic = False  # 是否强制重建网格。表示是否在每次推理时动态重建网格。默认为 False，即只有在网格尺寸变化时才会重建。
    export = False  # 是否为导出模式。

    """
    初始化操作：
        nc (int): 类别数，默认为 80。（COCO 数据集的类别数）
        anchors (list): 锚点列表，默认为空。用于生成不同尺度的边界框。
        ch (list): 输入通道数列表，默认为空。
        inplace (bool): 是否使用原地操作，默认为 True。
    """
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # 类别数，默认为 80。
        self.no = nc + 5  # 每个锚点的输出数，等于类别数加上 5（4 个坐标值和 1 个置信度）。
        self.nl = len(anchors)  # 检测层的数量，即锚点列表的长度。YOLOv5 通常有多个检测层，每个层负责不同的尺度。
        self.na = len(anchors[0]) // 2  # 每个检测层的锚点数量，等于 anchors[0] 的长度除以 2。因为每个锚点有两个值（宽度和高度），所以需要除以 2。
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # 这个列表用于存储每个检测层的网格信息。网格信息用于将预测的相对坐标转换为绝对坐标。
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # 这个列表用于存储每个检测层的锚点网格信息。锚点网格用于调整预测的边界框大小。
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1,
                                                                           2))  # 注册一个缓冲区（buffer），用于存储不参与梯度计算的张量，形状为 (nl, na, 2) nl 是检测层数，na 是每个检测层的锚点数，2 表示每个锚点的宽度和高度。
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # 卷积层列表，每个检测层对应一个卷积层，输出通道数为 no * na。即每个锚点的输出数乘以锚点数量。卷积核大小为 1x1。
        self.inplace = inplace  # 是否使用原地操作。原地操作可以节省内存，但可能会破坏输入数据。

    """
    forward 方法：
        对输入的 x 进行前向传播，生成检测框和置信度。
        首先，根据输入的 x 的形状，确定输出的形状，并初始化输出张量。
        然后，遍历每个检测层，执行卷积操作，并调整输出张量的形状。
        如果当前层是 YOLOv5 的训练模式，则直接返回输出张量。
        否则，执行推理操作，包括生成锚点网格，计算检测框和置信度，并返回结果。
    """
    def forward(self, x):
        """
        Args:
            x: x (list): 输入特征图列表，每个元素对应一个检测层的特征图。
        Returns:
            如果在训练模式下，返回 x。
            如果在推理模式下且导出模式开启，返回拼接后的检测结果。
            否则，返回拼接后的检测结果和输入特征图。
        """
        # 对每个检测层的特征图进行卷积操作。
        z = []  # 空列表，用于存储每个检测层的推理输出。
        # 遍历每个检测层。
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # 对第 i 个检测层的特征图 x[i] 应用卷积层 self.m[i]，得到卷积后的输出。
            bs, _, ny, nx = x[i].shape  # 获取卷积后特征图的形状，bs 是批量大小，ny 和 nx 分别是特征图的高度和宽度。
            # 将卷积后的特征图重新排列成形状为 (bs, na, no, ny, nx) 的张量。na 是锚点数量，no 是每个锚点的输出数。
            # permute(0, 1, 3, 4, 2)：交换维度顺序，使得最终的形状为 (bs, na, ny, nx, no)。这样可以方便后续的操作
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()  # contiguous()：确保张量在内存中是连续的，避免潜在的性能问题。
            # 推理模式
            if not self.training:  # inference
                # 如果需要动态重建网格 或 当前网格的尺寸与特征图的尺寸不匹配。
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    # 调用 _make_grid 方法生成新的网格和锚点网格。并更新 self.grid[i] 和 self.anchor_grid[i]。
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                # 如果是Segment类的实例，则表示该模型不仅预测边界框，还预测分割掩码（masks）
                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
    """ 生成网格和锚点网格的方法 """
    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        """
        Args:
            nx: nx (int): 网格的宽度，默认为 20。
            ny: ny (int): 网格的高度，默认为 20。
            i: i (int): 检测层的索引，默认为 0。
            torch_1_10: torch_1_10 (bool): 检查是否使用 PyTorch 1.10 及以上版本的meshgrid方法
        Returns:
            grid (Tensor): 网格张量，形状为 (1, na, ny, nx, 2)。
            anchor_grid (Tensor): 锚点网格张量，形状为 (1, na, ny, nx, 2)。
        """
        # 获取当前设备和数据类型。
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        # 定义网格的形状。
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        # 使用 torch.meshgrid 生成网格的 x 和 y 坐标。
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        # 将 x 和 y 坐标堆叠成网格张量，并扩展到指定的形状。
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        # 计算锚点网格，并扩展到指定的形状
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        # 返回生成的网格和锚点网格。
        return grid, anchor_grid



class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])



"""
yolov5的模型的基类，定义了一些通用的方法和属性，适用于所有基于YOLOv5的模型
"""
class BaseModel(nn.Module):
    # YOLOv5 base model

    # 模型的主前向传播方法。它调用了 _forward_once 方法来进行单尺度推理或训练。
    def forward(self, x, profile=False, visualize=False):
        """
        Args:
            x: 输入张量，形状为 [batch_size, channels, height, width] 的图像张量。
            profile：是否启用性能分析（如计算每层的 FLOPs 和推理时间）。
            visualize：是否启用特征图可视化。
        Returns:
        """
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        """ 这是模型的实际前向传播逻辑 """
        y, dt = [], []  # outputs 分别用于存储每一层的输出和性能分析的时间数据
        # 遍历模型的每个模块 m ，执行前向传播操作。
        for m in self.model:
            # # 如果 m.f != -1，则从之前的层获取输入x -> 跳跃连接
            if m.f != -1:
                """ 
                m.f：这是当前模块 m 的一个属性，通常是一个整数或整数列表。它表示了当前模块的输入来自模型中哪些层的输出。
                根据 m.f 的类型，灵活地选择当前模块的输入来源。
                如果 m.f 是一个整数，直接从 y[m.f] 获取输入；
                如果是整数列表，则从多个层获取输入，并将它们组合成一个列表。
                """
                # TODO 理解
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

            # 是否启用性能分析
            if profile:
                self._profile_one_layer(m, x, dt)
            # 每一层的前向传播操作
            x = m(x)  # run
            # m.i：这是当前模块 m 的索引，表示该模块在整个模型中的位置。
            # 通常，m.i 是一个整数，从 0 开始递增，对应于模型中每个模块的顺序。
            y.append(x if m.i in self.save else None)  # 将当前模块的输出添加到 y 中，如果该层需要保存输出。
            # 保存特征图
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        # TODO 为什么不是返回y,而是x
        """
        x：这是当前模块的输入或输出张量。在前向传播过程中，x 会随着每一层的计算不断更新，最终成为模型的最终输出。
        因此，x 包含了经过所有层处理后的特征图或预测结果。
        y：这是一个列表，用于存储每一层的输出（如果该层的输出需要被保存）。具体来说，y 中的元素可能是某个特定层的输出张量，也可能是 None（对于不需要保存的层）。
        y 的主要用途是为后续的跳跃连接（skip connections）或多尺度特征融合提供输入来源。
        """
        return x

    def _profile_one_layer(self, m, x, dt):
        """
        这个方法用于性能分析，记录每个层的 FLOPs（浮点运算次数）和推理时间。
        它通过多次运行该层来获得更准确的时间测量。
        """
        c = m == self.model[-1]  # 检查是否是最后一层，如果是，则复制输入以避免原地修改。
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # 使用 thop.profile 计算该层的 FLOPs。
        t = time_sync()
        # 记录该层的推理时间，通过多次运行该层来获得更准确的测量。
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        # 打印该层的性能信息，包括时间、FLOPs 和参数数量。
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        # 如果是最后一层，打印总的时间。
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        """
        这个方法用于融合卷积层和批归一化层（BatchNorm2d）。融合后的模型在推理时可以减少计算量，提高推理速度。
        """
        # 遍历模型的所有模块，查找 Conv 或 DWConv 层，并检查它们是否有 bn 属性。
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                # 如果存在 bn，则调用 fuse_conv_and_bn 函数融合卷积层和批归一化层。
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                # 删除 bn 属性
                delattr(m, 'bn')  # remove batchnorm
                # 并更新前向传播方法为 forward_fuse
                m.forward = m.forward_fuse  # update forward
        # 调用 info 方法打印融合后的模型信息
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        """ 这个方法用于打印模型的详细信息，包括层数、参数数量、FLOPs 等。可以通过设置 verbose=True 来获得更详细的输出。"""
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # 重写了 nn.Module 的 _apply 方法，用于应用一些变换（如 .to(), .cpu(), .cuda(), .half()）到模型中的张量。
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        # 如果是检测层或者分割层，则更新其 stride、grid 和 anchor_grid。
        if isinstance(m, (Detect, Segment)):
            # 确保其 stride、grid 和 anchor_grid 属性也被正确转换。
            # TODO 需要理解
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


# NOTE: 主要是对整个yolov5的网络结构进行解析，实例化后的model就是yaml文件代表的内容
class DetectionModel(BaseModel):
    # YOLOv5 detection model
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        """
        Args:
            cfg: 模型配置文件的路径或字典。
            ch: ：输入图像的通道数，默认为 3（RGB 图像）。
            nc: 类别数量，如果提供则覆盖配置文件中的值。
            anchors: 锚框（anchors），如果提供则覆盖配置文件中的值。
        """
        super().__init__()

        # 如果是字典，则直接使用该字典作为配置。
        if isinstance(cfg, dict):
            self.yaml = cfg  # 模型配置
        else:  # is *.yaml
            # 如果是路径，则会读取 YAML 文件；
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # 配置模型
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    # YOLOv5 segmentation model
    def __init__(self, cfg='yolov5s-seg.yaml', ch=3, nc=None, anchors=None):
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLOv5 classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml, model, number of classes, cutoff index
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        # Create a YOLOv5 classification model from a YOLOv5 detection model
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        # Create a YOLOv5 classification model from a *.yaml file
        self.model = None




# NOTE: 主要是对backbone和head进行解析，整个yolov5的网络结构这里
def parse_model(d, ch):  # model_dict, input_channels(3)
    """
    Args:
        d: 模型配置文件（yaml 文件）解析后的字典。
        ch: 输入通道数，通常是一个列表，表示每一层的输入通道数。
    Returns: 返回一个 PyTorch 模型（nn.Sequential）和一个保存层的索引列表（save）。
    """
    # Parse a YOLOv5 model.yaml dictionary
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    # 全激活函数
    if act:
        Conv.default_act = eval(act)  # 如果指定了激活函数，则将其设置为默认激活函数, 如 nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # 计算锚点框数量 na
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)  no = 输出通道数 no

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # 遍历模型配置中的 backbone 和 head 部分，逐层解析并构建模型
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args  [输入来源,模块的重复次数,
        # 模块类型, 模块的参数]
        m = eval(m) if isinstance(m, str) else m  # 使用 eval(m) 将字符串形式的模块名称（如 "Conv"）转换为实际的 PyTorch 模块（如 Conv）。
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # 深度缩放因子）用于调整模块的重复次数。
        # 如果模块是卷积类（如 Conv, Bottleneck, C3 等）
        if m in {
                Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x}:
            # 计算输入通道 c1 和输出通道 c2
            c1, c2 = ch[f], args[0]   # ch[f]就是说是ch[-1],上一层的输出， args[0]，参数中的第一个参数，即输出通道数。
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)  # 宽度缩放因子用于调整输出通道数，使模型可以灵活地缩放。

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # 如果是c3模块...， 还需要插入n，即模块的重复次数。
                n = 1
        # 如果模块是 BatchNorm2d，设置输入通道数。
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        # 如果模块是 Concat，计算输出通道数为所有输入通道数的和。
        elif m is Concat:
            c2 = sum(ch[x] for x in f)   # 特征融合
        # 如果模块是 Detect 或 Segment，处理锚点框和输出通道数。
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, 8)  # 使用 make_divisible 函数确保通道数是 8 的倍数，以优化硬件性能。


        # 如果模块是 Contract 或 Expand，调整通道数。
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
        # 根据模块类型和参数实例化模块，并计算参数数量。
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        # save 列表记录了需要保存的层索引，用于后续的特征融合或检测头处理。也就是说方便后续的特征融合
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        # 将每一层添加到 layers 列表中。
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        model.fuse()
